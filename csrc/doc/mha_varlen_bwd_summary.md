# mha_varlen_bwd.cpp 详解

## 1. 文件定位

[mha_varlen_bwd.cpp](../flash_attn_npu/mha_varlen_bwd.cpp) 是 FlashAttention NPU **反向传播（backward）** kernel 的核心实现文件，对应 Python 接口 `varlen_bwd`（支持变长/varlen/packed 输入），位于：

```
csrc/flash_attn_npu/mha_varlen_bwd.cpp
```

它将以下 FAG (FlashAttention Gradient) 组件组装成完整的反向 kernel：

| 组件 | 文件 | 职责 | 运行位置 |
|------|------|------|---------|
| Cube1 (Q*K^T, dOut*V^T) | [fag_mmad_cube1.hpp](../flash_attn_npu/fag_mmad_cube1.hpp) | 重算 S=dOut*V^T？不：S=Q*K^T 和 dP=dOut*V^T | Cube |
| Cube2 (dS*K) | [fag_mmad_cube2.hpp](../flash_attn_npu/fag_mmad_cube2.hpp) | dQ = dS * K | Cube |
| Cube3 (P^T*dOut, dS^T*Q) | [fag_mmad_cube3.hpp](../flash_attn_npu/fag_mmad_cube3.hpp) | dV = P^T*dOut, dK = dS^T*Q | Cube |
| Pre | [fag_epilogue_pre.hpp](../flash_attn_npu/fag_epilogue_pre.hpp) | dq/dk/dv workspace 清零 | Vector |
| Sfmg | [fag_epilogue_sfmg.hpp](../flash_attn_npu/fag_epilogue_sfmg.hpp) | SoftmaxGrad 预归约 D=rowsum(dP*P) | Vector |
| Op | [fag_epilogue_op.hpp](../flash_attn_npu/fag_epilogue_op.hpp) | 重算 P=Softmax(S)，计算 dS=P*(dP-D) | Vector |
| Post | [fag_epilogue_post.hpp](../flash_attn_npu/fag_epilogue_post.hpp) | scale+cast+写回 dq/dk/dv | Vector |
| Cube 地址生成 | [fag_common/cube_addr.h](../flash_attn_npu/fag_common/cube_addr.h) | 多核任务分配+GM地址映射 | Cube |
| Vector 地址生成 | [fag_common/vector_addr.h](../flash_attn_npu/fag_common/vector_addr.h) | 多核任务分配+子块地址映射 | Vector |
| 公共定义 | [fag_common/common_header.h](../flash_attn_npu/fag_common/common_header.h) | 常量/枚举/AddrInfo 结构体/tiling 索引 | — |

---

## 2. 反向数学公式

FlashAttention Backward 的核心计算（与前向共享 Q/K/V/O/dOut，输出 dQ/dK/dV）：

```
S = scale * Q * K^T                  // 重算 attention scores
P = softmax(mask(S))                 // 重算 attention probabilities
dP = dOut * V^T                      // dO 对 V 方向的中间梯度
D = rowsum(dP * P)                   // softmax backward 辅助项（逐行广播）
dS = P * (dP - D)                    // softmax 输入梯度
dQ = dS * K                          // Q 梯度
dV = P^T * dOut                      // V 梯度
dK = dS^T * Q                        // K 梯度
```

关键设计：**反向中重算 forward**（recomputation），不保存前向的 S 和 P 矩阵，节省显存。

---

## 3. 架构：Cube + Vector 双异构核协作

与前向 kernel（mha_fwd_kvcache.cpp）类似，反向 kernel 也使用 Atlas A2 的双异构核架构，但组件更复杂：

```
            同一个 AI Core (MIX_AIC_1_2: 1 Cube + 2 Vector sub-core)
┌────────────────────────────────────────────────┐
│  Cube 核              │  Vector 子核 (×2)       │
│                       │                         │
│  Cube1: Q*K^T → S    │  Pre: 清零workspace     │
│         dOut*V^T→dP──┼──►CUBE2VEC──►Op:重算P  │
│                       │            dS=P*(dP-D) │
│  Cube2: dS*K → dQ ◄──┼──VEC2CUBE◄──            │
│  Cube3: P^T*dOut→dV   │  Sfmg: D=rowsum(dP*P)  │
│         dS^T*Q → dK   │  Post: scale/cast/写回  │
│         ──CUBE2POST──►│                         │
└────────────────────────────────────────────────┘
```

- **跨核事件**：`CUBE2VEC(ID=7)`、`VEC2CUBE(ID=8)`、`CUBE2POST(ID=9)`
- **双缓冲**：CubeAddrInfo/VecAddrInfo 数组大小=2，通过 `taskId%2` 实现 ping-pong
- **核比例**：`KERNEL_TYPE_MIX_AIC_1_2` 声明 1 Cube : 2 Vector sub-core
- **Vector coreId 对齐**：Vector 侧 `coreId = GetBlockIdx()/2` 映射到 Cube 逻辑核

---

## 4. 反向与前向的关键区别

| 方面 | 前向 (mha_fwd_kvcache.cpp) | 反向 (mha_varlen_bwd.cpp) |
|------|--------------------------|--------------------------|
| Cube matmul 数量 | 2（QK 和 PV） | 3（Cube1 两次，Cube2 一次，Cube3 两次） |
| Vector epilogue 数量 | 3（InitOut/OnlineSoftmax/RescaleO） | 4（Pre/Sfmg/Op/Post） |
| 跨核同步信号 | 3 个（qkReady/softmaxReady/pvReady） | 3 个（CUBE2VEC/VEC2CUBE/CUBE2POST） |
| Q 加载方式 | 循环外 loadQGM 一次性加载 | 每个 taskId 由 CubeAddr 管理，Cube1 每次调用处理 |
| workspace 分区 | S/P/OTmp/OUpdate 4 区域 | dq/dk/dv/p/ds/mm1/mm2 7 个区域 |
| 输出写回 | O 直接写 GM（rescale 末块归一化） | dq/dk/dv fp32 workspace atomic add，Post 统一 cast+写回 |
| 流水线结构 | PRE_LAUNCH=2 三缓冲深度流水，KV循环外发射 | taskId 双缓冲 ping-pong，Cube1生产、Cube2/3消费上一轮 |
| Softmax 计算 | 在线 softmax（边算边累加） | 先 Sfmg 预归约 D，再 Op 重算 P 并计算 dS |

---

## 5. 文件结构

```
mha_varlen_bwd.cpp
├── #include 头文件
├── namespace FAG {
│   ├── class FAGKernel<7个模板参数>
│   │   ├── 类型别名（ElementA1/A2/A3... Layout...）
│   │   ├── struct Params（23个GM_ADDR成员）
│   │   ├── operator()(params)
│   │   │   ├── #ifdef __DAV_C220_CUBE__: Cube侧代码
│   │   │   │   ├── tiling解析 + workspace偏移
│   │   │   │   ├── TND/BSND布局判断
│   │   │   │   ├── CubeAddr初始化
│   │   │   │   ├── 构造3个BlockMmad
│   │   │   │   └── while(running)主循环
│   │   │   │       ├── addr_mapping → CubeAddrInfo[taskId%2]
│   │   │   │       ├── SetFlag灌泡
│   │   │   │       ├── Cube1(Q,K^T)→S; Cube1(dOut,V^T)→dP
│   │   │   │       ├── WaitFlag+CUBE2VEC通知
│   │   │   │       ├── if(taskId>0): WaitEvent(VEC2CUBE)
│   │   │   │       │   ├── Cube2(dS,K)→dQ
│   │   │   │       │   ├── Cube3(P^T,dOut)→dV
│   │   │   │       │   └── Cube3(dS^T,Q)→dK
│   │   │   │       └── CUBE2POST通知结束
│   │   │   └── #ifdef __DAV_C220_VEC__: Vector侧代码
│   │   │       ├── tiling解析
│   │   │       ├── Pre（清零）──Destroy
│   │   │       ├── Sfmg（预归约D）──Destroy
│   │   │       ├── SyncAll
│   │   │       ├── Op循环（while(running)）
│   │   │       │   ├── addr_mapping → VecAddrInfo
│   │   │       │   ├── WaitEvent(CUBE2VEC)
│   │   │       │   ├── epilogueFagOp（重算P+dS）
│   │   │       │   └── VEC2CUBE通知
│   │   │       ├── WaitEvent(CUBE2POST)
│   │   │       ├── SyncAll
│   │   │       └── Post（scale+cast+写回）
│   │   ├── SetFlag() / WaitFlag() 核内同步辅助
│   │   └── Arch::Resource resource
│   └── }
└── namespace FAG {
    └── __global__ void FAG<...>(...)
        ├── SetSyncBaseAddr + KERNEL_TASK_TYPE_DEFAULT
        ├── 组装Cube1/Cube2/Cube3 Gemm类型
        ├── 组装Pre/Sfmg/Op/Post Epilogue类型
        ├── 构造Params + FAGKernel实例
        └── 调用fag(params)
    }
```

---

## 6. 核心流程详解

### 6.1 双缓冲 Ping-Pong 流水

Cube 侧主循环是整个反向 kernel 的调度中枢：

```
taskId=0:
  addr_mapping→slot0
  Cube1(Q*K^T→S_slot0, dOut*V^T→dP_slot0) → CUBE2VEC
  Vector: Wait(CUBE2VEC) → Op(P,dS_slot0) → VEC2CUBE
  Cube侧 (taskId>0为false, 不做Cube2/3)

taskId=1:
  addr_mapping→slot1
  Cube1(Q*K^T→S_slot1, dOut*V^T→dP_slot1) → CUBE2VEC
  Vector: Wait(CUBE2VEC) → Op(P,dS_slot1) → VEC2CUBE
  Cube侧: Wait(VEC2CUBE from taskId=0)
          Cube2(dS_slot0*K→dQ)
          Cube3(P_slot0^T*dOut→dV)
          Cube3(dS_slot0^T*Q→dK)

taskId=2:
  addr_mapping→slot0（覆盖slot0，已被Cube2/3消费完毕）
  Cube1(...)→slot0 → CUBE2VEC
  Cube侧: Wait(VEC2CUBE from taskId=1)
          Cube2(dS_slot1*K→dQ)
          Cube3(P_slot1^T*dOut→dV)
          Cube3(dS_slot1^T*Q→dK)
...
```

### 6.2 Workspace 内存布局

```
workspace 指针
  │
  ├── TILING_P_WORKSPACE_OFFSET(=0):        P 矩阵（softmax概率，half）
  ├── TILING_DS_WORKSPACE_OFFSET(=1):       dS 矩阵（softmax输入梯度，half）
  ├── TILING_MM1_WORKSPACE_OFFSET:         dP = dOut*V^T（float）
  ├── TILING_MM2_WORKSPACE_OFFSET:         S = Q*K^T（float）
  ├── TILING_DQ_WORKSPACE_OFFSET:          dQ 累加区（fp32 atomic add）
  ├── TILING_DK_WORKSPACE_OFFSET:          dK 累加区（fp32 atomic add）
  ├── TILING_DV_WORKSPACE_OFFSET:          dV 累加区（fp32 atomic add）
  └── TILING_SFMG_WORKSPACE_OFFSET:        Sfmg D 辅助项
```

dQ/dK/dV 使用 fp32 累加是因为不同 K-block（对 dQ/dK）或 Q-block（对 dV）会通过 atomic add 累加到同一个输出位置。

### 6.3 Vector 侧四阶段

Vector 侧不是在一个大循环里完成所有工作，而是分成**4个顺序阶段**，每个阶段创建独立的 `TPipe` 并在结束后 `Destroy()`：

```
Pre ──► Sfmg ──► SyncAll ──► Op循环(与Cube流水) ──► Wait(CUBE2POST) ──► SyncAll ──► Post
清零    预归约D              重算P+dS（核心循环）    等Cube完成所有dQ/dK/dV    写回
```

- **Pre/Sfmg**：在 Op 循环之前执行，不参与跨核流水
- **Op**：核心循环，通过 CUBE2VEC/VEC2CUBE 与 Cube 流水
- **Post**：在所有 Cube 任务完成后执行，将 fp32 累加结果乘 scale 后 cast 到 half/bf16 写回 GM

### 6.4 Tiling 索引关键字段

从 `fag_common/common_header.h`：

| 索引 | 含义 |
|------|------|
| TILING_B(=5) | batch size |
| TILING_G(=9) | GQA group size g |
| TILING_N2(=8) | KV 头数 |
| TILING_D(=10) | head_dim |
| TILING_T1(=6)/T2(=7) | total Q/KV tokens |
| TILING_CORE_NUM(=3) | coreNum |
| TILING_DQ/DK/DV_WORKSPACE_OFFSET | dq/dk/dv 累加区偏移 |
| TILING_MM1/MM2_WORKSPACE_OFFSET | dP/S 中间结果偏移 |
| CONST_2(=2) | u64/u32 索引加倍因子 |

---

## 7. 例子 1：Prefill 训练反向（BSND + MHA + Causal）

**配置**：batch=2, seq_q=seq_k=1024(prefill), qHeads=kvHeads=32(MHA, g=1), headdim=128, causal, 8 AI Cores

**Tiling**：
- nheads=32, nheads_k=32, g=1
- 每个 128x128 子块为基本调度单元
- Q 方向 8 块（1024/128），K 方向 8 块，32 heads × 2 batch = 512 个 tile row
- 每轮 addr_mapping 最多生成 16 个 128x128 子块
- 总轮次约 (512×8/2)/16 ≈ 128 轮（causal 跳过上三角后约一半）

**流水时序**（前几轮）：

```
轮次(taskId)  →  0    1    2    3    4   ... 最后
Cube1(S,dP): [1]  [2]  [0]  [1]  [0] ...  [0]  (ping-pong slot)
Vec(Op):         [1]  [0]  [1]  [0]  [1] ...  [0]
Cube2/3(dQ,dV,dK):    [0]  [1]  [0]  [1] ...  [1]
                  发射  稳态 ←─────────────→ 排空
```

**Causal 下子块分配**：
```
       K: 0   1   2   3   4   5   6   7
Q=0: [  ✓ , ✗ , ✗ , ✗ , ✗ , ✗ , ✗ , ✗ ]  仅1块有效
Q=1: [ ✓ , ✓ , ✗ , ✗ , ✗ , ✗ , ✗ , ✗ ]  2块
Q=2: [ ✓ , ✓ , ✓ , ✗ , ✗ , ✗ , ✗ , ✗ ]  3块
...
Q=7: [ ✓ , ✓ , ✓ , ✓ , ✓ , ✓ , ✓ , ✓ ]  8块（全有效，fast path）
```
- `upperRight=true` 的子块（跨越对角线）需要 mask
- 完全在对角线下的块走无 mask 快速路径
- CubeAddr 和 VectorAddr 在 `addr_mapping` 中自动根据 causal 约束过滤子块

**dQ/dK/dV atomic add**：
- dQ 形状 [2,1024,32,128]：每个 Q 行需要累加所有 K block 的 dS*K 贡献，通过 dqWorkSpace fp32 原子加
- dK 形状 [2,1024,32,128]：每个 K 行需要累加所有 Q block 的 dS^T*Q 贡献，通过 dkWorkSpace fp32 原子加
- dV 形状 [2,1024,32,128]：每个 V 行需要累加所有 Q block 的 P^T*dOut 贡献，通过 dvWorkSpace fp32 原子加

---

## 8. 例子 2：变长 Decode+Prefill 混合批次（TND + GQA + 无 mask）

**配置**：batch=4(packed TND), 请求序列长度不同（prefill 512 + 3个decode 1），qHeads=64, kvHeads=8(GQA, g=8), headdim=128, 非causal, 8 AI Cores

**TND 布局处理**：
- cu_seq_qlen=[0, 512, 513, 514, 515]（cumulative）
- Vector 侧：`actucal_seq_q_addr = (int32*)cu_seq_qlen + 1`，跳过首个 0
- 每个 batch 的真实 qSeqlen 通过前缀差分获得：
  - batch0: 512-0=512
  - batch1: 513-512=1
  - batch2: 514-513=1
  - batch3: 515-514=1

**GQA 处理**：
- nheads=64, nheads_k=8, g=8
- 同一 KV head 对应 8 个 Q heads，在 addr_mapping 中通过 nheadsIdx/g 映射到 KV head

**任务划分**：
- batch0（prefill 512 tokens）：qSBlockNum=4（512/128），每KVhead对应8个Qhead组×8KVheads=64块→ 4×64=256 tiles
- batch1-3（decode 1 token each）：qSBlockNum=1，每个 batch 64 tiles → 3×64=192 tiles
- 总 tiles ≈ 448，每轮16个，约28轮

**流水时序**（与例1类似，但混合了长短序列）：

```
轮次 0-15:  主要处理batch0长序列（128行/块 × 多KV块）
轮次 16-27: 处理batch1/2/3短序列（1行/块）
最后一轮:    blockLength=0 → running=false
```

decode 块只有 1 行 Q（ky=1），Cube GEMM 会处理尾块；Vector 侧 VecBlockInfo 中 lengthy=1 标记实际长度。

---

## 9. 数据流/结构图

### 9.1 整体执行流程图

```
Vector core 侧:                              Cube core 侧:
─────────────────                            ─────────────────────────
SetSyncBaseAddr(fftsAddr)
KERNEL_TYPE_MIX_AIC_1_2
│
├─ Pre(TPipe1): 清零dq/dk/dv workspace
│    Destroy()
│
├─ Sfmg(TPipe2): 读dout+out
│    逐行计算 D=rowsum(dout*out)
│    → sfmg_workspace
│    Destroy()
│
├─ SyncAll()  ◄───────────────────────────────────┐
│                                                 │
├─ Op(TPipe3) while(running):                     │ while(running):
│    WaitEvent(CUBE2VEC) ◄─ CUBE2VEC ──┐          │  addr_mapping(slot[taskId%2])
│    vec_addr.addr_mapping              │          │  SetFlag() (灌泡)
│    epilogueFagOp(vecAddrInfo):        │          │  Cube1(Q,K^T)→S[slot]
│      SubGrapA: S*scale→P=Softmax     │          │  Cube1(dOut,V^T)→dP[slot]
│      SubGrapB: dS=P*(dP-D)           │          │  WaitFlag()
│    CrossCoreSetFlag(VEC2CUBE) ──VEC2CUBE──►     │  CrossCoreSetFlag(CUBE2VEC)
│    running=(blockLen>0)              │          │
│                                      │          │  WaitEvent(VEC2CUBE) ◄──┘
│                                      └──────────┤  if taskId>0:
│                                                 │    Cube2(dS,K)→dQ (atomic)
│                                                 │    WaitFlag+SetFlag
│                                                 │    Cube3(P^T,dOut)→dV (atomic)
│                                                 │    WaitFlag+SetFlag
│                                                 │    Cube3(dS^T,Q)→dK (atomic)
│                                                 │    WaitFlag
│                                                 │  running=(blockLen>0)
│                                                 │  taskId++
│                                                 │
├─ WaitEvent(CUBE2POST) ◄─ CUBE2POST ────────────── CrossCoreSetFlag(CUBE2POST)
│
├─ SyncAll()
│
└─ Post(TPipe4): 读dq/dk/dv workspace
     dq *= scaleValue (head_dim^-0.5 对反向也需要)
     fp32 → half/bf16 cast
     写回 GM dq/dk/dv
     Destroy()
```

### 9.2 三个 Cube Matmul 的形状与转置关系

```
Cube1 (A=RowMajor, B=ColumnMajor，即 B 转置):
  S  = Q * K^T     形状: [row(M), embed(K)] × [embed(K), kx(N)] → [row(M), kx(N)]
  dP = dOut * V^T  形状: [row(M), embedV(K)] × [embedV(K), kx(N)] → [row(M), kx(N)]
  L1TileShape = GemmShape<256, 128, 256>

Cube2 (A=RowMajor, B=RowMajor，不转置):
  dQ = dS * K      形状: [row(M), kx(K)] × [kx(K), embed(N)] → [row(M), embed(N)]
  L1TileShape = GemmShape<128, 128, 128>

Cube3 (A=ColumnMajor 即A转置, B=RowMajor 不转置):
  dV = P^T * dOut  形状: [kx(N), row(K)] × [row(K), embedV(M)] → [kx(N), embedV(M)]
  dK = dS^T * Q    形状: [kx(N), row(K)] × [row(K), embed(M)] → [kx(N), embed(M)]
  L1TileShape = GemmShape<256, 128, 256>
```

### 9.3 Workspace 各区域大小关系

```
每个 slot (ping/pong):
┌────────────┬────────────┬─────────────┐
│ S (mm2)    │ dP (mm1)   │ P/ dS 共享  │
│ 128*? x128 │ 128*? x128 │ (在固定偏移) │
│ float32    │ float32    │ half/fp16    │
└────────────┴────────────┴─────────────┘
  ↑ Cube1 写    ↑ Cube1 写    ↑ Vec_Op 写

跨 slot 共享（fp32 累加区，不 ping-pong）:
┌────────────┬────────────┬─────────────┐
│ dQ (fp32)  │ dK (fp32)  │ dV (fp32)   │
│ atomic add │ atomic add │ atomic add  │
│ 全部输出   │ 全部输出   │ 全部输出     │
└────────────┴────────────┴─────────────┘
  ↑ Cube2 写    ↑ Cube3(2nd)写 ↑ Cube3(1st)写
  ↑ Vec_Post 读并cast写回
```

### 9.4 与前向 kernel 流水线对比

```
前向（mha_fwd_kvcache.cpp）:
  Cube(QK)──qkReady──►Vec(SM)──softmaxReady──►Cube(PV)──pvReady──►Vec(RescaleO)
  PRE_LAUNCH=2 三缓冲深度流水（3个slot）

反向（mha_varlen_bwd.cpp）:
  阶段1: Vec Pre/Sfmg 顺序执行（不流水）
  阶段2: Cube1(S,dP)──CUBE2VEC──►Vec(Op:P+dS)──VEC2CUBE──►Cube2(dQ)+Cube3(dV,dK)
         taskId%2 双缓冲 ping-pong（2个slot）
  阶段3: Vec Post 执行（不流水，等所有Cube完成）
```

---

## 10. 关键设计要点

1. **重算 Forward（Activation Recomputation）**：反向不保存前向的 S 和 P，而是通过 Cube1 重新计算 S=Q*K^T，Vec_Op 重算 P=Softmax(S)。这是 FlashAttention 的标准反向策略，用算力换显存。
2. **fp32 累加 + atomic add**：dQ/dK/dV 先在 fp32 workspace 中原子加，最后由 Post epilogue 统一 cast+scale 写回 half/bf16，避免精度损失。
3. **Sfmg 预归约**：Sfmg epilogue 在 Op 循环前逐行计算 D=rowsum(dout*out)，利用前向输出 O 近似计算辅助项（减少 Op 中的归约开销）。
4. **双缓冲而非三缓冲**：反向使用 `taskId%2` 双缓冲（2 个 slot），因为生产-消费链是线性的（Cube1→Vec→Cube2/3），只需要 2 个 slot 覆盖 Cube1 生产和 Cube2/3 消费的重叠。
5. **GQA 支持**：通过 `nheadsIdx/g` 映射 Q head 到 KV head，addr_mapping 中 nheadsKIdx 正确路由 K/V 地址。
6. **Vector sub-core 对半拆分**：Vector 侧 `coreId = GetBlockIdx()/2` 将 2 个 sub-core 映射到同一逻辑核，VectorAddr 内部处理子块对半分配。
7. **TPipe 分阶段管理**：Pre/Sfmg/Op/Post 各用独立 TPipe，Destroy 后释放 UB 空间，保证下一阶段有足够 UB。
8. **AddrInfo 最多 16 个子块**：每轮 addr_mapping 最多生成 16 个 128x128 子块，控制 workspace per-slot 大小和 GEMM 调度粒度。
9. **kernel 类型 MIX_AIC_1_2**：宏声明 1 Cube : 2 Vector sub-core 的混合核类型，由 AscendC runtime 按此比例分配物理核。
10. **中间精度固定 float**：反向所有中间 matmul 输出均为 float（C1=C2=C3=float），不提供 low_precision 版本，与前向有 high/low prec 两个版本不同。

---

## 11. 注意点

1. `cubeAddrInfo[2]` 是双缓冲数组，`cubeAddrInfo[taskId%2]` 是写 slot（Cube1 写），`cubeAddrInfo[(taskId-1)%2]` 是读 slot（Cube2/3 读），两者不能搞错。
2. Cube1 连续调用两次（S 和 dP），共享同一个 BlockMmadFAGCube1 实例，通过传入不同的 A/B/C 指针计算不同的矩阵乘。
3. Cube3 也连续调用两次（dV 和 dK），因为 dV=P^T*dOut 和 dK=dS^T*Q 的矩阵布局相同（A转置B不转置），复用同一个 GEMM 模板。
4. Vec 侧 `GetBlockIdx()/2` 是 Vector sub-core 到逻辑 core 的映射；Cube 侧直接用 `GetBlockIdx()`。
5. TND 布局下 cu_seq_qlen/kvlen 第一个元素是 0，需要 `+1` 偏移跳过。
6. CUBE2POST 必须在 Cube 循环结束后（running=false）才发送，Vec Post 等此信号后才开始，确保 dQ/dK/dV fp32 累加全部完成。
7. Sfmg 阶段读取的是前向输出 `out`（即 O=P*V），不是 P 本身。这是一种优化：利用已有的 O 计算 D 的近似值，避免额外 workspace 占用。
8. 反向 `scaleValue` 在 Post 阶段才对 dQ/dK 乘（因为 dS 计算中的 scale 已在 S*scale 阶段应用，但对 dK 需要再乘一次）。
9. 当前反向不支持 dropout（drop_mask 参数预留但未使用）。
10. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 必须在 kernel 入口最开始调用，设置正确的 Cube:Vector 核比例。

---

## 12. 总结

[mha_varlen_bwd.cpp](../flash_attn_npu/mha_varlen_bwd.cpp) 是 FlashAttention NPU 反向传播的"总指挥"文件：

- **3 个 Cube 矩阵乘**（Cube1 双次调用: S/dP, Cube2: dQ, Cube3 双次调用: dV/dK）完成所有矩阵运算；
- **4 个 Vector epilogue**（Pre 清零 / Sfmg 预归约 / Op 重算P+dS / Post 写回）分阶段完成 softmax gradient 计算；
- **3 个跨核信号**（CUBE2VEC/VEC2CUBE/CUBE2POST）驱动 Cube↔Vector 生产消费流水；
- **双缓冲 ping-pong**（taskId%2）让 Cube1 生产和 Cube2/3 消费在不同 slot 上并行；
- **fp32 atomic add 累加区**保证 dQ/dK/dV 在多 block 累加时的精度；
- **重算 forward**策略避免保存 S/P 矩阵，大幅节省显存；
- 支持 **MHA/GQA/MQA**（通过 g 参数）、**BSND/TND**两种布局、**causal/no-mask**两种 mask 类型，共 8 种模板实例。
