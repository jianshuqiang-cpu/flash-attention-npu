# kernel_common.hpp 详解

## 1. 文件定位

[kernel_common.hpp](../flash_attn_npu/kernel_common.hpp) 是 FlashAttention NPU **前向推理** kernel 的公共定义头文件，位于：

```
csrc/flash_attn_npu/kernel_common.hpp
```

它被以下核心文件 `#include`：

- [mha_fwd_kvcache.cpp](../flash_attn_npu/mha_fwd_kvcache.cpp) — 前向 kernel 主实现，包含 `FAInferKernel` 类与全局 kernel 入口 `FAInfer`
- [flash_api.cpp](../flash_attn_npu/flash_api.cpp) — Host 侧 pybind 入口（也维护了一份 `GetQNBlockTile`/`GetQSBlockTile` 的等价副本）
- 以及被 epilogue、matmul 等子模块间接使用（通过 `mha_fwd_kvcache.cpp` 中引入）

它**不包含任何可执行计算逻辑**，只提供全前向路径共享的常量、枚举、POD 结构体和两个 tile 计算工具函数。

---

## 2. 为什么需要这个文件？

FlashAttention-NPU 前向 kernel 涉及：
- 双异构核（Cube 核做矩阵乘、Vector 核做 softmax/epilogue）；
- 多段流水线（QK → Softmax → PV → RescaleO）；
- 多种配置（causal/非 causal、MHA/GQA/MQA、BSND/TND 布局、paged KV cache）；
- L1/UB/GM 多级内存预算；
- Host 端 tiling 与 Device 端 kernel 必须使用**完全一致**的常量和分块规则。

`kernel_common.hpp` 将这些"全局配置"集中在一个轻量头文件里，确保 host 与 device 之间、Cube 核与 Vector 核之间看到的 tile 大小、同步 ID、枚举值完全一致，避免 magic number 散落各处造成不匹配。

---

## 3. 内容全景

文件内容可划分为四组：

| 分组 | 内容 | 关键标识符 |
|------|------|-----------|
| 同步与流水 | 跨核事件 ID、预发射深度 | `QK_READY_ID`、`SOFTMAX_READY_ID`、`PV_READY_ID`、`PRE_LAUNCH` |
| Tile/内存常量 | Q/K/V tile 尺寸、L1 容量、workspace 大小、mask 尺寸、对齐常量 | `Q_TILE_CEIL`、`MAX_KV_STACK_LEN`、`WORKSPACE_BLOCK_SIZE_DB`、`L1_MAX_SIZE`、`L1_MAX_N_NUM`、`DOUBLE_BUFFER`、`COMP_TRIU_MASK_DIM_LEN`、`N_SPLIT_HELPER`、`NUM_32/128/256` |
| 枚举类型 | 流水线类型、Mask 类型、输入布局 | `cvPipeLineType`、`MaskType`、`inputLayout`（在 `FaiKenel` 命名空间内） |
| 结构体/工具 | 参数打包结构体、tile 计算函数、通用模板 | `FAIKernelParams`、`GetQNBlockTile`、`GetQSBlockTile`、`AlignUp`/`Max`、`FaiKenel::BLOCK_SIZE` |

---

## 4. 常量逐组详解

### 4.1 跨核同步事件 ID（CrossCoreFlag）

```cpp
constexpr uint32_t QK_READY_ID     = 1;   // Cube 完成 Q*K^T
constexpr uint32_t SOFTMAX_READY_ID = 2;  // Vector 完成 softmax/mask
constexpr uint32_t PV_READY_ID     = 3;   // Cube 完成 P*V
```

Ascend C220 的 Cube 核与 Vector 核是**物理分离**的两种计算核，通过 `Arch::CrossCoreFlag` 做核间事件同步。三个 ID 构成三段式 producer-consumer 同步链：

```
  Cube 核                          Vector 核
┌─────────────┐   QK_READY(1)    ┌────────────────┐
│ Q*K^T matmul│ ───────────────> │ softmax + mask │
└─────────────┘                  └────────────────┘
                                         │
                              SOFTMAX_READY(2)
                                         v
┌─────────────┐   PV_READY(3)    ┌────────────────┐
│ P*V matmul  │ <─────────────── │ (等待 PV 完成)  │
└─────────────┘                  └────────────────┘
       │
       └─────────────────────────> online softmax rescale/accum
```

- Cube 在 Q*K^T 完成后 `CrossCoreSetFlag(qkReady)`；Vector 上的 softmax `CrossCoreWaitFlag(qkReady)`。
- Vector 在 softmax 写出 P 矩阵后 `CrossCoreSetFlag(softmaxReady)`；Cube 上的 PV matmul 据此开始。
- Cube 在 P*V 完成后 `CrossCoreSetFlag(pvReady)`；Vector 上的 RescaleO epilogue 据此做 O 累加更新。

三个 ID 互异，避免信号冲突。

### 4.2 深度流水参数 PRE_LAUNCH

```cpp
constexpr uint32_t PRE_LAUNCH = 2;
```

KV 外层循环的**预发射深度**。值为 2 意味着流水线上提前 2 个 KV stack tile 启动 QK matmul，总缓冲槽数 = `PRE_LAUNCH + 1 = 3`（三缓冲）。通过 `stackSeqCount % 3` 计算当前槽位，让 QK/Softmax/PV 三段在时间上深度重叠：

```
时间 →
KV tile 0: [QK][Softmax][PV     ]
KV tile 1:     [QK    ][Softmax][PV     ]
KV tile 2:         [QK ][Softmax][PV     ]   ← 三段完全重叠
KV tile 3:             [QK     ][Softmax][PV]
...
```

PV 读数据时使用 `(stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1)` 落后 2 个槽位读取。

### 4.3 Query Tile 与 Workspace

```cpp
constexpr uint32_t Q_TILE_CEIL          = 128;                      // Q 序列 M 维 tile = 128 行
constexpr uint32_t MAX_KV_STACK_LEN     = 512;                      // K/V stack tile 上限
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;  // 128*512 = 65536 元素
```

- `Q_TILE_CEIL=128` 是整个前向 kernel 最核心的 tile 常量：
  - `GetQSBlockTile` 直接返回 128（Q 序列方向每块 128 行）；
  - Cube GEMM L1 tile shape 为 `GemmShape<128, 128, 128>`，M 维固定 128；
  - Q 矩阵 L1 占用 = `128 * kDynNum * sizeof(ElementQ)`；
  - workspace 单块行数 = 128。
- `MAX_KV_STACK_LEN=512`：外层 KV 循环每次处理最多 512 个 K/V token；用于 KV 分块数计算 `CeilDiv(kvS, 512)`、V 矩阵 L1 预留、causal mask 快速路径判断。
- `WORKSPACE_BLOCK_SIZE_DB=65536`：单核单缓冲的 workspace 元素数，内部再划分 S（QK 分）、P（softmax 概率）、OTmp（PV 输出）、Update（online softmax 中间量）四个区域，实际 GM workspace 总量 = `blockDim × 此值 × sizeof(T) × (PRE_LAUNCH+1) × 4`。

### 4.4 L1 缓存相关

```cpp
constexpr uint32_t L1_MAX_SIZE   = 524288;  // 512 KB
constexpr uint32_t L1_MAX_N_NUM  = 128;     // N 维硬件上限
constexpr uint32_t DOUBLE_BUFFER = 2;       // V 双缓冲
```

L1 预算流程：
1. 先为 V 矩阵预留 `embedV × MAX_KV_STACK_LEN × sizeof(ElementV) × DOUBLE_BUFFER`；
2. 剩余空间分配给 Q 和 K；
3. 动态 N 维大小 `nDynNum = min(由内存预算反推值, L1_MAX_N_NUM)`，并向下对齐到 32 的倍数；
4. K 维 `kDynNum` 不低于 256。

### 4.5 Causal Mask

```cpp
constexpr uint32_t COMP_TRIU_MASK_DIM_LEN = 2048;
```

Host 侧会在 `is_causal=true` 时预生成一个 **2048×2048 uint8** 上三角矩阵（`at::triu(ones, 1)`，严格上三角=1，其余=0），整体拷贝到 NPU GM。Device 侧根据当前 Q/K tile 的坐标 `triUp`/`kvSStartIdx` 计算子窗口偏移，从这个大 mask 中切出当前 tile 所需的小块搬到 UB，转成 float 后在 softmax 前以加法 mask 形式将上三角位置置为 -inf。2048 表示**最大支持序列长度**。

### 4.6 对齐常量

| 常量 | 值 | 作用 |
|------|----|------|
| `NUM_32`  | 32  | Ascend C220 32 元素处理粒度（nDynNum 对齐、V 空间对齐） |
| `NUM_128` | 128 | embed 维度向上 RoundUp 对齐 |
| `NUM_256` | 256 | K 维 L1 tile 大小下限 |
| `FaiKenel::BLOCK_SIZE` | 16 | embed/row 数据搬运对齐单元（half 下 16 元素=32B） |
| `N_SPLIT_HELPER` | 2 | qNBlockTile 对齐到 2 的倍数，便于 sub-core 对半拆分 |

### 4.7 通用工具模板

```cpp
template<typename T> T AlignUp(T a, T b) { return (b==0)?0:(a+b-1)/b*b; }
template<typename T> T Max(T a, T b)     { return (a>b)?a:b; }
```

这两个是通用工具，但**当前 device 代码主要使用 AscendC 内置的 `RoundUp`/`CeilDiv` 和 `AscendC::Max`**，此处保留作为备用。

---

## 5. 枚举类型

所有枚举都在 `namespace KernelCommon::FaiKenel` 内，底层类型均为 `uint32_t`，方便直接作为模板参数传递。

### 5.1 MaskType

```cpp
enum class MaskType : uint32_t { NO_MASK=0, MASK_CAUSAL=1, MASK_SPEC=2 };
```

作为 `FAInferKernel` 类的模板参数 `MASK_TYPE`，通过 `if constexpr` 在编译期裁剪 mask 代码分支。Host 侧在所有 kernel 启动点根据 `is_causal` 选择 `MASK_CAUSAL` 或 `NO_MASK`。

> ⚠️ 反向路径 FAG 在 [fag_common/common_header.h](../flash_attn_npu/fag_common/common_header.h) 中有一套独立定义的同名枚举（只有 NO_MASK 和 MASK_CAUSAL 两个值，没有 MASK_SPEC），不要混用。

### 5.2 inputLayout

```cpp
enum class inputLayout : uint32_t { BSND=0, TND=1 };
```

作为 `FAInferKernel` 模板参数 `INPUT_LAYOUT`：
- **BSND**（Batch-Seqlen-Head-Dim）：标准 4D 布局，Q/K/V 为 `[B, S, N, D]`，qSeqlen/kvSeqlen 直接从 tiling 读取；用于定长/带 padding 的训练场景。
- **TND**（Token-Head-Dim）：packed 3D 布局 `[T, N, D]`，需通过 `actualQseqlen`（cumulative seqlen 数组）做前缀差分计算每个 batch 的真实序列长度；用于 variable-length 推理，避免 padding 浪费算力。

### 5.3 cvPipeLineType（预留）

```cpp
enum class cvPipeLineType : uint32_t { FAI_COMMON_NORMAL=0, FAI_COMMON_CHUNK_MASK=1 };
```

当前**未被实际使用**，为未来 chunk-mask 等特殊流水线模式预留。

---

## 6. FAIKernelParams 结构体

```cpp
struct FAIKernelParams {
    GM_ADDR q, k, v, mask, blockTables,
            actualQseqlen, actualKvseqlen,
            o, lse, workSpace, tiling;
};
```

这是 Device 侧 kernel 的**参数打包 POD**。全局 kernel 入口 `FAInfer` 为了符合 Ascend C kernel launch ABI，参数是 12 个独立的 `GM_ADDR`（即 `uint64_t` 全局地址）：

```cpp
__global__ void FAInfer(uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
    GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR workspace, GM_ADDR tiling);
```

在入口函数内部，除了 `fftsAddr` 外的 11 个地址被打包进 `FAIKernelParams`，再传给 `FAInferKernel::operator()(const FAIKernelParams&)` 使用。进入 operator() 后：

1. `params.tiling` 强转为 `__gm__ FAInferTilingData*`，读取所有标量分块参数；
2. `SetGlobalBuffer` 将 q/k/v/mask/blockTables/actualQseqlen/actualKvseqlen/o/lse 绑定到对应 `GlobalTensor`；
3. `params.workSpace` 按 tiling 中计算的大小动态切分为 4 个区域：

```
workSpace 指针
   │
   ├──► gS       (QK matmul 输出 = attention scores, mm1 输出)
   │
   ├──► gP       (softmax 输出 = attention probabilities, smOnline 输出)
   │
   ├──► gOTmp    (P*V 临时输出, mm2 输出)
   │
   └──► gOUpdate (online softmax 的 rowmax/rowsum/update 中间量)
```

每个区域大小 = `WORKSPACE_BLOCK_SIZE_DB × sizeof(T) × (PRE_LAUNCH+1)`（三缓冲）。

---

## 7. Tile 计算函数

### 7.1 GetQSBlockTile：序列维度分块

```cpp
__aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen) {
    return Q_TILE_CEIL;  // 固定 128
}
```

Q 序列方向（M 维，token 行）每块固定 **128 行**。参数 `kvSeqlen` 当前未参与调节（Host 侧副本注释 "kvSeqlen 暂未参与动态调节"），为未来自适应 tile 预留接口。

### 7.2 GetQNBlockTile：Head 维度分块（GQA 感知）

```cpp
__aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize) {
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    qNBlockTile = min(qNBlockTile, groupSize);
    qNBlockTile = max(qNBlockTile, 1);
    return qNBlockTile;
}
```

**参数**：
- `qSeqlen`：当前 batch 的实际 Q 序列长度；
- `groupSize = qHeads / kvHeads`：每个 KV head 对应的 Q head 数。
  - MHA：groupSize=1；GQA：groupSize=qHeads/kvHeads（如 8、4）；MQA：groupSize=qHeads。

**逻辑**：
1. `(128 / qSeqlen) / 2 * 2`：短序列（decode）让 qNBlockTile 更大（最多 128），一次处理更多 head；长序列（prefill）让 qNBlockTile 更小；向下对齐到 2 的倍数方便 sub-core 对半拆。
2. `min(..., groupSize)`：不超过 groupSize，保证一个 block 内的 Q heads 全部**属于同一个 KV head**（避免跨 KV head group 边界）。
3. `max(..., 1)`：至少处理 1 个 head。

**总任务数**计算：
```
qNBlockNumPerGroup = CeilDiv(groupSize, qNBlockTile);  // 每组内 Q head 分几块
curQNBlockNum      = qNBlockNumPerGroup * kvHeads;     // head 方向总块数
curQSBlockNum      = CeilDiv(qSeqlen, 128);            // 序列方向总块数
curTaskNum         = curQNBlockNum * curQSBlockNum;    // 总 task 数
```

---

## 8. 例 1：Prefill 长序列（BSND + MHA + Causal）

**配置**：
- batch = 2；qSeqlen = 1024（prefill）；kvSeqlen = 1024；
- qHeads = 32；kvHeads = 32 → groupSize = 1（MHA）；
- embed = 128；is_causal = true；layout = BSND。

**计算过程**：

1. **Causal mask**：Host 侧分配 2048×2048 uint8 上三角矩阵，通过 `at::triu(ones,1)` 预生成后拷贝到 NPU。
2. **qSBlockTile** = `GetQSBlockTile(...)` = **128**。
3. **qNBlockTile** = `GetQNBlockTile(1024, 1)`：
   - `(128/1024)/2*2 = 0`；
   - `min(0, 1) = 0`；
   - `max(0, 1) = 1` → **qNBlockTile = 1**（即每个 task 处理 1 个 Q head，符合 MHA 语义）。
4. **分块数量**：
   - qSBlockNum = CeilDiv(1024, 128) = **8**；
   - qNBlockNum = CeilDiv(1, 1) × 32 = **32**；
   - 总 task 数 = 8 × 32 × 2(batch) = **512**。
5. **KV 外层循环**：`kvSLoopNumTotal = CeilDiv(1024, 512) = 2` 次 KV stack tile。
6. **L1 分配**：V 预留 `128 × 512 × 2(half) × 2(双缓冲) = 256KB`，剩余 ~256KB 给 Q 和 K，可算出 `kDynNum=256`、`nDynNum=128`。
7. **跨核流水**：3 缓冲（PRE_LAUNCH+1=3）深度重叠 QK/Softmax/PV 三段。
8. **Causal fast path**：对完全在对角线下方的 KV stack tile（`kvSEndIdx <= triUp+1`）跳过 mask；只有跨越对角线的 tile 才从 2048×2048 mask 中取子窗应用。

**示意**（单个 batch、单个 head 的 task 划分）：

```
Q 序列 S (1024 行)
    ┌───── qSBlock=0 (行  0-127)  ──► 对应 kvS 块 0/1，因果仅需 mask kv 块 0 的上三角
    ├───── qSBlock=1 (行 128-255)
    ├───── qSBlock=2 (行 256-383)
    ├───── qSBlock=3 (行 384-511)  ──► kvS 块 0 完全在对角线下（无需 mask）
    ├───── ...
    └───── qSBlock=7 (行 896-1023)
×
Q head N (32 个)
    head0 head1 head2 ... head31 （每块 1 个 head）
= 8 × 32 = 256 tasks / batch
```

---

## 9. 例 2：Decode 短序列（TND + GQA + 无 mask）

**配置**：
- batch = 64（TND packed）；每个请求 qSeqlen = 1（decode，每次生成 1 个 token）；kvSeqlen 不等（平均 ~2048）；
- qHeads = 64；kvHeads = 8 → groupSize = 8（GQA 8:1）；
- embed = 128；is_causal = false（decdoe 单步无需 mask）；layout = TND。

**计算过程**：

1. **TND 布局**：输入是 packed `[T=64, N=64, D=128]`，kernel 通过 `gActualQseqlen` 前缀差分还原每个 batch 的真实 seqlen。
2. **qSBlockTile** = 128，但 qSeqlen=1 所以单块即可（最后一块尾块）。
3. **qNBlockTile** = `GetQNBlockTile(1, 8)`：
   - `(128/1)/2*2 = 128`；
   - `min(128, 8) = 8` → **qNBlockTile = 8**（一次处理 1 个 KV head 对应的全部 8 个 Q head，正好是一个 GQA group）。
4. **分块数量**（每个 batch）：
   - qSBlockNum = CeilDiv(1, 128) = 1；
   - qNBlockNumPerGroup = CeilDiv(8, 8) = 1；
   - qNBlockNum = 1 × 8(kvHeads) = 8；
   - 单 batch task = 1 × 8 = 8；总 task 数 = 64 × 8 = **512**。
5. **KV 外层循环**：对 kvSeqlen≈2048，`CeilDiv(2048, 512) = 4` 次 KV stack tile。
6. **Workspace**：128 行 × 512 列 × 4 区域 × 3 缓冲（注意 qSBlockSize 实际只有 1 行，但 workspace 按 tile 上限 128 分配）。
7. **无需 causal mask**（`NO_MASK`），softmax 路径直接走无 mask 快速路径。
8. **Block table**：通过 `gBlockTable[batch*pageStride + pageIdx]` 查找 paged KV cache 的物理页地址。

**示意**（单个 batch 的 task 划分）：

```
Q 序列 S ：只有 1 行（当前 decode token）
           └── qSBlock=0（1 行，不足 128，由硬件自动 tail 处理）

Q head N ：64 个 Q head 分成 8 个 KV group（每组 8 个 Q head）
           ├── group0 (QH 0-7, 对应 KVH0) ──► task (0,0)
           ├── group1 (QH 8-15,对应 KVH1) ──► task (0,1)
           ├── ...
           └── group7 (QH 56-63,对应 KVH7) ──► task (0,7)
= 1 × 8 = 8 tasks / batch
```

对比 Prefill：decode 下 qNBlockTile 从 1 变成 8，head 维一次处理整个 GQA group，减少 task 数、提高每个 task 的计算密度。

---

## 10. 数据流/结构图

### 10.1 三段跨核流水线时序（PRE_LAUNCH=2 三缓冲）

```
               ┌─────────── 三缓冲槽位: %3 循环 ───────────┐
               │  slot0     slot1     slot2                │
               └───────────────────────────────────────────┘

Cube(QK):   [QK0] [QK1] [QK2] [QK3] [QK4] [QK5] ...
              │     │     │     │     │     │
Vec(Softmax): │  [SM0] [SM1] [SM2] [SM3] [SM4] [SM5] ...
              │     │     │     │     │     │
Cube(PV):     │     │   [PV0] [PV1] [PV2] [PV3] [PV4] ...   ← PV 落后 PRE_LAUNCH=2 个槽
              │     │     │     │     │     │
Vec(Rescale): │     │     │   [UP0] [UP1] [UP2] [UP3] ...
              │     │     │     │
              ▼     ▼     ▼     ▼
            flag1 flag1 flag1 flag1 ... (QK_READY_ID)
                  flag2 flag2 flag2 ... (SOFTMAX_READY_ID)
                        flag3 flag3 ... (PV_READY_ID)
```

`QKk` 完成后 SetFlag(1) 触发 `SMk`；`SMk` 完成后 SetFlag(2) 触发 `PVk`；`PVk` 完成后 SetFlag(3) 触发 `UPk`（online softmax update/rescale）。

### 10.2 L1 内存预算

```
┌──────────────────── L1 总大小 512 KB ────────────────────┐
│                                                           │
│  ┌──────── V 预留 ────────┐  ┌──── Q+K 动态区 ─────────┐  │
│  │ embedV*512*sizeof(T)*2 │  │ 剩余 ~256KB             │  │
│  │ = 128*512*2B*2         │  │ → kDynNum ≥ 256         │  │
│  │ = 256 KB（双缓冲）     │  │ → nDynNum ≤ min(预算,128)│  │
│  └────────────────────────┘  └─────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### 10.3 Workspace 布局（单核单缓冲）

```
GM workSpace 指针
  │
  ▼
┌──────────────────────────┬──────────────────────┬──────────────────────┬─────────────────────────┐
│  gS (QK scores S)        │  gP (softmax probs P)│  gOTmp (PV output)   │  gOUpdate (m/l/dm 等)    │
│  128 × 512 × sizeof(S)   │  128 × 512 × half    │  128 × 128 × half    │  online softmax 中间量   │
│  (mm1 out)               │  (smOnline out)      │  (mm2 out)           │                          │
└──────────────────────────┴──────────────────────┴──────────────────────┴─────────────────────────┘
  ▲                                 ▲                         ▲                         ▲
  │ mm1OutSize                      │ smOnlineOutSize          │ mm2OutSize              │
  └───────────── × 3（PRE_LAUNCH+1 三缓冲）───────────────────────────────────────────────┘
```

Host 侧在 `flash_api.cpp` 中硬编码 `workspace_size = blockDim * 128*512 * sizeof(T) * 3 * 4`（对四个区域 × 三缓冲统一按最宽松尺寸分配）。

### 10.4 Host → Device 参数传递流程

```
┌───────────────── Host (flash_api.cpp) ─────────────────┐
│  q.data_ptr() / k.data_ptr() / v.data_ptr() / ...       │
│  tilingData (FAInferTilingData, host 上构造)            │
│  is_causal? (alloc 2048×2048 triu mask → device)        │
│  blockTables (paged KV page table)                      │
│  actualSeqlen (cumulative, for TND)                     │
└─────────────┬───────────────────────────────────────────┘
              │  kernel<<<blockDim, l2ctrl, stream>>>(...)
              ▼
┌───────────────── Device (mha_fwd_kvcache.cpp) ─────────┐
│  extern "C" __global__ void FAInfer(                    │
│      uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, ...,      │
│      GM_ADDR workspace, GM_ADDR tiling)                 │
│  {                                                      │
│      FAIKernelParams params{q,k,v,mask,blockTables,     │
│          actualQseqlen,actualKvseqlen,o,lse,            │
│          workspace, tiling};                            │
│      FAInferKernel<...> kernel;                         │
│      kernel(params);   // operator()(params)            │
│  }                                                      │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌──────── FAInferKernel::operator()(params) ──────────────┐
│  tiling = (__gm__ FAInferTilingData*)params.tiling;     │
│  gQ.SetGlobalBuffer(  (__gm__ ElementQ*)params.q );     │
│  gK/gV/gMask/gBlockTable/gActualQseqlen/... 同理        │
│  // workspace 切四区域：                                 │
│  gS      = params.workSpace;                            │
│  gP      = params.workSpace + mm1OutSize;               │
│  gOTmp   = params.workSpace + mm1OutSize + smOutSize;   │
│  gOUpdate= params.workSpace + mm1OutSize + smOutSize    │
│             + mm2OutSize;                               │
│  // 进入 KV 循环 → QK/Softmax/PV/Rescale 三段流水       │
└─────────────────────────────────────────────────────────┘
```

### 10.5 分块 tile 示意（BSND 布局下 Q 矩阵）

```
Q 矩阵 [S=qSeqlen, N=qHeads, D=embed]，按 (qSBlockTile=128, qNBlockTile) 二维分块
  每个 task 处理一个 (sBlock, nBlock) 子块：

                 head 维 N
        <─── qNBlockTile heads/block ───>
        ┌────────┬────────┬────────┬─────┐
     ┌──┤ (0,0)  │ (0,1)  │ (0,2)  │ ... │  qSBlock 0 (128 行)
     │  ├────────┼────────┼────────┼─────┤
     │  │ (1,0)  │ (1,1)  │ (1,2)  │ ... │  qSBlock 1 (128 行)
S 维 │  ├────────┼────────┼────────┼─────┤
     │  │ (2,0)  │  ...                                 qSBlockTile=128
     │  │ ...                                             (固定)
     ▼  └────────┴────────┴────────┴─────┘
        每个 task 的 GEMM shape = GemmShape<qSBlockSize(≤128), nDynNum(≤128), kDynNum(≥256)>
```

---

## 11. 注意点

1. **Host/Device 两份副本必须同步**：`flash_api.cpp` 中保留了 `GetQNBlockTile`/`GetQSBlockTile` 的等价副本（用 `std::min/std::max` 书写），两者逻辑修改时必须同步更新，否则会出现 host tiling 与 device 实际分块不一致的错误。
2. **COMP_TRIU_MASK_DIM_LEN=2048 是序列长度上限**：超过 2048 的序列无法复用预生成 mask，需要重新评估 mask 策略。
3. **前向/反向枚举独立**：`MaskType` 和 `inputLayout` 在反向 FAG 中是另一套定义（在 `fag_common/common_header.h`），不要跨前反向混用。
4. **AlignUp/Max 当前未被直接调用**：device 侧代码使用 AscendC 内置的 `RoundUp`/`CeilDiv`/`AscendC::Max`，这两个模板作为备用保留，删除前需确认没有未来路径会使用。
5. **Q_TILE_CEIL=128 是全局约束**：若需支持更大 tile（如 256），必须同步修改：
   - `GetQSBlockTile`/`GetQSBlockTile` 的 host 副本；
   - `GemmShape<128,128,128>` 等 L1 GEMM tile 配置；
   - workspace 大小估算（host 侧目前硬编码 `128*512`）；
   - 可能涉及 L1 预算公式与 epilogue UB 大小。
6. **N_SPLIT_HELPER=2 依赖 sub-core=2 的假设**：qNBlockTile 对齐到 2 的倍数是为了让 Vector 核 2 个 sub-core 能对半拆分，若 sub-core 数变化需重新评估。
7. **PRE_LAUNCH=2 → 三缓冲**：所有 workspace 偏移计算都基于 `% (PRE_LAUNCH+1)`，修改此值会影响所有槽位计算。

---

## 12. 总结

[kernel_common.hpp](../flash_attn_npu/kernel_common.hpp) 虽然只有不到 100 行，却是整个 FlashAttention NPU 前向推理 kernel 的"**基因表**"：

- **3 个跨核事件 ID** 定义了 Cube↔Vector 三段流水的同步契约；
- **Q_TILE_CEIL=128 + MAX_KV_STACK_LEN=512** 决定了序列维/KV 维 tile 大小，并衍生出 L1 tile shape、workspace 单块尺寸、外层 KV 循环次数；
- **PRE_LAUNCH=2** 决定三缓冲深度流水的槽位布局；
- **COMP_TRIU_MASK_DIM_LEN=2048** 决定了 causal mask 的支持上限；
- **MaskType/inputLayout** 两个枚举作为编译期模板参数，实现了不同 mask/布局组合的代码裁剪；
- **FAIKernelParams** 将 11 个 GM 地址打包，简化了 kernel 类内部传递；
- **GetQNBlockTile/GetQSBlockTile** 提供了 GQA 感知的 head 维动态分块规则。

理解这个文件，就拿到了阅读 `mha_fwd_kvcache.cpp`、[online_softmax.hpp](../flash_attn_npu/online_softmax.hpp)、[rescale_o.hpp](../flash_attn_npu/rescale_o.hpp)、qk/pv_matmul 等核心模块的"钥匙"：所有的 tile 大小、循环边界、缓冲槽位、workspace 偏移都是从这些常量推导出来的。
