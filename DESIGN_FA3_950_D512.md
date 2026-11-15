# Ascend950 FA3 D512：e94828b 设计与 fa7b92d 优化取舍

更新：2026-11-15。目标项目：`JY_Perf/Jy_ATK_TEST/flash-attention-npu`。

> 当前取舍：按用户决定移除 GQA/MQA 长 Q 尾块合并。Host、AICPU、kernel 和 FD
> 恢复整段 Q 长度决定 head 合并数的原有几何，删除新增 `fused_task.h`；原有短 Q
> 合并保留。单项回退性能记录见 `batchtest/tail_merge_ablation_d512_20261114_233832/REPORT.md`，
> 本次移除与验证记录见 `batchtest/remove_tail_merge_20261115/REPORT.md`。

> 状态（2026-11-14）：本设计已在工作区实现（实现记录与验收见
> `batchtest/d512_implementation_20261114/PHASE1_REPORT.md`、
> `PHASE2_REPORT.md`、`DESIGN_NOTES.md`）。功能验证已完成（D512 API 168/168、
> 旧维度 forward 回归），同步 sanitizer 验证尚未执行——**实现完成、验收未完成**。

| 项目 | 版本 |
|---|---|
| 当前 checkout | `e94828b40767e3cfa292a470a7577c85c978d0f8`，当前为 detached HEAD |
| 上一次分析的 checkout | `fa7b92d34ed10442ac103b3238c34d27f8047228` |
| 两者 merge-base | `de9d6a4e172a4a6b3ed9311eb0b781ba4991e47a` |
| 上版设计归档 | [DESIGN_FA3_950_D512_fa7b92d.md](DESIGN_FA3_950_D512_fa7b92d.md) |
| 分支对比记录 | 下文版本对比基于实施前快照；当前实现和验证状态以上方更新及对应报告为准 |

归档记录的是上一 checkout 的设计，其源码链接应结合 `git show fa7b92d:<path>` 查看。
两个分支的实施前快照均未实现本文的 D512 路径；当前工作区已完成 D512 实现，且按最新取舍移除了尾块合并。

## 1. 新结论

**继续采用 Q128/KV128、Q/K/V 完整驻留 L1、完整 FP32 输出常驻 UB、PV 按 128 列交付；保留当前任务内流水，按功能选择上一分支的优化。**

选入 D512 方案的旧分支优化：

1. **位置式 causal/SWA mask**：移除 D512 的 mask GM 读取，回收每 AIV 的 16 KiB mask 预留。
2. **紧凑 K L1 布局**：K 按实际 `128×512` 分配；Q/K/V 单槽、P 三槽，总 L1 480 KiB。
3. **单 head 的整块 strided Q 搬运**：减少 ND2NZ 命令，沿用当前 Q 单槽和同步。
4. **独立 256 B LSE scratch 的受限快路径**：沿用旧分支的适用条件和事件保护。

GQA/MQA 长 Q 尾块合并不再选入；保留原有任务划分和短 Q head 合并。

本版不引入 Q 双缓冲、跨任务 `PvTask` 环、三槽 final-stat 快照、首块 LO 延迟合并及融合 Cast。原因分别是资源预算、迁移范围以及 D512 单槽分片布局的约束，而不是认定这些优化在其他维度无效。

最终资源：**L1 480 KiB；每 AIV UB 242.25 KiB，剩余 5.75 KiB**。相对上版 D512 方案的 243.75 KiB，省掉的是跨任务 finalMax/finalSum 的 **1.5 KiB**。

目标范围：普通非量化、非 MLA 的 FP16/BF16 forward，精确 `Dq=Dk=Dv=512`；保留当前 `[1,256]`。不扩大到其他 257–511 维、不对称 D/Dv、FP8、FP32 输入、backward 或 FA4。现有 API 的其他限制继续成立。

## 2. 源码差异：上版方案哪些前提已不成立

| 项目 | 当前 e94828b | 上一 fa7b92d | 对本次设计的影响 |
|---|---|---|---|
| Q L1 | 单槽，`loadQGM` 使用槽 0 | 普通路径双槽，事件 0/5 轮转 | 保留当前单槽，无需迁入再删掉双槽 |
| K L1 | 每槽按 `256×192` 半精度预留 | 普通路径改为 `128×align16(D)` | 选入紧凑布局；D512 必须按真实大小分配 |
| 槽数来源 | kernel、QK/PV 构造器均读取 tiling/helper | 多处改成固定 constexpr | 保留当前参数传递，K/V 设置 1 即可；不要迁入固定双槽 |
| QK/PV L0 循环 | 按实际 M/N/K 计算循环数 | 普通路径固定 QK 的 M/N、PV 的 M/K 为 1 | 当前循环已能表达 D512；初版无需迁入这项标量优化 |
| Q load | 已在每任务 KV 循环外，单 head 仍走 multi-ND copy | 单 head 使用一次合法的 `[S,D]` strided copy | 迁入搬运形状优化；它减少命令，不减少有效 Q 字节数 |
| 流水边界 | 每任务运行 `kvSLoopCount+PRE_LAUNCH` 并 drain | 长 KV 跨任务流水，部分短 KV 保留任务内流水 | 保留当前生命周期；不需要 final-stat 快照 |
| mask | 2048×2048 uint8 缓存，按需 GM→UB；UB 预留 16 KiB | causal/SWA 在寄存器按位置比较 | 这是本次最有价值的迁移之一 |
| D256/Q128 UB | **248064 B = 242.25 KiB** | **233472 B = 228 KiB** | 上版 228 KiB 不能当作当前基线 |
| 任务几何 | 以整个 qSeqlen 决定 head 合并数，Host/AICPU/device 分别实现 | `fused_task.h` 区分完整 Q 块和 tail，可组内/跨组融合 | 保留原有几何，不迁移 full/tail 分离及新增共享头 |
| rescale | 完整 FP32 更新后独立 Cast | 有融合 Cast、首块 LO 保留、独立 LSE scratch | 当前独立 Cast 更适合 D512 分片；只选 LSE 小改动 |
| FlashDecode | Host 形状门控较窄，metadata 不支持显式 splits | 扩大 dense/varlen/metadata 支持 | D512 不附带整个 FD 功能扩展 |
| Softmax 的两处 PIPE_ALL | 当前 `online_softmax.hpp` 已无 PIPE_ALL | 提交说明曾包含删除 | 当前没有可重复迁移的这两处删除；kernel 收尾 barrier 保留 |
| 资源封装 | `ForwardResource950` | 相同 | 不改 TBuf/CATLASS 框架 |

对 `csrc/ascend950/flash_attn_npu_3` 排除 `autogen/` 后，两个 tree 的差异涉及 30 个文件、2296 行增加和 1744 行删除，包含 backward、编译组织、FD、Host 和多种 kernel 优化。因此不建议整包 cherry-pick。

### 当前源码定位

| 文件 | 当前关键位置 |
|---|---|
| [fwd_resource_plan.h](csrc/ascend950/flash_attn_npu_3/fwd_resource_plan.h) | `MakeUbPlan` 的 mask、LO/GO 布局；`L1Bytes`；Q64 回退 |
| [fwd_kernel.cpp](csrc/ascend950/flash_attn_npu_3/fwd_kernel.cpp) | 181 行读取槽数；395/410 行任务计数/解码；579 行 Q load；601 行任务内流水；783/850 行 flag 初始化/回收 |
| [qk_matmul.hpp](csrc/ascend950/flash_attn_npu_3/qk_matmul.hpp) | 157 行 helper 槽数；199 行 `loadQGM`；309 行 D 归约循环 |
| [pv_matmul.hpp](csrc/ascend950/flash_attn_npu_3/pv_matmul.hpp) | `operator()`：先搬完整 V，再进入 `nL0Itr` 输出列循环 |
| [online_softmax.hpp](csrc/ascend950/flash_attn_npu_3/online_softmax.hpp) | 117 行 mask UB handle；436–481 行 causal mask 搬运/同步；562–725 行 SWA |
| [rescale_o.hpp](csrc/ascend950/flash_attn_npu_3/rescale_o.hpp) | 173 行 `SubCoreCompute`；243 行独立 Cast；357 行 LSE；804 行最宽 256 分支 |
| [tiling.cpp](csrc/ascend950/flash_attn_npu_3/tiling.cpp) | 130 行 head tile；192–207 行 L1；400 行 FD base-task；562 行任务总数；627 行 tile 选择 |
| [fa_metadata.aicpu](csrc/ascend950/flash_attn_npu_3/fa_metadata.aicpu) | 26 行 head tile；102 行资源字段；140 行任务计数 |
| [mha_fwd.cpp](csrc/ascend950/flash_attn_npu_3/mha_fwd.cpp) | 240 行 D 上限；371/463 行 mask；430 行 FD 形状门控；534 行 DN |

## 3. 优化选择及依赖

“选入”表示纳入设计，尚未落地验证。下列提交是功能的来源，不是建议整提交套用。

| 来源 | 功能 | 选择 | 理由及迁移边界 |
|---|---|---|---|
| `0867a03` | 位置式 causal/SWA mask | **选入** | D512 路径使用；减少 mask GM 读取并释放 UB。提取 mask operator/VF 及位置参数，不带 DN/MHA 特化 |
| `bd74814` | 紧凑 K 布局 | **选入** | 通过 D512 的 Host/AICPU profile 写入实际 N/D；保留当前 helper 槽数 |
| `67de7a4` | 单 head 整块 Q copy | **选入** | 只取 `qNBlockSize==1` 的合法 strided copy；组内多 head 保留现有 multi-ND copy |
| `9cc7796` | full/tail 共享任务几何、组内尾块融合 | **不选** | 单项回退实测收益集中于特定任务分配；用户决定移除，保持原有几何以减少代码增量 |
| `bd74814` | 独立 LSE scratch | **选入小快路径** | 256 B；仅原条件下单 head 连续 LSE。其他情况保留当前对齐 fallback |
| `bd74814` | Q 双缓冲 | **不选** | D512 额外 128 KiB 后 L1 达到 608 KiB；不满足本方案预算 |
| `67de7a4` 等 | 跨任务 `PvTask` 环和 final-stat | **后续独立评估** | 不减少有效 Q/K/V 字节；引入状态生命周期变化。当前任务内流水足以支持 D512 |
| `9cc7796` | 跨 KV group 融合、分段 QK/PV/Fixpipe | **不选** | group 之间 KV 不共享，不能靠它减少有效 KV 字节；增加按列交付复杂度 |
| `bd74814` | 首块 LO 保留，第二块直接累加 | **不选** | 依赖两个完整 LO 槽；D512 只有一个 128 列 LO |
| `bd74814` | Rescale 与 Cast 融合 | **不照搬** | 列分片阶段打包 half 会覆盖仍活跃的 FP32 GO；保留最后完整 Cast |
| `bd74814` | Q-ready wait 后移、固定内层循环、snake/head-major 调度 | **后续独立评估** | 主要影响等待、标量和局部性，非本次最低 GM 有效字节的必要条件 |
| `67de7a4` / `9cc7796` | Host 长度读取合并、tiling cache | **后续独立评估** | 有 Eager/graph/stream 依赖，不是 D512 Cube 数据供给的前置条件 |
| `9943b57` | FD 扩大支持范围、metadata FD | **不并入本次** | 保留当前公开行为；只让当前合法的 FD 路径容纳 D512 |
| `d2798ec` | autogen 编译拆分 | **不并入本次** | 影响编译组织，与 D512 IO/资源正确性无关 |
| `0867a03` | DN/MHA D128 专用流水 | **不选** | D512 不进入 DN，更不是 D128 特化条件 |

最小依赖链是：资源 profile → D512 位置式 mask → P UB 单槽与 PV 四片交付 → D512 rescale → FD 容量 → 接口放行。Q copy 和 LSE 小优化可分别验证后并入。

## 4. 新分块和内存预算

### 4.1 直接放开 D512 会出现什么

当前 `L1Bytes(512,512)` 是 **688128 B = 672 KiB**。而且旧 K 每槽只有 `256×192×2=96 KiB`，小于完整 `128×512×2=128 KiB`；光减少槽数仍不合法，必须修正 K 形状。

当前 `MakeUbPlan(512,1)` 是 **379136 B = 370.25 KiB**，即使只保留一个完整 LO 仍超 UB。旧资源判断退到 Q64 后仍进入旧固定 UB layout，不能把这个回退视为已经支持 D512。

### 4.2 推荐 L1：480 KiB

`M=N=128, D=Dv=512`，Q/K/P/V 为 FP16/BF16。

| Buffer | 形状和槽数 | 字节范围，左闭右开 | 大小 |
|---|---|---|---:|
| Q | 128×512×1 | `[0,131072)` | 128 KiB |
| K | 128×512×1 | `[131072,262144)` | 128 KiB |
| P | 128×128×3 | `[262144,360448)` | 96 KiB |
| V | 128×512×1 | `[360448,491520)` | 128 KiB |
| 余量 | | `[491520,524288)` | 32 KiB |

Host/AICPU 统一写入：QK `M=128,N=128,KLeft=512,KRight=512`；PV `M=128,N=512,KLeft=128,KRight=128`；槽数 `Q/K/P/V=1/1/3/1`。Device 继续使用现有 helper 字段，无需把它们改成固定双槽。

L0 继续使用当前 128×128×128 的算术分片：QK 沿 D 做 4 次归约，PV 沿 Dv 做 4 个输出片。L0A/B 各双槽，每槽半精度 32 KiB；L0C 沿用现有 QK/PV 各双槽，每槽 64 KiB。无需 128×512 的 L0C。

### 4.3 推荐 UB：每 AIV 242.25 KiB

| Buffer | 形状/用途 | 起始字节 | 字节数 |
|---|---|---:|---:|
| Score | 2×64×128 FP32 | 0 | 65536 |
| P UB | 1×65×128 FP16/BF16，保留 padding | 65536 | 16640 |
| LO | 1×64×128 FP32 | 82176 | 32768 |
| GO | 1×64×512 FP32 | 114944 | 131072 |
| LM | 64 FP32 | 246016 | 256 |
| GM | 64 FP32 | 246272 | 256 |
| DM | 3×64 FP32 | 246528 | 768 |
| LL | 64 FP32 | 247296 | 256 |
| GL | 64 FP32 | 247552 | 256 |
| LSE scratch | 64 FP32 | 247808 | 256 |
| 结束 | | **248064** | **242.25 KiB** |

与上一版设计相同的 S/P/LO/GO 共占 240.25 KiB；当前任务内流水只需 1.75 KiB running stats 和 0.25 KiB LSE scratch。**DM 仍需要三槽**，不能因为没有跨任务流水而删除：它服务于当前 `PRE_LAUNCH=2` 的 KV 流水。

所有表中起点满足 32 B 对齐，剩余 5888 B。编译时还需确认具体 Copy/VF 对齐要求及额外 scratch。`UbPlan` 需区分 LO 与 GO 容量，不能继续用相同 `loBytes` 推导 GO 大小。

D512 不建立 mask UB handle；旧 D≤256 可保留 `mask` 字段与原布局。将 D512 plan 单独选出，避免通过全局更改 `MakeUbPlan` 顺便改变 D192/D256 的槽数和路由。

### 4.4 为什么选择位置式 mask，而不是原样套用上版资源

若保持当前 16 KiB mask，再加本方案 LO128 和独立 LSE scratch，需要 **258.25 KiB**，超出 UB。因此，在推荐的 Q128/LO128 结构下，回收 mask 是必要条件。

可行备选如下，均需实际修改布局才成立：

| 备选 | L1 | 每 AIV UB | 代价 |
|---|---:|---:|---|
| 推荐：Q128、位置式 mask、P UB 单槽、LO128 | 480 KiB | 242.25 KiB | 每 PV tile 四次交付 |
| Q128、保留 mask、P UB 单槽、LO64 | 480 KiB | 242.25 KiB | 八次交付；需改变 PV 列计算粒度和相关 stage 计数；仍有 mask GM 读取 |
| Q64、保留 mask、P UB 双槽、完整 LO512、Q 单槽 | 368 KiB | 194.5 KiB | 需同步缩小物理行容量及 UB slot stride；长 Q 增加 KV 重读 |

本次选择推荐方案。它同时消除 mask GM 读取、保留现有 PV 128 列 MMAD 粒度，并尽量减少长 Q 的 KV 重读。仅凭容量不能认定它延迟最低，单槽和握手成本仍需实测。

## 5. 计算与同步设计

### 5.1 保留任务内流水

保留当前 `for (kvSTileRelIdx < kvSLoopCount + PRE_LAUNCH)`，仍先生产 QK/Softmax，再消费延迟两轮的 PV/rescale，任务结束 drain 后进入下一个任务。Q 在任务开始加载一次，最后一次 QK 释放 Q 单槽。

QK 每个 KV tile 将完整 K 从 GM 搬入 L1 一次，沿 D 的 4 个 128 分片只做 L1→L0 搬运。四片归约完成后才进行一次 Softmax。PV 也在输出列循环外搬入完整 V 一次。

### 5.2 P UB 与 P L1 分别管理

- Score UB：`scoreSlot=tileSeq%2`。
- P L1 与 DM：`pL1Slot=tileSeq%3`。
- D512 P UB：固定槽 0，对应固定 `MTE3_V` 复用事件；旧路径仍使用原双槽。

写 P UB 前必须等待上次 UB→L1 DMA 读完；目标 P L1 槽仍通过原来的 ready/free 等待。地址和事件号要一起与 `ubSBufId` 解耦，不能只改地址。P L1 三槽不减，Score 双槽不减。

### 5.3 PV 四片交付与 rescale

```text
AIC                                      每个 AIV
完整 V GM→L1，一次
for c = 0,128,256,384:                    for c = 0,128,256,384:
    计算 P × V[:,c:c+128]
    等两个 AIV 的 LO free                    等 LO ready
    Fixpipe → 同一个 LO[本子核行,128]         GO[:,c:c+128] = first ? LO : dm*GO[:,c:c+128]+LO
    发布 LO ready                           在 Vector 读取完 LO 后发布 free
```

LO 目的列偏移始终为 0；V 源列偏移是 c；GO 更新目的列偏移也是 c。新增 VF 明确使用 `loStride=128`、`goStride=512`，不能把旧 `SubCoreCompute<256>` 传入 col=512。

沿用一个 PV flag 通道（5，第二 AIV 对应 21），每 KV tile 四次 ready/free，只允许一个未消费片。替换原先“第一片等待、最后一片通知”的协议，不能多发整块 ready。有效行数为 0 的 AIV 也完成四轮握手。

P/V 只能在最后一次 L1→L0 读取完成后释放。K/V 单槽的事件编号和 init/release 数量一致；无用的第二槽不得访问。P UB 单槽事件、跨核 LO flag、GO 输出 DMA 事件分别管理。

GO 的 `MTE3_V(4)` 防护仍按一次完整 PV 消费维护，四片内不重复消费同一个 token。任务退出/空区间初始化前完成最后的 rescale 和 DMA 防护。当前没有上一任务尚未完成的 Vector 更新，因此不需要 `SaveFinalStatistics`；若未来增加跨任务流水，必须重新加入快照和生命周期证明。

### 5.4 最后一轮输出

四片全部更新完 FP32 GO 后，使用本任务最终 GL 归一化；正常输出再对完整 GO 执行当前独立 Cast，复用原有舍入、grouped scatter、无效行清零和 LSE 行为。FD partial 保持 FP32。

不把旧分支的融合 Cast 插入每个列片：half packed 视图与 FP32 GO 别名，列片顺序打包会覆盖后续行仍需更新的 FP32 数据。整块 Cast 的大长度与别名条件仍需真实编译和 NPU 验证。

LSE 小优化仅用于上一分支的安全条件：单 head、没有 fully-masked 行、没有需要特殊清零的 SWA 行。使用独立 256 B scratch 和原有独立 DMA 事件；其他情况继续当前对齐 fallback，等待 O DMA 后才复用 GO。组内 tail 多 head 的 LSE 不擅自改成一块连续 GM store。

## 6. 位置式 mask 与组内 tail 的适配

### 6.1 Mask 参数必须使用有效序列长度

定义任务内逻辑行 `r`，每 head 有 `qRows` 行，实际 token 位置为：

```text
qAbs   = qStart + (r % qRows)
center = qAbs + actualSk - actualSq       # bottom-right 对齐
kAbs   = kvTileStart + column
causal = kAbs <= center
SWA    = (left<0 or kAbs >= center-left) and
         (right<0 or kAbs <= center+right)
```

选用旧分支 `ComputeScaleAndMaxPositions` / `Positions64` 的实现时，一起迁移 window 归一化/实际长度裁剪及 head 内行号回绕，不能只复制比较指令。负的 center、单边无限窗口、`Sq>Sk`、首尾 KV tile 和 GQA 跨 head 的行号都要覆盖。

当前 `mha_fwd.cpp` 有 metadata 与普通 Host 两处 `MakeDeviceTriuMask()`；D512 两处都应跳过，kernel 对应路径不得访问 mask 地址。D≤256 保留原逻辑时，mask 缓存和相关事件不能全局删除。

当前 mask 是按线程/device 缓存，首次创建才有生成与同步，不能把去掉它宣传为每次调用都省一次 mask 初始化。每次执行的 mask GM→UB 读取是否发生则取决于 causal/SWA 的边界 tile。

### 6.2 保留原有 head 合并与任务几何

`GetQNBlockTile()` 继续使用整个 qSeqlen，同一 batch 的完整块与尾块使用相同的 head 合并数：

```text
G = Hq/Hkv
g = GetQNBlockTile(Sq,G)
totalTasks = ceil(Sq/128) * Hkv * ceil(G/g)
```

长 Q 的 head 合并数为1，Q129/Hq16 为32个任务，Q257为48个任务，Q513为80个任务。
短 Q 沿用已有合并；head 范围始终位于同一 KV group。

Host 的 total/firstBatch task 数及 FD base-task 枚举、AICPU metadata、AICore 跨 batch
计数与解码均恢复原有代码，不新增 `fused_task.h`。D512 的 Q128 资源选择继续保留。

2026-11-14 单项回退实验中，Q129 的 GQA/MQA 耗时降低38.27%/34.52%，Q257/513基本持平。
该项收益未推广到多数场景，2026-11-15 按用户决定移除，历史实验数据保留供追溯。

## 7. GM 字节数与本次收益范围

普通 `num_splits=1`，对一个 m 行任务，实际访问 k 个 KV token：

```text
Q read      = 2*m*512
K+V read    = 2*k*(512+512)
mask read   = 0                          # D512 位置式 mask
S/P/LO/GO 中间结果 GM read/write = 0
O write     = 2*m*512
LSE write   = 4*m                        # 启用时
```

不包含 metadata、block table、对齐事务，也不等于 HBM 实测流量；L2 命中必须单独看。当前实现已经是 Q 常驻和片上 S/P/LO/GO 交互，本方案是在 D512 下保持这些性质，不能说这些全是从旧分支新引入的节省。

有效字节模型例子，无 mask、无 split、D512，非性能实测：

| 比较 | 条件 | 方案前后 K/V 有效读量 |
|---|---|---:|
| 保留 Q128，相对 Q64 | B=1,Hq=Hkv=1,Sq=1024,Sk=4096 | 128 MiB → 64 MiB；Q 均为 1 MiB |

causal/SWA 按实际访问的 KV tile 求和，不能套用固定减半比例。本文在原有任务划分下保持
Q128 和片上中间结果供给，不再主张长 Q 尾块合并的读取节省，也不声称全局 IO 最优。
M256 对应每 AIV 的 GO 已达 `128×512×4=256 KiB`，不能直接放入 248 KiB UB。

## 8. FlashDecode、metadata 与接口边界

当前 FD 的有效 Host 形状门控包含：非 local、paged KV、TND varlen Q、所有有效 Q 长度在 1–16、max KV≥1024，再由 tiler 判断任务数和可分配 core。显式 splits 的公开检查要求 paged KV+varlen Q；metadata 还拒绝显式 splits。普通 dense D512 可以通过非 FD 路径支持，不依赖扩大 FD 形状范围。

本次保持这些限制，验证当前合法 FD 场景的 D512。不要照搬上版设计中“dense/metadata FD 已存在”的前提。`num_splits=1` 是最低读取量基准；支持范围内的 auto/forced splits 另测。

FD combine 恢复原有任务解码时，D512 继续使用 Q128，不能套用旧大维度 Q64 的合并行数上限。
因此 `GetQNBlockTile` 的限制条件保留 D512 例外，与 Host 和主 kernel 一致；这是短 Q/资源
profile 的一致性要求，长 Q 尾块不再单独合并。

[fd_combine.hpp](csrc/ascend950/flash_attn_npu_3/fd_combine.hpp) 的 `MAX_HEAD_DIM` 及输出累加范围需要核对到 512。现有 32 KiB FP32 partial 输入窗口不必翻倍：8 行 D512 时每轮容纳 2 个 split，8×512 FP32 accumulator 为 16 KiB；连同现有 stats，总布局理论为 51488 B。保持输入窗口、按实际 D 计算批次，补充实际末端容量断言，不能仅改名义常量。

R 个不重叠 KV split 会使 Q 加载 R 次，K/V 总有效字节基本不变，并新增 partial O/LSE 写回及 combine 读取。partial O 每次有效读或写为 `4*R*m*512` 字节；workspace 分配仍需按 row capacity 和对齐计算。

公开 [flash_api.cpp](csrc/ascend950/flash_attn_npu_3/flash_api.cpp) combine 与 kernel 内部 FD combine 分别验证。metadata 普通 D512 路径更新资源 profile，task count 保留原有公式，不增加 metadata FD 功能或外部 ABI 字段。

普通路径仍可能分配旧 workspace 区；“不访问中间结果 GM”不等于“workspace 分配为零”。清理历史容量不并入本次最小增量。

## 9. 实现顺序与文件清单

| 步骤 | 文件 | 修改边界 |
|---|---|---|
| 1 | `fwd_resource_plan.h` | 精确 D512 plan：完整 L1、LO128/GO512、P UB 单槽、无 mask、可选 LSE scratch；旧 plan 不变 |
| 2 | `fwd_tiling_policy.h`、`tiling.cpp`、`fa_metadata.aicpu` | D512 优先选择 Q128 合法 profile，写实际 K 形状与活动槽数，避开旧资源回退 |
| 3 | `online_softmax.hpp`、`mha_fwd.cpp`、`fwd_kernel.cpp` | D512 位置式 mask；两处 Host mask 缓存调用跳过；P UB 单槽事件；保留旧路径 |
| 4 | `pv_matmul.hpp`、`rescale_o.hpp`、`fwd_kernel.cpp` | LO128 layout、四片 ready/free、独立 stride 的更新、完整 Cast、active flag init/release |
| 5 | `qk_matmul.hpp` | 单 head 整块 Q copy；保留当前 Q 单槽和原有等待，不带入整个双缓冲改动 |
| 6 | `rescale_o.hpp` | 选入 256 B LSE scratch 快路径及对应事件；其他 LSE 路径维持当前行为 |
| 7 | `fd_combine.hpp` | D512 实际容量、输入批次、partial 形状；保留现有 FD 支持范围 |
| 8 | `mha_fwd.cpp` | 实现和验证完成后最后放开精确 `(512,512)`，其余上限保持原样 |

生产修改沿用已有文件，不新增共享任务几何头。没有必要修改 `qk_matmul_DN.hpp`、backward、autogen、`fwd_resource.hpp` 或 Python 公共参数。实际新增行数以落地后的 diff 为准。

建议将资源/四片协议、位置式 mask、Q copy、LSE 优化分开记录验证结果，防止把容量正确和性能收益混成一个结论。

## 10. 本次检查与后续验收

已完成的设计检查：

- 使用当前真实 `fwd_resource_plan.h` 编译并运行独立 Host 小程序，确认 D256/Q128 UB=248064 B；直接代入 D512 的 UB=379136 B、L1=688128 B。
- 独立计算推荐布局，检查所有 offset 的 32 B 对齐、buffer 不重叠、UB=248064 B、L1=491520 B。
- 历史方案曾对 full/tail 模型做630组数学检查；该模型现已移除，记录不作为当前实现的验收依据。
- 上述是资源与数学模型检查，不是已修改的 Host/AICPU/AICore 实现测试，更不是 NPU 正确性或性能结果。

实现后必须完成：

1. **几何一致性**：Host/AICPU/device/FD 保持原有整段 Q 几何，逐项比对 ragged batch、尾块 head、qStart/headStart/rowNum 和 partial capacity，并保留短 Q 合并覆盖。
2. **O/LSE 正确性**：FP16/BF16；Q/KV 的 0、1、63/64/65、127/128/129、255/256/257 和长序列；MHA/GQA/MQA；dense/paged、已有存储布局；causal/SWA、全 masked、空 KV、softcap；普通 metadata/graph replay；合法 FD 和公开 combine。沿用真实长度 bounds 契约。
3. **同步与内存**：单/多 KV tile、四片交付、零行 AIV、首尾任务和重复调用；检查 P UB、P L1、K/V、LO、GO 和 LSE 的事件配对；sanitizer 正常退出、结构化错误为零且确认执行目标 kernel。
4. **基线回归**：D64/128/192/256 与 FP8/DN 的既有路由及资源不被 D512 条件污染；不在本次放开 backward。
5. **GM 与性能分别测量**：先核对 Q/K/V load 的循环位置和无中间 GM spill；再看 GM/L2/HBM 计数、Task Duration、AIC/AIV scalar、Cube/Vector/MTE 等待。按需比较 Q copy 和 LSE 开关；容量和理论字节数不代替时间验收。
6. **执行来源**：记录当前源码、实际加载 `.so`、kernel/tiling 和构建参数，在空闲设备做同配置 AB/BA；排除异常退出、共享负载和不完整导出。

本设计没有宣称 D512 已经支持，也没有把 fa7b92d 的历史验证结果当成 e94828b 上所选迁移组合的验证。
