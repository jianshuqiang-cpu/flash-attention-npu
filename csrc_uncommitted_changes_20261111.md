# Ascend 950 FA3：csrc 未提交修改逻辑与效果

本文记录 2026-11-11 工作区中 `csrc` 相对 `HEAD` 的未提交修改，供代码评审和后续维护使用。

- 对比基线：`d7686b42312523bb90b6c6f88a679ecb605152d0`。
- 取证方式：`git diff HEAD -- csrc`，包含暂存区和工作区相对 HEAD 的差异；生成本文时，`csrc` 下没有额外未跟踪文件，也没有已暂存修改。
- 修改范围：`csrc/ascend950/flash_attn_npu_3` 下 5 个文件；追加第 9 节的 icache prefetch 后，相对 HEAD 增加 578 行、删除 169 行。
- 第 1～8 节的性能和精度结果对应追加 prefetch 之前的[源码快照](JD/20261111/scalar_opt/final_source_files.json)。第 9 节随后修改了 `fwd_kernel.cpp`，该追加修改仅做编译检查，不沿用之前的运行验证结论。

## 1. 修改目标与总体结果

本次修改包含两项相互独立的工作：

1. **FP16/BF16 的 causal 和 SWA 路径去掉 triu 矩阵依赖。** 与 FP8 采用相同的按 Q/K 位置判断可见性的思路，在向量寄存器中生成比较谓词。SWA 的实现及独立性能结果见第 10 节。
2. **降低无 mask DN/MHA D128 路径的标量开销和执行时间。** 简化任务映射与固定资源参数计算，复用跨任务流水，并增加专用 rescale 和连续 LSE 写回。

性能验收对应给定 profile 中的 **T5 encoder 前向：BF16、TND、无 mask**。统一采样下，Task Duration 从 1043.090 µs 降到 943.222 µs，AIV scalar 从 734.669 µs 降到 353.099 µs；Task Duration 比同期 FlashAttentionScore 低 8.03%。

**这组性能收益不能归因于 causal 去矩阵化，因为该验收场景没有 causal mask。** causal 修改的直接效果是消除纯 causal 调用对矩阵生成、缓存和读取的依赖，已通过冷启动调用和数值校验。

## 2. 文件修改概览

| 文件 | 修改内容 | 作用 |
|---|---|---|
| [mha_fwd.cpp](csrc/ascend950/flash_attn_npu_3/mha_fwd.cpp) | 两条 Host 分支均移除 `MakeDeviceTriuMask` | causal/SWA 均不再分配/取得 mask 矩阵 |
| [fused_task.h](csrc/ascend950/flash_attn_npu_3/fused_task.h) | 新增 `MakeMhaShape`、`DecodeMha` | 在无跨组融合的 MHA 下简化任务形状和索引计算 |
| [fwd_kernel.cpp](csrc/ascend950/flash_attn_npu_3/fwd_kernel.cpp) | causal 位置参数；带条件保护的 DN/MHA D128 特化；轻量 PV 描述符；跨任务流水接入 | 减少通用控制逻辑，并让相邻 Q 任务的工作重叠 |
| [online_softmax.hpp](csrc/ascend950/flash_attn_npu_3/online_softmax.hpp) | 位置式 causal/SWA softmax；DN 完整 KV tile 快分支 | 去掉 causal/SWA mask 搬运；减少完整 KV tile 的动态分支与除法 |
| [rescale_o.hpp](csrc/ascend950/flash_attn_npu_3/rescale_o.hpp) | 单 head 连续 LSE 写回；新增 `RescaleDnMha128` | 减少通用布局、参数处理及统计量句柄切换开销 |

`device_triu_mask.h`、tiling 数据结构和 backward kernel 不在本次 `csrc` diff 中。既有跨任务流水、最终统计量快照等基础设施也不是全部在本次新增；这里主要将它们接入符合条件的 DN 路径。

## 3. FP16/BF16 causal/SWA 去矩阵化

### 3.1 Host：不再取得 mask 矩阵

`mha_fwd.cpp` 同时修改了两条入口分支：

| 分支 | 修改前生成条件 | 修改后生成条件 |
|---|---|---|
| 带 scheduler metadata | `hasMask && quantMode == 0` | 不生成矩阵 |
| Host 直接计算 tiling | `(is_causal || is_local) && quantMode == 0` | 不生成矩阵 |

causal/SWA 调用中的 `maskDevice` 均保持空指针。`hasMask` 仍参与 metadata 的 tiling 偏移选择，不能把它理解成“是否存在物理 mask 矩阵”；causal 仍然是一种逻辑 mask 模式。

原来的 [MakeDeviceTriuMask](csrc/ascend950/flash_attn_npu_3/device_triu_mask.h) 创建 `2048 × 2048` 的 uint8 矩阵，大小为 4 MiB，并具有线程内、按设备缓存。新逻辑让 causal/SWA 均不再触发该分配及初始化，而不是声称旧实现每次调用都会重新生成矩阵。头文件保留，但当前 FA3 前向不再调用该函数。

### 3.2 Kernel：根据实际 Q/K 位置判断可见性

`fwd_kernel.cpp` 不再绑定 `gMask` 或构造 GM mask tensor。causal 分支向 softmax 传递：

```text
qTileCenterStart = qSTileStart + kvSeqlen - qSeqlen
kvSStartIdx      = kvSTileIdx * kvBaseTile
```

对当前 Q tile 内的 token 行 `r` 和 KV 列 `k`，可见条件为：

```text
k <= qSTileStart + r + kvSeqlen - qSeqlen
```

这里保留了 Q/K 不等长时的右对齐 causal 语义。若当前 KV tile 对整个 Q tile 都可见，则直接使用原来的无 mask softmax 分支。

### 3.3 Vector：比较谓词替代矩阵读取

`online_softmax.hpp` 使用位置式重载，以及 `ComputeScaleAndMaxPositions`、`ComputeScaleAndMaxPositions64` 两个 VF，分别处理 KV 列数大于 64 和不大于 64 的情况；`IsSwa` 模板参数区分 causal 与 SWA。

处理顺序为：

1. 用 `Arange` 生成 KV 列的绝对位置；大于 64 列时另构造第二组列位置。
2. 对 score 执行 scale，以及按模板启用的 softcap。
3. 比较 `kvIndex <= qPosition`，使用 `Select` 将不可见位置替换为既有 `MIN_VALUE`。
4. 同时屏蔽 KV 尾部无效 lane，再执行 max、exp、sum 和 P 类型转换。

源码中的 `MIN_VALUE` 是 `-std::numeric_limits<float>::max()`，不是直接写入负无穷。完全无可见 KV 的行仍由既有输出处理维持 `O=0、LSE=+inf` 的接口约定。

对于合并多个 Q head 的逻辑行，`qRow` 从 `rowStart % qSBlockSize` 开始，并在达到 `qSBlockSize` 时回绕，避免把前一个 head 的 token 行号错误延续到下一个 head。

### 3.4 效果与边界

- causal 和 SWA 都不再生成或读取 triu GM 矩阵；SWA 额外使用左右窗口约束。
- 支持路径的数值行为通过 FP16/BF16、不同布局、GQA、Q/K 不等长、尾块和 softcap 用例校验。
- [冷启动测试](tests/test_flash_attn_npu_v3_950_causal_positions.py) 为每个组合启动新进程，避免旧的 triu 缓存掩盖首次调用行为；4 个组合覆盖两种 dtype 与 metadata 开关。
- [JD.py](JD.py) 仍会单独为 fusion 对照算子生成 triu，因此不能用整份 JD trace 中“出现过 triu”判断 FA3 内部仍生成矩阵。

## 4. DN/MHA D128：简化控制逻辑并接入跨任务流水

### 4.1 特化的启用条件

Host 的无 causal、无 SWA 路径可选择 DN 入口。`FAInferDn` 中既有的 FlashDecode 或跨组融合判断会先转入相应通用实现；进入 `RunFAInferDn` 后，新增分支继续检查完整资源计划：

| 条件 | 要求 |
|---|---|
| KV cache | `normalCache` |
| Head 关系 | `numHeads == kvHeads`，即 MHA |
| Q/K 和 V 的 head dimension | 均为 128 |
| Q、KV 基本 tile | 均为 128 |
| QK L1 参数 `(M, N, KLeft, KRight)` | `(128, 256, 128, 192)` |
| PV L1 参数 `(M, N, KLeft, KRight)` | `(128, 128, 128, 128)` |
| Q/K/V/P 的 L1 buffer 数 | `1 / 2 / 2 / 3` |

满足条件时调用 `operator()<true, false, false, true>`：开启跨任务流水、关闭 FlashDecode 和跨组融合、启用 `StaticDnMha128`。否则仍调用原通用版本。

这不是按 `t5_encoder` 名称或某个固定 batch/序列长度匹配的分支。序列长度、head 数、实际 stride 和尾块继续由输入决定。Paged cache、GQA、其他维度或资源计划不匹配时不进入该特化。

### 4.2 资源参数常量化与 MHA 任务映射

`StaticDnMha128` 将已由入口核验的维度、tile 和 buffer 数设为编译期可传播的常量，便于编译器简化地址与轮转计算；不改变 tiling ABI，也没有扩大缓冲区容量。

在无跨组融合的 MHA 中，每个 Q tile 对每个 head 恰好对应一个任务，所以新增映射可以写为：

```text
totalTasks = ceil(qSeqlen / qTile) * heads
qStart     = (taskIndex / heads) * qTile
qRows      = min(qTile, qSeqlen - qStart)
headStart  = taskIndex % heads
taskHeads  = 1
```

`MakeMhaShape` 和 `DecodeMha` 使用上述关系，避开通用 GQA/fusion 的分组计算。主循环也直接复用 `geometry.qStart` 和 `geometry.qRows`，减少重复推导 Q 偏移、总块数和末块长度。

### 4.3 跨 Q 任务保持流水连续

原 DN 默认在当前 Q 任务内部运行流水并收尾；新特化允许相邻 Q 任务共享连续的 `pipelineTileSeq`，复用已有跨任务调度，以减少任务边界的流水排空。

关键调整包括：

- `useCrossTaskPipeline` 不再统一排除 DN，由模板参数决定；普通 DN 默认仍保持原选择。
- DN QK 的轮转序号从任务内 `kvSTileRelIdx` 改为连续的 `pipelineTileSeq`，使 producer 和 consumer 对缓冲区索引的解释一致。
- 新增 `DnMhaPvTask`，只保存 V/O/LSE 偏移、行数、KV tile 信息和首尾标记；不保存该特化不使用的 mask、FD、分组分段字段。
- AIC 专用 `ConsumePv` 使用固定 D128 和对应的单级 L0 stage 增量；AIV 专用 `ConsumePv` 调用专用 rescale。
- 通用 `ConsumePv` 向 rescale 传入 `IsDN`，不再固定传 `false`，从而保留 DN 所需的 32 行对齐切分。

对应的缓冲区与同步关系为：

| Producer → Consumer | 缓冲区槽位 | 同步关系 |
|---|---|---|
| QK Fixpipe → softmax | `tileSeq % 2` | QK ready flag 为槽位号 |
| softmax P → PV | `tileSeq % 3` | P ready flag 为槽位号加 2 |
| PV Fixpipe → rescale | `tileSeq % 2` | PV ready flag 为槽位号加 5 |
| 最终 softmax max/sum → 最后一个 PV 的 rescale/LSE | `tileSeq % 3` | 复用既有最终统计量快照，在槽位复用前消费 |

稳定阶段，AIC 在 QK `t` 后消费 PV `t-2`，AIV 在 softmax `t` 前消费 PV `t-3`。遇到需要初始化输出的空 KV span 时，先排空待消费工作；所有任务结束后完成最终 drain。

最终统计量快照非常重要：后续 Q 任务可能先重置运行中的 max/sum，而前一个任务的最后一次 rescale 尚未执行。按 P 槽位保存的快照让延迟消费者读取到所属任务的统计量。

## 5. Softmax 与 rescale 的标量开销优化

### 5.1 DN 完整 KV tile 直接分发

DN softmax 中 `m` 表示当前 KV tile 的长度。新增 `m == 128` 分支，直接选择 `MAligendTileNum::Four`，并传入固定的 `mRound=128`、`mFirstTile=32`、`blockStride=65` 等参数。

完整 KV tile 不再执行原有 `m % mAligendTile`、`m / mAligendTile` 和多路尾块分发。非完整 tile 保留原计算，首个 KV tile 与后续更新仍使用不同模板实例；底层数值计算函数保持复用。

### 5.2 通用 LSE 的单 head 连续写回

`WriteGroupedLse` 增加快分支，条件为：只有一个 Q head、没有完全 masked 行，且没有需要处理的 SWA 无效行。

原分支先计算 LSE，再通过 `Brcb` 扩展数据，以多个短块写出。满足上述条件时，LSE 在 GM 中连续，可以直接计算到 GO scratch，然后执行一次连续 `DataCopyPad`。

这里使用受 `MTE3_V(EVENT_ID4)` 保护的 GO，而不是 LM。LM 同时服务后续 online softmax；若 LSE DMA 尚未结束就被下一次 softmax 覆盖，会破坏输出。其他分组或无效行情况继续走原分支。

### 5.3 专用 `RescaleDnMha128`

通用 rescale 需要处理多种列宽、分组输出、mask、partial output，以及运行统计量句柄切换。新函数利用已确认的 MHA/D128 条件：

- 每个任务只有一个 head，列宽固定 128；直接按 DN 的 32 行对齐规则切分到两个 AIV。
- 通过局部句柄读取最终统计量快照，不再替换和恢复 `gmUbTensor32`、`glUbTensor32` 成员句柄。
- 按首个/最后一个 KV tile 选择既有复制、rescale 或最终除法 VF，保留数值计算方式。
- O 按实际 `outputStride` 写回，LSE 按每个 AIV 的连续行区间写回。
- 输出 cast 保留原舍入选择：BF16 使用 `CAST_RINT`，FP16 使用 `CAST_NONE`。

同步上，非空 AIV 在重用 GO 前等待事件 4；O 的 DMA 完成后才允许把 GO 复用于 LSE；最后发布事件 4 供下次重用。没有有效行的 AIV 仍消费并释放 PV flag，然后返回，避免另一侧等待不到完成信号。

这部分对 AIV scalar 的改善最明显。诊断构建的前后对比中，仅从连续 LSE 版本切换到专用 rescale 后，AIV scalar 从 811.625 µs 降到 350.200 µs，vector 时间基本不变（549.738 → 549.523 µs）。这是组合中的一次消融实验，不能把所有改动收益简单相加，也不能把诊断构建代替正式构建验收。

## 6. 已验证的性能效果

### 6.1 原始瓶颈

给定 [原始 profile](JD/PROF_000001_20261111131444299_01504670ABEPRLDQ) 中，FAInferDn 相对 FlashAttentionScore：

- AIV scalar 高 31.63%。
- AIC scalar 高 6.49%。
- AIV 总时间高 2.51%。
- AIV icache miss rate 为 0.001，对照为 0。
- Vector、MAC 和主要搬运忙碌时间反而更低，因此优先处理标量控制与流水组织。

### 6.2 同协议正式构建对比

场景为 BF16/TND、`Q=K=V=[131475, 8, 128]`、512 个变长 batch、序列长度 8～276、无 mask、scale=1，并计算 LSE。设备为 Ascend950PR 的 device 2，CANN 9.2.0-beta.2，未锁频。

每个实现采集 10 组相邻的 ArgMax warmup → 目标 kernel，核验实际 launch 映射，全部目标样本取中位数。下表使用该协议重新采集的原版、优化版和 fusion，未混用原始 profile 的不同采样结果。

| 指标（µs） | 原 FA3 | 优化后 FA3 | FlashAttentionScore | 相对原 FA3 |
|---|---:|---:|---:|---:|
| Task Duration | 1043.090 | 943.222 | 1025.570 | −9.57% |
| AIV scalar | 734.669 | 353.099 | 558.341 | −51.94% |
| AIC scalar | 643.608 | 543.268 | 605.307 | −15.59% |

优化后 Task Duration 比 fusion 低 **8.03%**。AIV、AIC scalar 也分别比 fusion 低 36.76% 和 10.25%。全部样本及计数器见 [acceptance_metrics.json](JD/20261111/scalar_opt/acceptance_metrics.json)。

独立 Memory 采集显示 AIV 主存读取带宽从 0.287 降到 0.039 GB/s，接近 fusion 的 0.030 GB/s。这支持“减少标量控制相关访存”的判断，但该计数器不能单独区分栈访问和 metadata 访问，不能据此断言栈溢出是唯一原因。

### 6.3 完整 JD 命令复测

执行 `python JD.py --output=./JD/20261111/scalar_opt/final_jd`，保留原命令默认参数，使用新目录保存结果。以下是默认 20 次 kernel 采样的前向中位数，单位 µs；它与上一节采样协议不同，应分别比较。

| 场景 | 原 FA3（JD/4） | 优化后 FA3 | 同次 fusion | FA3 相对原版 |
|---|---:|---:|---:|---:|
| qwen3_3b | 54.694 | 55.210 | 46.913 | +0.94% |
| qwen3_4b | 76.198 | 76.169 | 81.168 | −0.04% |
| t5_encoder | 1047.824 | 959.772 | 1041.268 | −8.40% |
| t5_decoder | 223.791 | 221.436 | 255.397 | −1.05% |
| t5_cross | 754.036 | 735.452 | 757.632 | −2.46% |

原命令中 T5 encoder 前向比 fusion 快 **7.83%**。其余细节，包括 backward 和 Host 端时间，保存在 [comparison.csv](JD/20261111/scalar_opt/final_jd/comparison.csv)。约 1% 的变化没有被单独认定为稳定改善或退化。

## 7. 正确性、同步和复现依据

当时的正式构建已通过 **233 项测试**，日志见 [precision.log](JD/20261111/scalar_opt/precision.log)。下表记录历史验证范围；Q128 专项测试已随重复提交清理。

| 测试文件 | 用例数 | 主要覆盖 |
|---|---:|---|
| [JD forward](tests/test_flash_attn_npu_v3_950_jd_forward.py) | 5 | 五个真实 JD 场景，对照独立数值参考 |
| [Causal positions](tests/test_flash_attn_npu_v3_950_causal_positions.py) | 76 | FP16/BF16、布局、GQA、不同 Q/K 长度、softcap，以及 4 项冷启动无 triu 检查 |
| Q128（历史专项测试） | 128 | Paged/GQA、不同维度、mask 与 metadata/FD 等回退路径 |
| [DN MHA](tests/test_flash_attn_npu_v3_950_dn_mha.py) | 24 | 两种 dtype × 三种布局 × LSE 开关 × softcap 开关；空 KV、尾块、多任务波次和重复调用 |

另有 [Host 任务映射测试](tests/test_flash_attn_npu_v3_950_mha_tasks.cpp)，穷举 head 数 1～32、Q 长度 0～1025，比较新增 MHA 映射与通用非融合映射。

同步检查保留了 52 项结构性候选，flow analyzer 为 0 项发现；完整候选及人工数据流说明见 [sync_review.md](JD/20261111/scalar_opt/sync_review.md)。正则扫描结果不等价于同步正确性证明，不能将“flow analyzer 无发现”写成“静态检查完全无风险”。

验证产物的 FA3 二进制 SHA256 为：

```text
e27ea813d2cc56685d5ee3d0d6be4c017e159d8fd8ab76067efcd3143dfe5c91
```

可追溯资料包括[构建依赖检查日志](JD/20261111/scalar_opt/build_dependency.log)、[源码/二进制身份记录](JD/20261111/scalar_opt/final_identity.json)、[原命令结果](JD/20261111/scalar_opt/final_jd/results.json)以及[完整性能分析](JD/20261111/scalar_opt/analysis.md)。这些 `JD` 路径是当前工作区的验证产物，分享本文或提交到其他环境时需要另行保留相应证据文件。

本次文档生成重新核对了 diff、源码快照和已有验证记录，没有重新运行 NPU 测试或性能采集。上述结果适用于该源码快照；后续修改代码后需要重新验证。

## 8. 当前效果的适用边界

- 主要性能结论是所给 T5 encoder 无 mask 前向场景的结果，不表示所有 shape、dtype 和布局都获得同等幅度收益。
- Qwen3_3b 前向仍比同期 fusion 慢约 17.69%；backward kernel 未在本次修改中优化。
- AIV icache miss rate 仍为 0.001，高于 fusion 的 0；这项差距没有消除。
- 相对原 FA3，部分忙碌时间增加：AIC MTE2 为 795.911 → 830.902 µs，Fixpipe 为 414.630 → 463.936 µs，AIV MTE3 为 328.780 → 405.550 µs，但仍低于同期 fusion。流水线计数器存在重叠，不能相加解释端到端延迟，也不能声称所有指标都下降。
- 汇编对比中的 AIV 入口栈分配仍为 2568 字节；入口包含通用回退路径，不能把静态代码大小或栈帧大小直接当作热点执行开销。实测 scalar 时间和消融实验是性能判断的主要依据。

## 9. 追加：950 FA3 icache prefetch（仅编译检查）

参考 [icache_prefetch_20261110.patch](icache_prefetch_20261110.patch) 中 910 FA3 的实现，在 950 的 [fwd_kernel.cpp](csrc/ascend950/flash_attn_npu_3/fwd_kernel.cpp) 接入 `AscendC::ICachePreLoad(2)`。

- 为 `FAIKernel950` 增加默认关闭的编译期模板参数 `EnableICachePrefetch`。
- `RunFAInfer` 和 `RunFAInferDn` 根据 `TND + normalCache + NO_MASK + !HasSoftcap` 启用该参数。
- 仅在 AIV、非 FlashDecode 的实例中，于 softmax 准备和 QK-ready 等待之前执行 prefetch；通过 `if constexpr` 消除不适用实例中的调用，不新增运行时 prefetch 分支。
- 覆盖 FP16/BF16 的普通及 DN 前向路径，LSE 开关均可；未扩展到单独实现的 FP8 kernel，也未修改 backward。
- 本机 CANN 的 DAV_3510 实现支持该 API，底层调用 `preload`。

按本次要求，仅构建 `FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3` 的完整扩展，不运行精度测试或性能采集。因此尚无证据说明 prefetch 对 icache miss rate 或 Task Duration 的实际收益；第 6～8 节数据仍属于追加前版本。

编译状态：完整 950 v3 扩展编译、链接及原位产物更新成功，构建退出码为 0；未运行精度或性能验证。日志：`/tmp/fa3_950_icache_prefetch_build.log`。

追加 prefetch 后的二进制 SHA256：`0e051052c1b70dacd76539111b561fc996e9b7ed4cac28019356c75214588aa6`。

本节记录首次追加 prefetch 时的范围。此后现有源码已将条件放宽为 `TND && !HasSoftcap`；下面 SWA 对比的基线保留了这些既有修改，不使用本节的旧二进制。

## 10. 追加：FP16/BF16 SWA 去矩阵化及随机窗口对比

参考 FP8 的位置式窗口判断，SWA 现在也不再需要 triu 矩阵。右对齐 Q 位置为 `center = q + actual_k_len - actual_q_len`，可见条件为 `center - left <= k <= center + right`。VF 用无符号距离比较合并上下界；负窗口和超长窗口按实际 Q/K 长度裁剪。

- `mha_fwd.cpp` 移除普通 tiling 与 metadata 两条路径的矩阵生成，保留逻辑 mask 类型及 metadata 偏移。
- `fwd_kernel.cpp` 移除 GM mask 绑定和只服务 mask DMA 的 V→MTE2 event 4/5，向边界 tile 的 softmax 传递位置和窗口参数。
- `online_softmax.hpp` 复用位置式 VF，删除 Pre/Next mask 搬运、UB mask 绑定和六个旧矩阵 mask VF。完整可见 tile、GQA 行回绕、softcap、尾块及全遮挡行处理保留。共享 UB resource plan 未重新压缩。

新增 [随机用例脚本](benchmarks/swa_jd_cases.py)、[采集入口](benchmarks/collect_swa_jd.py)、[采集 runner](benchmarks/swa_jd_runner.py) 和 [使用说明](benchmarks/swa_jd.md)。复用 JD.py 的五个 shape、输入种子和长度生成逻辑，固定窗口种子 `20261111`，每个 shape 随机抽三组窗口，覆盖 BF16/FP16，共 30 例。causal 的实际右窗口仍为 0，传入值和实际值均记录；短序列允许重复抽样。

前后均以每例 10 组相邻预热/目标调用的 kernel 时间中位数比较。下表为三个随机窗口的耗时下降比例中位数：

| 场景 | BF16 | FP16 |
|---|---:|---:|
| qwen3_3b | 40.72% | 39.16% |
| qwen3_4b | 35.31% | 33.10% |
| t5_encoder | 31.92% | 30.82% |
| t5_decoder | 1.66% | 0.78% |
| t5_cross | 0.35% | 0.35% |

测量未锁频，短序列的小幅变化应视为收益有限或基本持平；不包含旧实现首次创建 4 MiB mask 的额外开销。完整数据、频率观察和统计边界见 [SWA 分析报告](JD/20261111/swa_positions/analysis.md)，逐例耗时见 [对比表](JD/20261111/swa_positions/comparison/comparison.md)。

验证：完整 950/v3 扩展编译通过；两轮回归为 196 passed 和 92 passed（其中 32 项重复，共 256 个不同测试）；30 组 JD 场景的抽样行输出与基线逐元素相同，LSE 通过独立参考校验。新增 [SWA 测试](tests/test_flash_attn_npu_v3_950_swa_positions.py) 包含四项独立进程的冷启动无 triu 检查。性能 trace 中，基线两个进程共出现两个 Triu kernel，候选为零。

基线 SHA256：`796cfb3545c3d6272098011891368fefe9dc45de63f767c99c15ecdfde96a799`；候选 SHA256：`7efc520a1636dc54220c03e721cc08b244bf86d8c3b2e17411237be56ff01bf6`。本节是 SWA 前向的独立结果，不替代第 6～8 节无 mask 场景的历史验收记录，也不声明 backward 加速。

## 11. 追加：回收 UB mask 预留，Dv192 使用双 PV buffer

第 10 节移除了 mask 矩阵读写，但资源规划仍保留 16 KiB 地址区间。本次修改
`fwd_resource_plan.h`，删除 `UbPlan.mask`，将两套布局中的 `finalMax/finalSum`
前移，保留其三槽快照及生命周期。相同 Dv/Q tile 下，FP16/BF16 的 No mask、
causal、SWA 统一回收这部分空间；FP8/backward 不在本次布局修改范围。

每个 AIV 按当前 248 KiB 预算：

| Dv / Q tile | PV buffer 前→后 | UB 占用前→后 | 修改后尾部剩余 |
|---|---|---|---:|
| 128 / 128 | 2→2 | 243.25→227.25 KiB | 20.75 KiB |
| 160 / 128 | 2→2 | 235.75→219.75 KiB | 28.25 KiB |
| 192 / 128 | 1→2 | 211.75→243.75 KiB | 4.25 KiB |
| 256 / 128 | 1→1 | 243.75→227.75 KiB | 20.25 KiB |
| 192 / 64 | 2→2 | 243.25→227.25 KiB | 20.75 KiB |

Dv 指 V/output 的 head dimension。Dv192 第二槽需要 48 KiB，回收 mask 后能够容纳；
Dv256 双槽仍需 291.75 KiB，不能启用。已有 `otmpStages_` 控制 buffer 轮转和同步旗标，
因此无需新增同步协议。D128/小 Q 历史布局的其他间隙保持原样，未计入表中尾部剩余。

现有 SWA tiling 策略还会随容量判断改变：Dv192 的有限窗口宽度在 [256,511]，且任务数
足够时，从 Q64 切为 Q128。不同 mask 场景可能因此选到不同 Q tile，进而使用不同布局。

在设备 7 上，主套件 B=4、Sq=Sk=1024、Hq=Hkv=8、Dqk=128，测得：

| Dv192 场景 | BF16 前→后（us） | 耗时下降 | FP16 前→后（us） | 耗时下降 |
|---|---:|---:|---:|---:|
| No mask，Q128 单槽→双槽 | 109.246→107.231 | 1.84% | 108.929→107.105 | 1.67% |
| causal，Q128 单槽→双槽 | 64.087→61.650 | 3.80% | 63.523→61.977 | 2.43% |
| SWA(641,63)，Q128 单槽→双槽 | 69.805→69.319 | 0.70% | 69.907→69.391 | 0.74% |
| SWA(257,63)，Q64→Q128 | 61.194→56.636 | 7.45% | 61.618→56.825 | 7.78% |

最后一行两版实际都为双槽，收益包含更大的 Q tile，不能归因于单槽改双槽。
小于约 1% 的变化按基本持平理解，其余也只代表当前形状。

JD 五个 Dv128 场景原本已使用双 PV buffer。此次随机窗口对照中，**Qwen3_3b 的
BF16/FP16 分别回退 2.54%/2.50%**；Qwen3_4b 耗时下降 1.22%/0.75%，T5 三个场景
基本持平。本次不是全场景加速，Qwen3_3b 回退原因尚未专项定位。

验证：950/v3 扩展及 AICPU metadata 编译通过；2 项 CPU 资源/调度测试和 180 项 NPU
回归通过；36 个性能用例的抽样输出与基线逐元素相同，LSE 通过独立参考。新增边界测试
覆盖 Dv192/208、SWA 宽度 255/321、GQA、变长尾块和重复调用。

正式性能分为完整 20 例主套件和完整 10 例 JD 套件，各例保留 10 组 ArgMax warmup/target，
取全部目标耗时中位数。前后 launch 映射完整，未锁频，两个 die 的频率记录均为 1650 MHz；
主套件最大 spread 为 4.26%/3.72%，JD 为 2.42%/3.79%。六个 Q64 对照波动较大，
其中确认轮两例超过 10% 门限，不作稳定收益或回退结论。此前频率记录不完整的采集和
失败的 msprof op 回退探针均保留，未混入正式结果，也未删除单个样本。

方法和复现脚本见 [UB/PV benchmark 说明](benchmarks/ub_pv.md)，完整结果、限制和身份见
[UB/PV 分析报告](JD/20261111/ub_pv_double/analysis.md)，逐例数据见
[性能对比索引](JD/20261111/ub_pv_double/comparison/comparison.md)。

本次基线二进制 SHA256：`7efc520a1636dc54220c03e721cc08b244bf86d8c3b2e17411237be56ff01bf6`；
候选：`d06f4eb438cdf5bae0c3301a6f5836f2e8fb220c82ab9cf6023f01a3f0b251ed`。


## 12. 追加：950 FA3 autogen 编译优化

继续使用 autogen 显式实例化，将 FP16/BF16 的每个 dtype/layout 重编译单元按
mask/执行路径和 softcap 拆为八个叶子，每个只包含四种 LSE/cache 组合。
六个原入口变为轻量 host 路由；FP8 的三个单元只包含自身实现。兼容同步写入的 backward
拆分后，生成器统一输出 111 个单元，加 API/combine 共 113 个 ASC 单元，AICPU metadata
继续单独编译。kernel 叶子移除 torch/extension.h；计算、tiling 和设备侧流水保持不变。

生成器仅更新内容变化的文件，避免重复运行破坏增量缓存。setup.py 增加 --parallel/MAX_JOBS
并发控制，自动识别 affinity 与继承的 cgroup-v2 CPU 配额；本机显示 256 个逻辑 CPU，但
当前用户配额为 16 核，默认因此使用 16 个 worker。日志按单元记录编译耗时，输出加锁。

| 构建 | 实测耗时 |
|---|---:|
| 原始 11 个 ASC 单元，全量 | 1361.98s，约 22 分 42 秒 |
| 最终 113 个 ASC 单元，全量 | 406.59s，约 6 分 47 秒 |
| 最终代码，无改动增量 | 5.20s，113 个对象全部复用 |

原始测量期间与中间构建存在 CPU 配额争用，最终轮在本任务其他编译结束后执行；这些是
共享机观测值，不作为严格受控的加速倍率。全量均不复用对象文件，不清空操作系统页缓存。

验证：5 项 CPU 测试通过，包含 384 组 forward 与 48 组 backward 路由；353 项 NPU 回归
通过，覆盖 FP16/BF16、FP8 与全部 backward 模板的梯度参考。最后全量重建后又通过
64 项 NPU 冒烟及 5 项 CPU 检查。两版 312 个 kernel 入口集合完全一致；数值计算源码未变。

构建方法见 [autogen 说明](csrc/ascend950/flash_attn_npu_3/autogen/README.md)，完整耗时、
验证记录与测量边界见 [编译优化报告](JD/20261111/compile_autogen/analysis.md)。


## 13. 追加：供给流水、PV/LSE与任务遍历优化（2026-11-12）

当前为候选47，完整设计、各阶段对照和验证范围见[供给流水优化说明](supply_pipeline_optimization_20261112.md)。相对前一完整版本25，本轮新增：

- `rescale_o.hpp`：FP16/BF16双OTmp槽首块结果保留到第二块融合读取，消除LO→GO复制；需要立即清零、单槽、单KV块、FP8和DN专用helper保留旧流程。消费后才释放两个PV槽，保持GO复用保护。
- `fwd_resource_plan.h`、`rescale_o.hpp`：增加256 B独立LSE scratch，单头有效行的LSE计算不再等待O DMA读取GO结束；MTE3_V(6)保护scratch复用。No mask/causal/SWA共享布局，所有FP16/BF16布局相对第11节多0.25 KiB，单双PV槽和QueryTile边界不变。Dv192/Q128总占用244 KiB、尾部剩余4 KiB。
- `fwd_kernel.cpp`：实际Q长度相同时复用任务几何；非FD/非DN/非跨组融合路径使用单调snake分核，矩形单头任务按head-major解码。任务集合与AIC/AIV顺序一致，物理/实际长度与地址推进仍独立处理，无模型或固定shape分支。
- 保留先前阶段的单头解码化简、普通Q双buffer/紧凑K、Q-ready等待下移、固定L0单轮维度、OTmp索引化简和最终Rescale/Cast融合；未保留无收益的额外K/V预取或P四槽方案。

设备5最终交错测量的hot两轮均值如下（μs）：

| JD 场景 | 最初基线（μs） | 上一阶段25（μs） | 当前47（μs） | FAScore（μs） | 相对最初基线 |
|---|---:|---:|---:|---:|---:|
| qwen3_3b | 54.623 | 50.633 | 45.368 | 47.841 | -16.94% |
| qwen3_4b | 78.151 | 74.679 | 63.192 | 81.785 | -19.14% |
| t5_decoder | 221.376 | 159.706 | 137.928 | 255.774 | -37.70% |
| t5_encoder | 958.354 | 未测 | 955.385 | 1046.003 | -0.31% |
| t5_cross | 738.910 | 未测 | 735.382 | 761.864 | -0.48% |

3B相对上一版本25降低10.40%；追加独立进程首调中位数当前107.694 μs（n=6）、FAScore107.782 μs（n=5），按持平处理。encoder/cross不足0.5%的自身变化按基本持平处理。所有采集前后快照clean，但设备不独占、不锁频，不能排除间隙内干扰。

52个受影响TU完整构建通过；1113项前向回归通过、48项backward排除；新增资源容量/对齐/非重叠检查通过。racecheck/memcheck各4目标launch、4clean、0结构化错误、exit0且应用正常退出，仅覆盖BF16/TND/causal/block0多波。完整库与动态检查目标device ELF一致。当前扩展SHA256为`781cf108cbe04eabfd7d2f5caf166a02c764f65897fdd9fe6fffdb8db0b94e94`。
