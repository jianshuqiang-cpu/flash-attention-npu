# Ascend 950 FA4 相对 PR 458 的前向优化

比较基准为 PR 458 的提交 `bf744323e111b7f970c0fbc7332398a129a5a9d3`。分支 `fa4-950-onerec-integrated` 从 `pr458-local-sync` 的 `fd04c54` 切出，保留 PR 458 的祖先历史。源码和测试的最终提交为 `e687464`。相对基准的代码差异集中在 Ascend 950 FA4 的 18 个源码文件和一个测试文件。[PR 458](https://github.com/MinghuasLab/flash-attention-npu/pull/458)

| 提交 | 在本分支中的作用 |
|---|---|
| `d3ccfdb` | 移植 `78f83d8` 中的位置掩码、DN 完整 tile、Rescale 门控和 LSE 写回，整理 GQA 尾块的任务接口。 |
| `fd04c54` | 在已有跨任务流水中接入跨 KV head 分段，统一任务切分，优化 batch 定位并修复固定布局 KV 地址。 |
| `e687464` | 恢复 PR 的单 Q head 搬运，融合最终 Cast，化简 L0 循环，引入紧凑 K 槽和双 Q 槽，扩展回归用例。 |

## 掩码、softmax 和输出

| 修改 | 相对基准的行为变化及适用条件 |
|---|---|
| Causal / SWA 位置掩码 | Host 取消设备 triu 矩阵的构造和传递，移除 `device_triu_mask.h` 和相应的掩码大小常量。边界 tile 的 softmax 按 Q/K 序列位置与左右窗口判断可见性，随之移除原掩码从 GM 到 UB 的搬运及同步。Q 和 KV 长度不同时仍按各自的序列位置对齐。 |
| DN 完整 KV tile | DN softmax 在本次 KV tile 恰为 128 个 token 时使用已知的分块参数，省去通用尾块的运行时分支。部分 tile 继续按有效长度处理。 |
| Rescale 跳过条件 | 基准已有 `DmAllOne` 检查。当前仅在普通前向满足 `flashDecodeFlag == 0 && maxKvSeqlen > 4096` 时打开该检查，减少短 KV 路径上的检查开销。 |
| 单 head LSE 写回 | 当前段只有一个 Q head，且输出行连续、无需逐行处理掩码时，将 LSE 写到有 DMA 事件保护的输出暂存区，再一次连续写回 GM。其他情况保留逐行处理。 |
| 最终 Cast 融合 | 普通 FP16/BF16 前向在最后一个 KV tile 的归一化中直接转换为目标输出类型。FlashDecode 需要后续 combine 的部分输出继续以 FP32 写出。这个改动可独立于 L0 化简。 |

对应实现见 [Host 前向](csrc/ascend950/flash_attn_npu_4/mha_fwd.cpp)、[softmax](csrc/ascend950/flash_attn_npu_4/online_softmax.hpp) 和 [RescaleO 与 LSE 写回](csrc/ascend950/flash_attn_npu_4/rescale_o.hpp)。

## 任务切分、流水与地址

- **同一 KV head 的 GQA 尾块。** PR 已支持由多个 Q head 共享一个 KV head 的尾块。本分支在统一的 `MakeShape/Decode` 中按尾部有效 Q 行数和 tile 容量选取组合宽度。以 OneRec Qwen3 4B 为例，PR 基准与本分支的分组调度均为 1152 个任务。
- **跨多个 KV head 的尾任务。** Host tiling、AICPU metadata 和 AICore 共用任务几何。分组上限结合 batch、KV head 数、可用核数及 16 行对齐后的容量。一个任务可拥有多个 KV head，执行时每个段只使用一个 KV head，段内 Q heads 共享 K/V。每段分别定位 K/V、softmax 统计槽、O 和 LSE，并进入已有的 QK → softmax → 延迟 PV 流水。
- **与 PR 的实现关系。** PR `189c83d` 引入的跨任务流水仍参与执行。本分支替换了 `bf74432` 的 `FusedGroups/Segmented` 搬运方式，移除 `fused_copy.hpp` 中的 `CopyFusedSegmentToUB`。每个核心处理完最后一个任务的最后一段，或遇到空段时，按流水依赖排空待处理的 PV。Host、AICPU 与设备端必须对每批任务数、首 head 和段宽度采用同一规则。
- **batch 定位和缓存的物理步长。** 两个 metadata 生成入口记录各 batch 的任务数是否相等。相等时设备端从任务编号直接算 batch ID，不等时逐批推进。TND 使用前缀和定位，固定布局使用物理分配步长，分页布局使用相应批次的页表。特别是带 padding 的连续 KV cache，批间地址按分配容量推进，实际有效长度仅决定 attention 范围。
- **metadata 语义。** `uniformBatchTaskCount` 使用原预留位置，零值走逐批定位。`headGroupBudget` 取代 PR 的布尔量 `fusedCrossGroup`，表达允许的分组宽度。切换版本时应使用对应版本生成的 scheduler metadata，尤其是启用跨 head 分组的任务。

设备公共头中的 `GetQNBlockTile` 保留原有的组内宽度计算，和新的跨 head 调度分别维护。固定布局调度继续通过 `kv_seqlen_allocated` 传入物理 KV 长度。[核端公共函数](csrc/ascend950/flash_attn_npu_4/kernel_common.hpp) · [启动参数](csrc/ascend950/flash_attn_npu_4/fwd_dispatch.hpp)

对应实现见 [共享任务几何](csrc/ascend950/flash_attn_npu_4/fused_task.h)、[Host tiling](csrc/ascend950/flash_attn_npu_4/tiling.cpp)、[AICPU metadata](csrc/ascend950/flash_attn_npu_4/fa_metadata.aicpu)、[tiling 字段](csrc/ascend950/flash_attn_npu_4/tilingdata.h) 和 [设备前向主循环](csrc/ascend950/flash_attn_npu_4/fwd_kernel.cpp)。

## Cube 搬运、L0 循环和 L1 布局

| 修改 | 实现和依赖 |
|---|---|
| 恢复单 Q head 的搬运 | `qNBlockSize == 1` 时，把带 token stride 的 `[S,D]` 作为一个矩阵做 ND2NZ 搬运。该快路径来自 PR 基准，集成期间曾移除，最终恢复并适配当前 Q 槽索引。多 head 使用对应的多矩阵搬运路径。 |
| L0 单轮化简 | 对普通 FP16/BF16，有效 Q 段行数和 KV tile 各自不超过 128。ND/DN QK 将 M、N 循环固定为一轮，PV 将 M、K 循环固定为一轮。主机和 AICPU 共用上限，核端以静态断言核对所需的 L0 tile 大小。FP8 继续按实际尺寸计算循环次数。 |
| 紧凑 K 槽 | Host 与 AICPU 共用 `MakeQkL1Layout`。在等宽且 `1 ≤ Dq = Dk = Dv ≤ 256` 的受支持输入上，每个 K 槽按一个 KV tile 和 `align16(D)` 分配。普通 FP16/BF16 的计算粒度为 KV128，FP8 的 KV tile 为 256。 |
| 双 Q 槽 | Q 在两个 L1 槽间按非空段轮换。同一段的各 KV tile 复用该段的 Q，最后一次 L1 消费后释放对应槽，两个槽使用独立的 MTE1→MTE2 空闲事件 0 和 5。普通 ND/DN 和 FP8 的装载与消费使用同一槽号。普通核在确定 P/V 基址前校验 L1 总容量，单槽 metadata 仍按单槽寻址。等宽 FP16/BF16、D=256 时 Q/K/V 各占 128 KiB、P 占 96 KiB，共 480 KiB。 |

对应实现见 [QK 与单 head 搬运](csrc/ascend950/flash_attn_npu_4/qk_matmul.hpp)、[DN QK](csrc/ascend950/flash_attn_npu_4/qk_matmul_DN.hpp)、[PV](csrc/ascend950/flash_attn_npu_4/pv_matmul.hpp) 和 [FP8 核的 Q 槽切换](csrc/ascend950/flash_attn_npu_4/fwd_kernel_quant.cpp)。

## 回归验证和性能解读

新增的回归用例用 CPU FP32 对照 O 和 LSE，覆盖短 Q 跨 KV head、奇数 head 数、不同 Q/K/V 宽度组合、等宽 D=193/256 下跨 KV tile 的 Q 槽切换，以及有效 KV 长度小于固定缓存容量的 batch 地址。用例分别走预计算 metadata 与 Host tiling。该源码此前在 Ascend 950DT、CANN 9.2.1 上运行 metadata 和 FP8 两组测试，结果为 191 项通过、11 项跳过。源码提交后在该分支复跑新增的两类 metadata 用例，5 个用例全部通过。[metadata 测试](tests/test_flash_attn_npu_v4_metadata.py) · [FP8 测试](tests/test_flash_attn_npu_v4_fp8.py)

OneRec 的双 Q 槽单项对照以“已恢复单 head 搬运、已包含 Cast 和 L0 的单 Q 版本”为基线。两张卡上，T5 Encoder 主核耗时下降 5.14% / 5.23%，Qwen3 3B 下降 2.45% / 2.97%，Qwen3 4B 下降 5.47% / 6.25%。T5 Decoder 主核耗时上升 5.04% / 4.94%，T5 Cross 两张卡方向不同。每次调用前均 flush 256 MiB L2；这些百分比仅反映双 Q 阶段的前向主核变化。各阶段采用不同基线，整分支相对原 PR 的总收益需由两端直接对测。

查看完整源码差异和运行对应回归：

```bash
git diff --stat bf744323e111b7f970c0fbc7332398a129a5a9d3..e687464
git diff bf744323e111b7f970c0fbc7332398a129a5a9d3..e687464 -- csrc/ascend950/flash_attn_npu_4 tests/test_flash_attn_npu_v4_metadata.py
python -m pytest -q tests/test_flash_attn_npu_v4_metadata.py tests/test_flash_attn_npu_v4_fp8.py
```
