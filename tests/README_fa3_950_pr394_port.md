# PR #394 → Ascend950 FA3 性能迁移

当前状态：迁移代码已实现并完成 950 原生编译、CPU 策略验证和 NPU 回归；本地扩展已安装为最终版本。性能收益依赖输入，以下同时记录收益与回退。

## 版本与环境

- 来源：[PR #394](https://github.com/MinghuasLab/flash-attention-npu/pull/394)，head `e5fc5f390162a232fa0589ae2ee86749fda7b6b3`。
- 目标：`JY_Perf/flash-attention-npu`，起始 HEAD `02f99629c79ab43cfe4683e79e32c179b2aebe40`。
- 基线：在 `/tmp/fa950-pr394-baseline/source` 隔离工作区从上述 HEAD 重新编译，仅构建 950 v3。
- 环境：Ascend950PR，CANN `/usr/local/Ascend/cann-9.2.0-beta.2`（安装信息 version=9.2.0，innerversion=V100R001C12SPC001B237），torch 2.10.0+cpu，torch_npu 2.10.0，Python 3.12。
- 激活：`source /usr/local/Ascend/cann/set_env.sh`，`source /home/jianshuqiang/repo/flash-attention-npu/.venv/bin/activate`。
- 早期性能使用物理设备 2；最终表格使用物理设备 3 上串行运行的基线/最终版。NPU 回归使用物理设备 5。原有未提交测试和 `.gitignore` 修改保留。
- 最终扩展 SHA256：`800b9ba32c74b2165fd981893cfbb1b0b34d03d69d4d62720699769c9a673d56`。

## 已实现的迁移

| 改动 | 实现位置 | 950 适配要点 |
|---|---|---|
| 共享任务划分 | `fused_task.h`，Host tiling、AICPU、AICore、FD base-task 枚举 | 区分完整 Q tile 和尾块，支持 Q64/Q128；FP8 保留原任务几何 |
| GQA 尾块组内融合 | `fused_task.h`、`fwd_kernel.cpp` | 按实际尾块行数选择 head 数；不以整条 Q 长度限制尾块 |
| 短 Q 跨 KV group 融合 | `qk_matmul.hpp`、`pv_matmul.hpp`、`fused_copy.hpp` | 每组使用正确的 K/V；Q/P 在 L1 中分段对齐，S/O 在两个 AIV 的 UB 中紧凑存储 |
| 紧凑向量计算 | `online_softmax.hpp` | 合并 Softmax/rescale，P 按组散写到 L1；分段可跨 AIV 有效行分界 |
| 分段与跨任务流水配合 | `fwd_kernel.cpp` | PvTask 保留每组行数、组数和 V head stride；首段消费 ready，末段发布 ready；L0 stage 计数包含各分段 |
| tiling tensor 缓存 | `fwd_tiling_cache.hpp`、`mha_fwd.cpp` | 线程局部、最多 32 项，键包含 device、stream 和完整已生成 tiling；每次仍校验当前长度，缓存 tensor 不再写入 |
| FD 编译期特化 | `fwd_kernel.cpp` | 分离普通/FD 实例，未跨组的实例也移除分段逻辑；DN 路径保持独立 |
| 减少重复标量计算 | `fwd_kernel.cpp` | 当前 batch 的任务形状只在 batch 切换时更新 |

原有 950 RegBase、L0C→UB、UB→L1 数据通路继续使用；没有搬入 910 的 GM 中转流水，也没有重写数学语义。

## 性能选路

第一轮全量跨组候选通过精度，但并非所有场景都更快。相同协议下，重编基线与该候选的 kernel 中位数分别为：Qwen3 4B 95.553→79.373 µs，T5 decoder 243.365→213.284 µs，T5 cross 749.979→1109.478 µs。最后一例由原 DN 路径切换到普通布局，分段计算的成本超过向量融合收益。

因此最终策略为：

- 完整 Q tile 保持逐 head 调度，GQA 尾块可以在同一 KV group 内合并。
- 跨 KV group 仅对短 Q（`q < qBaseTile`）、batch 足够覆盖 AIC、分段对齐后 L1/UB 容量允许的场景开启。
- 无 mask 且存在多个 KV tile 时保留 DN 路径，不开启跨组融合。
- 长 Q 的 MHA 尾块保留原逐 head 任务；这是目标平台 A/B 后的选择，不宣称照搬 910 全部启用条件。
- FP8 保留既有分组。950 的 Q tile 和 FD 调度依赖实际 KV 长度，因此没有套用 910 的“只回读 Q 长度”快路径。

## 验证

最终选路版本验证：

- CPU 策略回归 **23 项通过**，包括 25,600 组几何枚举与 UBSan：逐 `(query, head)` 计数验证无重复、无遗漏，检查组边界及双 AIV/L1 容量。
- NPU 回归 **764 项全部通过**（42.52 s）：既有 migration、Q128、pipeline、FD consistency/layout 701 项；新增融合任务测试 57 项；graph metadata 6 项。
- 新增 57 项包括输出/LSE 与直接 tiling/metadata 一致性 36 项，BSND/BNSD、分页 BBND/BNBD、独立 K/V page stride 16 项，缓存原地长度修改/双 stream/容量淘汰 1 项，ragged/空 Q/空 KV 4 项。
- 原生编译成功，`git diff --check` 通过。测试日志见 `batchtest/pr394_950/regression-final.log`、`cpu-final.log`。

当前保留的 NPU 回归入口如下。上面的 764 项是历史验证结果，其中 migration、Q128 专项测试已随重复提交清理，不包含在下面的命令中。

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python -m pytest -q -x \
  tests/test_flash_attn_npu_v3_950_fused_tasks.py \
  tests/test_flash_attn_npu_v3_950_pipeline.py \
  tests/test_flash_attn_npu_v3_950_fd_consistency.py \
  tests/test_flash_attn_npu_v3_950_fd_layout.py \
  tests/test_flash_attn_npu_v3_graph.py
```

新增测试入口：

```bash
python -m pytest -q tests/test_flash_attn_npu_v3_950_fused_policy.py
ASCEND_RT_VISIBLE_DEVICES=5 python -m pytest -q tests/test_flash_attn_npu_v3_950_fused_tasks.py
```

构建沿用项目入口：

```bash
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3 \
FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace
```

## 性能协议与证据

使用本工作区 `JD.py` 的五个 OneRec 场景和同一输入生成函数，只测 FA3 forward，保留 autograd。预热 10 次；单次同步 E2E 采样 5×30 次，另测批量摊销；profiler 采集 15 次 attention kernel，取中位数。输入生成/上传不计入计时，采用热缓存。E2E 包含两次 NPU Event record 与每次同步，不能和其它计时口径直接比较。

最终性能结果（负数表示耗时下降）：

| 场景 | 基线 kernel (µs) | 最终 kernel (µs) | kernel 耗时变化 | 基线 E2E (µs) | 最终 E2E (µs) | E2E 耗时变化 |
|---|---:|---:|---:|---:|---:|---:|
| qwen3_3b | 53.972 | 58.990 | +9.3% | 183.556 | 167.716 | -8.6% |
| qwen3_4b | 94.213 | 79.328 | -15.8% | 228.553 | 193.115 | -15.5% |
| t5_encoder | 1040.627 | 1046.414 | +0.6% | 1226.742 | 1262.305 | +2.9% |
| t5_decoder | 243.261 | 223.808 | -8.0% | 378.333 | 329.665 | -12.9% |
| t5_cross | 751.770 | 756.081 | +0.6% | 891.113 | 865.119 | -2.9% |

Qwen3 4B 与 T5 decoder 的 kernel 分别降低 15.8%、8.0%；T5 cross 已消除第一版全量跨组导致的明显回退。Qwen3 3B 的 kernel 仍比基线慢 9.3%，但包含 tiling 缓存收益的 E2E 降低 8.6%；T5 encoder E2E 增加 2.9%。因此不能将本次迁移描述为所有场景 kernel 均加速；后续若以 Qwen3 3B 单 kernel 为硬性指标，需要继续分析普通 MHA 调度/生成代码的额外成本。

设备 2 复测期间出现其他用户压测进程，部分 kernel 耗时翻倍；该轮标记为 `contended`，不用于上述结论。最终结果换到设备 3 串行采集。机器仍是共享环境，细小差异不作为普遍加速/回退结论；这里如实保留采样值。

原始数据保存在 `batchtest/pr394_950/`：

- `baseline_device3.json`、`final_device3.json`：最终比较，含二进制 SHA256、输入长度和逐次采样。
- `baseline.json`、`fusion_all_groups.json`、`fusion_cache_fd.json`：早期候选，解释选路依据。
- `final_device2.json`、`baseline_device2_contended.json`、`final_device2_contended.json`：最终版首次采集及受干扰的复测，不替换最终比较。
- `profiler/<阶段>/<场景>/kernel_details.csv`：已复制原始 profiler CSV 到工作区；JSON 同时保留采集时的原始路径。
- `bench_forward.py`、`comparison.md`、`build-gated.log`、`regression-final.log`、`cpu-final.log`：采集脚本、比较表及验证日志。

该归档目录沿用仓库已有忽略规则；没有修改 `.gitignore`。完整构建和 profiler 中间文件在 `/tmp/fa950-pr394-baseline`、`/tmp/fa950-pr394-work`。

复现性能（输出目录必须不存在，先激活环境）：

```bash
python batchtest/pr394_950/bench_forward.py "$PWD" /tmp/fa950-final-reproduce --device 3
```
