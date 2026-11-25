# v7 算子 vs 反量化前 bf16 路径（性能对比，最终版）

口径：
- **bf16**：未量化基线，即 `csrc/ascend950/flash_attn_npu_3` 原始路径（bf16 KV + host tiling），
  用 `b0e5e56` 独立 worktree 编译验证与本包一致（差异 ±3% 噪声内）。
- **v7**：混存 KV（bf16 Q + fp8 KV）+ `scheduler_metadata`（AICPU tiling），FD 形态经
  `prepare_fd_metadata` 静态化。
- **wall**：端到端单次调用（含 Python/pybind、C++ host、下发、device、同步），同进程交替 A/B，
  10~30 轮中位数。
- **kernel**：profiler `Task Duration` 汇总（只含 device 时间）。
- **NPUGraph**：`graph.replay()` 的 wall。
- 精度：v7 与 host tiling 的 fp8 路径**逐位一致**；与 bf16 的差异是纯 KV 量化误差
  rel ≈ 3.65e-2 ~ 3.80e-2（实现误差另测 2.1e-3）。

| 场景 | bf16 wall | bf16 kernel | v7 wall | v7 kernel | wall 提升 | kernel 提升 | NPUGraph | graph 提升 |
|---|---|---|---|---|---|---|---|---|
| prefill 4096/8192 SWA512 | 0.351 ms | 0.171 ms | **0.233 ms** | 0.183 ms | **1.5x** | 0.93x | 0.202 ms | **1.74x** |
| prefill 4096/8192 全注意力 | 1.503 ms | 1.326 ms | **1.194 ms** | 1.146 ms | **1.26x** | **1.16x** | 1.167 ms | **1.29x** |
| decode 12x4/8192 SWA512 | 0.179 ms | 0.0113 ms | **0.057 ms** | 0.0136 ms | **3.12x** | 0.83x | 0.033 ms | **5.43x** |
| decode 12x4/131072 SWA512 | 0.175 ms | 0.0112 ms | **0.058 ms** | 0.0136 ms | **3.04x** | 0.82x | 0.033 ms | **5.30x** |
| decode 12x4/32768 全注意力（FD） | 0.473 ms | 0.323 ms | **0.234 ms** | 0.186 ms | **2.02x** | **1.73x** | 0.190 ms | **2.49x** |
| decode 12x4/131072 全注意力（FD） | 1.337 ms | 1.155 ms | **0.700 ms** | 0.634 ms | **1.91x** | **1.82x** | 0.664 ms | **2.01x** |

## 结论

1. **生产形态（step5 decode，12x4 + SWA512）**：wall 0.175~0.179 ms -> 0.057 ms（**3.0~3.1x**），
   取 NPUGraph 重放后 0.033 ms（**5.3~5.4x**）。该形态 kernel 只有 0.014 ms，端到端收益来自
   去掉每次约 0.17 ms 的 host tiling；剩余耗时主要是 launch/同步/调用固定开销。
2. **全注意力形态**：fp8 KV 的带宽收益显现，kernel 1.16x（prefill）/ 1.73~1.82x（FD decode），
   叠加 host tiling 去除后端到端 1.26x / 1.91~2.02x。
3. **滑窗小 kernel 的 kern 列略慢**（0.82~0.93x）：fp8 反量化在几十微秒级 kernel 上占优的
   带宽已被固定开销淹没，属预期；wall 仍是大幅提升。
4. bf16 路径本身无回归。

## 波动说明

共享机器，跨进程有 ±10% 漂移。最小形态（decode SWA，wall ~0.06 ms）单次对比波动较大，
上表取 30 轮交替中位数；实测区间：bf16 0.167~0.203 ms、v7 0.052~0.069 ms。
