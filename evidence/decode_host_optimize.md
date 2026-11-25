# SWA decode 专项优化（v8 -> v9）：把端到端拆到可优化的最小项

## 1. 先测清楚时间去哪了（decode 12x4 / kv8192 / SWA512）

| 组成 | 耗时 | 说明 |
|---|---|---|
| C++ 主体（校验+tiling 读取+分配+下发） | **~5.0 us** | `chrono` 打点：entry→tiling 1.8、tiling→alloc 1.4、alloc→launch 1.7 |
| pybind 38 参数转换 | ~7 us | 与 C++ 主体分开测出 |
| Python 公共包装 + impl 前处理 | ~10 us | `_can_use_eager_fast`、布局探测、16 张量连续性检查 |
| device 侧 kernel 下发延迟 | **~16 us** | 用“空 kernel + sync”标定 |
| kernel 执行 | 13.6 us | bf16 同形态 11.2 us（fp8 反量化开销） |
| `torch.npu.synchronize()` | ~5 us | 标定值 |

结论：kernel 只占约 1/4，**固定开销（下发延迟 + 调用路径）才是大头**；下发延迟 16 us 与
NPUGraph replay 的下限一致，属 device 侧不可压；可压的是调用路径 ~17 us。

## 2. 优化：新增 lean 推理入口

`flash_attn_with_kvcache_lean(...)`：跳过通用包装的布局探测、参数归一化和 eager 派发检查，
直接调用 kernel binding；只填 decode 必需的参数。

调用方契约（写在 docstring 里）：paged cache、全部张量连续、传入 `scheduler_metadata`
（需要 FD 时先过 `prepare_fd_metadata`）、不支持 append-KV / rotary / 显式 num_splits。

调用路径耗时：public 约 22 us -> lean 约 9 us。

验证：新增 `test_mixed_kv_lean_entry_matches_public_api`（lean 与公共 API、以及 graph 重放
三者逐位一致），全套 **138 passed / 86 skipped**。

## 3. 结果（同进程，15~20 轮中位数，wall 含 sync）

| 场景 | bf16 优化前 | 公共 API + metadata | **lean** | NPUGraph replay | lean/bf16 | graph/bf16 |
|---|---|---|---|---|---|---|
| prefill 4096/8192 SWA512 | 0.330 ms | 0.217 ms | **0.208 ms** | 0.199 ms | 1.59x | 1.66x |
| prefill 4096/8192 全注意力 | 1.474 ms | 1.184 ms | **1.173 ms** | 1.164 ms | 1.26x | 1.27x |
| decode 12x4/8192 SWA512 | 0.173 ms | 0.045 ms | **0.036 ms** | 0.030 ms | **4.81x** | **5.73x** |
| decode 12x4/131072 SWA512 | 0.171 ms | 0.045 ms | **0.036 ms** | 0.030 ms | **4.82x** | **5.75x** |
| decode 12x4/32768 全注意力（FD） | 0.437 ms | 0.214 ms | **0.205 ms** | 0.190 ms | 2.14x | 2.29x |
| decode 12x4/131072 全注意力（FD） | 1.270 ms | 0.683 ms | **0.688 ms** | 0.682 ms | 1.85x | 1.86x |

（末行 lean 与公共 API 持平：FD 形态由 kernel 0.63 ms 主导，调用开销可忽略。）

## 4. 为什么 SWA decode 的 kernel 比 bf16 慢

decode SWA 只有一个 512 token 的窗口、12 个 task（batch 12 x kv_head 1），28 个 AIC 只能
用上 12 个；kernel 仅 13.6 us（bf16 11.2 us），差值是 fp8 反量化。要再快只能把 KV 窗口再切分
给更多核（等价于对 SWA 启用 FlashDecode），需要 FD kernel 支持窗口 + 每 task 的 KV 起始偏移，
属 kernel 级改动，风险高，本轮未做。

## 5. 当前 decode SWA 的构成（lean，36 us）

`device 下发延迟 16 + kernel 13.6 + sync 4.7 + 调用路径 ~9 ≈ 43 us`（实测 36~40 us，部分重叠）。
NPUGraph replay 30 us 已等于“下发延迟 + kernel + sync”的下限，因此生产部署建议：
固定 shape 用 lean 入口 + NPUGraph 捕获；需要每次改 cache_seqlens 的场景注意 metadata 也需重建。
