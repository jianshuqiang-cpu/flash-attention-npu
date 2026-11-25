# 第二轮性能优化：host 开销与选核竞态（v6 -> v7）

## 1. 修复：AICPU tiling 读回竞态（真实缺陷，不只是性能）

`get_scheduler_metadata` 生成 metadata 后要读回 AICPU 写好的 tiling（`qL1BufNum` 决定是否
走 paired Q-head kernel）。原先只做 `aclrtSynchronizeStream`，而 AICPU kernel 可能由框架在
私有 stream 上派发，导致读回偶发得到全 0：

- 现象：同一 shape 的 prefill 全注意力 kernel 时间在 1.14 ms / 1.44 ms 之间跳动
  （paired kernel vs generic kernel）；结果仍位级正确，只是静默变慢 ~26%；
- 修法：`flash_attn_metadata_flags()` 改为 `aclrtSynchronizeDevice()`（每次生成 metadata
  只调用一次，在 capture 之外）；`prepare_fd_metadata()` 覆盖写 tiling 前也先同步设备，
  避免与 AICPU 的写入竞争；
- 复现验证：连续 3 个独立进程，`paired_prefill` 恒为 True，prefill 全注意力 wall
  1.188 / 1.188 / 1.184 ms（修复前 1.48 ms 与 1.19 ms 随机出现）。

## 2. LSE 初始化省略

`mha_fwd` 原先每次调用都 `softmaxlse.fill_(inf)`（一次额外 device kernel）。
只有两类形态需要该初始化：FlashDecode 的 partial 合并，以及 `seqused_q` 截断导致部分行不写。
现在仅在 `flashDecodeEnabled || seqused_q` 时填充。

验证：`mixed_kv` 全套（含 host tiling 与 metadata 两条路径**逐位比对**、以及 CPU 参考
的 LSE `assert_close`）137 passed / 86 skipped —— 两条路径在 metadata 路径不填充的前提下
仍逐位一致，说明 kernel 覆盖了全部 LSE 行。

## 3. Python 调用路径瘦身

| 改动 | 说明 |
|---|---|
| `_contiguous_all()` | 一次循环处理 16 个可选张量，替代 6 个生成器 + 10 次辅助函数调用 |
| `_prepare_forward_qkv` 快路径 | paged + 连续 packed cache（BBND）时直接返回，跳过 `stride()` 探测 |
| `_can_use_eager_forward` | 用显式循环 + `t.__class__` 替代生成器 `all()` 与全局 `type` 查找 |

enqueue 侧（不含 device）实测：public 包装 28.2us -> 约 21us，impl 前处理 9.4us -> 约 3us。

## 4. 结果（device 2，同进程，10 轮中位数）

| 场景 | bf16 优化前 | v5（v6 包） | **v7** | v7/bf16 | v7/bf16（NPUGraph replay） |
|---|---|---|---|---|---|
| prefill 4096/8192 SWA512 | 0.351 ms | 0.262 ms | **0.225 ms** | 1.56x | 1.73x |
| prefill 4096/8192 全注意力 | 1.499 ms | 1.484 ms* | **1.189 ms** | 1.26x | 1.29x |
| decode 12x4/8192 SWA512 | 0.180 ms | 0.061 ms | **0.050 ms** | 3.62x | 5.61x |
| decode 12x4/131072 SWA512 | 0.186 ms | 0.063 ms | **0.049 ms** | 3.77x | 5.84x |
| decode 12x4/32768 全注意力（FD） | 0.446 ms | 0.218 ms | **0.218 ms** | 2.05x | 2.33x |
| decode 12x4/131072 全注意力（FD） | 1.284 ms | 0.697 ms | **0.679 ms** | 1.89x | 1.97x |

\* v6 的 prefill 全注意力受上述竞态影响，时而 1.19 ms、时而 1.48 ms。

kernel 时间（v7）：0.183 / 1.141 / 0.014 / 0.014 / 0.171 / 0.625 ms。
小 shape 的端到端耗时已接近固定开销下限：decode SWA 形态 wall 0.049 ms 中 kernel 仅
0.014 ms，其余为 launch/同步与 Python/pybind 开销；改用 NPUGraph replay 可进一步降到 0.032 ms。

## 5. 已尝试但放弃

- 复用两个空 accum 输出张量（省 2 次 `at::empty`）：会让 114 个用例失败（输出别名被上层
  框架视为无效），已回退；
- `kvBaseTile` 对 mixed KV 也用 256：与 PR 作者按 L1 预算确定的分块冲突，风险大于收益，未采用。
