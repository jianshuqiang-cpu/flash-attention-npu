# 去掉 metadata 构建里的设备级同步（v9 -> v10）

## 1. 问题

`get_scheduler_metadata` 需要读回 AICPU 写好的 tiling（`qL1BufNum` 决定是否走 paired kernel），
上一版为规避竞态用了 `aclrtSynchronizeDevice()`。设备级同步会等待**该设备上所有 stream** 的
在飞任务，实测：

| 场景 | metadata 构建耗时 |
|---|---|
| 设备空闲 | 0.157 ms |
| 另一条 stream 上有 matmul 在跑 | **1.160 ms** |

即 host 被无关任务拖住（这正是 profiling 里看到的 sync 卡顿）。

## 2. 两步修复

### 2.1 设备同步 -> 当前 stream 同步

metadata 的 AICPU 派发本身已经用事件把 AICPU stream 桥接回当前 stream
（`recordEvent(cur)` -> `waitEvent(aicpu)` -> kernel -> `recordEvent(aicpu)` -> `waitEvent(cur)`），
所以等待**当前 stream** 就足以保证 tiling 可见。改为 `aclrtSynchronizeStream(current)`，
并加入一次校验式兜底：若读回的 `qL1BufNum` 不是 1/2（说明竞争未排除），再做一次设备同步重读。
`prepare_fd_metadata` 里覆盖写 tiling 前的设备同步也一并删除——H2D 与 AICPU 写在同一 stream
上天然有序。

### 2.2 decode 形态彻底不读回（零同步）

paired 流水线要求每个 head 至少一个完整 128 行 Q tile（`fullQ128Tasks >= 2*blockDim`，其中
`fullQ128Tasks = Σ activeQ/128 × numHeads`）。`max_seqlen_q < 128` 时该值必为 0，
`paired_prefill` 必然为 False，因此在 Python 侧直接置位，**不访问设备、不产生任何同步**。
只有 `max_seqlen_q >= 128` 的 prefill 形态才做一次当前 stream 同步的读回。

## 3. 结果

| 指标 | 修复前 | 修复后 |
|---|---|---|
| metadata 构建（设备空闲） | 0.157 ms | **0.059 ms** |
| metadata 构建（另有 stream 忙） | 1.160 ms | **0.087 ms** |
| decode 形态是否触碰设备 | 是（设备同步） | **否（零同步）** |
| prefill `paired_prefill` | True | True（仍正确，3 次进程验证） |

## 4. 追加发现：非 FD 形态的 metadata 可以跨 KV 长度复用

`cache_seqlens` 增长只影响 kernel 运行时读取的 `seqlens_k`，非 FlashDecode 的 tiling 与它无关。
实测用 `cache_seqlens=512` 生成的 metadata，在 2048 / 4096 / 8192 上运行，与 host tiling
**逐位一致**（decode SWA、prefill SWA、prefill 全注意力均是）。

因此 **稳态 decode 可以完全不做 per-step metadata 构建**（构建一次，随 KV 增长一直复用）：
per-step 开销从「57 us metadata + 36 us forward」降到「36 us forward」。

限制：FlashDecode 形态（长上下文全注意力 decode）的 tiling 含 `fdDecodeSchedules` / 启动核数 /
workspace，确实依赖 `cache_seqlens`，长度变化时必须重建并重跑 `prepare_fd_metadata`。

新增用例 `test_mixed_kv_metadata_reuse_across_kv_lengths` 覆盖该复用语义；全套
**139 passed / 86 skipped**。
