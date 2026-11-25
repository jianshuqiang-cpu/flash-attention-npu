# 入图（scheduler_metadata / NPUGraph）补全 — 改动与实测（v5：paired prefill + FlashDecode 均已生效）

## 改动（相对 PR #466 head b0e5e56）

| 文件 | 改动 |
|---|---|
| `csrc/.../flash_api.cpp` | `get_scheduler_metadata` 接受 `quant_mode=2`（bf16 Q + fp8 KV，Dqk=Dv=192）；`FAMetadataArgs` 增 `softcapActive`；新增 `flash_attn_metadata_flags()`（host 读回 tiling 中的启动标志，非 capture 安全，只在生 metadata 时调用） |
| `csrc/.../fa_metadata_args.h` | `FAMetadataArgs` 增加 `softcapActive` 字段 |
| `csrc/.../fa_metadata.aicpu` | 镜像 host tiling 的 mixed-KV tile 策略：paired-prefill 波次（`qBaseTile=QueryTile` + `qL1BufNum=2`）与 small-MTP（`maxQ∈[5,8] & GQA16 → qBaseTile=128`） |
| `csrc/.../mha_fwd.cpp` | 去掉混存 KV 的 `!scheduler_metadata_` 限制；新增可选 `paired_prefill` / `fd_active_cores` / `fd_workspace_bytes` 入参（metadata 路径据此选 kernel、定启动核数与 workspace，全程无 D2H） |
| `csrc/.../flash_api.cpp` | 新增 `flash_attn_prepare_fd_metadata()`：capture 之外跑一次 host tiling，把 FD tiling H2D 覆盖进 metadata，并返回 {fdActiveCoreNum, workSpaceSize, flashDecodeFlag} |
| `flash_attn_npu_3/flash_attn_npu_interface_950.py` | `get_scheduler_metadata(..., kv_dtype=torch.float8_e4m3fn)` → quant_mode=2；生成后一次性读回启动标志并缓存到 metadata 张量属性（供 forward 选核，forward 内不再有 D2H）；新增 `prepare_fd_metadata()`（一次 host tiling + 上传 + 缓存 FD 启动形状）；混存 KV + 显式 metadata 时跳过 host tiling |
| `tests/test_flash_attn_npu_v3_mixed_kv_graph.py` | 新增：metadata↔host 位级一致、NPUGraph 捕获/重放一致、CPU 参考精度、paired prefill 标志用例 |

## 验证

- 全套回归：`mixed_kv_graph` 6 + `mixed_kv` 109 + `graph` 7 + `metadata` 15 = **137 passed / 86 skipped**
- 位级一致性：**全部 6 类场景 max|diff|=0**（含 FlashDecode），NPUGraph 捕获/重放同样逐位一致

## 端到端耗时（device 2，交替 A/B，8 轮中位数）

| 场景 | host tiling | + metadata | 提升 | 位级 |
|---|---|---|---|---|
| prefill 4096/8192 SWA512 | 0.365 ms | 0.252 ms | 1.45x | 是 |
| prefill 4096/8192 全注意力 | 1.334 ms | 1.218 ms | 1.10x | 是 |
| decode 12x4/8192 SWA512 | 0.192 ms | 0.076 ms | 2.52x | 是 |
| decode 12x4/131072 SWA512 | 0.198 ms | 0.085 ms | 2.34x | 是 |
| decode 12x4/32768 全注意力（FD） | 0.355 ms | 0.234 ms | 1.52x | 是 |
| decode 12x4/131072 全注意力（FD） | 0.816 ms | 0.696 ms | 1.17x | 是 |

NPUGraph replay 耗时（同机）：decode 131072 全注意力 0.703 ms、decode 32768 全注意力 0.203 ms、
decode 131072 滑窗 0.039 ms。

## 定位过程（保留以便复现）

1. 用 `FA_DUMP_TILING` 把 host / metadata 两条路径的 `FAInferTilingData` 各 dump 一份，
   逐字段比对：66 个标量中仅 3 个不同（`workSpaceSize`、`windowSizeLeft`、`quantMode`），
   且三者对非 FlashDecode 路径均无效 → 说明 tiling 语义一致。
2. 用 `FA_DUMP_ARGS` dump 启动参数，发现唯一差异是 `paired_query_heads`：
   host=1、metadata=0。根因是 `mha_fwd.cpp` 用**本地零初始化**的 `tilingData.qL1BufNum`
   推导该标志（metadata 路径 tiling 在显存里）→ 恒为 0，退回 generic kernel（结果同位，
   但慢 ~20%）。
3. 修法：metadata 生成时（capture 之外）一次性 D2H 读回 `qL1BufNum`，缓存为张量属性
   `md.paired_prefill`；forward 直接把该 bool 传入 C++（`std::optional<bool>`）。
   capture 期间没有任何同步或 D2H。

## FlashDecode 静态化设计（已实现）

问题：FD 的启动核数 `fdActiveCoreNum` 与 workspace 大小依赖每批 KV 长度，capture 期间不能做
host tiling / D2H，因此 metadata 路径原先关闭 FD（长上下文全注意力 decode 慢 1.5x）。

方案（把"一次性的不确定"挪到 capture 之外，而不是把 300 行 tiling 移植进 AICPU）：

1. 新增 `prepare_fd_metadata(md, q, k, v, cache_seqlens, ...)`：在 capture 之外复用现有 host
   tiling（`FAInferTiling::DoTiling`，含 `ScheduleKvTail` / `FillFdCombineCoreSchedules`），
   若判定 `flashDecodeFlag` 则把该 tiling **H2D 覆盖写进 metadata** 的 tiling 区。
2. 同时返回 `{fdActiveCoreNum, workSpaceSize, flashDecodeFlag}`，由 Python 缓存为
   `md.fd_active_cores` / `md.fd_workspace_bytes` / `md.flash_decode`。
3. forward 在 metadata 分支直接用这三个标量：`flashDecodeEnabled=true`、
   `launchBlockDim=fd_active_cores`、按 `fd_workspace_bytes` 分配 workspace；**capture 内零 D2H、
   零 host tiling**（workspace 大小对同一 metadata 恒定，重放稳定）。
4. 未调用 `prepare_fd_metadata` 时行为不变（FD 关闭，结果正确、较慢），因此向后兼容。

已知边界：metadata（含 FD tiling 与 kvCum）与生成时的 `cache_seqlens` 快照绑定；解码 KV 长度
推进后需重新生成 metadata 并重跑 `prepare_fd_metadata`。
