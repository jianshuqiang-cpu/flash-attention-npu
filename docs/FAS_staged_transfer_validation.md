# FA3 / Ascend 910：FAS 分阶段搬运

参考：`ops-transformer/attention/flash_attention_score/docs/FAS_910_S1S2与分阶段搬运计算.md`。

## 搬运与计算

保留 `QK → GM(S, fp32) → Vector softmax → GM(P, fp16/bf16) → PV → GM(O, fp32) → Vector rescale` 四阶段流水。Q 在一个 S1 任务内只搬入 L1 一次，各 S2 块复用；后续 S2 块更新 max/sum，并将前一块 O 按 `exp(old_max-new_max)` 缩放，最终块再归一化。

- S1 的容量从 128 行增至 256 行，短序列的 GQA head 也在同一 256 行预算内打包。append-KV 使用 128 行，为 ND 写回缓冲保留 L1 空间。
- S2 每阶段使用 512 列（128×4）。短于块宽的有效区间一次完成 softmax；超过 512 的序列按阶段在线更新，尾块只搬运有效区间。
- Q/K 的 L1 空间按 head dimension 分配。D≤128 时，K 一次搬运最多 512 列；L0 仍按最多 128×128×128 计算，直到所有 M/N/K 子块读完才释放 K 缓冲。
- Vector1 保持最多 8192 个 fp32 元素的工作块，Vector2 独立按输出维度分行。扩大 Cube 块不会同步扩大单次 Vector 工作块。
- host、AICPU metadata、设备端共享 `fa_tile_config.h` 中的容量常量。S/P/O workspace 槽位同步扩容；LSE、max/sum 以及 O-update 的 Vector 分区也同步调整。

这实现了文档中的分层搬运和在线更新机制；没有直接复用 FAS 的 Matmul 高阶 API，也没有逐一复制其所有模板选择条件。现有 Catlass L0 计算、mask、paged-KV、softcap、append-KV 和 split-KV 接口继续使用。

## L1 容量例子

fp16/bf16，D=Dv=128，无 append：

| S2 块 | Q 常驻 | K/P 复用的双缓冲池 | V | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 512 列 | 64 KiB | 256 KiB | 128 KiB | 448 KiB |

L0A/L0B/L0C 保持原有双缓冲容量。S2 方向不分给不同核，除原有 flash-decode split-KV 场景外，多核任务仍沿 batch/head/S1 分配。

## 验证方法

环境：Ascend 910B3，CANN 9.1.0 beta1，Python 3.11，PyTorch 2.10.0，仓库本地 FA3 扩展。

原二进制 SHA256：`9ea950bdde0a13c0145d07bac2d45672a9eaa6a7caf7469b76c8a24f083d8e54`。

最终验证二进制 SHA256：`91bd192c538fe2f29dec571ebb483f8c588ef1a06ea67fb658f19345d95343bd`。

固定种子 `20261022`。a.py 的 50 个 BSND 用例在物理卡 7，b.py 的 50 个随机 TND 用例和脚本当前启用的 1 个固定 TND 用例在物理卡 3。每次仅向进程暴露目标卡，脚本使用 `--device npu:0`，避免 profiler/随机数初始化访问其它占用中的卡。

性能以原脚本采集的第三次迭代 FA3 kernel Duration 为准，比较同一用例的修改前后耗时，不使用 FA3/FAS 比值代替自身加速比。重复 NPU event 计时作为辅助检查。原脚本的 shape、输入种子、参考计算和精度阈值未修改。

强制构建：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_FORCE_REBUILD=TRUE \
  /root/miniconda3/envs/py311/bin/python setup.py build_ext -j 8
```

新增的 36 个 S2 边界用例覆盖 fp16/bf16、KV 长度 512/513/1023/1024/1025/2049，以及 dense、causal、paged-local 三种模式。它们与 18 个 Q-tile 边界用例合计 54 项，全部通过。

V3 全量回归结果为 `599 passed, 10 skipped`，耗时 1903.74 秒。执行范围包括前向、metadata、graph、反向确定性、Q-tile 和新增 S2 分阶段边界测试；无失败用例。

## 性能结果

原脚本 profiler 的 FA3 kernel Duration 汇总如下：

| 脚本 | 用例数 | 修改前总耗时 | 修改后总耗时 | 修改前均值 | 修改后均值 | 加速比 | 更快用例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a.py` | 50 | 111.021 ms | 109.959 ms | 2.220 ms | 2.199 ms | 1.0097× | 33/50 |
| `b.py` | 51 | 169.199 ms | 122.138 ms | 3.318 ms | 2.395 ms | 1.3853× | 50/51 |

重复 NPU event 计时也得到相同方向：`a.py` 汇总 116.995 ms → 115.978 ms（1.0088×），`b.py` 汇总 212.846 ms → 127.051 ms（1.6753×）。Profiler CSV 与 event JSON 保存在 `/tmp/fas_staged_20261022/{baseline,final_q256}`。
