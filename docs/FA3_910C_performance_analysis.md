# FA3 分阶段搬运在 910C 上的性能分析

## 结论

当前 910C 数据不能直接证明同一份 FA3 内核在 910C 上退化。910C profile
中的设备二进制与本仓库在 910B 上运行的设备二进制具有不同的 kernel ABI，
应先统一源码、构建产物和运行时加载路径，再做架构间比较。

统一二进制后，优先检查分阶段搬运产生的 GM workspace 流量和同步等待。
当前 kernel 的 910B profile 已显示它主要受 Scalar、同步和 MTE 限制，而不是
Cube 计算限制。910C 使用更多核会扩大同时活跃的 workspace 工作集，可能加重
L2 冲突和 GM 搬运竞争。

## 可复现数据

比较对象：

- 910C：`910C-FA3/OpBasicInfo.csv`
- 910B：seed 42、同一条 `b.py --num-cases 1` 用例，profile 位于
  `/tmp/fas_staged_20261022/910b_seed42/op/OPPROF_20261023095821_EDKANQINKGOBDKTF`

| 指标 | 910B | 910C | 变化 |
|---|---:|---:|---:|
| kernel duration | 2359.627 us | 3405.676 us | 910C 慢 44.33% |
| Cube / Vector 数 | 20 / 40 | 24 / 48 | 910C 多 20% |
| 每 Cube 平均周期 | 4.146 M | 6.091 M | 910C 多 46.90% |
| 按每核任务数折算的每任务周期 | 1.00 | 1.763 | 910C 多约 76.3% |

因此问题不在于 910C 没有使用足够多的核。更多核并没有抵消单任务执行效率的
下降。

## 二进制不一致

profile 中的设备二进制：

| 项目 | 910B | 910C |
|---|---|---|
| SHA-256 | `a45262dda239b003e2a6f6fa8406c26cbad8da9fb489094851a699d282bde647` | `ecf41de2f431a4984bd355b61db20968053870fcd3d20ceb0c2089cb41800030` |
| 大小 | 1,586,224 B | 1,691,808 B |
| 编译器时间戳 | 2026-05-09 | 2026-07-30 |

差异不仅来自针对不同设备重新编译。ELF 的 `.ascend.meta.*` section 显示：

- 910B kernel 参数包含 `FAInferTilingParams`，模板在 LSE mode 后只有当前 v3
  的 `HAS_SOFTCAP` 开关。
- 910C kernel 没有 `FAInferTilingParams` 参数，并在 LSE mode 后带有 4 个额外
  bool 模板参数。这与旧 `csrc/ascend910/flash_attn_npu` 中的
  `HAS_SOFTCAP/RETURN_SOFTMAX/HAS_DROPOUT/HAS_ALIBI` 形态一致。

当前 `setup.py` 对 `flash_attn_npu_3.flash_attn_npu_3` 明确选择
`csrc/ascend910/flash_attn_npu_3`。910C 环境需检查是否使用了旧 checkout、旧 wheel、
未清理的 build cache，或从其他目录导入了扩展。

在 910C 构建和运行目录执行：

```bash
git rev-parse HEAD
git status --short
sha256sum csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp

python - <<'PY'
import hashlib
import os
import flash_attn_npu_3
import flash_attn_npu_3.flash_attn_npu_3 as ext

print("package:", flash_attn_npu_3.__file__)
print("extension:", ext.__file__)
with open(ext.__file__, "rb") as f:
    print("extension sha256:", hashlib.sha256(f.read()).hexdigest())
print("cwd:", os.getcwd())
PY
```

清理 `build/`、旧 `.so` 和已安装的旧 wheel 后，只构建 v3；再次 profile 后确认
`OpBasicInfo.csv` 的 Op Name 中出现 `FAInferTilingParams`。

## 现有实现的瓶颈

910B 的有效 PMU 数据表明：

- AIC Scalar 占 72.48%，其中 event id2 等待约 1306 us；Cube 计算只占 16.37%。
- AIC MTE2 占 52.28%，Fixpipe 占 44.97%。
- AIV Vector 占约 58.5%，Scalar 占约 54.4%，`wait_ib` 约 1018 us。
- AIV MTE2/MTE3 分别占约 36.0%/25.8%，MTE3 wait ratio 约 48.8%。

这些指标说明主要时间花在跨 Cube/Vector 的等待、S/P/O/O-update 的 GM 中转以及
相应的数据搬运上。单纯增加 Cube 并行度不会直接解决瓶颈。

Q tile 从 128 增到 256 后，固定 workspace 的理论分配为每个 Cube 5.25 MiB：

```text
256 * 512 elements * (S:4 + P:2 + O:4 + update:4 bytes) * 3 slots
```

20 核共 105 MiB，24 核共 126 MiB。分配空间并不等于全部产生实际 GM 流量，
但 24 核会产生更大的并发活跃工作集。910C profile 的 AIV write L2 hit rate 平均
为 56.85%，910B 为 82.32%，这个差值支持 L2/写流量竞争假设。

910C 的其余 PMU 字段目前不可用于定量结论：`aic_time/aiv_time` 和所有利用率均为
0 或 NA，Cube FOPS 也只有明显不合理的 114688。`OpBasicInfo.csv` 同时缺失 op name、
block dim、device 和频率。应使用与 910C 驱动/固件匹配的 CANN/msprof 重新采集
`Default` 指标，确认这些字段正常后再比较 Pipe/MTE stall。

## 优化顺序

1. **统一构建和加载结果。** 这是当前最高优先级。记录 git commit、CANN 版本、
   Python package 路径、host `.so` SHA，以及 profile dump 中 device binary SHA。
2. **做核数扫描。** 在 host tiling 与 kernel launch 中同时把 AIC 数限制为
   12/16/20/24，比较 duration、AIV write L2 hit、MTE2/MTE3 wait。若 20 核快于
   24 核且 L2 命中率回升，再加入 910C 专用的自动核数选择；不要未经测量直接
   固定为 20 核。
3. **做 Q tile 扫描。** 保持源码与环境相同，对这个 seed 42 场景比较 Q=128 和
   Q=256。Q=256 在 910B 上减少任务数并显著加速，但更大的 S/P workspace 可能在
   910C 上触发不同的 L2 行为。最终可按 SoC 和 shape 选择 tile。
4. **压缩 workspace 地址跨度。** S/P 需要 `Q*512`，但 O/O-update 只需要
   `Q*head_dim`；O-update 当前没有使用 3 个 prelaunch slot，却按 3 slot 分配。
   将四个区域改为各自的 stride/size，避免按最大 S 区域统一留洞。此项先验证
   cache set/TLB 指标，单独减少未访问的分配字节不会必然降低 kernel duration。
5. **消除 GM 中转和同步。** 收益最大的结构性优化是让一个 128-column 子块的
   softmax/P 或 O update 尽量驻留片上，减少 `S -> GM -> P -> GM -> O -> GM`；其次
   是合并 event wait、减少 AIV `wait_ib` 和 AIC event id2 等待。每次只改一个阶段，
   用 PipeUtilization 和 L2Cache 验证。

建议先完成步骤 1，并采集 12/16/20/24 核与 Q=128/256 的 8 点矩阵。只有这组数据
能够区分“错误二进制”、“核数导致的 L2 竞争”和“Q=256 tile 不适配 910C”。

