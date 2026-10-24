# FA3 TND 短序列 S=3、H=16 专用优化设计

## 1. 文档目的

本文记录 `flash_attn_npu_3` 在 Ascend 910B3 上针对以下窄场景的性能分析、方案演进、最终实现和验证结果：

```text
layout       = TND varlen
batch        = 512
total_q      = 1536
total_kv     = 1536
q_seqlen     = 3（每条序列）
kv_seqlen    = 3（每条序列）
q_heads      = 16
kv_heads     = 16
head_dim     = 128
dtype        = bfloat16
causal       = true
softmax_scale= 1.0
softcap      = 0.0
```

对应复现命令：

```bash
cd /home/j00574704/repo/Q_TILE_256/flash-attention-npu
/root/miniconda3/envs/py311/bin/python b_old.py \
  --device npu:6 --seed 20261022 --num-cases 0
```

最终目标不是提高 Cube 算术利用率，而是消除远大于有效计算量的 AIC/AIV 跨核等待。

## 2. 结论摘要

最终实现为一个严格 shape 门控的 Vector-only 前向路径：

- AIC 在读取 tiling 后直接返回，不再启动 QK/PV Cube 计算。
- 每个 Vector 子核负责 8 个 head。
- 一个 head 的 3 行 Q/K/V 搬入 UB 后，直接完成 6 个 causal QK 点积、3 行 softmax、3 行 PV 和 LSE。
- 输出直接写回 GM，不再经由 `S/P/OTmp` workspace。
- QK-ready、softmax-ready、PV-ready 等跨核 flag 不再出现在该路径中。

最终结果：

| 指标 | 优化前/中间 Cube 方案 | 最终 Vector-only |
|---|---:|---:|
| 目标 kernel | 约 2.46--2.80 ms | 0.221 ms |
| msopprof task duration | 2456.809 us | 225.265 us |
| `aiv_scalar_wait_ib_time` | 1889.812 us | 6.487 us |
| `aic_scalar_wait_id2_time` | 1874.609 us | 不再产生该跨核等待列 |
| AIC Cube active | 25.023 us | 0 us |
| fusion baseline | 约 0.38 ms | V3 比 fusion 快约 42% |

## 3. 原始瓶颈

### 3.1 原始任务粒度

原始调度按 batch、head 拆任务。目标 shape 有：

$$
N_{task}=B\times H=512\times16=8192
$$

每个任务的有效矩阵却非常小：

- QK：$3\times128$ 乘 $128\times3$；
- PV：$3\times3$ 乘 $3\times128$；
- causal 模式真正需要的 QK 元素只有 $1+2+3=6$ 个。

对 Cube 而言，M/N 方向至少按 16 对齐。每个 head 的 QK 会落到类似 $16\times16$ 的物理 tile，绝大多数计算是 padding。更重要的是，每个微小任务仍然需要完整的跨核流水同步。

### 3.2 原始跨核数据流

```mermaid
flowchart LR
    QK["AIC: QK Cube"] -->|"S 写 GM workspace"| S[(S workspace)]
    S -->|"qkReady / wait id"| SM["AIV: mask + online softmax"]
    SM -->|"P 写 GM workspace"| P[(P workspace)]
    P -->|"softmaxReady / wait id"| PV["AIC: PV Cube"]
    PV -->|"OTmp 写 GM workspace"| O[(OTmp workspace)]
    O -->|"pvReady / wait id"| RS["AIV: rescale + output/LSE"]
```

该流水适合较大的 Q/KV tile，因为 Cube 和 Vector 可以通过多任务 ping-pong 隐藏延迟；但在 `S=3` 时，每阶段计算只有几十微秒甚至更少，固定同步和 GM 往返成为主耗时。

### 3.3 Profiling 证据

合并到“一条序列一个任务、任务内循环 16 head”的 Cube 版本中，msopprof 仍显示：

| Pipe 指标（每核平均） | 时间 |
|---|---:|
| AIC total | 2449.065 us |
| AIC Cube active | 25.023 us |
| AIC `scalar_wait_id2` | 1874.609 us |
| AIV total | 2450.342 us |
| AIV Vector active | 160.158 us |
| AIV `scalar_wait_ib` | 1889.812 us |

Cube 有效时间只占总时间约 1%，Vector 有效时间也远小于等待时间。因此，继续优化 Cube 指令本身不会解决主要问题。

## 4. 方案演进

### 4.1 方案 A：一条序列的 16 个 head 合并为一个任务

第一步按用户建议改变任务所有权：

- 原来：一个任务处理一个 `(batch, head)`；
- 修改后：一个任务处理一条序列的全部 16 个 head；
- `totalTaskNum` 从 8192 降为 512；
- 16 个 QK 完成后只发送一次 `qkReady`；
- 16 个 head 的 softmax 完成后只发送一次 `softmaxReady`；
- PV 同样在一个任务内循环 16 个 head。

这一版把 kernel 从约 2.80 ms 降到约 2.46 ms，证明任务合并方向正确，但约 1.88 ms 的两类等待仍然存在。原因是阶段间依赖仍然是严格的：Vector 必须等全部 QK，Cube PV 又必须等 Vector softmax。

### 4.2 方案 B：16 份 padding 后的小 GEMM 批处理

随后验证了与“16 个 `16×16` 小 GEMM/IterateBatch”语义等价的 full-L0 实验：

- Q/K 或 P/V 一次装入 L1；
- 使用完整 L0A/L0B/L0C 容纳 16 个独立小矩阵；
- 连续执行 16 个 MMAD；
- 一次性写回 16 份结果。

结果反而退化：

| 指标 | 一任务 16-head Cube | full-L0 批处理 |
|---|---:|---:|
| task duration | 2456.809 us | 2833.797 us |
| AIC `scalar_wait_id2` | 1874.609 us | 2355.900 us |
| AIV `scalar_wait_ib` | 1889.812 us | 2344.777 us |
| AIC Cube active | 25.023 us | 25.031 us |

退化原因不是 GEMM 算得更多，而是 full-L0 方案需要先排空并独占现有 ping-pong L0 槽位，破坏了原先多任务间的搬运/计算重叠。Cube active 几乎没有变化，跨核等待却进一步上升。

因此，该方案在算法语义上成立，但不适合作为当前内核结构中的最终实现。

### 4.3 方案 C：Vector-only 专用路径

`S=3` 时每个 head 的 causal QK 只有 6 个长度为 128 的点积。与两次 Cube 调度、三段 GM workspace 和多次跨核握手相比，直接在 Vector 上完成这些点积更便宜。

最终选择：

```mermaid
flowchart LR
    GM["GM: Q/K/V"] -->|"3 行 × 128"| UB["AIV UB"]
    UB --> QK["6 个 causal dot"]
    QK --> SM["3 行 fp32 softmax"]
    SM --> PV["Vector PV + normalize"]
    PV --> OUT["GM: O + LSE"]
```

整个链路在同一个 Vector 子核内完成，不再依赖 AIC 结果。

## 5. 最终实现详解

### 5.1 专用路径触发条件

AICPU metadata 只有在以下条件全部成立时才设置 `tinyBatch16Flag`：

1. Q 是 varlen/TND；
2. KV 也是 varlen/TND；
3. 非 paged KV；
4. 非 append-KV；
5. causal mask；
6. `q_heads == 16`；
7. `kv_heads == 16`；
8. `embeddingSize == embeddingSizeV == 128`；
9. 每个 batch 的 Q 长度都等于 3；
10. 每个 batch 的 KV 长度都等于 3。

Kernel 侧还要求：

```cpp
tinyBatch16Flag != 0U && softcapValue == 0.0f
```

`softcap != 0` 时不进入 Vector-only 计算，而是保留批量 Cube/Vector 兼容路径。

### 5.2 metadata 契约修复

`FlashAttnVarlenFunc` 已经把 `cu_seqlens_k` 求差为每 batch 的 `cache_seqlens`，AICPU 使用后者读取实际 KV 长度。但 scheduler 仍需知道 KV 的物理布局是否为 TND。

原先 Python 调用把 `cu_seqlens_k` 参数传成 `None`，导致：

```text
metaArgs.isVarlenKv = 0
```

即使实际调用是 TND varlen KV，专用分支也永远无法命中。现在保留 `cache_seqlens` 作为长度输入，同时把 `cu_seqlens_k` 作为布局存在性标志传入 scheduler：

```python
cache_seqlens=cu_seqlens_k[1:] - cu_seqlens_k[:-1]
cu_seqlens_k=cu_seqlens_k
```

这样既不会重复解析累计长度，又能严格区分 TND KV 和固定长度 KV，避免相似 shape 误命中。

### 5.3 tiling ABI

`FAInferTilingData` 原有一个 32-bit padding 字段。实现复用该槽位：

```cpp
uint32_t tinyBatch16Flag;
```

因此：

- tiling 结构体总大小不变；
- metadata buffer 大小不变；
- 不需要扩大 Host/AICPU/Device 间的 ABI；
- Host tiling 和 AICPU tiling 都填充同一字段。

命中时 `totalTaskNum` 设为 batch 数，即本用例的 512，而不是 8192。

### 5.4 核与 head 的映射

910B3 的 mixed kernel 中，一个逻辑 AI Core 对应两个 Vector 子核：

```cpp
subBlockIdx = GetSubBlockIdx();       // 0 或 1
headStart   = subBlockIdx * 8;        // 0 或 8
logicalCore = GetBlockIdx() / GetSubBlockNum();
```

映射关系：

| Vector 子核 | 负责的 head |
|---|---|
| sub-block 0 | head 0--7 |
| sub-block 1 | head 8--15 |

每个逻辑 core 以跨步方式处理 batch：

```cpp
for (batchIdx = logicalCore; batchIdx < batch; batchIdx += logicalCoreNum)
```

每个 `(batch, head)` 只由一个 Vector 子核处理，不存在写冲突，也不需要跨核归并。

### 5.5 TND GM 布局和搬运

输入布局为：

```text
Q/K/V: [total_tokens, 16, 128]
```

由于每条序列严格为 3 token：

```cpp
tokenBase = batchIdx * 3;
inputOffset = tokenBase * 16 * 128 + head * 128;
```

同一 head 的相邻 token 在 GM 中间隔 15 个其他 head，因此使用三段二维 `DataCopyPad`：

```text
blockCount = 3
blockLen   = 128 * sizeof(bfloat16)
srcStride  = (16 * 128 - 128) * sizeof(bfloat16) = 3840 bytes
```

Q、K、V 各搬运 3×128 个 BF16 元素到 UB，随后转为 FP32 计算。

### 5.6 UB 划分

专用路径在共享 UB 中使用固定偏移，最大偏移约 16 KiB：

| Buffer | 类型 | 字节偏移 | 用途 |
|---|---|---:|---|
| `q16` | BF16/FP16 | 0 | 3×128 Q |
| `k16` | BF16/FP16 | 1024 | 3×128 K |
| `v16` | BF16/FP16 | 2048 | 3×128 V |
| `p16` | BF16/FP16 | 3072 | 量化后的非归一化 P |
| `q32` | FP32 | 4096 | Q 计算副本 |
| `k32` | FP32 | 6144 | K 计算副本 |
| `v32` | FP32 | 8192 | V 计算副本 |
| `tmp32` | FP32 | 10240 | 点积/PV 临时向量 |
| `dot32` | FP32 | 11264 | 6 个 QK 点积 |
| `score32` | FP32 | 12288 | 3×8 对齐的 score/exp |
| `quantP32` | FP32 | 12800 | P 量化后回转 FP32 |
| `out32` | FP32 | 13312 | 3×128 输出累加 |
| `out16` | BF16/FP16 | 15360 | 输出写回缓冲 |
| `lse32` | FP32 | 16384 | 3 个 LSE |

按 head 串行复用这些 buffer，避免为 8 个 head 同时复制大块 UB，从而保持较低的 UB 占用。

### 5.7 QK：只计算 causal 有效元素

3×3 causal score 矩阵为：

$$
S=\begin{bmatrix}
q_0k_0^T & -\infty & -\infty \\
q_1k_0^T & q_1k_1^T & -\infty \\
q_2k_0^T & q_2k_1^T & q_2k_2^T
\end{bmatrix}\cdot scale
$$

实现只循环 `kRow <= qRow`，因此只计算：

```text
(0,0)
(1,0) (1,1)
(2,0) (2,1) (2,2)
```

每个长度 128 的点积通过以下步骤完成：

1. `Mul<float>` 得到 128 个逐元素乘积；
2. 前 64 个元素与后 64 个元素相加；
3. `WholeReduceSum` 对 64 个元素求和。

每个 head 一共执行 6 个点积，不计算上三角元素，也没有 16×16 Cube padding。

### 5.8 Softmax 数值流程

每行分别计算最大值：

$$
m_i=\max_{0\le j\le i}S_{ij}
$$

再计算非归一化概率及分母：

$$
P'_{ij}=\exp(S_{ij}-m_i),\qquad
l_i=\sum_{j=0}^{i}P'_{ij}
$$

`score32` 使用 8 个 FP32 元素作为一行的 pitch，总计 3×8=24 个元素，以满足 Vector 指令的对齐要求。padding 位置初始化为 `-80`，但不会参与分母或 PV。

为贴近通用路径的数值顺序，非归一化的 $P'$ 会先：

```text
FP32 -> ElementP(BF16/FP16) -> FP32
```

PV 使用量化后的 $P'$，分母 $l_i$ 保持 FP32。这样复现了通用路径“P 以低精度写入 workspace、PV 后再用 FP32 分母归一化”的主要舍入点。

### 5.9 PV 和归一化

对第 $i$ 行：

$$
O_i=\frac{\sum_{j=0}^{i}\operatorname{quant}(P'_{ij})V_j}{l_i}
$$

实现使用 `Muls + Add` 在 FP32 中累加：

```text
row 0: P00 * V0
row 1: P10 * V0 + P11 * V1
row 2: P20 * V0 + P21 * V1 + P22 * V2
```

每行完成后乘 `1 / denom`，最后一次性把 3×128 FP32 输出转为输入 dtype。

### 5.10 LSE 和输出写回

LSE 为：

$$
LSE_i=\log(l_i)+m_i
$$

输出布局：

```text
O:   [total_tokens, 16, 128]
LSE: [16, total_tokens]
```

O 使用和输入相同的二维跨 head stride 写回；LSE 直接写入当前 head 对应的连续 3-token 区间。

### 5.11 同步模型变化

最终路径仍需要本 Vector 子核内部的流水同步：

- MTE2 → Vector：等待 Q/K/V 搬入 UB；
- Vector barrier：保证 Cast、Mul、Reduce、Exp 和累加的局部依赖；
- Vector → MTE3：保证 O/LSE 计算完成后写回；
- MTE3 → Vector：保证 UB buffer 在下一个 head 复用前已释放。

但它不再需要：

- `qkReady`；
- `softmaxReady`；
- `pvReady`；
- AIC/AIV `wait_id1/id2/id3`；
- S/P/OTmp workspace 阶段同步。

最终仍有约 6.49 us 的 `aiv_scalar_wait_ib_time`，这是 Vector 指令发射/局部流水行为，已从原来的约 1.89 ms 降低两个数量级以上。`aiv_scalar_wait_time` 中还包含本核 MTE event 等待，不能与已消除的跨核 `wait_id2` 混为一谈。

## 6. 代码改动映射

| 文件 | 作用 |
|---|---|
| `flash_attn_npu_3/flash_attn_npu_interface.py` | 把 `cu_seqlens_k` 作为 KV varlen/TND 布局标志传给 scheduler |
| `csrc/ascend910/flash_attn_npu_3/tilingdata.h` | 复用 padding 字段增加 `tinyBatch16Flag` |
| `csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu` | 在设备侧检查精确 shape/长度并生成 512-task tiling |
| `csrc/ascend910/flash_attn_npu_3/flash_api.cpp` | Host tiling 路径执行同样的门控与任务合并 |
| `csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp` | 实现 Vector-only QK/softmax/PV/LSE，并在通用初始化前返回 |
| `csrc/ascend910/flash_attn_npu_3/online_softmax.hpp` | 保留 tiny batch Cube fallback 的 3×3 causal mask 片上生成 |
| `csrc/ascend910/flash_attn_npu_3/pv_matmul.hpp` | batch fallback 中仅第一个 head 等待一次 softmax-ready |
| `csrc/ascend910/flash_attn_npu_3/rescale_o.hpp` | batch fallback 中针对 3-token LSE 直接做 strided GM 写回 |

## 7. 性能验证

### 7.1 最终 `b_old.py` 结果

NPU 6 上最终复跑：

```text
v3       = 0.221 ms
fusion   = 0.383 ms
delta    = -0.162 ms
ratio    = 0.577x
max_abs  = 7.812500e-03
max_rel  = 7.633588e-03
```

相对最初约 2.80 ms，目标 kernel 提升约 12.7 倍；相对同次 fusion baseline 快约 42.3%。

### 7.2 最新普通 profiler

```text
kernel duration       = 220.744 us
AIC MAC               = 0.000 us
AIC scalar            = 3.681 us
AIC MTE1              = 0.003 us
AIC MTE2              = 0.002 us
AIC Fixpipe           = 0.001 us
AIV total             = 213.490 us
AIV Vector            = 123.920 us
```

AIC 只执行 tiling 分支和返回逻辑，没有进行 Cube 计算或 workspace 搬运。

### 7.3 最终 PipeUtilization

```text
task duration                 = 225.264511 us
AIC time/core avg             = 10.044972 us
AIC Cube/core avg             = 0.000000 us
AIV time/sub-core avg         = 216.003861 us
AIV Vector/sub-core avg       = 123.919554 us
AIV scalar_wait_ib avg        = 6.486694 us
AIC scalar_wait_id2           = 不存在
```

最终 CSV 不再包含 `_id*` 跨核等待列，因为该专用路径没有发出对应的跨核 wait 指令。

### 7.4 Profiling 文件

- Cube 合并方案：`profile_tiny_batch16_msopprof_app/OPPROF_20261023230158_MQFIWIHXRDMHJIJG/PipeUtilization.csv`
- full-L0 失败方案：`profile_tiny_batch16_msopprof_batchl0_exact/OPPROF_20261024103938_BFDGMIFVGUCWQGZX/PipeUtilization.csv`
- 最终 Vector-only：`profile_tiny_batch16_msopprof_vector_app/OPPROF_20261024110424_UPMQYSYLPLMBMYSZ/PipeUtilization.csv`
- 最新普通 profiler：`batchtest/profile_am_varlen_50_fa3/fa_01/c89d486a88db_1986232_20261024112000506_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`

## 8. 精度说明

最终输出相对 `npu_fusion_attention`：

```text
max_abs = 0.0078125
max_rel = 0.007633588
```

误差来源主要是：

1. Vector FP32 点积的归约顺序与 Cube 累加顺序不同；
2. 为匹配通用路径，P 在 PV 前显式量化到 BF16 再回转 FP32；
3. 最终 O 再从 FP32 转为 BF16。

结果稳定在 BF16 误差量级。连续多次复跑得到相同的 max-abs/max-rel。

## 9. 构建与复现

### 9.1 编译

```bash
cd /home/j00574704/repo/Q_TILE_256/flash-attention-npu
FLASH_ATTN_BUILD_VERSION=v3 \
FLASH_ATTN_BUILD_NPU=910 \
/root/miniconda3/envs/py311/bin/python setup.py build_ext --inplace -j 8
```

需要排除旧对象影响时：

```bash
FLASH_ATTN_BUILD_VERSION=v3 \
FLASH_ATTN_BUILD_NPU=910 \
FLASH_ATTN_FORCE_REBUILD=TRUE \
/root/miniconda3/envs/py311/bin/python setup.py build_ext --inplace -j 8
```

### 9.2 目标回归

```bash
/root/miniconda3/envs/py311/bin/python b_old.py \
  --device npu:6 --seed 20261022 --num-cases 0
```

验收建议：

- V3 kernel 约 0.22 ms；
- `max_abs <= 0.0078125`；
- 普通 profiler 中 AIC MAC 为 0；
- PipeUtilization 中 AIC Cube 为 0；
- 不再出现 `aic_scalar_wait_id2_time`；
- `aiv_scalar_wait_ib_time` 应为个位数微秒量级，而不是毫秒量级。

## 10. 适用范围和限制

该优化是有意设计的窄特化，不是通用 FlashAttention 替代实现。

不满足以下任一条件时，不进入 Vector-only 路径：

- Q/KV 同时为 TND varlen；
- 每条 Q/KV 长度都严格为 3；
- MHA 16×16 heads；
- D/DV 都为 128；
- causal；
- 非 paged、非 append-KV；
- `softcap == 0`。

特别注意：

- 不能仅检查 `max_seqlen == 3`，必须检查每一条序列都是 3，否则 `batchIdx * 3` 的直接寻址不成立；
- 不能用于 GQA/MQA，因为当前 K/V head 直接按 Q head 索引；
- 不能用于非 causal，因为当前只计算下三角 6 个 score；
- 不能用于 `softcap > 0`，专用路径没有 tanh softcap；
- 不能直接扩展到 D≠128，点积折半和 UB 布局都写死为 128；
- 不应在没有重新 profiling 的情况下把 `HEADS_PER_SUBBLOCK` 从 8 改为其他值。

## 11. 后续可选优化

当前 0.22 ms 中，Vector active 约 124 us，Scalar 约 100 us。后续优化重点已经从跨核等待转为单核指令组织：

1. 尝试把同一 sub-block 的多个 head 做双缓冲，重叠下一 head 的 Q/K/V MTE2 和当前 head 计算；
2. 减少 `GetValue/SetValue` 标量访问，使用更连续的 Vector reduction/广播；
3. 合并 Q/K/V 搬运或调整 UB 排布，降低每 head 的 event 数量；
4. 评估 8-head 并行批量 Vector dot，但必须控制 UB 和寄存器压力；
5. 为 `S=2/4` 建立独立成本模型，不应直接复用 `S=3` 常量；
6. 如果未来硬件提供低开销小矩阵 batch Cube 指令，再重新比较 Vector-only 与 IterateBatch。

在现有 910B3/CANN 环境中，重新引入 Cube 不应是第一优先级，因为最终 profile 已证明该 shape 的主要收益来自彻底删除 AIC/AIV 阶段边界。

## 12. Python 调用示例

```python
import torch
import torch_npu
from flash_attn_npu_3 import flash_attn_varlen_func

device = "npu:6"
torch.npu.set_device(device)

batch = 512
heads = 16
dim = 128
tokens = batch * 3

q = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
cu = torch.arange(0, tokens + 1, 3, dtype=torch.int32, device=device)

out, lse, *_ = flash_attn_varlen_func(
    q,
    k,
    v,
    cu,
    cu,
    3,
    3,
    softmax_scale=1.0,
    causal=True,
    window_size=(-1, -1),
)

assert out.shape == q.shape
assert lse.shape == (heads, tokens)
```

输入支持 `torch.float16` 和 `torch.bfloat16`；本次性能与精度数据均来自 `torch.bfloat16`。
