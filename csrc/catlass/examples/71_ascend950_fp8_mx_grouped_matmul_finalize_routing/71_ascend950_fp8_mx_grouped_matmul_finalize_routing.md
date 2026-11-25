# CATLASS Ascend950Fp8MxGroupedMatmul_FinalizeRouting

## 详细设计

### 算子流程

```text
(A * MxScaleA) @ (B * MxScaleB) + Bias → C(m,n)    [AIC: Grouped MX FP8 Matmul]
        ↓ Write to GM Workspace
   C_workspace(m,n)
        ↓ AIV: Clear Output
   out(batch,n) = 0
        ↓ AIV: SharedInput (optional)
   out[offset:offset+bsdp, :] += weight * Cast<BF16→FP32>(SharedInput)
        ↓ AIV: Logit Weighting
   Ĉ[p, :] = logit[p] * C[p, :]
        ↓ AIV: Scatter Add
   out[rowIndex[p], :] += Ĉ[p, :]
```

其中：

- AIC 侧完成 MX FP8 反量化 + 分组矩阵乘 + 可选 bias 累加，结果写入 GM workspace
- AIV 侧完成后处理：输出清零、可选共享专家赋值、Logit 加权、Scatter Add 聚合

### 模板组装

#### 确定性版

| 组件           | 模板类                                                                                                            | 说明                                                       |
| -------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Kernel         | [GroupedMxMatmulFinalizeRoutingTla](../../include/catlass/gemm/kernel/grouped_mx_matmul_finalize_routing_tla.hpp) | AIC/AIV双核协作，按group遍历，CrossCore Flag流水线同步     |
| BlockMmad      | [BlockMmadTla](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)                                      | MX量化矩阵乘，DispatchPolicy=`MmadMx<Ascend950, true, 16>` |
| TileCopy       | [PackedMxTileCopyTla](../../include/catlass/gemm/block/block_mmad.hpp)                                            | GM→L1→UB数据搬运                                           |
| BlockEpilogue  | [BlockEpilogueFinalizeRouting](../../include/catlass/epilogue/block/block_epilogue_finalize_routing.hpp)          | 输出清零 + SharedInput赋值 + Logit加权 + Scatter Add       |
| BlockScheduler | [ColumnBlockSwizzle](../../include/catlass/gemm/block/block_swizzle.hpp)                                          | 按列分块调度，分配到各AICore                               |

#### 非确定性版（no_deter）

| 组件 | 模板类 | 说明 |
|------|--------|------|
| Kernel | [GroupedMxMatmulFinalizeRoutingNoDeterTla](../../include/catlass/gemm/kernel/grouped_mx_matmul_finalize_routing_no_deter_tla.hpp) | AIC/AIV双核协作，采用ASWT滚动调度，CrossCore Flag流水线同步 |
| BlockMmad | [BlockMmadTla](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp) | MX量化矩阵乘，DispatchPolicy=`MmadMx<Ascend950, true, 16>`（与确定性版相同） |
| TileCopy | [PackedMxTileCopyTla](../../include/catlass/gemm/block/block_mmad.hpp) | GM→L1→UB数据搬运（与确定性版相同） |
| BlockEpilogue | [BlockEpilogueFinalizeRoutingNoDeter](../../include/catlass/epilogue/block/block_epilogue_finalize_routing_no_deter.hpp) | 输出清零 + SharedInput赋值 + Logit加权 + Scatter Add，AIV按M维切分 |
| BlockScheduler | [GemmGroupedAswtTailSplitSwizzle](../../include/catlass/gemm/block/block_swizzle_grouped_aswt.hpp) | 滚动核分配 + 窗口调度，支持尾块多核拆分 |

### AIC/AIV双核协作

- **AIC**：遍历所有group，对每个group按BlockScheduler分配tile任务（确定性版使用`ColumnBlockSwizzle`，非确定性版使用`GemmGroupedAswtTailSplitSwizzle`），执行MX FP8矩阵乘（含可选bias），结果写入GM workspace。通过`CrossCoreSetFlag(6)`通知AIV结果就绪，`CrossCoreWaitFlag(4)`等待AIV消费完毕。
- **AIV**：
  1. **预处理阶段**：清零输出区域（`ClearOutTile`），可选地加载共享专家输出（`AssignSharedInputTile`）
  2. **后处理阶段**：通过`CrossCoreWaitFlag(6)`等待AIC完成当前tile，从workspace读取GMM结果，执行Logit加权（`Muls`）+ Scatter Add（原子累加到输出），完成后`CrossCoreSetFlag(4)`通知AIC

#### 同步协议

```text
AIC:  ──[计算 tile 0]──SetFlag(6)──[计算 tile 1]──SetFlag(6)──...──WaitFlag(4)──
AIV:  ──WaitFlag(6)──[后处理 tile 0]──SetFlag(4)──WaitFlag(6)──[后处理 tile 1]──...
```

- Flag ID `4`（`AIV_SYNC_AIC_FLAG`）：AIV → AIC 方向
- Flag ID `6`（`AIC_SYNC_AIV_FLAG`）：AIC → AIV 方向
- 使用 `AIC_SYNC_AIV_MODE_2` 模式，实现tile粒度的流水线化交替执行

#### AIV SubBlock分裂

- **确定性版**：每个AIC核配2个AIV子核（通过`GetSubBlockIdx`），**N维**一分为二并行处理，`vecTileShape`为`MatrixShape<L1_M, L1_N/2>`。
- **非确定性版（no_deter）**：每个AIC核配2个AIV子核，**M维**一分为二并行处理，`vecTileShape`为`MatrixShape<L1_M/2, L1_N>`。

### Finalize Routing详细设计

#### 概述

Finalize Routing是MoE（Mixture of Experts）推理中的关键后处理步骤，将分组矩阵乘的结果按路由权重加权后聚合到输出张量。整个过程在AIV上执行，分为四个阶段：

```text
GMM结果 (workspace, FP32)
    ↓ ClearOutTile        清零输出区域
    ↓ AssignSharedInput   (可选) 共享专家输出赋值
    ↓ LogitScatterAdd     Logit加权 + Scatter Add聚合
out (FP32)
```

#### 阶段一：ClearOutTile — 清零输出区域

对输出张量`out(batch, n)`执行零初始化：

1. **UB清零**：在UB中使用`Duplicate`生成全零buffer
2. **分批写回GM**：按`MAX_CLEAR_GM_COUNT`（50K元素）分批将零值写入GM
3. **并行策略**：所有AIV worker轮询处理batch行

#### 阶段二：AssignSharedInputTile — 共享专家输出赋值（可选）

当`SharedInput`不为空时，对`j ∈ [0, bsdp)`执行：

$$\text{out}[j + \text{sharedInputOffset}, :] = \text{sharedInputWeight} \times \text{Cast}_{\text{BF16} \to \text{FP32}}(\text{SharedInput}[j, :])$$

实现步骤：

1. 从GM加载BF16共享输入到UB
2. Cast到FP32
3. `Muls`乘以权重系数`sharedInputWeight`
4. 写回到输出矩阵对应行
5. 按`MAX_SOLVE_SHARED_INPUT_COUNT`（20K元素）分批处理

#### 阶段三：LogitScatterAddTile — Logit加权 + Scatter Add

这是整个finalize routing的核心计算，对每个token `p ∈ [0, m)`执行：

1. **加载**：从GM workspace读取GMM结果`C[p, :]`、Logit向量、RowIndex向量到UB
2. **Brcb广播**：对Logit做`Brcb`（Broadcast Row to Column Block），将1D logit标量广播为与矩阵列对齐的格式。当行数超过`MAX_VECTOR_REPEAT_COUNT=255`时，拆成两次Brcb
3. **逐元素乘法**：`C[p, :] *= logit[p]`，使用`Mul`向量指令完成logit加权
4. **原子Scatter Add**：根据`RowIndex[p]`将加权后的行原子累加到输出矩阵对应行。使用`SetAtomicAdd` + `DisableDmaAtomic`包裹

#### 常量表

| 常量                           | 值    | 含义                            |
| ------------------------------ | ----- | ------------------------------- |
| `EPILOGUE_TILE_N`              | `256` | AIV后处理N维度分块大小          |
| `MAX_CLEAR_GM_COUNT`           | `50K` | 单次清零GM的最大元素数          |
| `MAX_SOLVE_SHARED_INPUT_COUNT` | `20K` | 单次处理SharedInput的最大元素数 |
| `MAX_VECTOR_REPEAT_COUNT`      | `255` | Brcb单次最大重复次数            |
| `AIV_SYNC_AIC_FLAG`            | `4`   | AIV→AIC同步Flag ID              |
| `AIC_SYNC_AIV_FLAG`            | `6`   | AIC→AIV同步Flag ID              |

### 计算公式

#### 符号约定

| 符号           | 含义                                                                 |
| -------------- | -------------------------------------------------------------------- |
| $e$            | 专家数，即groupList的长度                                            |
| $m$            | 总token数，即A的第一维                                               |
| $k$            | 隐藏维度，即A的第二维                                                |
| $n$            | 输出维度                                                             |
| $\text{batch}$ | 输出批次大小                                                         |
| $\text{bsdp}$  | 共享专家行数，$\text{bsdp} = \text{batch} / \text{dataParallelSize}$ |
| $s_i$          | 第 $i$ 个专家处理的token起始索引                                     |
| $m_i$          | 第 $i$ 个专家处理的token数                                           |

#### 第一步：分组矩阵乘法（MX FP8）

对每个专家 $i \in \{0, 1, \ldots, e-1\}$：

$$C_{[s_i:s_i+m_i,\; :]} = \hat{A}_{[s_i:s_i+m_i,\; :]} \times \hat{B}_i + \text{bias}_i$$

其中 $\hat{A}$、$\hat{B}$ 为MX反量化后的FP32矩阵。

#### 第二步：共享专家输出赋值

$$\text{out}[j, :] = \mathbf{0}, \quad j = 0, 1, \ldots, \text{batch}-1$$

若SharedInput不为空，对 $j \in [0,\; \text{bsdp})$：

$$\text{out}[j + \text{sharedInputOffset}, :] = \text{sharedInputWeight} \times \text{Cast}_{\text{BF16} \to \text{FP32}}(\text{SharedInput}[j, :])$$

#### 第三步：Logit加权

$$\hat{C}[p, :] = \text{logit}[p] \times C[p, :], \quad p = 0, 1, \ldots, m-1$$

#### 第四步：Scatter Add聚合

$$\text{out}[\text{rowIndex}[p], :] \mathrel{+}= \hat{C}[p, :], \quad p = 0, 1, \ldots, m-1$$

#### 完整公式

$$\text{out}[r, :] = \sum_{\substack{p=0 \\ \text{rowIndex}[p] \,=\, r}}^{m-1} \text{logit}[p] \cdot C[p, :] \;+\; \mathbb{1}_{\text{shared}}(r) \cdot \text{sharedInputWeight} \cdot \text{Cast}_{\text{BF16} \to \text{FP32}}(\text{SharedInput}[r - \text{sharedInputOffset}, :])$$

其中指示函数 $\mathbb{1}_{\text{shared}}(r)$ 定义为：

$$\mathbb{1}_{\text{shared}}(r) = \begin{cases} 1, & \text{SharedInput} \neq \text{nullptr} \;\wedge\; r \in [\text{sharedInputOffset},\; \text{sharedInputOffset} + \text{bsdp}) \\ 0, & \text{otherwise} \end{cases}$$

### Tile参数

| 层级   | Shape (M×N×K)   |
| ------ | --------------- |
| L1Tile | 256 × 256 × 256 |
| L0Tile | 256 × 256 × 128 |

### 非确定性版（no_deter）设计差异

非确定性版（`fp8_mx_grouped_matmul_finalize_routing_no_deter.cpp`）与确定性版共享相同的计算逻辑（MX FP8反量化 + 分组矩阵乘 + Finalize Routing），仅在**调度方式**和**AIV后处理切分方向**上有所不同。

#### 调度器：GemmGroupedAswtTailSplitSwizzle

确定性版使用 `ColumnBlockSwizzle` 按列分块调度，每个 group 的 tile 按固定列顺序分配给 AICore。

非确定性版改用 [GemmGroupedAswtTailSplitSwizzle](../../include/catlass/gemm/block/block_swizzle_grouped_aswt.hpp)，主要特性：

1. **滚动核分配**：`startBlockIdx_` 跨 group 滚动，避免所有 group 的起始核固定对齐，提升多核利用率
2. **窗口调度**：通过滑动窗口限制同时在飞的 tile 数，降低资源竞争
3. **尾块多核拆分（UpdateTailTile）**：最后一个 group 的尾部 tile 在满足条件时可拆分到多个核并行处理，改善负载均衡
4. **AswtBlockShape**：`GetBlockShape` 返回 `AswtBlockShape{m, n, mOffset, nOffset}`，tile 坐标需加上 mOffset/nOffset 偏移

#### AIV后处理切分方向

| 对比项 | 确定性版 | 非确定性版 |
|--------|---------|-----------|
| AIV SubBlock切分 | N维一分为二（2个子核各处理 N/2 列） | M维一分为二（2个子核各处理 M/2 行） |
| `vecTileShape` | `MatrixShape<L1_M, L1_N/2>` | `MatrixShape<L1_M/2, L1_N>` |
| BlockEpilogue | `BlockEpilogueFinalizeRouting` | `BlockEpilogueFinalizeRoutingNoDeter` |

#### 同步协议

AIC/AIV 的 CrossCore Flag 同步协议（Flag 4/6、`AIC_SYNC_AIV_MODE_2`）与确定性版完全一致，workspace 布局和数据流也相同。

#### 适用场景

- **确定性版**：适用于对输出确定性（bit-exact reproducibility）有严格要求的场景
- **非确定性版**：适用于对确定性无要求、追求更高多核利用率和吞吐的场景，尤其在 group 数较多或尾部 tile 负载不均时优势更明显
