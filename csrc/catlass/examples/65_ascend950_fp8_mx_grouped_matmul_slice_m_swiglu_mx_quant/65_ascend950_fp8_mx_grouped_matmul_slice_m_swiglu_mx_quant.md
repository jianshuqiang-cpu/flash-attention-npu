# CATLASS Ascend950Fp8MxGroupedMatmulSliceM_Swiglu_MxQuant

## 详细设计

### 算子流程

```text
(A * MxScaleA) @ (B * MxScaleB) → C(m,n)
        ↓ Slice n
   Act(m,n/2)    Gate(m,n/2)
        ↓ SwiGLU
   S = Swish(Act) * Gate
        ↓ MX Quant
   Q(m,n/2), QScale(m,ceil(n/2/64),2)
```

Swish公式为$Swish(x)=\frac{x}{1+e^{-x}}$

MX Quant过程如下:

$shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rfloor - emax$

$QScale = 2 ^ {shared\_exp}$

$Q_i = quantize\_to\_element\_format(S_i/Qscale), \space i\space from\space 1\space to\space blocksize$

- $emax$: 对应数据类型的最大正则数的指数位。

  |   DataType    | emax |
  | :-----------: | :--: |
  | FLOAT8_E4M3FN |  8   |
  |  FLOAT8_E5M2  |  15  |

- $blocksize$：指每次量化的元素个数，仅支持32。

### 模板组装

| 组件           | 模板类                                                                                                                       | 说明                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Kernel         | [GroupedMxMatmulSliceMSwigluMxQuantTla](../../include/catlass/gemm/kernel/grouped_mx_matmul_slice_m_swiglu_mx_quant_tla.hpp) | AIC/AIV双核协作，按group遍历，Slice-M分组调度              |
| BlockMmad      | [BlockMmadTla](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)                                                 | MX量化矩阵乘，DispatchPolicy=`MmadMx<Ascend950, true, 16>` |
| TileCopy       | [PackedMxTileCopyTlaToUB](../../include/catlass/gemm/block/block_mmad.hpp)                                                   | GM→L1→UB数据搬运，`SPLIT_M`模式                            |
| BlockEpilogue  | [BlockEpilogue\<BlockEpilogueSwigluMxQuant\>](../../include/catlass/epilogue/block/block_epilogue_swiglu_mx_quant.hpp)       | SwiGLU激活 + MX FP8量化输出                                |
| BlockScheduler | [GemmIdentityBlockSwizzle\<3,1\>](../../include/catlass/gemm/block/block_swizzle.hpp)                                        | 按(M,N)平铺分块，轮转分配到各AICore                        |

### AIC/AIV双核协作

- **AIC**：执行MX矩阵乘，对每个Block分别加载B的Act列（`[0, N/2)`）和Gate列（`[N/2, N)`）执行matmul计算，共两次matmul计算。结果写入UB上，之后进行AIV操作。写入UB时，按M分到当前AIC核对应的两个AIV核上。
- **AIV**：接收AIC计算结果，在同一UB上执行 `Swish(Act) × Gate → MXQuant`，将量化结果Q和QScale写回GM，继续下一Block计算。

### MX Quant详细设计

#### 概述

MX Quant的数学定义为：对每32个元素组成的block，取绝对值最大值，计算共享指数 `shared_exp = floor(log2(max|S_i|)) - emax`，然后以 `2^shared_exp` 为缩放因子将数据量化到FP8。

当前实现中**并未直接计算 `log2` 和 `2^x`**，而是完全在BF16指数域（exponent field）上进行位操作来等价实现上述数学语义，利用硬件向量指令高效完成。整个过程分为三个阶段：

```text
SwiGLU输出 (BF16)
    ↓ ComputeMaxExp      提取每个block的最大指数域
    ↓ ComputeScale        在指数域上计算shared_exp、scale、half_scale
    ↓ QuantToFp8          用half_scale缩放数据后Cast到FP8
Q (FP8), QScale (E8M0)
```

#### 阶段一：ComputeMaxExp — 提取块内最大指数域

对SwiGLU输出（已转为BF16）的每个32元素block：

1. **提取指数域**：对每个BF16元素做 `And(data, 0x7F80)`，掩码掉符号位和尾数位，仅保留指数域。BF16的指数域值与数值的绝对值大小单调对应，因此取指数域的最大值等价于取绝对值最大值对应的指数。
2. **块内归约**：对32个元素的指数域执行 `ReduceMax`，得到该block的 `max_exp`。

此步骤等价于公式中的 `floor(log2(max_i(|S_i|)))`，但避免了显式的 `log2` 运算。

#### 阶段二：ComputeScale — 计算量化Scale与反量化HalfScale

对每个block的 `max_exp` 进行如下处理：

| 步骤              | 操作                                                            | 说明                                                               |
| ----------------- | --------------------------------------------------------------- | ------------------------------------------------------------------ |
| 特殊值判断        | `cmpResult = (max_exp != 0x7F80)`                               | 排除Inf/NaN                                                        |
| 零值判断          | `zeroMask = (max_exp != 0)`                                     | 排除全零block                                                      |
| 计算shared_exp    | `shared_exp = max_exp - EMAX`                                   | `EMAX = emax << 7`，在BF16指数域上等价于 `floor(log2(max)) - emax` |
| 计算QScale (E8M0) | `scale_value = shared_exp >> 7`                                 | 右移7位提取E8M0格式的scale字节                                     |
| 特殊值修正        | 若为Inf/NaN → `scale_value = 0xFF`；若为零 → `scale_value = 0`  | 处理边界情况                                                       |
| 计算HalfScale     | `half_scale_val = BF16_EXP_BIAS(0x7F00) - shared_exp`           | 构造一个BF16数，其指数域表示 `2^(-shared_exp)`，即 `1/QScale`      |
| HalfScale特殊修正 | Inf/NaN → `0x7F81`；零 → `0`；`shared_exp == 0x7F00` → `0x0040` | 处理边界情况                                                       |

**关键设计**：`half_scale_val` 是一个合法的BF16位模式，其指数域为 `0x7F00 - shared_exp`。当用它乘以原数据时，硬件BF16乘法器自动完成 `S_i × 2^(-shared_exp)` = `S_i / QScale` 的效果，无需显式计算 `2^x` 或做浮点除法。

#### 阶段三：QuantToFp8 — 缩放并Cast到FP8

1. **缩放**：将SwiGLU输出的BF16数据与对应block的 `half_scale`（BF16格式）做向量乘法 `data_bf16 × half_scale_bf16`。由于 `half_scale` 的指数域编码了 `-shared_exp`，此乘法等价于 `S_i / QScale`。
2. **类型转换**：将缩放后的BF16结果先Cast到FP32，再通过硬件Cast指令以饱和模式（`SatMode::SAT`，`RoundMode::CAST_RINT`）转换到目标FP8类型（`float8_e4m3fn` 或 `float8_e5m2`），自动完成clamp和舍入。

此步骤等价于公式中的 `quantize_to_element_format(S_i / QScale)`。

#### 常量表

| 常量                    | 值                 | 含义                                                |
| ----------------------- | ------------------ | --------------------------------------------------- |
| `MAX_EXP_FOR_BF16`      | `0x7F80`           | BF16指数域掩码（Inf/NaN的指数域值）                 |
| `BF16_EXP_BIAS`         | `0x7F00`           | BF16指数偏置（对应指数0）                           |
| `EMAX_SHIFTED`          | `0x0400`或`0x0780` | 对应数据类型的最大正则数的指数位                    |
| `SHR_NUM_FOR_BF16`      | `7`                | BF16指数域到E8M0 scale的右移位数                    |
| `MAX_EXP_FOR_FP8`       | `0xFF`             | E8M0格式的Inf/NaN标记                               |
| `NAN_CUSTOMIZATION`     | `0x7F81`           | NaN场景下的half_scale特殊值                         |
| `SPECIAL_EXP_THRESHOLD` | `0x0040`           | shared_exp恰好等于BF16_EXP_BIAS时的half_scale修正值 |

### Tile参数

| 层级   | Shape (M×N×K)   |
| ------ | --------------- |
| L1Tile | 128 × 256 × 256 |
| L0Tile | 128 × 256 × 128 |
