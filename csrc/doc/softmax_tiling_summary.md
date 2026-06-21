# softmax_tiling.cpp 详细解析

> **文件位置**: `csrc/flash_attn_npu/softmax_tiling.cpp`
> **运行位置**: Host 侧（CPU 端），**不是** Device 侧 kernel
> **作用**: 为 CANN 内置 softmax/softmax_grad 算子计算 UB 分块参数（Tiling）
> **调用方**: `fag_tiling.cpp::FAGTiling::GetFATilingParam()`（反向传播 tiling 阶段）

---

## 1. 文件定位

本文件在 **Host 侧**（CPU）运行，不执行任何 NPU 计算。它的唯一职责是：在 kernel launch 之前，根据张量形状、数据类型、UB workspace 大小，计算出 softmax/softmax_grad 算子沿行维的分块参数 `SoftMaxTiling`，供 Device 侧 NPU kernel 使用。

### 关键区分：前向 softmax vs 反向 softmax

| 场景 | softmax 实现位置 | Tiling 计算 |
|------|-----------------|------------|
| **前向推理**（mha_fwd_kvcache） | `online_softmax.hpp`（Device 侧，手写 kernel，全 float 精度） | 不需要本文件，online_softmax 内联计算 tile |
| **反向传播**（mha_bwd / mha_varlen_bwd） | 调用 CANN 内置 softmax/softmax_grad 算子 | **需要本文件**（Host 侧算 tiling 传给内置算子） |

前向 FlashAttention 的 softmax 已经完全由 `online_softmax.hpp` 在 NPU 上实现并与 PV 计算紧密流水线化，不需要外部 tiling。反向传播中 softmax 梯度部分复用了 CANN 的内置算子，需要 Host 侧预先计算分块参数。

### 编译方式

本文件通过 `#include "softmax_tiling.cpp"` 被 `fag_tiling.cpp` 文本包含，而 `fag_tiling.cpp` 又被 `flash_api.cpp` 包含，最终由 bisheng 编译器以 **unity build**（单编译单元）方式编译。setup.py 仅将 `flash_api.cpp` 作为源文件传入编译器。

### 调用链

```
flash_api.cpp (Host API: mha_bwd / mha_varlen_bwd)
  └── FAGTiling::GetFATilingParam()          [fag_tiling.cpp]
        ├── SoftMaxTilingFunc(softmaxShape, 4, 33KB, ...)         ← 本文件
        └── SoftMaxGradTilingFunc(softmaxGradShape, 4, ~37KB, ...) ← 本文件
              ↓
        将 tiling 结果打包进 TilingData → 传给 NPU kernel
```

---

## 2. Tiling 解决什么问题

Softmax 对张量**最后一维**做归约（逐行计算 max→exp→sum→div）。NPU 的 UB 容量有限（典型 ~33-40KB），当行数很多时无法一次性将所有行放入 UB，需要沿行维 M 切分成多个 **split**：

```
srcM 行 × srcK 列（展平后的 2D 矩阵）:
┌────────────────────────┐
│  split 0 (baseM 行)     │ ← 一次 UB 处理
├────────────────────────┤
│  split 1 (baseM 行)     │ ← 一次 UB 处理
├────────────────────────┤
│         ...            │   rangeM 个完整 split
├────────────────────────┤
│  tail (tailM 行)        │ ← 尾块（srcM % baseM 行，可能为0）
└────────────────────────┘
```

Tiling 的核心是计算出 **baseM**（每个 split 的行数），使得 `baseM × srcK`（split 数据）+ `baseM × elementNumPerBlk`（reduce 缓冲）+ 对齐余量能装进 UB workspace。

---

## 3. SoftMaxTiling 输出字段（16 个 uint32_t）

`SoftMaxTiling` 定义在 CANN toolchain 外部头文件（`-ltiling_api`），本文件中的 `SoftMaxTilingLocal` 是其镜像结构体。字段含义：

| 字段 | 含义 | 计算公式 |
|------|------|---------|
| `srcM` | 展平后行数 | `prod(前N-1维)` |
| `srcK` | 归约轴长度（列数） | `最后一维大小` |
| `srcSize` | 总元素数 | `srcM × srcK` |
| `outMaxM` | reduce 输出最大行数 | `= srcM` |
| `outMaxK` | reduce 输出块列数 | `= elementNumPerBlk` |
| `outMaxSize` | reduce 输出最大元素数 | `srcM × elementNumPerBlk` |
| `splitM` | **每 split 行数（核心参数 baseM）** | UB 容量估算+对齐 |
| `splitK` | 每 split 列数 | `= srcK`（整行处理） |
| `splitSize` | 每 split 元素数 | `baseM × srcK` |
| `reduceM` | reduce 阶段行数 | `= baseM` |
| `reduceK` | reduce 阶段块列数 | `= elementNumPerBlk` |
| `reduceSize` | reduce 阶段元素数 | `baseM × elementNumPerBlk` |
| `rangeM` | 完整 split 数量 | `srcM / baseM` |
| `tailM` | 尾块行数 | `srcM % baseM`（0=无尾块） |
| `tailSplitSize` | 尾块数据元素数 | `tailM × srcK` |
| `tailReduceSize` | 尾块 reduce 元素数 | `tailM × elementNumPerBlk` |

---

## 4. 核心算法流程

### 4.1 baseM 估算公式

```
elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE(32) / dataTypeSize
                  = 32/4 = 8 (float)
                  = 32/2 = 16 (half)

workLocalSize = localWorkSpaceSize / 4  (字节→float元素数)

baseM = workLocalSize / (每行列缓冲需求)
```

**每行列缓冲需求**（分母的含义——处理 1 行所需的 float 元素缓冲）：

| 函数 | dataTypeSize | 分母公式 | 含义 |
|------|-------------|---------|------|
| SoftMaxTilingFunc | float(4) | `8 + srcK + 64` | 1个reduce缓冲 + 1个数据缓冲 + 对齐 |
| SoftMaxGradTilingFunc | float(4) | `8 + srcK + 64` | （同前向，当前实际使用） |
| SoftMaxGradTilingFunc | half(2) | `16×2 + srcK×3 + 64` | 2个reduce缓冲 + 3个数据缓冲 + 对齐 |

反向 half 精度需要更多缓冲，因为梯度公式 `dX = Y * (dY - sum(dY * Y))` 需要同时保留 Y（softmax 输出）、dY（上游梯度）、dY*Y（中间乘积）共 3 份行数据。

### 4.2 对齐调整流程

```
baseM 估算值
   │
   ▼
【对齐1】若 baseM < srcM 且 > 8:
         baseM 向下对齐到 BASIC_TILE_NUM(8) 的倍数
   │
   ▼
【AdjustToBasicBlockBaseM】（仅当 baseM>8, srcM%8==0, srcK%64==0 时）:
   步骤1: baseM 向下对齐到 8 的倍数
   步骤2: while (srcM % baseM != 0) baseM -= 8   (保证整除 srcM)
   步骤3: while (baseM*srcK >= 64*256=16384) baseM /= 2  (硬件repeat≤255)
   │
   ▼
最终 baseM → 计算所有 split/reduce/range/tail 字段
```

### 4.3 数据流图

```
Host CPU:
  srcShape (N-D)
     │
     ▼ GetLastAxisShapeND()
  {srcM, srcK}  (展平为2D)
     │
     ├─ elementNumPerBlk = 32 / dataTypeSize
     ├─ workLocalSize = localWorkSpaceSize / 4
     │
     ▼ baseM 估算
  baseM = min(workLocalSize / (reduce_buf + data_buf + align), srcM)
     │
     ▼ 对齐到8的倍数 + AdjustToBasicBlockBaseM
  aligned baseM
     │
     ▼ 填充16个字段
  SoftMaxTiling {srcM, srcK, splitM=baseM, rangeM=srcM/baseM, tailM=srcM%baseM, ...}
     │
     ▼ (通过 TilingData 传递)
  NPU Device kernel 使用 tiling 参数控制 UB 分块
```

---

## 5. 常量说明

| 常量 | 值 | 含义 |
|------|---|------|
| `SOFTMAX_DEFAULT_BLK_SIZE` | 32 | 默认块大小（字节），即一个向量寄存器宽度 32B |
| `SOFTMAX_TMPBUFFER_COUNT` | 2 | softmax 前向临时缓冲数（split 数据 + reduce 结果） |
| `SOFTMAX_HALF_SIZE` | 2 | half/bf16 元素字节数 |
| `SOFTMAX_FLOAT_SIZE` | 4 | float 元素字节数 |
| `SOFTMAXGRAD_TMPBUFFER_COUNT` | 3 | softmax 反向临时缓冲数（比前向多 1 个，用于 dY*Y 中间结果） |
| `BASIC_TILE_NUM` | 8 | 基础 tile 数 = 32B / 4B = 8 个 float 元素 |
| `SOFTMAX_BASICBLOCK_MIN_SIZE` | 256 | basicblock 最小尺寸（与硬件 repeat 限制相关） |
| `SOFTMAX_BASICBLOCK_UNIT` | 64 | basicblock 对齐单位（元素数） |

---

## 6. 例子 1：Softmax 前向 Tiling（实际调用参数）

来自 `fag_tiling.cpp` 中 `GetFATilingParam()` 的实际调用。

### 输入参数

| 参数 | 值 | 来源 |
|------|-----|------|
| `srcShape` | `{64, 128}` | 固定以 64 行 × 128 列估算（s1VecSize=64, s2VecSize=128） |
| `dataTypeSize` | 4 (float) | `sizeof(float)` |
| `localWorkSpaceSize` | 33 × 1024 = 33792 字节 | `tmpBufferSize = 33*1024` |

### 逐步计算

**Step 1: 展平 shape**
```
srcShape = {64, 128}（已经是2D）
srcK = 128（最后一维）
srcM = 64×128/128 = 64
```

**Step 2: 计算基本参数**
```
elementNumPerBlk = 32 / 4 = 8 (float 元素/块)
workLocalSize = 33792 / 4 = 8448 (float 元素数)
```

**Step 3: 估算 baseM**
```
每行列缓冲需求 = elementNumPerBlk + srcK + 64 = 8 + 128 + 64 = 200 float元素
baseM = min(8448 / 200, 64) = min(42, 64) = 42
```

**Step 4: 对齐到 8 的倍数**
```
baseM=42 < srcM=64, >8 → 42/8*8 = 40
```

**Step 5: AdjustToBasicBlockBaseM(40, 64, 128)**
```
条件: baseM=40>8, srcM=64%8=0, srcK=128%64=0 → 进入basicblock
步骤1: baseM = 40/8*8 = 40
步骤2: while(64%40≠0):   64%40=24≠0 → baseM=32
                          64%32=0 → 退出
步骤3: while(32*128=4096 >= 16384?) → 4096<16384 → 不进入
最终 baseM = 32
```

**Step 6: 填充输出字段**

| 字段 | 值 | 字段 | 值 |
|------|-----|------|-----|
| srcM | 64 | srcK | 128 |
| srcSize | 8192 | outMaxM | 64 |
| outMaxK | 8 | outMaxSize | 512 |
| **splitM** | **32** | splitK | 128 |
| splitSize | 4096 | reduceM | 32 |
| reduceK | 8 | reduceSize | 256 |
| rangeM | 2 | tailM | 0 |
| tailSplitSize | 0 | tailReduceSize | 0 |

### 结果图示

```
UB workspace (33KB = 8448 float elements)
┌─────────────────────────────────────────────┐
│ split 0: 32行 × 128列 = 4096 float (16KB)   │  ← goUbTensor 等价区域
│ reduce:  32行 × 8块    = 256 float  (1KB)   │  ← reduce 临时缓冲
│ 对齐余量 + 其他:        ~4096 float (16KB)   │  ← 工作余量
└─────────────────────────────────────────────┘

矩阵分块 (64行 × 128列):
┌──────────────────┐  ← split 0 (32行)
│  rows [0:32)     │    rangeM=2 个完整 split
├──────────────────┤  ← split 1 (32行)
│  rows [32:64)    │    tailM=0 (无尾块)
└──────────────────┘
```

---

## 7. 例子 2：Softmax 反向 Grad Tiling（headdim=128 实际调用）

来自 `fag_tiling.cpp` 中 `GetFATilingParam()` 的 SoftMaxGradTilingFunc 调用，headdim=128。

### 输入参数（从 fag_tiling.cpp 推导）

| 参数 | 值 | 推导 |
|------|-----|------|
| `srcShape` | `{48, 128}` | singleLoopNBurstNum=24KB/4/128=48 行, headdim=128 |
| `dataTypeSize` | 4 (float) | 传入 sizeof(float) |
| `localWorkSpaceSize` | ~37880 字节 | tempBufferLen = 40KB - outputBufferLen ≈ 40960-3080=37880 |

其中：
- `inputBufferLen = 24*1024 = 24576`，`castBufferLen = 48*1024 = 49152`
- `outputBufferLen = ceil(49152/128) * 8 = 385*8 = 3080`
- `tempBufferLen = 40960 - 3080 = 37880`
- `singleLoopNBurstNum = 24576 / 4 / 128 = 48`

### 逐步计算

**Step 1: 展平 shape**
```
srcShape = {48, 128}（已经是2D）
srcK = 128, srcM = 48
```

**Step 2: 基本参数**
```
elementNumPerBlk = 32/4 = 8
workLocalSize = 37880/4 = 9470
dataTypeSize=4≠2 → 走else分支（同前向公式）
```

**Step 3: 估算 baseM**
```
每行列缓冲需求 = 8 + 128 + 64 = 200
baseM_initial = min(9470/200, 48) = min(47, 48) = 47
```

**Step 4: 对齐**
```
baseM=47 < srcM=48, >8 → 47/8*8 = 40
```

**Step 5: AdjustToBasicBlockBaseM(40, 48, 128)**
```
条件: 40>8, 48%8=0, 128%64=0 → 进入basicblock
步骤1: baseM = 40
步骤2: while(48%40≠0):   48%40=8≠0 → baseM=32
                          48%32=16≠0 → baseM=24
                          48%24=0 → 退出
步骤3: while(24*128=3072 >= 16384?) → 3072<16384 → 不进入
最终 baseM = 24
```

**Step 6: 输出字段**

| 字段 | 值 | 字段 | 值 |
|------|-----|------|-----|
| srcM | 48 | srcK | 128 |
| srcSize | 6144 | outMaxM | 48 |
| outMaxK | 8 | outMaxSize | 384 |
| **splitM** | **24** | splitK | 128 |
| splitSize | 3072 | reduceM | 24 |
| reduceK | 8 | reduceSize | 192 |
| rangeM | 2 | tailM | 0 |
| tailSplitSize | 0 | tailReduceSize | 0 |

### 结果图示

```
softmax grad 反向计算矩阵 (48行 × 128列):
┌──────────────────┐  ← split 0 (24行)
│  rows [0:24)     │    rangeM=2 splits
├──────────────────┤  ← split 1 (24行)
│  rows [24:48)    │    tailM=0
└──────────────────┘

每 split 在 UB 中的缓冲布局 (~37KB):
┌──────────────────────────────────────────┐
│ Y (softmax输出):     24×128 = 3072 float │  ← srcK缓冲
│ dY (上游梯度):       24×128 = 3072 float │  ← (grad额外需要)
│ dY*Y (中间乘积):     24×128 = 3072 float │  ← (grad额外需要)
│ reduce (sum缓冲):    24×8   = 192 float  │  ← elementNumPerBlk缓冲
│ 其他工作余量:         ~剩余空间           │
└──────────────────────────────────────────┘
```

---

## 8. 函数逐个解析

### 8.1 `GetLastAxisShapeND<T>()`

模板函数，将任意维度的 shape 展平为 2D `{srcM, srcK}`：
- `srcK = shape.back()`（最后一维 = softmax 归约轴）
- `srcM = totalElements / srcK`（前面所有维展平为行）

模板参数 T 兼容 `uint32_t`（前向调用）和 `int64_t`（反向调用，因为反向 shape 来自 `int64_t` 张量维度）。

### 8.2 `AdjustToBasicBlockBaseM()`

basicblock 是 CANN AscendC 中一种特殊的向量化执行模式，需要满足对齐约束才能启用：
- 条件：baseM > 8，srcM 能被 8 整除，srcK 能被 64 整除
- 三步调整确保：对齐到 8 → 整除 srcM → 不超过硬件 repeat=255 限制

```
AdjustToBasicBlockBaseM 决策树:

baseM > 8?
   ├─ No → 不调整（baseM太小，basicblock不适用）
   └─ Yes
       srcM % 8 == 0?
       ├─ No → 不调整
       └─ Yes
           srcK % 64 == 0?
           ├─ No → 不调整
           └─ Yes → 进入basicblock调整：
                   1. baseM = floor(baseM/8)*8   (对齐8)
                   2. while srcM%baseM≠0: baseM-=8 (整除srcM)
                   3. while baseM*srcK>=16384: baseM/=2 (repeat≤255)
```

### 8.3 `SoftMaxTilingFunc()`

前向 softmax tiling，7 步：
1. 展平 N-D → 2D
2. 计算 elementNumPerBlk（每块元素数）
3. 转换 workspace 单位（字节 → float 元素数）
4. 估算 baseM = min(workLocalSize / (reduce_buf + data_buf + align), srcM)
5. 对齐 baseM 到 8 的倍数
6. basicblock 调整
7. 填充 16 个输出字段

### 8.4 `SoftMaxGradTilingFunc()`

反向 softmax_grad tiling，与前向逻辑几乎一致，唯一区别在 Step 4：
- half 精度(dataTypeSize==2)：分母 = elementNumPerBlk×2 + srcK×3 + 64（多 1 reduce + 2 数据缓冲）
- float 精度（当前实际使用）：分母同前向 = elementNumPerBlk + srcK + 64

### 8.5 `printSoftmaxTilingData()`

调试打印函数，输出所有 16 个 tiling 字段值。**当前无任何调用方**（死代码），类似的 `printFAGTilingData` 在 flash_api.cpp 中的调用也被注释掉。保留是为了开发调试时可临时启用。

---

## 9. 设计亮点

1. **Host/Device 分离**：Tiling 计算在 Host 侧完成纯数学运算，Device 侧 kernel 直接使用结果，避免 NPU 上做分支判断。
2. **模板兼容多类型**：`GetLastAxisShapeND<T>` 同时支持 `vector<uint32_t>` 和 `vector<int64_t>`，对应前/反向不同的 shape 来源。
3. **UB 容量感知**：baseM 由 `workLocalSize / per_row_cost` 直接推导，物理意义明确（每行列缓冲需求）。
4. **多级对齐**：先对齐到 BASIC_TILE_NUM(8)，再做 basicblock 三步骤整，最后检查硬件 repeat 限制。
5. **反向缓冲区分**：half 精度反向需要 3 个数据缓冲 vs 前向 1 个，通过 SOFTMAX_TMPBUFFER_COUNT/SOFTMAXGRAD_TMPBUFFER_COUNT 常量清晰表达。
6. **整除保证**：AdjustToBasicBlockBaseM 步骤 2 的 `while(srcM%baseM≠0)` 确保 rangeM 个完整 split 能均匀覆盖所有行，减少尾块处理开销。
7. **零运行时开销**：所有计算都是编译期/启动期整数运算，不影响 NPU 执行性能。

---

## 10. 注意事项

- **本文件是 Host 侧代码**：不在 NPU 上运行，不要与 online_softmax.hpp（Device 侧 kernel）混淆。前向 FlashAttention 不使用本文件。
- **SoftMaxTiling 来自外部 CANN 库**：类型定义不在本代码库中（链接 `-ltiling_api`），字段顺序和大小通过 `SoftMaxTilingLocal` 镜像验证——修改字段必须同步两边。
- **当前调用始终使用 float 路径**：fag_tiling.cpp 传入 `sizeof(float)=4`，SoftMaxGradTilingFunc 的 half 分支（dataTypeSize==2）是预留路径，未被实际使用。
- **`srcShape` 是估算形状而非实际张量形状**：SoftMaxTilingFunc 传入固定 `{64, 128}` 做 UB 容量估算，不是实际的 softmax 输入形状（实际形状在 kernel 内部通过 tilingdata 传入）。
- **`#include <iomanip>` 未被使用**：文件包含 `<iomanip>` 但代码中没有使用任何 iomanip 操纵符（如 setw/setprecision），可移除但不影响编译。
- **`printSoftmaxTilingData` 缺少 `<iostream>` 包含**：该函数使用 `std::cout`/`std::endl` 但未 `#include <iostream>`，目前因 unity build 中其他文件间接包含而能编译，但独立看是不完整的。由于是死代码，不影响功能。
- **`SoftMaxTilingLocal` 未被使用**：本地镜像结构体仅用于文档/理解目的，实际代码直接操作外部 `SoftMaxTiling&` 引用。
- **workLocalSize 单位假设**：`workLocalSize = localWorkSpaceSize / SOFTMAX_FLOAT_SIZE` 将字节数除以 4 得到 float 元素数，但当 dataTypeSize=2(half)时，half 元素数应为字节/2；不过分母中 `elementNumPerBlk * SOFTMAX_TMPBUFFER_COUNT` 等也按 float 宽度估算，整体单位一致，公式自洽。

---

## 11. 总结

`softmax_tiling.cpp` 以约 180 行 Host 侧 C++ 代码实现了 CANN softmax/softmax_grad 算子的 UB 分块参数计算：

1. **核心任务**：根据 UB workspace 大小和张量形状，计算最优的每 split 行数 baseM
2. **两个 Tiling 函数**：`SoftMaxTilingFunc`（前向）和 `SoftMaxGradTilingFunc`（反向，支持 half 多缓冲路径）
3. **关键参数 baseM**：受 UB 容量约束，经"估算→8对齐→basicblock调整"三阶段确定
4. **16 字段输出**：完整描述 split/reduce/range/tail 分块信息供 Device 侧 kernel 使用
5. **仅用于反向传播**：前向 FlashAttention 使用手写 online_softmax kernel，不需要本文件

它是 FlashAttention NPU 反向传播 Host API 调用链中的一个纯数学辅助模块，通过 unity build 嵌入 `flash_api.cpp`，在 kernel launch 前为 CANN 内置 softmax 算子准备分块参数。
