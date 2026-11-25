# Conv算子开发指南

该文档面向Conv算子的开发场景，以Basic Conv2D为切入点，说明从问题定义、模板参数选择到Kernel组装和Host调用的完整流程，帮助开发者快速上手Conv2D算子的开发与调测。

## Conv计算原理

**问题定义**

对于一个2D卷积，输入（Fmap）的shape为 $(N, C_{\text{in}}, H_i, W_i)$，权重（Filter）的shape为 $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$，输出（Output）的shape为 $(N, C_{\text{out}}, H_o, W_o)$。输出中每个位置的计算公式为：

$$
\text{out}(n, c_{\text{out}}, h_o, w_o) = \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} \text{fmap}(n, c_{\text{in}}, h_i+i, w_i+j) \cdot \text{filter}(c_{\text{out}}, c_{\text{in}}, i, j)
$$

其中输出空间维度由输入、padding、dilation和stride共同决定：

$$
\begin{aligned}
H_o &= \lfloor (H_i + \text{padTop} + \text{padBottom} - \text{dilationH} \times (K_h - 1) - 1) / \text{strideH} \rfloor + 1 \\
W_o &= \lfloor (W_i + \text{padLeft} + \text{padRight} - \text{dilationW} \times (K_w - 1) - 1) / \text{strideW} \rfloor + 1
\end{aligned}
$$

Conv2D本质上可以转化为矩阵乘（Im2col + GEMM），Catlass中通过tile级别的BlockConv2d组件直接实现滑窗计算，避免显式Im2col重排。

**数据排布**

Conv2D涉及三种数据布局，均为5维Tensor：

| 张量   | Layout         | 维度含义                     | 默认C0 |
| ------ | -------------- | ---------------------------- | ------ |
| Fmap   | `NC1HWC0`      | `[Batch, Cin1, Hi, Wi, C0]`  | 16     |
| Filter | `CI1KHKWCOCI0` | `[Cin1, Kh, Kw, Cout, C0]`   | 16     |
| Output | `NC1HWC0`      | `[Batch, Cout1, Ho, Wo, C0]` | 16     |

其中`C0 = 16`为硬件对齐粒度（`BYTE_PER_C0 / sizeof(fp16)`），`Cin1 = CeilDiv(Cin, C0)`，`Cout1 = CeilDiv(Cout, C0)`。

Catlass中对应的Layout类型定义在`include/catlass/layout/tensor.hpp`：

- `layout::NC1HWC0`— Fmap和Output使用，strides为`[Cin1*Hi*Wi*C0, Hi*Wi*C0, Wi*C0, C0, 1]`
- `layout::CI1KHKWCOCI0`— Filter使用，strides为`[Kh*Kw*Cout*C0, Kw*Cout*C0, Cout*C0, C0, 1]`

## 完整代码解析

**1. 参数定义与命令行解析**

`Options`结构体定义并解析卷积参数：

```cpp
struct Options {
    uint32_t dataSizes[5] = {2, 33, 43, 112, 80}; // {batch, hi, wi, cin, cout}
    uint8_t filterSizes[2] = {3, 3};               // {kh, kw}
    uint8_t pads[4] = {2, 2, 2, 2};                // {padLeft, padRight, padTop, padBottom}
    uint8_t strides[2] = {2, 2};                   // {strideH, strideW}
    uint8_t dilations[2] = {1, 1};                 // {dilationH, dilationW}
    int32_t deviceId{0};
    Catlass::Conv2dParams problemParams{};
    // Parse() 从argv解析各参数后调用:
    // problemParams = Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);
};
```

`Conv2dParams`（`include/catlass/conv_coord.hpp`）根据输入shape自动计算以下导出量：

| 字段              | 含义             | 计算方式                      |
| ----------------- | ---------------- | ----------------------------- |
| `cin1()`          | 输入C1维         | `CeilDiv(cin, C0)`            |
| `cout()`          | 输出通道         | 用户指定                      |
| `cout1()`         | 输出C1维         | `CeilDiv(cout, C0)`           |
| `coutRound()`     | 对齐后的输出通道 | `(cout + C0 - 1) / C0 * C0`   |
| `ho()`/`wo()`     | 输出空间维度     | 由pad/stride/dilation公式计算 |
| `postIm2colShape` | 任务空间         | `{Batch, Ho, Wo, Cout, Cin1}` |

**2. 环境初始化与内存分配**

```cpp
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(options.deviceId));
ACL_CHECK(aclrtCreateStream(&stream));

// 从Conv2dParams中取出各维度
uint32_t c0 = options.problemParams.C0;        // 16
uint32_t batch = options.problemParams.batch();
uint32_t hi = options.problemParams.hi();
uint32_t wi = options.problemParams.wi();
uint32_t cin1 = options.problemParams.cin1();
uint32_t ho = options.problemParams.ho();
uint32_t wo = options.problemParams.wo();
uint32_t cout1 = options.problemParams.cout1();
uint32_t cout = options.problemParams.cout();
uint32_t coutRound = options.problemParams.coutRound();
uint32_t kh = options.problemParams.kh();
uint32_t kw = options.problemParams.kw();

// 计算各Tensor所需大小（单位：元素个数）
size_t lenFmap = batch * cin1 * hi * wi * c0;
size_t lenFilter = cin1 * kh * kw * cout * c0;
size_t lenOutput = batch * ho * wo * coutRound;
```

注意`lenFilter`中的`cout`是原始通道数（未对齐），因为Filter Layout的最后一维是C0=16，物理大小是`cin1 * kh * kw * cout * c0`。

**3. Host数据准备**

```cpp
using LayoutFmap = layout::NC1HWC0;
using LayoutFilter = layout::CI1KHKWCOCI0;
using LayoutOutput = layout::NC1HWC0;

LayoutFmap layoutFmap{batch, cin1, hi, wi, c0};
LayoutFilter layoutFilter{cin1, kh, kw, cout, c0};
LayoutOutput layoutOutput{batch, cout1, ho, wo, c0};

std::vector<fp16_t> hostFmap(lenFmap);
std::vector<fp16_t> hostFilter(lenFilter);
golden::FillRandomData<fp16_t>(hostFmap, -5.0f, 5.0f);
golden::FillRandomData<fp16_t>(hostFilter, -5.0f, 5.0f);
```

Layout对象用于计算各Tensor内部的偏移量，Host数据通过随机数填充后拷贝到Device。

**4. 模板参数选择**

这是Conv2D算子开发的核心环节，需要根据目标shape和硬件约束选择一组模板参数。

- **4.1 Architecture Tag**

```cpp
using ArchTag = Arch::AtlasA2;
```

指定目标昇腾芯片架构，Catlass目前主要支持`Arch::AtlasA2`。

- **4.2 Dispatch Policy — 流水策略**

```cpp
constexpr uint32_t L1A_STAGES = 2;
constexpr uint32_t L1B_STAGES = 2;
constexpr uint32_t L0A_STAGES = 2;
constexpr uint32_t L0B_STAGES = 2;
constexpr uint32_t L0C_STAGES = 1;
constexpr bool ENABLE_UNIT_FLAG = false;

using DispatchPolicy =
    Conv::ConvAtlasA2Pingpong<L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG>;
```

`ConvAtlasA2Pingpong`控制各级存储（L1/L0A/L0B/L0C）的Double Buffer stage数：

| 参数               | 默认值 | 说明                          |
| ------------------ | ------ | ----------------------------- |
| `L1A_STAGES`       | 2      | Fmap从GM到L1的多buffer深度    |
| `L1B_STAGES`       | 2      | Filter从GM到L1的多buffer深度  |
| `L0A_STAGES`       | 2      | Fmap从L1到L0A的多buffer深度   |
| `L0B_STAGES`       | 2      | Filter从L1到L0B的多buffer深度 |
| `L0C_STAGES`       | 1      | L0C输出buffer深度             |
| `ENABLE_UNIT_FLAG` | false  | 是否启用unit flag             |

`BlockConv2d`内部会对stage数做cap：`L1A_STAGES = min(DispatchPolicy::L1A_STAGES, MAX_STAGES)`，`MAX_STAGES = 2`。

- **4.3 Tile Shape — 分块形状**

```cpp
using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<8, 12, 8>;   // (hoBlock, woBlock, cin1BlockSmall)
using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<96, 8>;  // (coutBlock, cin1BlockBig)
using L0TileShape = Catlass::Conv2dL0Shape<16, 96, 16>;         // (mL0, nL0, kL0)
```

这三个Tile Shape决定了卷积计算在各级存储上的分块粒度：

| Tile Shape                      | 含义                   | 作用                                            |
| ------------------------------- | ---------------------- | ----------------------------------------------- |
| `FmapL1TileShape<Ho, Wo, Cin1>` | Fmap加载到L1的基本块   | 每次加载`[Ho, Wo, Cin1, C0]`的Fmap tile         |
| `FilterL1TileShape<Cout, Cin1>` | Filter加载到L1的基本块 | 每次加载`[Cin1, Kh, Kw, Cout, C0]`的Filter tile |
| `L0TileShape<M, N, K>`          | L0A/L0B上的Mmad tile   | 每次Matrix Multiply的计算粒度                   |

注意`FmapL1TileShape::Cin1`和`FilterL1TileShape::Cin1`的关系：`K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1`，表示每加载一块Filter需要在K方向循环多次Fmap tile。

- **4.4 数据类型与Layout组合**

```cpp
using FmapType = Gemm::GemmType<half, LayoutFmap>;
using FilterType = Gemm::GemmType<half, LayoutFilter>;
using OutputType = Gemm::GemmType<half, LayoutOutput>;
```

将元素类型和Layout组合为GemmType，传递给BlockConv2d。

- **4.5 BlockConv与Kernel组装**

```cpp
using BlockConv2d = Conv::Block::BlockConv2d<
    DispatchPolicy, FmapL1TileShape, FilterL1TileShape, L0TileShape,
    FmapType, FilterType, OutputType>;
using BlockEpilogue = void;

using BlockScheduler = typename Conv::Block::Conv2dIdentityBlockSwizzle<3, 0>;

using Conv2dKernel = Conv::Kernel::BasicConv2d<BlockConv2d, BlockEpilogue, BlockScheduler>;
using Conv2dAdapter = Conv::Device::DeviceConv<Conv2dKernel>;
```

| 组件             | 作用                                                               |
| ---------------- | ------------------------------------------------------------------ |
| `BlockConv2d`    | 实现单次`Fmap_tile * Filter_tile -> Output_tile`的矩阵乘计算       |
| `BlockEpilogue`  | 后处理（当前为`void`，无后处理）                                   |
| `BlockScheduler` | 任务切分策略，决定每个AI Core处理哪块输出tile                      |
| `BasicConv2d`    | Kernel级封装，包含循环调度和边界处理                               |
| `DeviceConv`     | Host端适配器，封装CanImplement、GetWorkspaceSize、Initialize和执行 |

**5. 任务调度与边界处理**

`Conv2dIdentityBlockSwizzle<3,0>`定义任务切分方式。Kernel内部（`basic_conv2d.hpp`）的循环逻辑：

```cpp
for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < loops; loopIdx += AscendC::GetBlockNum()) {
    // 1. 获取当前块的逻辑坐标
    Conv2dCoord blockCoord = conv2dBlockScheduler.GetBlockCoord(loopIdx);
    // 2. 获取实际块大小（处理边界）
    Conv2dCoord actualBlockShape = conv2dBlockScheduler.GetActualBlockShape(blockCoord);
    // 3. 计算Fmap在H/W方向的有效范围（含边界padding处理）
    // 4. 计算各张量的GM偏移量
    // 5. 调用blockConv2d执行
    blockConv2d(gmFmap[...], layoutFmap, gmFilter[...], layoutFilter,
                gmOutput[...], layoutOutput, actualConv2dBlockShape, blockPadList);
}
```

边界处理逻辑：根据`hoStart / woStart`、stride、dilation和padding计算对应的`hiStart / hiEnd`和`wiStart / wiEnd`，当起始位置超出边界时记录padding量并裁剪范围。

**6. 执行与精度验证**

```cpp
Conv2dKernel::Arguments arguments{options.problemParams, deviceFmap, deviceFilter, deviceOutput};
Conv2dAdapter conv2d_op;
conv2d_op.CanImplement(arguments);
size_t sizeWorkspace = conv2d_op.GetWorkspaceSize(arguments);
// 分配workspace（如有）
conv2d_op.Initialize(arguments, deviceWorkspace);
conv2d_op(stream, aicCoreNum);
ACL_CHECK(aclrtSynchronizeStream(stream));

// 结果拷回Host并与golden对比
std::vector<fp16_t> hostOutput(lenOutput);
ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
std::vector<float> hostGolden(lenOutput);
golden::ComputeConv2d(options.problemParams, hostFmap, layoutFmap, hostFilter, layoutFilter,
                      hostGolden, layoutOutput);
std::vector<uint64_t> errorIndices = golden::CompareData(hostOutput, hostGolden, cin1 * kh * kw * c0);
```

-`CanImplement`：检查当前shape是否在Kernel支持范围内（BasicConv2d始终返回true）
\-`GetWorkspaceSize`：返回所需workspace大小（BasicConv2d返回0）
\-`golden::ComputeConv2d`：CPU参考实现，用于精度对比

**7. 资源释放**

```cpp
ACL_CHECK(aclrtFree(deviceFmap));
ACL_CHECK(aclrtFree(deviceFilter));
ACL_CHECK(aclrtFree(deviceOutput));
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(options.deviceId));
ACL_CHECK(aclFinalize());
```

## Kernel执行工作流

`BasicConv2d`的Kernel实现在`include/catlass/conv/kernel/basic_conv2d.hpp`，其核心流程如下：

```text
[GM] Fmap ──GM→L1──→ [L1] Fmap Tile ──L1→L0A──→ [L0A] zN format
[GM] Filter ──GM→L1──→ [L1] Filter Tile ──L1→L0B──→ [L0B] nZ format
                                                     ↓
                                              [Cube] Mmad (L0A × L0B → L0C)
                                                     ↓
[GM] Output ←──L0C→GM── [L0C] Accumulator
```

工作流说明:

1. **GM → L1**：Fmap和Filter按tile从Global Memory搬运到L1，搬运过程中完成格式转换（随路NZ）
2. **L1 → L0A/L0B**：从L1到L0A/L0B的再搬运，进一步调整为Cube计算所需格式
3. **Cube Mmad**：L0A × L0B 矩阵乘，结果累加到L0C
4. **L0C → GM**：将L0C结果写回Global Memory
5. **K循环**：在`Cin1`维度上循环，累加多个K分块的结果

## 总结

本文档完整介绍了CATLASS中Basic Conv2D算子的开发流程。
