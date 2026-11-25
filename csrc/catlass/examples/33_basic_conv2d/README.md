# BasicConv2d Example Readme

## 代码组织

```text
├── 33_basic_conv2d
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_conv2d.cpp # 主文件
```

## 功能介绍

- 该算子完成2D版本的卷积计算
- 昇腾亲和的特征图尺寸表达是：`(N, C1, H, W, C0)`，其中:
  - N: 批量Batch大小
  - C1: `C1 = CeilDiv(C, C0)`，其中`C`为输入特征图的通道数，`C0`为16
  - H: 特征图高度
  - W: 特征图宽度
  - C0: `C0`为16
 - 昇腾亲和的卷积核尺寸表达是：`(Cin, Kh, Kw, Cout, C0)`，其中:
  - Cin: `C1 = CeilDiv(C, C0)`
  - Kh: 卷积核高度
  - Kw: 卷积核宽度
  - Cout: 输出通道数
  - C0: `C0`为16
- 需满足以下基础约束：
  - 膨胀系数`dilations`和卷积核计算步幅`strides`均不能为零
  - 卷积核的有效感受野大小不能超过输入特征图的大小，需满足：
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`

    其中`hi`，`wi`为输入特征图的高度和宽度，`kh`, `kw`为卷积核的高度和宽度，`dilationH`, `dilationW`为上述两方向上的膨胀系数，`padTop`, `padBottom`, `padLeft`, `padRight`为上、下、左、右填充大小。

- 考虑到空间分配，需满足下述条件（为做区分下述公式中小写的符号为运行时常量，反之是编译期常量）：
  - `L1_STAGES * FmapSize + L1_STAGES * FilterSize <= L1_SIZE`
    其中`L1_STAGES`在开double-buffer的情形下为2，不启用为1，L1_SIZE是512K（AtlasA2/A3），FmapSize和FilterSize的具体计算公式为：
    - `FmapSize = Cin1 * hi * wi * C0 * sizeof(ElementFmap)`, `Cin1`为`FmapL1TileShape`下的Tiling常量（输入通道数），hi和wi由`FmapL1TileShape`下的`Ho`和`Wo`（输出特征尺寸）可反推得到，逆运算为：
      1. `hi = (Ho - 1) * strideH + dilationH * (kh - 1) + 1`
      2. `wi = (Wo - 1) * strideW + dilationW * (kw - 1) + 1`

    - `FilterSize = Cin1 * kh * kw * Cout * C0`，`Cin1`为`FilterL1TileShape`下的Tiling常量（输入通道数）, `Cout`是`FilterL1TileShape`下`Cout`（输出通道数）对齐到`C0`后的值

  - `L0A_STAGES * FmapL0ASize <= L0A_SIZE`, 其中`FmapL1ASize`具体为`Ho * Wo * max(L0K, kh * kw * C0) * sizeof(ElementFmap)`，`L0K`为`L0TileShape`下的Tiling常量, `L0A_SIZE`等于64K（AtlasA2/A3）
  - `L0B_STAGES * FilterL0BSize <= L0B_SIZE`, 其中`FilterL0BSize`具体为`max(L0K, kh * kw * C0) * CoutL0 * sizeof(ElementFilter)`，`L0K`为`L0TileShape`下的Tiling常量, `CoutL0`是`L0N`对齐到`C0`的大小，`L0B_SIZE`也等于64K（AtlasA2/A3）
  - `Ho * Wo * Cout * sizeof(ElementOut) <= L0C_SIZE`, 其中`L0C_SIZE`等于128K（AtlasA2/A3），`Ho`, `Wo`, `Cout`均为`FilterL1TileShape`中的Tiling常量。
  （样例中ElementFmap，ElementFilter和ElementOutput为fp16类型）



## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 33_basic_conv2d
cd ./output/bin
# 可执行文件名 |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# Device ID可选，默认为0
./33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

执行结果如下，表明精度验证通过。

```text
Compare success.
```
