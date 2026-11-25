# Mxfp8 FlashAttention Infer Example Readme

> **注意**：本样例位于 `experimental/` 目录下，如需编译运行，请先将样例目录拷贝至 `examples/` 下，并在 `examples/CMakeLists.txt` 中添加样例名称 `ascend950_fp8_mx_flash_attention_infer`。

## 代码组织

```text
experimental
├── attention
│   ├── ascend950_fp8_mx_flash_attention_infer
│   │   ├── CMakeLists.txt           # CMake编译文件
│   │   ├── gen_data.py              # 数据生成脚本
│   │   ├── fai_kernel_utils.h       # Kernel辅助文件
│   │   ├── tiling_data_def.h        # Tiling数据结构定义
│   │   ├── fai.cpp                  # 主程序入口
│   │   ├── fai_kernel.h             # Kernel实现
│   │   ├── fai_tiling.h             # Tiling计算实现
│   │   └── README.md
```

## 功能说明

本样例基于[49_ascend950_flash_attention_infer](../../../examples/49_ascend950_flash_attention_infer/README.md)适配mxfp8输入，支持Q/K/V为`float8_e4m3_t`类型，并需要对应 `float8_e8m0_t`类型缩放因子输入。在非PA场景下，以BNSD为例，Q/K scale的shape为\[B, N, S, D/64, 2\],V scale的shape为\[B, N, S/64, D, 2\]。另外，本样例额外支持P静态量化开关，搭配P scale输入。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../../docs/zh/1_Practice/01_quick_start.md#编译执行)

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例

```text
batch=1          # batch大小
qSeqlen=177      # query序列长度
kvSeqlen=512     # key/value序列长度
numHeads=1       # query head数量
kvHeads=1        # key/value head数量
headSize=128     # embeddingSize
isVariedLen=0    # 是否使用变长序列，当前仅支持0
maskType=1       # mask类型，0表示无mask，1表示使用mask
dtype="half"     # 输出数据类型，支持"half"或"bf16"
cacheMode=1      # 缓存模式，0表示非Paged Attention，1表示Paged Attention
usePscale=1      # 是否增加P矩阵dequant因子输入，1表示增加
device=0

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh ascend950_fp8_mx_flash_attention_infer -DCATLASS_ARCH=3510
}

function gen_data() {
    python3 examples/ascend950_fp8_mx_flash_attention_infer/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType $cacheMode "$dtype" $usePscale
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./ascend950_fp8_mx_flash_attention_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType $cacheMode $usePscale --device $device --dtype $dtype
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 样例约束

- `kvSeqlen`仅支持为64的倍数
- `embeddingSize`仅支持64/128
- `isVariedLen`当前仅支持为0
