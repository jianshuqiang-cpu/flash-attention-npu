# FlashAttentionInfer Example Readme

## 代码组织

```text
├── 70_ascend950_flash_attention_chunk_prefill
│   ├── CMakeLists.txt # CMake编译文件
│   ├── fai_kernel.cpp
│   ├── fai_tiling.cpp
│   ├── fai_tilingdata.h
│   ├── fai.cpp
│   ├── gen_data.py
│   ├──kernel_common.hpp
│   └── README.md
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#算子编译)

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例

```text
batch=1
qSeqlen=567
kvSeqlen=1000
numblocks=8
numHeads=8
kvHeads=1
qkHeadSize=128
vHeadSize=128
isVariedLen=0
dtype="half"
device=0
innerPrec=0 # 仅gen_data.py使用，对应kernel逻辑固定为0
blocksize=128
cacheLayout="nd" # "nd"/"nz"

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh -DCATLASS_ARCH=3510 70_ascend950_flash_attention_chunk_prefill --clean
}

function gen_data() {
    rm -rf examples/70_ascend950_flash_attention_chunk_prefill/data
    python3 examples/70_ascend950_flash_attention_chunk_prefill/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $qkHeadSize $vHeadSize $isVariedLen "$dtype" \
        $numblocks $innerPrec $blocksize "$cacheLayout"
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' qkHeadSize=' $qkHeadSize  ' vHeadSize=' $vHeadSize
    cd output/bin/
    ./70_ascend950_flash_attention_chunk_prefill $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $qkHeadSize $vHeadSize $isVariedLen \
        $numblocks $blocksize --dtype $dtype --cache_layout $cacheLayout --device $device
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```

## 已支持特性

|            特性             |          对应参数          |
| :-------------------------: | :------------------------: |
|          数据类型           |    dtype="half"/"bf16"     |
|      不同batch序列可变      |      isVariedLen=0/1       |
|          blocksize          | blocksize=128/256/512/1024 |
|         qk_head_dim         |   qkHeadSize=64/128/192    |
|         v_head_dim          |      vHeadSize=64/128      |
| kvlayout支持PAGE_ND/PAGE_NZ |   cacheLayout="nd"/"nz"    |
