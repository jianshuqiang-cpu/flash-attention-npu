# FlashAttentionInfer Example Readme
## 代码组织
```
├── 200_ascend950_flash_attention_chunk_prefill
│   ├── CMakeLists.txt # CMake编译文件
│   ├── fai_kernel.cpp
│   ├── fai_tiling.cpp
│   ├── fai_tilingdata.h
│   ├── fai.cpp
│   ├── gen_data.py
│   ├── kernel_common.hpp
│   └── README.md
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。   
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例
```
batch=4
qSeqlen=512
kvSeqlen=1024
numblocks=4096
numHeads=4
kvHeads=2
qkHeadSize=128
vHeadSize=64
isVariedLen=0
maskType=0
dtype="half"
device=0
kvCacheType=1
enableUnitFlag=0
layout=1
lseFlag=0
innerPrec=0
blocksize=512
sink_flag=0
preToken=0
nextToken=${qSeqlen}
sparseMode=3
cacheLayout="nd"
isBnNBsD=0
globalWindowSize=4
localWindowSize=1024

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh -DCATLASS_ARCH=3510 200_ascend950_flash_attention_chunk_prefill --clean
}

function gen_data() {
    rm -rf examples/200_ascend950_flash_attention_chunk_prefill/data
    python3 examples/200_ascend950_flash_attention_chunk_prefill/gen_data_fa.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $qkHeadSize $vHeadSize $isVariedLen $maskType "$dtype" $kvCacheType \
        $layout $numblocks $innerPrec $lseFlag $blocksize $sink_flag  $preToken $nextToken "$cacheLayout" $isBnNBsD $globalWindowSize $localWindowSize
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' qkHeadSize=' $qkHeadSize  ' vHeadSize=' $vHeadSize  ' innerPrec=' $innerPrec
    cd output/bin/
    ./200_ascend950_flash_attention_chunk_prefill $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $qkHeadSize $vHeadSize $isVariedLen \
            $maskType $kvCacheType $isBnNBsD $layout $numblocks $innerPrec $lseFlag $blocksize $sink_flag \
            $preToken $nextToken $sparseMode $globalWindowSize $localWindowSize --device $device --dtype $dtype --cache_layout $cacheLayout 
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。
```
Compare success.
```
参数说明如下：

|        参数         |                             说明                             |                 支持范围                  |
| :-----------------: | :----------------------------------------------------------: | :--------------------------------------: |
|       `batch`       |                           batch数                            |                   >=1                    |
|      `qSeqlen`      |               每个batch的q输入的最大序列长度                 |                   >=1                    |
|     `kvSeqlen`      |               每个batch的kv输入的最大序列长度                |                   >=1                    |
|     `numblocks`     |                PA场景下用来表示kv的block块数                 |                   >=1                    |
|     `numHeads`      |              q输入的头数，需为kv头数的整数倍                 |                   >=1                    |
|      `kvHeads`      |                         kv输入的头数                         |                   >=1                    |
|    `qkHeadSize`     |                        qk输入的headdim                        |               64 / 128 / 192             |
|     `vHeadSize`     |                        v输入的headdim                        |                 64 / 128                 |
|    `isVariedLen`    |             生成可变长序列，0表示定长，1表示变长             |                  0 / 1                   |
|     `maskType`      |      稀疏模式，0表示nomask，1表示casual mask，4表示swamask       |               0 / 1 / 4                 |
|       `dtype`       |                     qkv输入数据类型                      |            "half" / "bf16"             |
|      `device`       |                         npu设备id                          |                   >=0                    |
|    `kvCacheType`    |            kv输入是否使能PA，0表示normal，1表示paged            |                  0 / 1                   |
|  `enableUnitFlag`   |                        使能单元标志                        |                  0 / 1                   |
|      `layout`       | qkv输入layout，0表示BSND，1表示TND，使能BSND时isVariedLen只能为0 |                  0 / 1                   |
|      `lseFlag`      |                         LSE标志                          |                  0 / 1                   |
|     `innerPrec`     |  softmax阶段计算精度，0表示fp32精度，1表示fp16精度  |                  0 / 1                   |
|     `blocksize`     |              page场景block块的大小               |       128 / 256 / 512 / 1024        |
|     `sink_flag`     |                        sink标志                         |                  0 / 1                   |
|     `preToken`      |                         preToken                          |                   >=0                    |
|     `nextToken`     |                         nextToken                         |                   >=0                    |
|    `sparseMode`     |                         稀疏模式                         |                   0 / 3                   |
|    `cacheLayout`    | kv输入的layout格式，PAGE场景支持"nd"/"nz"，TND场景支持"tnd_nd"/"tnd_nz" | "nd" / "nz" / "tnd_nd" / "tnd_nz" |
|     `isBnNBsD`      | page场景kvshape，0表示[blocknum, blocksize, headnum, headdim]，1表示[blocknum, headnum, blocksize, headdim] |                  0 / 1                   |
| `globalWindowSize`  |                    swa固定窗口大小                     |                   4                   |
| `localWindowSize`   |                    swa可见窗口大小                     |             1024 / 2048             |


## 已支持特性

|            特性             |          对应参数          |
| :-------------------------: | :------------------------: |
|          数据类型           |    dtype="half"/"bf16"     |
|      不同batch序列可变      |      isVariedLen=0/1       |
|          blocksize          | blocksize=128/256/512/1024 |
|         qk_head_dim         |   qkHeadSize=64/128/192    |
|         v_head_dim          |      vHeadSize=64/128      |
|      kvlayout支持PAGE       |   kvCacheType=1    |
|      layout支持TND        | layout=1 | 
|      kv输入支持nd/nz格式        | cacheLayout=“nd”/"nz" | 
|      支持casual mask        | maskType=1 | 
|      支持SWA        | maskType=4 (仅支持PA，BnNBsD场景) | 