# FlashAttentionInfer Example Readme

## 代码组织

```text
├── 23_flash_attention_infer
│   ├── CMakeLists.txt # CMake构建配置文件
│   ├── gen_data.py
│   ├── kernel_common.hpp
│   ├── main.cpp
│   ├── fai_kernel.cpp
│   ├── fai_tiling.cpp
│   └── README.md
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例

```text
batch=1           # batch大小
qSeqlen=177       # query序列长度
kvSeqlen=512      # key/value序列长度
numHeads=1        # query head数量
kvHeads=1         # key/value head数量
headSize=128      # embeddingSize
isVariedLen=0     # 是否使用变长序列，当前仅支持0
maskType=1        # mask类型，0表示无mask，1表示使用mask
dtype="bf16"      # 数据类型，支持"half"或"bf16"
cacheMode=1       # 缓存模式，0表示非Paged Attention，1表示Paged Attention
device=0          # 设备ID

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh 23_flash_attention_infer
}

function gen_data() {
    python3 examples/23_flash_attention_infer/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType "$dtype" $cacheMode
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./23_flash_attention_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType --device $device --dtype $dtype
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```
