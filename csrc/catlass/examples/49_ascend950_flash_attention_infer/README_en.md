# Flash Attention Infer Example Readme

## Code Organization

```text
├── 49_ascend950_flash_attention_infer
│   ├── CMakeLists.txt           # CMake build file
│   ├── gen_data.py              # Data generation script
│   ├── fai_kernel_utils.h       # Kernel auxiliary file
│   ├── tiling_data_def.h        # Tiling data structure definition
│   ├── fai.cpp                  # Main program entry
│   ├── fai_kernel.h             # Kernel implementation
│   ├── fai_tiling.h             # Tiling computing implementation
│   └── README.md
```

## Examples

- After obtaining the code, build the corresponding operator executable file. For details, see [quickstart](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).

- Run `gen_data.py` to generate a test sample. The test sample needs to be input from the command line. After the command is executed, a data directory is generated in the current path, including the input data of the operator and the golden data used for precision verification.
- Then, execute the operator. Note that the input shape of the operator must be the same as the shape of the data generated in the first step.

The following is a complete shell script example:

```text
batch=1          # Batch size
qSeqlen=177      # Query sequence length
kvSeqlen=512     # Key/Value sequence length
numHeads=1       # Number of query heads
kvHeads=1        # Number of key/value heads
headSize=128     # embeddingSize
isVariedLen=0    # Whether to use variable-length sequences. Currently, only 0 is supported.
maskType=1       # Mask type. 0 indicates no mask, and 1 indicates that a mask is used.
dtype="half"     # Data type. The value can be "half" or "bf16".
cacheMode=1      # Cache mode. 0 indicates non-paged attention, and 1 indicates paged attention.
device=0

function build() {
    rm -rf build
    rm -rf output
    bash scripts/build.sh 49_ascend950_flash_attention_infer -DCATLASS_ARCH=3510
}

function gen_data() {
    python3 examples/49_ascend950_flash_attention_infer/gen_data.py $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType $cacheMode "$dtype"
    echo "Data gen finished"
}

function run_kernel() {
    echo 'Case: B=' $batch ' qS=' $qSeqlen ' kvS=' $kvSeqlen ' qN=' $numHeads ' kvN=' $kvHeads ' D=' $headSize ' mask=' $maskType
    cd output/bin/
    ./49_ascend950_flash_attention_infer $batch $qSeqlen $kvSeqlen $numHeads $kvHeads $headSize $isVariedLen $maskType $cacheMode --device $device --dtype $dtype
}

build
gen_data
run_kernel
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```
