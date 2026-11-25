# MLA Example Readme

## Code Organization

```text
├── 19_mla
│   ├── CMakeLists.txt # CMake build file
│   ├── gen_data.py
│   ├── kernel_common.hpp # Common variables and macros shared across kernel implementations
│   ├── main.cpp
│   ├── mla_kernel.cpp # MLA TP 2/4/8 template
│   ├── mla_kernel_tp1_spec.cpp # MLA TP 1 template
│   └── README.md
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Step 1: Execute `gen_data.py` with the command line to generate test vectors and validation assets.

```bash
# Execute in the ./examples/19_mla directory.
python gen_data.py 1 1 128 16 16 128 half
# Input parameters map to: batchSize, qSeqlen, kvSeqlen, qheadNum, numBlock, and blockSize.
# qSeqlen: Number of tokens to be inferred. Supported range is 1 to 4 (covers standard decode and MTP scenarios).
# kvSeqlen: Total sequence length of the input.
# blockSize: Currently accepts a fixed value of 128.
# The final parameter explicitly defines the data type: "half" or "bf16".
```

After the command is executed, a data directory is generated in the current path, containing the operator input data and the golden data used for accuracy verification.

```text
├── data
│   ├── block_table.bin
│   ├── golden.bin
│   ├── k.bin
│   ├── k_rope.bin
│   ├── kv_seqlen.bin
│   ├── q.bin
│   ├── q_ntokens.bin
│   ├── q_rope.bin
│   └── q_seqlen.bin
```

Step 2: Execute the operator. Note that the input shape of the operator must match the shape of the data generated in the first step.

```bash
# Compile the specified test case from the root of the CATLASS repository
bash scripts/build.sh 19_mla
cd output/bin
./19_mla 1 1 128 16 16 128
# Parameters must align with the data generator execution.
# Full syntax signature: batchSize, qSeqlen, kvSeqlen, qheadNum, numBlock, blockSize [--dtype DTYPE --datapath DATA_PATH --device DEVICE_ID] Options: --dtype defaults to half, --datapath defaults to ../../examples/19_mla/data, --device defaults to 0.
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
