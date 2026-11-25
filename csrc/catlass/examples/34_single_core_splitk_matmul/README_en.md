# SingleSplitK_Matmul Example Readme

## Description

- Operator function: Optimized matrix multiplication computation. For details about the optimization strategy, see the [Single-Core Split-K Policy Description](./34_single_splitk_matmul.md).

## Parameters

The command-line arguments for this sample include $M$, $N$, $K$, and `deviceId`, adhering to the identical structural criteria used in [00_basic_matmul Parameters](../00_basic_matmul/README.md#parameters).
The underlying operator prototype is structured as follows:

| Name | Class     | Data Type        | Dimensions | Format | Description                           |
| ---- | --------- | ---------------- | ---------- | ------ | ------------------------------------- |
| matA | inTensor  | fp16\|bf16\|fp32 | [m, k]     | ND\|NZ | Left matrix; supports transposition.  |
| matB | inTensor  | fp16\|bf16\|fp32 | [k, n]     | ND\|NZ | Right matrix; supports transposition. |
| matC | outTensor | fp16\|bf16\|fp32 | [m, n]     | ND     | Output matrix.                        |

## Constraints

None.

## Code Organization

The sample code is organized as follows:

```text
├── 34_single_splitk_matmul
│   ├── CMakeLists.txt           # CMake build file
│   ├── single_core_splitk.cpp   # Main file
│   └── README.md
```

## Example

Compile the sample code to generate the corresponding operator executable:

```bash
# Compile a specified test case.
bash scripts/build.sh 34_single_core_splitk_matmul
```

Go to the compilation directory `output/bin` of the executable file and run the operator sample program. Similar to the [00_basic_matmul](../00_basic_matmul/README.md) implementation, verification workloads are randomly initialized based on the dimensions provided in the command line:

```bash
cd output/bin
# Executable name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./34_single_core_splitk_matmul 256 512 1024 0
```

• 256: matrix m-axis

• 512: n-axis

• 1024: k-axis

• 0: Device ID (optional). Defaults to 0.

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
