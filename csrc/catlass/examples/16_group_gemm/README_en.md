# GroupGemm Example Readme

## Code Organization

```text
├── 16_group_gemm
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── group_gemm.cpp # Main file
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 16_group_gemm
cd output/bin
# Executable file name | Number of matrices | Matrix m-axis group | n-axis group | k-axis group | Device ID
# The device ID is optional. The default value is 0.
./16_group_gemm 3 "128,256,512" "256,512,128" "512,256,128" 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
