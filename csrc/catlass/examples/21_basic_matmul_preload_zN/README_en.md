# BasicMatmulPreloadZN Example Readme

## Code Organization

```text
├── 21_basic_matmul_preload_zN
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── basic_matmul_preload_zN.cpp # Main file
```

## Function

This operator supports NZ format input for matrix B (zN for non-transposed usage and nZ for transposed usage) based on `00_basic_matmul`, and uses the `DispatchPolicy` of `MmadAtlasA2Preload`.

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 21_basic_matmul_preload_zN
cd /output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./21_basic_matmul_preload_zN 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
