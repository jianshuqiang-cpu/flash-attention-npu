# BasicConv2dTla Example Readme

## Code Organization

```text
├── 56_ascend950_basic_conv2d_tla
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── basic_conv2d_tla.cpp # Main file
```

## Description

- Function: performs basic convolution computation.

## Remarks

The overall design of this test case is the same as that of [_basic_matmul](../33_basic_conv2d/README.md). The difference is that TLA-related abstraction is used. Therefore, related examples are provided for description.

## Example

- After obtaining the code, compile the corresponding operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#operator-compilation). This test case is an Ascend 950 operator. During compilation, you need to add -DCATLASS_ARCH=3510.
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 56_ascend950_basic_conv2d_tla -DCATLASS_ARCH=3510
cd ./output/bin
# Executable file name |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# The device ID is optional. The default value is 0.
./56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
