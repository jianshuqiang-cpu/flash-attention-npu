# Basic Conv2d Example Readme

## Code Organization

```text
├── 33_basic_conv2d
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│ └── basic_conv2d.cpp # Main file
```

## Example

- After obtaining the code, build the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 33_basic_conv2d
cd ./output/bin
# Executable file name |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# The device ID is optional. The default value is 0.
./33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```
