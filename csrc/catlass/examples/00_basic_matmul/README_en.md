# BasicMatmul Example Readme

## Description

- Function: performs basic matrix multiplication.
- Formula:

  $$
    \begin{aligned}
    C &= A \times B \\
    C_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j}
    \end{aligned}
  $$

  where $A$ and $B$ are input matrices in the shape of `(m, k)` and `(k, n)`, respectively. $C$ is the output matrix in the shape of `(m, n)`.

## Parameters

The following are the running parameters of this example:

| Parameter  | Description                                                                                                           | Constraints                              |
| ---------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `m`        | Number of rows in the left matrix A in matrix multiplication                                                          | -                                        |
| `n`        | Number of columns in the right matrix B in matrix multiplication                                                      | -                                        |
| `k`        | Number of columns in the left matrix A in matrix multiplication<br>(That is, the number of rows in the right matrix.) | -                                        |
| `deviceId` | ID of the used NPU card (default: 0)                                                                                  | Within the valid range of the device NPU |

The key template parameters involved in BasicMatmul are as follows:

| Parameter  | Description                    | Valid Range                                     |
| ---------- | ------------------------------ | ----------------------------------------------- |
| `ElementA` | Data type of the left matrix   | `float` \| `fp16_t` \| `bfloat16_t` \| `int8_t` |
| `ElementB` | Data type of the right matrix  | `float` \| `fp16_t` \| `bfloat16_t` \| `int8_t` |
| `ElementC` | Data type of the result matrix | `float` \| `fp16_t` \| `bfloat16_t` \| `int8_t` |
| `LayoutA`  | Layout of the left matrix      | `layout::RowMajor` \| `layout::ColumnMajor`     |
| `LayoutB`  | Layout of the right matrix     | `layout::RowMajor` \| `layout::ColumnMajor`     |
| `LayoutC`  | Layout of the result matrix    | `layout::RowMajor`                              |

## Constraints

The types of the left matrix, right matrix, and result matrix must meet the following mapping conditions:

| `ElementA`   | `ElementB`   | `ElementC`                          |
| ------------ | ------------ | ----------------------------------- |
| `float`      | `float`      | `float` \| `fp16_t` \| `bfloat16_t` |
| `fp16_t`     | `fp16_t`     | `float` \| `fp16_t` \| `bfloat16_t` |
| `bfloat16_t` | `bfloat16_t` | `float` \| `fp16_t` \| `bfloat16_t` |
| `int8_t`     | `int8_t`     | `int32_t`                           |

## Code Organization

```text
├── 00_basic_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── basic_matmul.cpp # Main file
```

## Example

1. Compile the sample code and generate the corresponding operator executable file.

    ```bash
    bash scripts/build.sh 00_basic_matmul
    ```

2. Go to the compilation directory `output/bin` of the executable file and run the operator sample program. The test sample data is randomly generated, and the size is specified by the command line input.

    ```bash
    cd output/bin
    ./00_basic_matmul 256 512 1024 0
    ```

    • 256: matrix m-axis

    • 512: n-axis

    • 1024: k-axis

    • 0: Device ID (optional). Defaults to 0.

    If the following result is displayed, the sample is successfully executed.

    ```text
    Compare success.
    ```
