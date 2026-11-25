# Basic Matmul TLA GEMV Example Readme

Note: The community package does not support the 950 capability currently. Stay tuned for future versions.

## Code Organization

```text
├── 50_ascend950_basic_matmul_gemv
│   ├── CMakeLists.txt     # CMake build file
│   ├── README.md
│ └── basic_matmul_tla.cpp # Main file
```

## Example

- After obtaining the code, build the corresponding operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution). This test case is an Ascend 950 operator. During compilation, you need to add -DCATLASS_ARCH=3510.
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 50_ascend950_basic_matmul_gemv -DCATLASS_ARCH=3510
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./50_ascend950_basic_matmul_gemv 1 128 127 0
```

If the following information is displayed, the accuracy comparison is successful.

```text
Compare success.
```

## Instructions

The `DispatchPolicy MmadPingpong` used by `BasicMatmul` by default supports the following template parameters:

| Template Parameter | Default Value | Parameters                                                                                              |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------------- |
| ArchTag            | None          | Specifies the architecture model.                                                                       |
| enableUnitFlag     | false         | Whether to enable Unitflag. This parameter must be set to `false` when the L0C multi-buffer is enabled. |
| useHF32            | false         | Whether to enable HF32. Only the float type is supported.                                               |
| l0CStages          | 1             | Specifies the number of L0C buffers. Setting this parameter to `2` enables the L0C dual-buffer.         |
| enableL1Resident   | false         | Whether to enable L1 resident.                                                                          |
| l1AStages          | 1             | Number of buffers for loading matrix A to L1.                                                           |
| l1BStages          | 1             | Number of buffers for loading matrix B to L1.                                                           |
| l0AStages          | 1             | Number of buffers for loading matrix A to L0.                                                           |
| l0BStages          | 1             | Number of buffers for loading matrix B to L0.                                                           |

Assume that the matrix shape is `M N K`, the tile size on L1 is `m1 n1 k1`, the number of blocks in the M direction is `mTiles = CeilDiv(M, m1)`, the number of blocks in the N direction is `nTiles = CeilDiv(N, n1)`, and the total number of tasks is `taskBlocks = mTiles × nTiles`. In the following two cases, `enableL1Resident` can be enabled:

1. `mTiles = 11`, `nTiles > CoreNum`, and `K < 2 * k1`. In this case, you can also set `l0CStages=2` (`enableUnitFlag`must be disabled). If the space is insufficient and`l0CStages=2`cannot be set, set`n1` to half of the original value.

2. `nTiles = 1`, `mTiles > CoreNum`, and `K < 2 * k1`. In this case, you can also set `l0CStages=2` (`enableUnitFlag` must be disabled). If the space is insufficient and `l0CStages=2` cannot be set, set `m1` to half of the original value.
