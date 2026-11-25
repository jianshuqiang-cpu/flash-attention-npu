# StreamkMatmul Example Readme

## Code Organization

```text
├── 37_streamk_matmul
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── streamk_matmul.cpp # Main file
```

## Template Description

This template is a multi-core split-K template for the tail block. It tiles the K-dimension of the tail block to generate more task blocks, so that more cores can be used to compute the tail block.
During tiling, the computation workload of the tail block is evenly distributed to all compute cores.

```sh
# Compiling a specified case
bash scripts/build.sh 37_streamk_matmul
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./37_streamk_matmul 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```

## Usage Scenarios

Assume that the matrix shape is `M N K`, the tile size on L1 is `m1 n1 k1`, and the number of AI Cores is `C`. The number of basic blocks that can be partitioned is `B = CeilDiv(M, m1) × CeilDiv(N, n1)`, and the number of computation rounds is `B / C`. If `B % C > 0`, a tail round needs to be calculated.
When `B/C > 1` and `B % C ≤ C × 0.8·, this template can be used to obtain better performance.

**To test the performance, you are advised to comment out the accuracy comparison code.**
