# CATLASS Matmul Fixpipe Opti Example Introduction

## Prototype Design

| Name | Type      | Data Type | Dimension | Format | Description   |
| ---- | --------- | --------- | --------- | ------ | ------------- |
| matA | inTensor  | half      | [m, k]    | ND     | Left matrix   |
| matB | inTensor  | half      | [n, k]    | ND     | Right matrix  |
| matC | outTensor | float     | [m, n]    | ND     | Output matrix |

## Sample Implementation

The CATLASS [`46_ascend950_matmul_fixpipe_opti` example](./README.md) operator is an Ascend-friendly Matmul operator implemented based on the CATLASS Gemm API. It is optimized for Fixpipe misaligned transfer scenarios. The key operator components include the following parts:

- Example: [matmul_fixpipe_opti.cpp](./matmul_fixpipe_opti.cpp)
- Kernel implementation:
  - Main kernel file: [matmul_mix_fixpipe_opti.hpp](../../include/catlass/gemm/kernel/matmul_mix_fixpipe_opti.hpp)

- **Block components**, including:
  - General mmad component: [block_mmad_pingpong_tla.hpp](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)
  - Post-processing component optimized for Fixpipe: [block_epilogue_fixpipe.hpp](../../include/catlass/epilogue/block/block_epilogue_fixpipe.hpp)
  - ASWT core distribution strategy: [BlockSchedulerAswt](../../include/catlass/gemm/block/block_scheduler_aswt.hpp)

## Solution Design

Currently, Fixpipe has a performance issue with misaligned writes along the N axis. In Matmul scenarios with small K and large M/N values, this can easily lead to a Fixpipe-bound situation. To address this issue, Fixpipe can leverage new features of the Ascend 950 hardware (Ascend 950 adds a new data path from the L0C Buffer to UB). By enabling dualDstCtrl, the result data in the Cube core's L0C Buffer is split into two paths and written in parallel to the dedicated UBs of two Vector cores (one Cube core corresponds to two Vector cores). Then, the DataCopyPad basic instruction is used to transfer data from UB to Global Memory. The UB of each Vector core independently supports Double Buffering to achieve pipeline overlap between read and write operations. While transferring data to Global Memory, it continuously receives data from the L0C Buffer, effectively improving data throughput efficiency.

<img src="../../docs/assets/images/fixpipe-opti.png" width="50%">

## Performance Benefits

With the same tileShape and scheduler strategy, the performance comparison and benefits of this operator (matmul_fixpipe_opti) versus the traditional operator method (matmul_fixpipe) that uses Fixpipe to directly transfer to Global Memory are shown in the table below.

| [M, N, K]         | matmul_fixpipe | matmul_fixpipe_opti | Acceleration Ratio | Remarks       |
| ----------------- | -------------- | ------------------- | ------------------ | ------------- |
| [567, 488, 399]   | 6.53us         | 6.34us              | 1.03               | MN misaligned |
| [1226, 1557, 399] | 15.06us        | 12.05us             | 1.25               | MN misaligned |
| [2058, 2038, 256] | 20.69us        | 12.64us             | 1.64               | MN misaligned |
| [2048, 2048, 256] | 11.97us        | 12.09us             | 0.99               | MN aligned    |
| [2058, 2048, 256] | 13.81us        | 12.38us             | 1.12               | M misaligned  |
| [2048, 2038, 256] | 19.09us        | 12.35us             | 1.55               | N misaligned  |

The table shows that in M/N misaligned scenarios, especially when the N axis is misaligned, the performance of matmul_fixpipe_opti achieves a positive improvement compared to the basic version matmul_fixpipe. Moreover, the larger the M/N values and the smaller the K value, the more significant the performance improvement.

### Description

- L1TileShape: [256, 256, 128]
- L0TileShape: [256, 256, 64]
- Scheduler policy: [ASWT](../../docs/en/2_Design/01_kernel_design/05_aswt.md)
- Test environment: NPU model is Ascend 950, and CANN package version is 9.0.0.
