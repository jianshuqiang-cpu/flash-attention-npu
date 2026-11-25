# Using Ascend C Operator Debugging APIs in a CATLASS Sample Project

Ascend C operator debugging APIs are the debugging capabilities of Ascend C. They can be used to print internal kernel information ([printf](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0193.html)) and view tensor content ([DumpTensor](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0192.html)).

## Examples

The following uses `00_basic_matmul` as an example to demonstrate the test process based on [Ascend C operator debugging APIs](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0192.html).

### Inserting Debugging Code

Add the debugging API call at the desired level. For example, add the following code to the kernel function in `include/catlass/gemm/kernel/basic_matmul.hpp`:

```diff
// include/catlass/gemm/kernel/basic_matmul.hpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIC>(Params const &params) {
    BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
+   AscendC::printf("CoreLoops is %d\n", coreLoops);
    Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad(resource);

    // Represent the full gm
    AscendC::GlobalTensor<ElementA> gmA;
    gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
    AscendC::GlobalTensor<ElementB> gmB;
    gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
+   AscendC::DumpTensor(gmA, coreLoops, 16);
    for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        // Compute block location
        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

        // Compute initial location in logical coordinates
        MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
        MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
        MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
        int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
        int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
        int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

        // Compute block-scoped matrix multiply-add
        blockMmad(gmA[gmOffsetA], params.layoutA,
                    gmB[gmOffsetB], params.layoutB,
                    gmC[gmOffsetC], params.layoutC,
                    actualBlockShape);
    }
}
```

### Build and Execution

1. Build the operator sample by referring to [Quick Start](../01_quick_start.md). In the current version, no additional build option is required. If the debugging API is called in the code, the compiler is automatically enabled.

    ```bash
    bash scripts/build.sh 00_basic_matmul
    ```

2. Switch to the `output/bin` directory where the executable file is compiled and run the operator sample program.

    ```bash
    cd output/bin
    # Executable file name | Matrix M-axis | N-axis | K-axis | Device ID (optional)
    ./00_basic_matmul 256 512 1024 0
    ```

- ⚠ Precautions
  - [`DumpTensor`](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0192.html) **does not support printing values on `L0A`, `L0B`, and `FixPipe`. In particular, on `Ascend 950PR/Ascend 950DT`**, values on `L1` cannot be printed.

### Output Example (For Reference Only. The Actual Output May Vary Depending on the Hardware and Operator Implementation.)

```bash
./00_basic_matmul 256 512 1024 0
opType=device_gemm, DumpHead: AIC-0, CoreType=AIC, block dim=24, total_block_num=24, block_remain_len=1048408, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CoreLoops is 4
DumpTensor: desc=4, addr=c0013000, data_type=float16, position=GM
[3.402344, -1.056641, 2.830078, 2.984375, 4.117188, -3.025391, -1.647461, 2.681641, -2.222656, 0.539551, -0.226074, 1.289062, -1.352539, 0.134033, 4.523438, 4.160156]
... # Each cube core outputs information once.
Compare success.
```
