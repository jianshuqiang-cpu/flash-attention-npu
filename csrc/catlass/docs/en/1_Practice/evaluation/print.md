# Device-side Printing in a CATLASS Sample Project

The compiler provides the device-side printing function `cce::printf` for debugging. You can use it in the same way as using `printf` in the C standard library.

- `cube/vector/mix` operators are supported.
- Formatted strings are supported.
- Common integers, floating-point numbers, pointers, and characters can be printed.

  - ⚠️ **Note** This feature has limited functionality. You are advised to use the [`AscendC debugging API`](ascendc_dump.md) for printing and debugging.

## Example

The following uses `09_splitk_matmul` as an example to describe how to print information on the device.

### Inserting Code for Printing

Add the code for printing to the code segment to debug.

```diff
// include/catlass/gemm/kernel/splitk_matmul.hpp
// ...
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint64_t elementCount, uint32_t splitkFactor)
    {
        // The vec mte processes 256 bytes of data at a time.
        constexpr uint32_t ELE_PER_VECTOR_BLOCK = 256 / sizeof(ElementAccumulator);
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint64_t taskPerAiv =
            (elementCount / aivNum + ELE_PER_VECTOR_BLOCK - 1) / ELE_PER_VECTOR_BLOCK * ELE_PER_VECTOR_BLOCK;
        if (taskPerAiv == 0) taskPerAiv = ELE_PER_VECTOR_BLOCK;
        uint32_t tileLen;
        if (taskPerAiv > COMPUTE_LENGTH) {
            tileLen = COMPUTE_LENGTH;
        } else {
            tileLen = taskPerAiv;
        }
+       cce::printf("tileLen:%d\n", tileLen);
// ...
    }
```

### Build and Execution

1. Following [Quick Start](../01_quick_start.md), enable the tool's build switch `--enable_print` to enable the device-side printing feature.

```bash
bash scripts/build.sh --enable_print 09_splitk_matmul
```

2. Switch to the `output/bin` directory where the executable file is built and execute the operator sample program.

```bash
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID (optional)
./09_splitk_matmul 256 512 1024 0
```

- ⚠ Precautions
  - Currently, `device-side printing` supports only the values on `GM`, `UB`, and `SB (Scalar Buffer)`.

### Output Example

Result

```bash
./09_splitk_matmul 256 512 1024 0
-----------------------------------------------------------------------------
---------------------------------HiIPU Print---------------------------------
-----------------------------------------------------------------------------
==> Logical Block 0
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

==> Logical Block 1
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

... # Omitted here

==> Logical Block 23
=> Physical Block

=> Physical Block
tileLen:2752

=> Physical Block
tileLen:2752

Compare success.
```
