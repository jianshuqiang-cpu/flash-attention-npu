# CHANGELOG

## CATLASS 1.X

### CATLASS 1.5.0

- Key Features
  - Added support for the Ascend 950 architecture and its underlying template components.
  - TLA enhancement: introduced origin_shape, added APIs such as TileView, MakeTensorLike, and Tensor operator(), and improved layout and tensor expressions.
  - Matmul generalization project: The W8A8 Per-Token + Per-Channel dynamic quantization path and batch compilation are supported, and related design documents are supplemented.
  - FixPipe capability extension: Added/enhanced combined templates and examples such as Matmul FixPipe optimization and GMM + FixPipe + Dequant
  - Adapted to CANN 9.0.0.beta2. When g++ is linked with the BiSheng toolchain, profapi needs to be explicitly linked (compiler non-compatibility change description).
  - Added unit test and CI adaptation for Ascend 950.
- More Examples
  - [Ascend 950 Basic Matmul](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/43_ascend950_basic_matmul/README.md)
  - [Ascend 950 Matmul FixPipe Optimization](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/46_ascend950_matmul_fixpipe_opti/README.md)
  - [Ascend950 Grouped Matmul SliceM Per-Token Dequant](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/47_ascend950_grouped_matmul_slice_m_per_token_dequant/README.md)
  - [Ascend 950 Grouped Matmul Per-Tensor & Per-Channel Dequant](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant/README.md)
  - [Ascend 950 Flash Attention Inference](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/49_ascend950_flash_attention_infer/README.md)
  - [Ascend 950 Basic Matmul GEMV](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/50_ascend950_basic_matmul_gemv/README.md)
  - [Ascend 950 Quant Matmul Per-Group & Per-Block TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/51_ascend950_quant_matmul_per_group_per_block_tla/README.md) (Per-Group x Per-Block Quantization)
  - [Quant Optimized Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/42_quant_optimized_matmul_tla/README.md)
  - [Quant Matmul Full LoadA TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/44_quant_matmul_full_loadA_tla/README.md)
  - [Strided Batched Matmul TLA](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/45_strided_batched_matmul_tla/README.md)
  - [Matmul Generalization Project: dynamic W8A8 per-token quantization](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/103_dynamic_optimized_quant_matmul_per_token_basic/README.md)
- Tool Support
  - [shared\_lib](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/shared_lib/README.md): Added soname to output artifacts; shared libraries depended on by Python extensions are now switched to `.so` with version information.
  - [Python extension](https://gitcode.com/cann/catlass/tree/v1.5.0/examples/python_extension/README.md): Supports passing build options to `build.sh` and supports asynchronous mode; updated device-side pointer access methods.
  - [msTuner\_CATLASS](https://gitcode.com/cann/catlass/tree/v1.5.0/tools/tuner/README.md): Extended GEMM configuration and search space; [MatmulGelu](https://gitcode.com/cann/catlass/blob/v1.5.0/examples/27_matmul_gelu/README.md): example added for tuning demonstration.
- Documentation
  - Adjusted documentation directory and resource paths (e.g.,`figures` directory), with error fixes and content revisions.
  - Added related documentation for the generalization project, such as MultiCoreSplitK, StreamK, and Single Core Split-K.
  - Fixed incorrect statements in [ascendc\_dump documentation](https://gitcode.com/cann/catlass/blob/v1.5.0/docs/1_Practice/evaluation_tools/ascendc_dump.md)
- Bug Fixes & Optimizations
  - Fixed issues with TLA OriginShape and Flash Attention Golden, etc. Improved implementation paths for Ascend 950 FA's Block/Epilogue.
  - Adjusted blockLen calculation logic in [CopyGmToL1](https://gitcode.com/cann/catlass/blob/v1.5.0/include/catlass/gemm/tile/copy_gm_to_l1.hpp), eliminating unnecessary rounding up for alignment with `C0_NUM_PER_FRACTAL`.
  - NaN Special: Added zero-clear for Ki=0 scenarios in `exp11` and related paths to prevent dirty data from affecting model accuracy.
  - Fixed issues like CopyL0CToDstQuantMode. Corrected the Ascend 950 architecture identifier from 3501 to 3510.
  - Fixed indirect header file references, License comments, and multiple documentation typos. Continuously eliminated code specification warnings and style issues.

### CATLASS 1.4.0

- Key Features
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md) added:
    - `LocalPaddingCPaddingCommonMatmul` template, which uses local workspace to pad matrix C.
- More Examples
  - [StreamK Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/37_streamk_matmul/README.md)
  - [W4A4 Low-Precision Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/38_w4a4_matmul_per_token_per_channel_dequant/README.md)
  - [Matmul Operator L2 Tiling + Staggered Core Distribution](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/39_big_matmul_tla/README.md)
  - [Sparse Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/41_sparse_matmul_tla/README.md)
- Tool Support
  - Added the [shared\_lib](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/shared_lib/basic_matmul_shared_lib.cpp) usage example.
- Documentation
  - Added [detailed design document for the single-core split-K optimized matmul operator](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/contents/example_design/34_single_splitk_matmul.md), introducing the design concept and code breakdown for single-core split-K matrix multiplication.
  - Added [performance data display for Matmul/GroupedMatmul operator on the homepage](https://gitcode.com/cann/catlass/blob/v1.4.0/README.md).
  - [msdebug document](https://gitcode.com/cann/catlass/blob/v1.4.0/docs/tools/msdebug.md) added driver support check.
- Bug Fixes & Optimizations
  - Optimized the [grouped\_matmul\_slice\_m](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/02_grouped_matmul_slice_m/README.md) example to support segmented `groupList` input.
  - Added 32Byte alignment constraint for `TileShape` in BlockMmad.
  - Fixed the data length issue in the [w4a8 matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/32_w4a8_matmul/w4a8.cpp) example.
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.4.0/examples/102_dynamic_optimized_matmul/README.md) now supports scenarios where stride is greater than shape.
  - [msTuner\_CATLASS](https://gitcode.com/cann/catlass/tree/v1.4.0/tools/tuner/README.md) now supports tiling tuning for `quant Matmul`.
  - Fix the improper use of `cmath` on the device.
  - Fixed accuracy issues and naming errors in [Matmul Silu](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/28_matmul_silu/README.md).
  - Fixed vector synchronization issues in [cast\_int4\_to\_int8](https://gitcode.com/cann/catlass/blob/v1.4.0/include/catlass/gemm/tile/cast_int4_to_int8.hpp) component.
  - Fixed half data type usage issues in the [w8a16 Matmul](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/30_w8a16_matmul/w8a16_matmul.cpp) operator.
  - Fixed logic issues in the [Single-Core Split-K Matmul Operator](https://gitcode.com/cann/catlass/blob/v1.4.0/examples/34_single_core_splitk_matmul/single_core_splitk.cpp) when `RemovePaddingNDAndCastC` is empty.

### CATLASS 1.3.0

- Key Features
  - Lowered the minimum required `CMake` version from 3.22 to 3.16.
  - Added support for [`FixPipe` inline quantization](https://gitcode.com/cann/catlass/tree/v1.3.0/include/catlass/gemm/tile/tile_copy.hpp#L373).
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md) added:
    - `PaddingCommonMatmul`
    - `SmallMatmul`
    - `PaddingMultiCoreSplitkMatmul`
    - `PaddingStreamkMatmul`
    - `Single-Core Split-K Template Series`
    - `Dynamic Swizzle`
- More Examples
  - [Matmul operator for INT4 dequantization](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/32_w4a8_matmul/README.md)
  - [2D convolution operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/33_basic_conv2d/README.md)
  - [Single-core split-K optimized matmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/34_single_core_splitk_matmul/README.md)
- Tool Support
  - Added the [msOpGen](https://www.hiascend.com/document/detail/en/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0018.html) tool code example [basic_matmul](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/basic_matmul_aclnn.cpp) and [access document](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/advanced/basic_matmul_aclnn/README.md).
  - [msTuner_CATLASS](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md) added:
    - `GroupedMatmulSliceM operator`
    - `OptimizedMatmul operator`
- Documentation
  - Added [detailed design document for the INT8 type dequantization GroupedMatmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/example_design/10_grouped_matmul_slice_m_per_token_dequant.md), introducing the design concept and code breakdown for groupMatmul + epilogue type operators.
  - Added [matrix multiplication template summary document](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/matmul_template_summary.md),summarizing existing Matmul template designs in the template library.
  - Added [CommonMatmul description document](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/doc/CommonMatmul.md), introducing the basic templates in the generalized Matmul project.
- Bug Fixes & Optimizations
  - Fixed memory issues introduced when copying masks during softmax in the [Flash Attention inference operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/23_flash_attention_infer/README.md).
  - Fixed documentation errors.
    - [catlass_optimize_guidance.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/catlass_optimize_guidance.md)
    - [api.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/contents/advanced/api.md)
    - [Template Library Quick Start.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/quickstart.md)
    - [tutorials.md](https://gitcode.com/cann/catlass/tree/v1.3.0/docs/tutorials.md)
  - [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/102_dynamic_optimized_matmul/README.md) updates:
    - Modified `TilingParams` reading method for better readability.
    - Optimized the original `Splitk ReduceAdd` to fully utilize the UB space.
    - Added Python environment detection in `CMakeLists.txt`.
  - Fixed the issue where the [optimized Matmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/06_optimized_matmul/README.md) did not support PADDING_NZ in the kernel.
  - Optimized and reconstructed the [FP8 dequantized Matmul operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/29_a2_fp8_e4m3_matmul/README.md) to better comply with the Prologue paradigm.
  - Fixed the accuracy issues of the [Matmul bias operator](https://gitcode.com/cann/catlass/tree/v1.3.0/examples/20_matmul_bias/README.md) and added verification and interception for bf16.
  - Optimized simulation compilation logic and compilation issues in the A3 environment; the compilation logic for `simulator` mode is now the same as for the board-on mode.
  - [msTuner_CATLASS tool](https://gitcode.com/cann/catlass/tree/v1.3.0/tools/tuner/README.md) updates:
    - Added security checks for interface replacement, illegal characters, and maximum `groupCount` value.
    - Fixed the issue where the default `ffts_addr` parameter was blocked when dispatching some operators.
  - Changed the default number of consecutive times the cross-core flag can be set to avoid system freezes after exceeding the limit.

### CATLASS 1.2.0

- Key Features
  - Support for passing the compute platform architecture during operator compilation.
  - Added [Matmul generalization project](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul/README.md) example:
    - Automatically determines Tiling parameters based on feature sizes.
    - You can select the best template from a set of preset operator templates.

- More Examples
  - [Flash Attention Inference Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/23_flash_attention_infer/README.md)
  - [3D Convolution Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/24_conv_bias/README.md)
  - [Matrix A Full Load Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/25_matmul_full_loadA/README.md)
  - [Small Matrix Optimization Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/31_small_matmul/README.md)
  - [MatmulRelu Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/26_matmul_relu/README.md)
  - [MatmulGelu Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/27_matmul_gelu/README.md)
  - [MatmulSilu Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/28_matmul_silu/README.md)
  - [FP8 Dequantized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/29_a2_fp8_e4m3_matmul/README.md)
  - [INT8 Dequantized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/30_w8a16_matmul/README.md)

- Tool Support
  - Updated the content of [Python API Calling](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/python_extension/README.md):
    - Adjusted the project organization structure.
    - Added support for transposed cases.
  - Added [`msTuner_CATLASS`](https://gitcode.com/cann/catlass/tree/v1.2.0/tools/tuner/README.md) tool for automatic tiling tuning, which runs exhaustively within the search space to collect performance data.
  - Added support for enabling the [`msSanitizer`](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0039.html) address sanitizer tool (add --enable_mssanitizer to compile options).

- Documentation
  - Added [`catlass_optimize_guidance.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/contents/advanced/catlass_optimize_guidance.md) document, introducing common optimization methods for `Gemm` operators empowered by CATLASS.

- Bug Fixes & Optimizations
  - Optimized the [`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md) operator implementation to support any padding combinations.
  - Fixed the issue where the `msTuner_CATLASS` tool could not obtain the actual running `DeviceId` when the `ASCEND_RT_VISIBLE_DEVICES` environment variable was enabled.
  - Fixed the `Set/Wait` mismatch exception in the [PFA operator example](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/19_mla/README.md)for single-row data scenarios.
  - Fixed `Workspace` size calculation errors in the [`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul/README.md) operator example.
  - Isolated the code segments that enable the `AscendC::Dump` and `AscendC::print` functions.
  - Fixed the output zero-clearing behavior of the [`GroupedMatmulSliceK`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/05_grouped_matmul_slice_k/README.md) operator for the special case Ki=0, and adjusted the true value comparison logic to full (M, N, K) dimensions.
  - Fixed errors in documentation such as [`performance_tools.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tools/performance_tools.md) and [`tutorials.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tutorials.md).

### CATLASS 1.1.0

- Key Features
  - Provides engineering components for calling CATLASS operators from Python.
    - Can compile [pybind](https://github.com/pybind/pybind11) extensions and [PyTorch](https://pytorch.org/) extensions.
  - Supports operator simulation (by enabling the `--simulator` option during compilation).
  - Compilation process adapted to the [Bisheng](https://www.hiascend.com/cann/bisheng) compiler.

- More Examples
  - [MatmulBias Operator with Bias](https://gitcode.com/cann/catlass/blob/v1.1.0/examples/20_matmul_bias/README.md)
  - [Preload-Optimized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/21_basic_matmul_preload_zN/README.md) (co-developed with iFLYTEK)
  - [Split-K-Optimized Matmul Operator](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/22_padding_splitk_matmul/README.md) (co-developed with iFLYTEK)

- Tool Support
  - Added support for using [`AscendC::Dump`](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html) and [`AscendC::printf`](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0075.html) for print debugging.
    - Added `--enable_ascendc_dump` and `--enable_print` to the compilation options to enable the preceding functions.
    - See [ascendc_dump](https://gitcode.com/cann/catlass/tree/v1.1.0/docs/tools/ascendc_dump.md) and [print](https://gitcode.com/cann/catlass/tree/v1.1.0/docs/tools/print.md).

- Documentation
  - Added [Tutorials quick start examples](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tutorials.md).
  - Added documentation for operator performance tuning using the [msProf tool](https://www.hiascend.com/document/detail/en/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0082.html): [msProf](https://gitee.com/ascend/catlass/tree/v1.1.0/docs/tools/performance_tools.md) performance tuning.

- Bug Fixes & Optimizations
  - Optimized the AIC program at the `Kernel` level, adding `PIPE_ALL` to avoid impact on the whole network.
  - Optimized the implementation of the [optimized Matmul](https://gitcode.com/cann/catlass/tree/v1.1.0/examples/06_optimized_matmul/README.md) operator to disable the AIV core in unnecessary padding scenarios.
  - Fixed the error that occurs when `nextBlock` is preloaded at the `Block` layer.
  - Isolated the `inline` definition of `AscendC` on the kernel side to prevent some standard libraries from being unavailable during heterogeneous programming.
  - Modified the redefinition issue of the `l2offset` setting.

### CATLASS 1.0.0

- Official open-source release of the [CATLASS](https://gitcode.com/cann/catlass/) template library

- Key Features
  - Provides hierarchical operator development capabilities including Kernel, Block, Tile, and Basic layers.

- Sample Reference
  - Provides operator examples including basic Matmul and various different optimization strategies.
