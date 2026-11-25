# CATLASS Sample Design Document

This document summarizes the design ideas and code breakdown of certain samples. You can refer to the specific content as needed.

- [102_DynamicOptimizedMatmul](../../../../examples/102_dynamic_optimized_matmul/README_en.md) - Dynamically determines the tiling parameters based on shapes and attempts to select the best template for execution to maximize performance.
  - [CommonMatmul](../../../../examples/102_dynamic_optimized_matmul/docs/en/CommonMatmul_en.md)
  - [MultiCoreSplitkMatmul](../../../../examples/102_dynamic_optimized_matmul/docs/en/MultiCoreSplitkMatmul_en.md)
  - [StreamkMatmul](../../../../examples/102_dynamic_optimized_matmul/docs/en/StreamkMatmul_en.md)
- [10_grouped_matmul_slice_m_per_token_dequant](../../../../examples/10_grouped_matmul_slice_m_per_token_dequant/10_grouped_matmul_slice_m_per_token_dequant.md) - Disassembles sample 10 in the template library, including prototype design, sample implementation, example assembly, and kernel implementation. It is instructive for those who want to understand the implementation of operators of the "groupMatmul + epilogue" type.
- [19_mla](../../../../examples/19_mla/mla.md) - Disassembles sample 19 in the template library and explains the implementation of the Atlas A2-aware Flash-MLA operator.
- [34_single_splitk_matmul](../../../../examples/34_single_core_splitk_matmul/34_single_splitk_matmul.md) - Disassembles sample 34 in the template library, which is a single-core split-K matrix multiplication sample, and explains the algorithm implementation and benefit evaluation range.
- [44_quant_matmul_full_loadA_tla](../../../../examples/44_quant_matmul_full_loadA_tla/44_quant_matmul_full_loadA_tla.md) - Disassembles sample 44 in the template library, which is the implementation of the fully-loaded matrix A matmul sample under quantization.
- [49_ascend950_flash_attention_infer](../../../../examples/49_ascend950_flash_attention_infer/flash_attention_infer.md) - Disassembles sample 49 in the template library and explains the implementation of the Ascend 950-aware FlashAttention inference operator.
- [52_quant_multi_core_splitk_matmul_tla](../../../../examples/52_quant_multi_core_splitk_matmul_tla/52_quant_multi_core_splitk_matmul_tla.md) - Disassembles sample 52 in the template library, which is the implementation of the multi-core split-K sample under quantization.
