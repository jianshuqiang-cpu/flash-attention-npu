# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
from catlass_cppgen.op.gemm import Gemm
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import ColumnMajor, RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import (
    MmadPingpong,
    MmadPingpongTlaV2,
)
from catlass_cppgen.kernel.gemm import (
    BasicMatmulKernel,
    BatchedMatmulKernel,
    StridedBatchedMatmulKernel,
    BasicMatmulTlaVisitorKernel,
    MultiCoreSplitkMatmulKernel,
    StreamkMatmulKernel,
    TailMultiCoreSplitkMatmulKernel,
)

from assertion_helper import TestAssertions


def find_kernel_by_type(kernels, kernel_type):
    """根据类型查找 kernel"""
    for kernel in kernels:
        if isinstance(kernel, kernel_type):
            return kernel
    return None


class TestGemm(unittest.TestCase):
    def setUp(self):
        self.check = TestAssertions(self)

    def test_basic_matmul_kernel(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor, A=a, B=b
        )
        kernels = gemm_plan.get_kernels()

        # 根据类型查找 kernel，而不是使用固定索引
        basic_kernel = find_kernel_by_type(kernels, BasicMatmulKernel)
        if basic_kernel is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        # 基础特征检查
        self.assertEqual(type(basic_kernel), BasicMatmulKernel)
        self.assertFalse(basic_kernel.relu_enable)  # default to False

        basic_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
        )

        params = basic_kernel.gen_params_device(def_mode=False)
        self.check.test_params(
            params,
            (
                "problemShape",
                "deviceA",
                "layoutA",
                "deviceB",
                "layoutB",
                "deviceC",
                "layoutC",
                "deviceBias",
            ),
        )

        kernel_str = basic_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BasicMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_batched_matmul(self):
        a = OpTensor.from_shape_stride(
            shape=(8, 128, 256), stride=(32768, 256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(8, 256, 384), stride=(98304, 384, 1), dtype=DataType.FLOAT
        )
        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950, element_C=DataType.FLOAT, core_num=8, A=a, B=b
        )
        kernels = gemm_plan.get_kernels()

        batched_kernel = find_kernel_by_type(kernels, BatchedMatmulKernel)
        if batched_kernel is None:
            raise ValueError(
                f"No kernel named 'BatchedMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        # 基础特征检查
        self.assertEqual(type(batched_kernel), BatchedMatmulKernel)
        self.assertFalse(batched_kernel.relu_enable)  # default to False

        batched_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(
                arch_tag=Arch.Ascend950, enable_unit_flag=True
            ),
        )

        kernel_str = batched_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BatchedMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_strided_batched_matmul(self):
        a = OpTensor.from_shape_stride(
            shape=(3, 128, 256),
            stride=(256, 3 * 256, 1),
            dtype=DataType.FLOAT16,
        )
        b = OpTensor.from_shape_stride(
            shape=(3, 256, 384),
            stride=(256 * 384, 384, 1),
            dtype=DataType.FLOAT16,
        )
        gemm_plan = Gemm(
            atlas_arch=Arch.AtlasA2,
            element_C=DataType.FLOAT16,
            core_num=8,
            A=a,
            B=b,
            strided_batched=True,
            transA=True,
            transB=False,
        )
        kernel = find_kernel_by_type(
            gemm_plan.get_kernels(),
            StridedBatchedMatmulKernel,
        )
        self.assertIsNotNone(kernel)

        kernel_str = kernel.gen_kernel_template()
        layout_str = kernel.gen_layout_template()

        self.check.test_kernel(kernel_str, "StridedBatchedMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpongTlaV2")
        self.check.test_arch_tag(kernel_str, Arch.AtlasA2)
        self.assertIn("constexpr bool transA = true;", layout_str)
        self.assertIn("constexpr bool transB = false;", layout_str)
        self.assertIn(
            "strideBatchA = transA ? static_cast<int64_t>(k)",
            layout_str,
        )
        self.assertIn(
            "strideMA = transA ? static_cast<int64_t>(batchCount) * k",
            layout_str,
        )
        self.assertNotIn("lda", kernel.gen_params_device())
        self.assertNotIn("strideA", kernel.gen_params_device())
        self.assertIn("_TN_", kernel.gen_kernel_name())

    def test_strided_batched_matmul_with_tune(self):
        a = OpTensor.from_shape_stride(
            shape=(3, 128, 256),
            stride=(256, 1, 384),
            dtype=DataType.FLOAT,
        )
        b = OpTensor.from_shape_stride(
            shape=(3, 256, 384),
            stride=(384, 1152, 1),
            dtype=DataType.FLOAT,
        )
        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element_C=DataType.FLOAT,
            core_num=8,
            A=a,
            B=b,
            strided_batched=True,
            transA=True,        
            transB=False,
            # Note; transA and transB means whether it 
            # is transposed with the batch axis. 
            # If you want to set the Row/ColumnMajor, 
            # you need to construct correct op tensor or 
            # directly set `layout_A` and `layout_B` 
            # with messy `strides` that cannot be auto derived
        )
        kernel = find_kernel_by_type(
            gemm_plan.get_kernels(),
            StridedBatchedMatmulKernel,
        )
        self.assertIsNotNone(kernel)
        self.assertEqual(type(kernel), StridedBatchedMatmulKernel)

        kernel.tune(
            GemmShape(256, 256, 128),
            GemmShape(256, 256, 32),
            dispatch_policy=MmadPingpongTlaV2(
                arch_tag=Arch.Ascend950, enable_unit_flag=True
            ),
        )

        kernel_str = kernel.gen_kernel_template()
        layout_str = kernel.gen_layout_template()

        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 128))
        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 32), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "ColumnMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "StridedBatchedMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpongTlaV2")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)
        self.assertIn("constexpr bool transA = true;", layout_str)
        self.assertIn("constexpr bool transB = false;", layout_str)

        params = kernel.gen_params_device(def_mode=False)
        self.check.test_params(
            params,
            (
                "batchCount",
                "problemShape",
                "deviceA",
                "layoutA",
                "deviceB",
                "layoutB",
                "deviceC",
                "layoutC",
            ),
        )

    def test_basic_matmul_kernel_with_relu(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor, A=a, B=b
        )
        kernels = gemm_plan.get_kernels()

        basic_kernel = find_kernel_by_type(kernels, BasicMatmulKernel)
        if basic_kernel is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(basic_kernel), BasicMatmulKernel)
        self.assertFalse(basic_kernel.relu_enable)

        basic_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
            relu_enable=True,
        )
        self.assertTrue(basic_kernel.relu_enable)

        kernel_str = basic_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BasicMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_basic_matmul_kernel_with_is_hf32(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor, A=a, B=b
        )
        kernels = gemm_plan.get_kernels()

        basic_kernel = find_kernel_by_type(kernels, BasicMatmulKernel)
        if basic_kernel is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(basic_kernel), BasicMatmulKernel)

        basic_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
            is_hf32=True,
        )
        self.assertTrue(basic_kernel.dispatch_policy[0].use_hf32_mode)

        kernel_str = basic_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BasicMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_matmul_with_bias(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )
        bias = OpTensor.from_shape_stride(
            shape=(384,), stride=(1,), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            A=a,
            B=b,
            Bias=bias,
        )
        kernels = gemm_plan.get_kernels()

        basic_kernel = find_kernel_by_type(kernels, BasicMatmulKernel)
        if basic_kernel is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(basic_kernel), BasicMatmulKernel)

        basic_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
        )

        params = basic_kernel.gen_params_device(def_mode=False)
        self.check.test_params(
            params,
            (
                "problemShape",
                "deviceA",
                "layoutA",
                "deviceB",
                "layoutB",
                "deviceC",
                "layoutC",
                "deviceBias",
            ),
        )

        kernel_str = basic_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BasicMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_tile_shape_with_bias(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )
        bias = OpTensor.from_shape_stride(
            shape=(384,), stride=(1,), dtype=DataType.FLOAT
        )

        gemm_plan_without_bias = Gemm(
            atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor, A=a, B=b
        )
        kernels_without_bias = gemm_plan_without_bias.get_kernels()
        basic_kernel_without_bias = find_kernel_by_type(
            kernels_without_bias, BasicMatmulKernel
        )
        if basic_kernel_without_bias is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels_without_bias]}"
            )

        default_shape_no_bias = basic_kernel_without_bias.get_default_tile_shape()
        self.assertEqual(
            default_shape_no_bias, (GemmShape(256, 256, 128), GemmShape(256, 256, 32))
        )

        gemm_plan_with_bias = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            A=a,
            B=b,
            Bias=bias,
        )
        kernels_with_bias = gemm_plan_with_bias.get_kernels()
        basic_kernel_with_bias = find_kernel_by_type(
            kernels_with_bias, BasicMatmulKernel
        )
        if basic_kernel_with_bias is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels_with_bias]}"
            )

        default_shape_with_bias = basic_kernel_with_bias.get_default_tile_shape()
        self.assertEqual(
            default_shape_with_bias, (GemmShape(240, 256, 128), GemmShape(240, 256, 32))
        )

    def test_streamk_matmul(self):
        a = OpTensor.from_shape_stride(
            shape=(1280, 3000), stride=(3000, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(3000, 256), stride=(256, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()

        streamk_kernel = find_kernel_by_type(kernels, StreamkMatmulKernel)
        if streamk_kernel is None:
            raise ValueError(
                f"No kernel named 'StreamkMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(streamk_kernel), StreamkMatmulKernel)
        self.assertEqual(streamk_kernel.slice_axis, "K")
        self.assertFalse(streamk_kernel.relu_enable)

        streamk_kernel.tune(
            GemmShape(256, 256, 128),
            GemmShape(256, 256, 32),
            dispatch_policy=MmadPingpong(
                arch_tag=Arch.Ascend950, enable_unit_flag=True
            ),
        )

        kernel_str = streamk_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 128))
        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 32), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "StreamkMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_multi_core_splitk_matmul_kernel(self):
        a = OpTensor.from_shape_stride(
            shape=(256, 3000), stride=(3000, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(3000, 256), stride=(256, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()

        splitk_kernel = find_kernel_by_type(kernels, MultiCoreSplitkMatmulKernel)
        if splitk_kernel is None:
            raise ValueError(
                f"No kernel named 'MultiCoreSplitkMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(splitk_kernel), MultiCoreSplitkMatmulKernel)
        self.assertEqual(splitk_kernel.slice_axis, "K")
        self.assertFalse(splitk_kernel.relu_enable)

        splitk_kernel.tune(
            GemmShape(256, 256, 128),
            GemmShape(256, 256, 32),
            dispatch_policy=MmadPingpong(
                arch_tag=Arch.Ascend950, enable_unit_flag=True
            ),
        )

        kernel_str = splitk_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 128))
        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 32), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "MultiCoreSplitkMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_tail_multi_core_splitk_matmul_kernel(self):
        a = OpTensor.from_shape_stride(
            shape=(768, 3000), stride=(3000, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(3000, 768), stride=(768, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()

        tail_splitk_kernel = find_kernel_by_type(
            kernels, TailMultiCoreSplitkMatmulKernel
        )
        if tail_splitk_kernel is None:
            raise ValueError(
                f"No kernel named 'TailMultiCoreSplitkMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(tail_splitk_kernel), TailMultiCoreSplitkMatmulKernel)
        self.assertEqual(tail_splitk_kernel.slice_axis, "K")
        self.assertFalse(tail_splitk_kernel.relu_enable)

        tail_splitk_kernel.tune(
            GemmShape(256, 256, 128),
            GemmShape(256, 256, 32),
            dispatch_policy=MmadPingpong(
                arch_tag=Arch.Ascend950, enable_unit_flag=True
            ),
        )

        kernel_str = tail_splitk_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 128))
        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 32), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "TailMultiCoreSplitkMatmulTla")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_basic_matmul_tla_visitor_kernel(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        function_source = """
def epilogue(accum):
    temp = accum
    return temp
"""
        example_inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 384), stride=(384, 1), dtype=DataType.FLOAT
            ),
            "temp": OpTensor.from_shape_stride(
                shape=(128, 384), stride=(384, 1), dtype=DataType.FLOAT
            ),
        }

        evg_config = {
            "fn_src": function_source,
            "example_inputs": example_inputs,
        }

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            evg_config=evg_config,
            element=DataType.FLOAT,
            layout=RowMajor,
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()

        visitor_kernel = find_kernel_by_type(kernels, BasicMatmulTlaVisitorKernel)
        if visitor_kernel is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulTlaVisitorKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(visitor_kernel), BasicMatmulTlaVisitorKernel)
        self.assertEqual(visitor_kernel.slice_axis, None)
        self.assertFalse(visitor_kernel.relu_enable)

        visitor_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
        )

        kernel_str = visitor_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BasicMatmulTlaVisitor")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_basic_matmul_tla_visitor_kernel_with_constant(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        function_source = """
def epilogue(accum):
    scale = constant(0.1, "float")
    result = accum * scale
    return result
"""
        example_inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 384), stride=(384, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 384), stride=(384, 1), dtype=DataType.FLOAT
            ),
        }

        evg_config = {
            "fn_src": function_source,
            "example_inputs": example_inputs,
        }

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            evg_config=evg_config,
            element=DataType.FLOAT,
            layout=RowMajor,
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()

        visitor_kernel = find_kernel_by_type(kernels, BasicMatmulTlaVisitorKernel)
        if visitor_kernel is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulTlaVisitorKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(visitor_kernel), BasicMatmulTlaVisitorKernel)

        visitor_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
        )

        kernel_str = visitor_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BasicMatmulTlaVisitor")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_to_evg_method(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950, element=DataType.FLOAT, layout=RowMajor, A=a, B=b
        )
        kernels = gemm_plan.get_kernels()

        basic_kernel = find_kernel_by_type(kernels, BasicMatmulKernel)
        if basic_kernel is None:
            raise ValueError(
                f"No kernel named 'BasicMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(basic_kernel), BasicMatmulKernel)

        is_support_evg = bool(getattr(basic_kernel, "is_support_evg", False))
        self.assertTrue(is_support_evg)

        function_source = """
def epilogue(accum, bias):
    result = accum + bias
    return result
"""
        example_inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 384), stride=(384, 1), dtype=DataType.FLOAT
            ),
            "bias": OpTensor.from_shape_stride(
                shape=(1, 384), stride=(384, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 384), stride=(384, 1), dtype=DataType.FLOAT
            ),
        }

        evg_config = {
            "fn_src": function_source,
            "example_inputs": example_inputs,
        }

        basic_kernel.tune(
            GemmShape(128, 256, 64),
            GemmShape(128, 256, 64),
            dispatch_policy=MmadPingpong(arch_tag=Arch.Ascend950),
        )

        evg_kernel = basic_kernel.to_evg(evg_config)
        self.assertIsNotNone(evg_kernel)
        self.assertIsInstance(evg_kernel, BasicMatmulTlaVisitorKernel)
        self.assertIsNotNone(evg_kernel.evg)

        kernel_str = evg_kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_str, GemmShape(128, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_layout(kernel_str, "RowMajor")
        self.check.test_layout(kernel_str, "RowMajor", "TagB")
        self.check.test_layout(kernel_str, "RowMajor", "TagC")
        self.check.test_kernel(kernel_str, "BasicMatmulTlaVisitor")
        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_to_evg_unsupported_kernel(self):
        a = OpTensor.from_shape_stride(
            shape=(256, 3000), stride=(3000, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(3000, 256), stride=(256, 1), dtype=DataType.FLOAT
        )

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()

        splitk_kernel = find_kernel_by_type(kernels, MultiCoreSplitkMatmulKernel)
        if splitk_kernel is None:
            raise ValueError(
                f"No kernel named 'MultiCoreSplitkMatmulKernel' found, available kernel list: {[type(k).__name__ for k in kernels]}"
            )

        self.assertEqual(type(splitk_kernel), MultiCoreSplitkMatmulKernel)

        is_support_evg = bool(getattr(splitk_kernel, "is_support_evg", False))
        self.assertFalse(is_support_evg)

        evg_config = {
            "fn_src": "def epilogue(accum): return accum\n",
            "example_inputs": {
                "accum": OpTensor.from_shape_stride(
                    shape=(256, 256), stride=(256, 1), dtype=DataType.FLOAT
                ),
                "result": OpTensor.from_shape_stride(
                    shape=(256, 256), stride=(256, 1), dtype=DataType.FLOAT
                ),
            },
        }

        result = splitk_kernel.to_evg(evg_config)
        self.assertIsNone(result)


######################################


if __name__ == "__main__":
    unittest.main()
