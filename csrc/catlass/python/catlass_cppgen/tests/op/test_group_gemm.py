# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
import re
from catlass_cppgen.op.group_gemm import GroupGemm

from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor, ColumnMajor, VectorLayout
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.kernel.group_gemm.grouped_matmul_slice_m import (
    GroupedMatmulSliceMKernel,
)
from catlass_cppgen.common.op_tensor import OpTensor

from assertion_helper import TestAssertions


class TestGroupGemm(unittest.TestCase):
    def setUp(self):
        self.check = TestAssertions(self)

    def test_group_gemm(self):
        problemCount = 4
        M, K, N = 128, 256, 384
        a = OpTensor.from_shape_stride(
            shape=(M, K), stride=(K, 1), dtype=DataType.FLOAT
        )
        # B 是 3D tensor: [problemCount, k, n] = [4, 256, 384]
        b = OpTensor.from_shape_stride(
            shape=(problemCount, K, N), stride=(K * N, N, 1), dtype=DataType.FLOAT
        )
        # 创建 groupList OpTensor（一维，int64_t 类型）
        groupList = OpTensor(
            dtype=DataType.INT64,
            layout=VectorLayout(problemCount),  # 4 个 group
            shape=(problemCount,),
        )

        group_gemm_plan = GroupGemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=a,
            B=b,
            groupList=groupList,
        )
        kernels = group_gemm_plan.get_kernels()
        self.assertEqual(len(kernels), 1)
        self.assertIsInstance(kernels[0], GroupedMatmulSliceMKernel)

        group_gemm_kernel = kernels[0]
        group_gemm_kernel.tune(GemmShape(256, 256, 128), GemmShape(256, 256, 64))

        kernel_str = group_gemm_kernel.gen_kernel_template()
        input_str = group_gemm_kernel.gen_input_template()
        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 128))
        self.check.test_tileshape(kernel_str, GemmShape(256, 256, 64), pos="L0")
        self.check.test_element_dtype(kernel_str, DataType.FLOAT)
        self.check.test_kernel(kernel_str, "GroupedMatmulSliceMTla")
        self._test_problem_count(input_str, problemCount)

        self.check.test_dispatch_policy(kernel_str, "MmadPingpong")
        self.check.test_arch_tag(kernel_str, Arch.Ascend950)

    def test_group_gemm_codegen_float16(self):
        groupList = OpTensor.from_shape_stride(
            shape=(2,), stride=(1,), dtype=DataType.INT64
        )
        A = OpTensor.from_shape_stride(
            shape=(64, 512), stride=(512, 1), dtype=DataType.FLOAT16
        )
        B = OpTensor.from_shape_stride(
            shape=(2, 512, 256), stride=(131072, 256, 1), dtype=DataType.FLOAT16
        )

        gemm_plan = GroupGemm(
            atlas_arch=Arch.AtlasA2,
            element=DataType.FLOAT16,
            layout=RowMajor,
            core_num=8,
            A=A,
            B=B,
            groupList=groupList,
        )
        kernels = gemm_plan.get_kernels()
        gemm_kernel = kernels[0]
        gemm_kernel.tune(GemmShape(256, 256, 128), GemmShape(256, 256, 64))

        kernel_str = gemm_kernel.gen_kernel_template()
        self.check.test_element_dtype(kernel_str, DataType.FLOAT16)

    def test_group_gemm_codegen_column_major(self):
        groupList = OpTensor.from_shape_stride(
            shape=(2,), stride=(1,), dtype=DataType.INT64
        )
        A = OpTensor.from_shape_stride(
            shape=(64, 512), stride=(1, 64), dtype=DataType.FLOAT
        )
        B = OpTensor.from_shape_stride(
            shape=(2, 512, 256), stride=(131072, 1, 512), dtype=DataType.FLOAT
        )

        gemm_plan = GroupGemm(
            atlas_arch=Arch.AtlasA2,
            element=DataType.FLOAT,
            layout=ColumnMajor,
            core_num=8,
            A=A,
            B=B,
            groupList=groupList,
        )
        kernels = gemm_plan.get_kernels()
        self.assertEqual(len(kernels), 1)
        self.assertIsInstance(kernels[0], GroupedMatmulSliceMKernel)

    def test_group_gemm_missing_grouplist(self):
        A = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        B = OpTensor.from_shape_stride(
            shape=(256, 512), stride=(512, 1), dtype=DataType.FLOAT
        )

        with self.assertRaises(ValueError):
            GroupGemm(
                atlas_arch=Arch.AtlasA2,
                element=DataType.FLOAT,
                layout=RowMajor,
                core_num=8,
                A=A,
                B=B,
            )

    def test_group_gemm_grouplist_shape_mismatch(self):
        groupList = OpTensor.from_shape_stride(
            shape=(3,), stride=(1,), dtype=DataType.INT64
        )
        A = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        B = OpTensor.from_shape_stride(
            shape=(2, 256, 512), stride=(131072, 512, 1), dtype=DataType.FLOAT
        )

        with self.assertRaises(ValueError):
            GroupGemm(
                atlas_arch=Arch.AtlasA2,
                element=DataType.FLOAT,
                layout=RowMajor,
                core_num=8,
                A=A,
                B=B,
                groupList=groupList,
            )

    def test_group_gemm_a_not_2d(self):
        groupList = OpTensor.from_shape_stride(
            shape=(2,), stride=(1,), dtype=DataType.INT64
        )
        A = OpTensor.from_shape_stride(
            shape=(2, 128, 256), stride=(32768, 256, 1), dtype=DataType.FLOAT
        )
        B = OpTensor.from_shape_stride(
            shape=(2, 256, 512), stride=(131072, 512, 1), dtype=DataType.FLOAT
        )

        with self.assertRaises(ValueError):
            GroupGemm(
                atlas_arch=Arch.AtlasA2,
                element=DataType.FLOAT,
                layout=RowMajor,
                core_num=8,
                A=A,
                B=B,
                groupList=groupList,
            )

    def test_group_gemm_can_implement(self):
        groupList = OpTensor.from_shape_stride(
            shape=(2,), stride=(1,), dtype=DataType.INT64
        )
        A = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        B = OpTensor.from_shape_stride(
            shape=(2, 256, 512), stride=(131072, 512, 1), dtype=DataType.FLOAT
        )

        gemm_plan = GroupGemm(
            atlas_arch=Arch.AtlasA2,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=A,
            B=B,
            groupList=groupList,
        )
        self.assertTrue(gemm_plan.can_implement())

        gemm_plan2 = GroupGemm(
            atlas_arch=Arch.AtlasA2,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=A,
            B=B,
            groupList=groupList,
            alpha=2.0,
            beta=0.5,
        )
        self.assertFalse(gemm_plan2.can_implement())

    def test_group_gemm_includes(self):
        groupList = OpTensor.from_shape_stride(
            shape=(2,), stride=(1,), dtype=DataType.INT64
        )
        A = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        B = OpTensor.from_shape_stride(
            shape=(2, 256, 512), stride=(131072, 512, 1), dtype=DataType.FLOAT
        )

        gemm_plan = GroupGemm(
            atlas_arch=Arch.AtlasA2,
            element=DataType.FLOAT,
            layout=RowMajor,
            core_num=8,
            A=A,
            B=B,
            groupList=groupList,
        )
        kernels = gemm_plan.get_kernels()
        includes = kernels[0].gen_includes()
        self.assertIn("catlass/catlass.hpp", includes)
        self.assertIn("catlass/gemm/kernel/grouped_matmul_slice_m_tla.hpp", includes)

    def _test_problem_count(self, template_str: str, expected_count: int):
        match = re.search(r"uint32_t\s+problemCount\s*=\s*(\d+);", template_str)
        self.assertIsNotNone(match, "problemCount not found in template")
        self.assertEqual(int(match.group(1)), expected_count)


if __name__ == "__main__":
    unittest.main()
