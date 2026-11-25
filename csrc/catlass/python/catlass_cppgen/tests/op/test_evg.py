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
from catlass_cppgen.catlass.library import BroadcastType, EpilogueOp
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.evg_extension import evg

from assertion_helper import TestEvgAssertions


class TestBaseOp(unittest.TestCase):
    def setUp(self):
        self.check = TestEvgAssertions(self)

    def test_codegen_with_evg_direct(self):
        fn_src = """
def epilogue(accum, bias):
    result = accum + bias
    return result
"""
        example_inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "bias": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }

        from catlass_cppgen.kernel.gemm.basic_matmul_tla_visitor import (
            BasicMatmulTlaVisitorKernel,
        )

        kernel = BasicMatmulTlaVisitorKernel(
            element_accumulator=DataType.FLOAT,
            element_A=DataType.FLOAT,
            element_B=DataType.FLOAT,
            element_C=DataType.FLOAT,
            element_Bias=DataType.FLOAT,
            layout_A=RowMajor((128, 256)),
            layout_B=RowMajor((256, 384)),
            arch_tag=Arch.Ascend950,
            M=128,
            K=256,
            N=384,
            evg={
                "fn_src": fn_src,
                "example_inputs": example_inputs,
            },
        )

        evg_template = kernel.gen_evg_template()
        kernel_template = kernel.gen_kernel_template()

        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "A")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "B")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "C")
        self.check.test_layout(kernel_template, "RowMajor", "TagA")
        self.check.test_layout(kernel_template, "RowMajor", "TagB")
        self.check.test_layout(kernel_template, "RowMajor", "TagC")

        self.check.test_arch_tag(kernel_template, Arch.Ascend950)

        self.check.test_accu_dtype(evg_template, DataType.FLOAT)
        self.check.test_layout(evg_template, "RowMajor", "TagBias")
        self.check.test_boardcast(evg_template, "Bias", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_template, EpilogueOp.Add)
        self.check.test_tree_visitor(
            evg_template, ("Compute0, Accum, Bias", "Result, EVGCompute0")
        )

    def test_codegen_without_evg(self):
        from catlass_cppgen.kernel.gemm.basic_matmul_tla_visitor import (
            BasicMatmulTlaVisitorKernel,
        )

        kernel = BasicMatmulTlaVisitorKernel(
            element_accumulator=DataType.FLOAT,
            element_A=DataType.FLOAT,
            element_B=DataType.FLOAT,
            element_C=DataType.FLOAT,
            element_Bias=DataType.FLOAT,
            layout_A=RowMajor((128, 256)),
            layout_B=RowMajor((256, 384)),
            arch_tag=Arch.Ascend950,
            M=128,
            K=256,
            N=384,
            evg=None,
        )

        kernel_template = kernel.gen_kernel_template()

        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "A")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "B")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "C")
        self.check.test_layout(kernel_template, "RowMajor", "TagA")
        self.check.test_layout(kernel_template, "RowMajor", "TagB")
        self.check.test_layout(kernel_template, "RowMajor", "TagC")
        self.check.test_arch_tag(kernel_template, Arch.Ascend950)

    def test_codegen_with_evg(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        fn_src = """
def epilogue(accum, bias):
    result = accum + bias
    return result
"""
        example_inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "bias": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            evg_config={"fn_src": fn_src, "example_inputs": example_inputs},
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()
        kernel = kernels[0]

        from catlass_cppgen.kernel.gemm.basic_matmul_tla_visitor import (
            BasicMatmulTlaVisitorKernel,
        )

        self.assertIsInstance(kernel, BasicMatmulTlaVisitorKernel)

        evg_template = kernel.gen_evg_template()
        kernel_template = kernel.gen_kernel_template()

        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "A")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "B")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "C")
        self.check.test_layout(kernel_template, "RowMajor", "TagA")
        self.check.test_layout(kernel_template, "RowMajor", "TagB")
        self.check.test_layout(kernel_template, "RowMajor", "TagC")
        self.check.test_arch_tag(kernel_template, Arch.Ascend950)

        self.check.test_accu_dtype(evg_template, DataType.FLOAT)
        self.check.test_layout(evg_template, "RowMajor", "TagBias")
        self.check.test_boardcast(evg_template, "Bias", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_template, EpilogueOp.Add)
        self.check.test_tree_visitor(
            evg_template, ("Compute0, Accum, Bias", "Result, EVGCompute0")
        )

    def test_tune_functionality(self):
        a = OpTensor.from_shape_stride(
            shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
        )
        b = OpTensor.from_shape_stride(
            shape=(256, 384), stride=(384, 1), dtype=DataType.FLOAT
        )

        fn_src = """
def epilogue(accum, bias):
    result = accum + bias
    return result
"""
        example_inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "bias": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }

        gemm_plan = Gemm(
            atlas_arch=Arch.Ascend950,
            element=DataType.FLOAT,
            layout=RowMajor,
            evg_config={"fn_src": fn_src, "example_inputs": example_inputs},
            A=a,
            B=b,
        )
        kernels = gemm_plan.get_kernels()
        kernel = kernels[0]

        default_l1, default_l0 = kernel.get_default_tile_shape()
        self.assertEqual(default_l1, GemmShape(256, 256, 128))
        self.assertEqual(default_l0, GemmShape(256, 256, 32))

        new_l1 = GemmShape(128, 256, 64)
        new_l0 = GemmShape(128, 256, 64)
        kernel.tune(l1_tile_shape=new_l1, l0_tile_shape=new_l0)
        self.assertEqual(kernel.l1_tile_shape, new_l1)
        self.assertEqual(kernel.l0_tile_shape, new_l0)

        evg_template = kernel.gen_evg_template()
        kernel_template = kernel.gen_kernel_template()

        self.check.test_tileshape(kernel_template, GemmShape(128, 256, 64))
        self.check.test_tileshape(kernel_template, GemmShape(128, 256, 64), pos="L0")

        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "A")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "B")
        self.check.test_element_dtype(kernel_template, DataType.FLOAT, "C")
        self.check.test_layout(kernel_template, "RowMajor", "TagA")
        self.check.test_layout(kernel_template, "RowMajor", "TagB")
        self.check.test_layout(kernel_template, "RowMajor", "TagC")
        self.check.test_arch_tag(kernel_template, Arch.Ascend950)

        self.check.test_accu_dtype(evg_template, DataType.FLOAT)
        self.check.test_layout(evg_template, "RowMajor", "TagBias")
        self.check.test_boardcast(evg_template, "Bias", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_template, EpilogueOp.Add)
        self.check.test_tree_visitor(
            evg_template, ("Compute0, Accum, Bias", "Result, EVGCompute0")
        )


######################


class TestEvgOp(unittest.TestCase):
    def setUp(self):
        self.check = TestEvgAssertions(self)

    def _build_evg(
        self, fn_src: str, inputs: str, output_dtype: DataType = DataType.FLOAT
    ):
        callback_name, evg_args, evg_str, _ = evg(fn_src=fn_src, example_inputs=inputs)
        self.assertEqual(callback_name, "EVGResult")  # fixed callback

        return evg_str, evg_args

    def test_evg_add(self):
        fn_src = """
def epilogue(accum, bias):
    result = accum + bias
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "bias": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_layout(evg_str, "RowMajor", "TagBias")
        self.check.test_boardcast(evg_str, "Bias", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_str, EpilogueOp.Add)
        self.check.test_tree_visitor(
            evg_str, ("Compute0, Accum, Bias", "Result, EVGCompute0")
        )

    def test_evg_mul(self):
        fn_src = """
def epilogue(accum, scale):
    result = accum * scale
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "scale": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_boardcast(evg_str, "Scale", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_str, EpilogueOp.Mul)
        self.check.test_tree_visitor(
            evg_str, ("Compute0, Accum, Scale", "Result, EVGCompute0")
        )

    def test_evg_relu(self):
        fn_src = """
def epilogue(accum):
    result = relu(accum)
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_visitor_compute(evg_str, EpilogueOp.Relu)
        self.check.test_tree_visitor(
            evg_str, ("Compute0, Accum", "Result, EVGCompute0")
        )

    def test_evg_leaky_relu(self):
        fn_src = """
def epilogue(accum, alpha):
    result = leakyRelu(accum, alpha)
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "alpha": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_boardcast(evg_str, "Alpha", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_str, EpilogueOp.LeakyRelu)
        self.check.test_tree_visitor(
            evg_str, ("Compute0, Accum, Alpha", "Result, EVGCompute0")
        )

    def test_evg_prelu(self):
        fn_src = """
def epilogue(accum, weight):
    result = Prelu(accum, weight)
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "weight": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_boardcast(evg_str, "Weight", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_str, EpilogueOp.Prelu)
        self.check.test_tree_visitor(
            evg_str, ("Compute0, Accum, Weight", "Result, EVGCompute0")
        )

    def test_evg_sigmoid(self):
        fn_src = """
def epilogue(accum):
    result = sigmoid(accum)
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_visitor_compute(evg_str, EpilogueOp.Sigmoid)
        self.check.test_tree_visitor(
            evg_str, ("Compute0, Accum", "Result, EVGCompute0")
        )

    def test_evg_cast(self):
        fn_src = """
def epilogue(accum):
    result = cast(accum, "half", "float")
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT16
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_visitor_compute(evg_str, EpilogueOp.Cast)
        self.check.test_tree_visitor(
            evg_str, ("Compute0, Accum", "Result, EVGCompute0")
        )

    def test_evg_relu_add(self):
        fn_src = """
def epilogue(accum, bias):
    relu_result = relu(accum)
    result = relu_result + bias
    return result
"""
        inputs = {
            "accum": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "bias": OpTensor.from_shape_stride(
                shape=(1, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
            "result": OpTensor.from_shape_stride(
                shape=(128, 256), stride=(256, 1), dtype=DataType.FLOAT
            ),
        }
        evg_str, _ = self._build_evg(fn_src, inputs)
        self.check.test_accu_dtype(evg_str, DataType.FLOAT)
        self.check.test_boardcast(evg_str, "Bias", BroadcastType.RowBroadcast)
        self.check.test_visitor_compute(evg_str, [EpilogueOp.Relu, EpilogueOp.Add])
        self.check.test_tree_visitor(
            evg_str,
            ("Compute0, Accum", "Compute1, EVGCompute0, Bias", "Result, EVGCompute1"),
        )


######################################


if __name__ == "__main__":
    unittest.main()
