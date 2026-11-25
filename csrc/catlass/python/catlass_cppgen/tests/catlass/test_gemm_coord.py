# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
from catlass_cppgen.catlass.gemm_coord import GemmShape, GemmCoord, Shape
import random


class TestGemmCoord(unittest.TestCase):
    def test_gemm_shape_str(self):
        m, n, k = [random.randint(0, 512)] * 3
        gemm_shape = GemmShape(m, n, k)
        self.assertEqual(str(gemm_shape), f"GemmShape<{m}, {n}, {k}>")

    def test_gemm_shape_tla(self):
        m, n, k = [random.randint(0, 512)] * 3
        gemm_shape = GemmShape(m, n, k)
        shape = gemm_shape.tla()
        self.assertIsInstance(shape, Shape)
        self.assertEqual(shape.m, m)
        self.assertEqual(shape.n, n)
        self.assertEqual(shape.k, k)

    def test_shape_str(self):
        m, n, k = [random.randint(0, 512)] * 3
        shape = Shape(m, n, k)
        expected = f"Shape<Int<{m}>, Int<{n}>, Int<{k}>>"
        self.assertEqual(str(shape), expected)

    def test_gemm_coord_creation(self):
        m, n, k = [random.randint(0, 512)] * 3
        coord = GemmCoord(m, n, k)
        self.assertEqual(coord.m, m)
        self.assertEqual(coord.n, n)
        self.assertEqual(coord.k, k)


if __name__ == "__main__":
    unittest.main()
