# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
import ctypes
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor, ColumnMajor
import random


class TestOpTensor(unittest.TestCase):
    def test_op_tensor_creation(self):
        """测试 OpTensor 的创建"""
        dtype = DataType.FLOAT
        layout = RowMajor((128, 256))
        tensor = OpTensor(dtype, layout)
        self.assertEqual(tensor.dtype, dtype)
        self.assertEqual(tensor.layout, layout)

    def test_op_tensor_with_data_ptr(self):
        """测试带 data_ptr 的 OpTensor 创建"""
        dtype = DataType.FLOAT
        layout = RowMajor((128, 256))
        data_ptr = ctypes.c_void_p(0x12345678)
        tensor = OpTensor(dtype, layout, data_ptr)
        self.assertEqual(tensor.dtype, dtype)
        self.assertEqual(tensor.layout, layout)

    def test_op_tensor_shape_property(self):
        """测试 shape 属性"""
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = RowMajor((m, n))
        tensor = OpTensor(DataType.FLOAT, layout)
        self.assertEqual(tensor.shape, (m, n))

    def test_op_tensor_stride_property(self):
        """测试 stride 属性"""
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = RowMajor((m, n))
        tensor = OpTensor(DataType.FLOAT, layout)
        self.assertEqual(tensor.stride, (n, 1))

    def test_op_tensor_capacity_property(self):
        """测试 capacity 属性"""
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = RowMajor((m, n))
        tensor = OpTensor(DataType.FLOAT, layout)
        self.assertEqual(tensor.capacity, m * n)

    def test_op_tensor_column_major(self):
        """测试使用 ColumnMajor layout"""
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = ColumnMajor((m, n))
        tensor = OpTensor(DataType.FLOAT16, layout)
        self.assertEqual(tensor.shape, (m, n))
        self.assertEqual(tensor.stride, (1, m))
        self.assertEqual(tensor.capacity, m * n)

    def test_op_tensor_different_dtypes(self):
        """测试不同的数据类型"""
        layout = RowMajor((64, 128))
        for dtype in [DataType.FLOAT, DataType.FLOAT16, DataType.INT8, DataType.INT32]:
            tensor = OpTensor(dtype, layout)
            self.assertEqual(tensor.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
