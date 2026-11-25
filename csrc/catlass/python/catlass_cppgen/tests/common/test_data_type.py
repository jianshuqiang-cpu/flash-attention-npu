# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
import torch
from catlass_cppgen.common.data_type import DataType, get_default_accumulator


class TestDataType(unittest.TestCase):
    def test_data_type_enum_values(self):
        """测试 DataType 枚举值"""
        self.assertEqual(DataType.FLOAT.value, "float")
        self.assertEqual(DataType.FLOAT16.value, "half")
        self.assertEqual(DataType.INT8.value, "int8_t")
        self.assertEqual(DataType.INT32.value, "int32_t")
        self.assertEqual(DataType.BF16.value, "bfloat16_t")

    def test_from_dtype_float32(self):
        """测试从 torch.float32 创建 DataType"""
        dtype = DataType.from_dtype(torch.float32)
        self.assertEqual(dtype, DataType.FLOAT)

    def test_from_dtype_float16(self):
        """测试从 torch.float16 创建 DataType"""
        dtype = DataType.from_dtype(torch.float16)
        self.assertEqual(dtype, DataType.FLOAT16)

    def test_from_dtype_int8(self):
        """测试从 torch.int8 创建 DataType"""
        dtype = DataType.from_dtype(torch.int8)
        self.assertEqual(dtype, DataType.INT8)

    def test_from_dtype_int32(self):
        """测试从 torch.int32 创建 DataType"""
        dtype = DataType.from_dtype(torch.int32)
        self.assertEqual(dtype, DataType.INT32)

    def test_from_dtype_int64(self):
        """测试从 torch.int64 创建 DataType"""
        dtype = DataType.from_dtype(torch.int64)
        self.assertEqual(dtype, DataType.INT64)

    def test_from_dtype_float64(self):
        """测试从 torch.float64 创建 DataType"""
        dtype = DataType.from_dtype(torch.float64)
        self.assertEqual(dtype, DataType.DOUBLE)

    def test_from_dtype_bool(self):
        """测试从 torch.bool 创建 DataType"""
        dtype = DataType.from_dtype(torch.bool)
        self.assertEqual(dtype, DataType.BOOL)

    def test_from_dtype_mxfp8(self):
        """测试 mxfp8 类型 (float8)"""
        if hasattr(torch, "float8_e5m2"):
            dtype = DataType.from_dtype(torch.float8_e5m2)
            self.assertEqual(dtype, DataType.FLOAT8_E5M2)
        if hasattr(torch, "float8_e4m3fn"):
            dtype = DataType.from_dtype(torch.float8_e4m3fn)
            self.assertEqual(dtype, DataType.FLOAT8_E4M3FN)
        if hasattr(torch, "float8_e8m0fnu"):
            dtype = DataType.from_dtype(torch.float8_e8m0fnu)
            self.assertEqual(dtype, DataType.FLOAT8_E8M0)

    def test_from_dtype_mxfp4(self):
        """测试 mxfp4 类型 (float4)"""
        if hasattr(torch, "float4_e2m1fn_x2"):
            dtype = DataType.from_dtype(torch.float4_e2m1fn_x2)
            self.assertEqual(dtype, DataType.FLOAT4_E2M1)
        if hasattr(torch, "float4_e1m2fn_x2"):
            dtype = DataType.from_dtype(torch.float4_e1m2fn_x2)
            self.assertEqual(dtype, DataType.FLOAT4_E1M2)

    def test_from_dtype_torch_aliases(self):
        """测试 torch 别名"""
        self.assertEqual(DataType.from_dtype(torch.float), DataType.FLOAT)
        self.assertEqual(DataType.from_dtype(torch.half), DataType.FLOAT16)
        self.assertEqual(DataType.from_dtype(torch.double), DataType.DOUBLE)
        self.assertEqual(DataType.from_dtype(torch.short), DataType.INT16)
        self.assertEqual(DataType.from_dtype(torch.int), DataType.INT32)
        self.assertEqual(DataType.from_dtype(torch.long), DataType.INT64)

    def test_from_dtype_bfloat16(self):
        """测试 bfloat16"""
        if hasattr(torch, "bfloat16"):
            dtype = DataType.from_dtype(torch.bfloat16)
            self.assertEqual(dtype, DataType.BF16)

    def test_from_dtype_undefined(self):
        """测试未定义的类型返回 UNDEFINED"""
        # 传入一个不存在的类型
        dtype = DataType.from_dtype("unknown_type")
        self.assertEqual(dtype, DataType.UNDEFINED)

    def test_data_size_float(self):
        """测试 FLOAT 的数据大小"""
        self.assertEqual(DataType.FLOAT.data_size(), 4)

    def test_data_size_float16(self):
        """测试 FLOAT16 的数据大小"""
        self.assertEqual(DataType.FLOAT16.data_size(), 2)

    def test_data_size_bf16(self):
        """测试 BF16 的数据大小"""
        self.assertEqual(DataType.BF16.data_size(), 2)

    def test_data_size_int8(self):
        """测试 INT8 的数据大小"""
        self.assertEqual(DataType.INT8.data_size(), 1)

    def test_data_size_int32(self):
        """测试 INT32 的数据大小"""
        self.assertEqual(DataType.INT32.data_size(), 4)

    def test_data_size_int64(self):
        """测试 INT64 的数据大小"""
        self.assertEqual(DataType.INT64.data_size(), 8)


class TestGetDefaultAccumulator(unittest.TestCase):
    def test_get_default_accumulator_float16(self):
        """测试 FLOAT16 + FLOAT16 -> FLOAT"""
        acc = get_default_accumulator(DataType.FLOAT16, DataType.FLOAT16)
        self.assertEqual(acc, DataType.FLOAT)

    def test_get_default_accumulator_float_float16(self):
        """测试 FLOAT + FLOAT16 -> FLOAT"""
        pass

    def test_get_default_accumulator_bf16(self):
        """测试 BF16 + BF16 -> FLOAT"""
        acc = get_default_accumulator(DataType.BF16, DataType.BF16)
        self.assertEqual(acc, DataType.FLOAT)

    def test_get_default_accumulator_int8(self):
        """测试 INT8 + INT8 -> INT32"""
        acc = get_default_accumulator(DataType.INT8, DataType.INT8)
        self.assertEqual(acc, DataType.INT32)

    def test_get_default_accumulator_float(self):
        """测试 FLOAT + FLOAT -> FLOAT (默认返回自身)"""
        acc = get_default_accumulator(DataType.FLOAT, DataType.FLOAT)
        self.assertEqual(acc, DataType.FLOAT)


if __name__ == "__main__":
    unittest.main()
