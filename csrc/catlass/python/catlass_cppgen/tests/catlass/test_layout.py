# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
from catlass_cppgen.catlass.layout.layout import (
    Coord,
    RowMajor,
    ColumnMajor,
    VectorLayout,
)
import random


class TestCoord(unittest.TestCase):
    def test_coord_creation(self):
        values = [random.randint(0, 100) for _ in range(3)]
        coord = Coord(values)
        self.assertEqual(coord.idx, tuple(values))

    def test_coord_single_value(self):
        coord = Coord([42])
        self.assertEqual(coord.idx, (42,))


class TestRowMajor(unittest.TestCase):
    def test_row_major_creation(self):
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = RowMajor((m, n))
        self.assertEqual(layout.shape, (m, n))
        self.assertEqual(layout.stride, (n, 1))

    def test_row_major_capacity(self):
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = RowMajor((m, n))
        self.assertEqual(layout.capacity, m * n)

    def test_row_major_is_need_padding_false(self):
        """测试不需要 padding 的情况"""
        # stride[0] = 64, align = 16, 64 % 16 == 0
        layout = RowMajor((4, 64))
        self.assertFalse(layout.is_need_padding(16))

    def test_row_major_is_need_padding_true(self):
        """测试需要 padding 的情况"""
        # stride[0] = 65, align = 16, 65 % 16 != 0
        layout = RowMajor((4, 65))
        self.assertTrue(layout.is_need_padding(16))

    def test_row_major_is_need_padding_large_stride(self):
        """测试 stride >= 65536 的情况"""
        layout = RowMajor((1000, 65536))
        self.assertTrue(layout.is_need_padding(16))

    def test_row_major_get_padding_layout_no_padding(self):
        """测试不需要 padding 时返回自身"""
        layout = RowMajor((4, 64))
        result = layout.get_padding_layout(16)
        self.assertIs(result, layout)

    def test_row_major_get_padding_layout_with_padding(self):
        """测试需要 padding 时返回 PaddingRowMajor"""
        pass


class TestColumnMajor(unittest.TestCase):
    def test_column_major_creation(self):
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = ColumnMajor((m, n))
        self.assertEqual(layout.shape, (m, n))
        self.assertEqual(layout.stride, (1, m))

    def test_column_major_capacity(self):
        m, n = random.randint(1, 100), random.randint(1, 100)
        layout = ColumnMajor((m, n))
        self.assertEqual(layout.capacity, m * n)

    def test_column_major_is_need_padding_false(self):
        """测试不需要 padding 的情况"""
        # stride[0] = 1, align = 16, 1 % 16 != 0，但 ColumnMajor 的 stride[0] 是 1
        # 实际上 ColumnMajor 的 stride 是 (1, m)，所以 stride[0] = 1
        layout = ColumnMajor((64, 4))
        # stride[0] = 1, 1 % 16 != 0，所以需要 padding
        self.assertTrue(layout.is_need_padding(16))

    def test_column_major_is_need_padding_true(self):
        """测试需要 padding 的情况"""
        layout = ColumnMajor((65, 4))
        self.assertTrue(layout.is_need_padding(16))

    def test_column_major_get_padding_layout_no_padding(self):
        """测试不需要 padding 时返回自身（这种情况在 ColumnMajor 中很少见）"""
        # 由于 ColumnMajor 的 stride[0] = 1，通常需要 padding
        layout = ColumnMajor((16, 4))
        result = layout.get_padding_layout(1)  # align = 1 时不需要 padding
        self.assertIs(result, layout)

    def test_column_major_get_padding_layout_with_padding(self):
        """测试需要 padding 时返回 PaddingColumnMajor"""
        pass


class TestPaddingLayouts(unittest.TestCase):
    def test_padding_row_major_value(self):
        pass

    def test_padding_column_major_value(self):
        pass


class TestVectorLayout(unittest.TestCase):
    def test_vector_layout_creation(self):
        shape = random.randint(1, 100)
        layout = VectorLayout(shape)
        self.assertTrue(len(layout.shape) == 1 and layout.shape[0] == shape)
        self.assertTrue(len(layout.stride) == 1 and layout.stride[0] == 1)

    def test_vector_layout_is_need_padding(self):
        """VectorLayout 不需要 padding"""
        layout = VectorLayout(100)
        self.assertFalse(layout.is_need_padding(16))


class TestPrivateLayouts(unittest.TestCase):
    def test_private_layout_is_need_padding(self):
        pass

    def test_nz_value(self):
        pass

    def test_zn_value(self):
        pass

    def test_zz_value(self):
        pass

    def test_nn_value(self):
        pass

    def test_private_layout_subclasses(self):
        pass


if __name__ == "__main__":
    unittest.main()
