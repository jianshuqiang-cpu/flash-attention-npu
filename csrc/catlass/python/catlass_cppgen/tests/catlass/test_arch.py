# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import unittest
from catlass_cppgen.catlass.arch.arch import Arch


class TestArch(unittest.TestCase):
    def test_arch_atlas_a2(self):
        arch = Arch.AtlasA2
        self.assertEqual(arch.value, "Arch::AtlasA2")

    def test_arch_ascend950(self):
        arch = Arch.Ascend950
        self.assertEqual(arch.value, "Arch::Ascend950")

    def test_arch_enum_values(self):
        """测试所有 Arch 枚举值"""
        expected_values = {
            Arch.AtlasA2: "Arch::AtlasA2",
            Arch.Ascend950: "Arch::Ascend950",
        }
        for arch, expected_value in expected_values.items():
            self.assertEqual(arch.value, expected_value)


if __name__ == "__main__":
    unittest.main()
