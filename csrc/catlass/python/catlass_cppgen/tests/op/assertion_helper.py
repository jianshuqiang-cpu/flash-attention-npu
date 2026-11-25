#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from collections.abc import Iterable

import re
import unittest
from typing import Union, List

from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.library import (
    BroadcastTag,
    BroadcastType,
    EpilogueOp,
    EpilogueOpTag,
)


class TestAssertions:
    def __init__(self, test_case: unittest.TestCase):
        self.t = test_case

    def test_params(self, params_str: str, params_list: Union[list, tuple]):
        actual_params_list = [p.strip() for p in params_str.split(",")]
        self.t.assertEqual(len(actual_params_list), len(params_list))
        for params, actual_params in zip(params_list, actual_params_list):
            self.t.assertEqual(params, actual_params)

    def test_tileshape(self, kernel_str: str, tiling: GemmShape, pos: str = "L1"):
        self.t.assertIn(pos, ("L1", "L0"), "argument 'pos' can only be 'L1' or 'L0'")

        pattern = (
            r"using\s+L1TileShape\s*=\s*Shape<Int<(\d+)>,\s*Int<(\d+)>,\s*Int<(\d+)>>;"
            if pos == "L1"
            else r"using\s+L0TileShape\s*=\s*Shape<Int<(\d+)>,\s*Int<(\d+)>,\s*Int<(\d+)>>;"
        )
        match_tiling = re.search(pattern, kernel_str)
        self.t.assertIsNotNone(match_tiling)
        self.t.assertEqual(match_tiling.group(1), str(tiling.m))
        self.t.assertEqual(match_tiling.group(2), str(tiling.n))
        self.t.assertEqual(match_tiling.group(3), str(tiling.k))

    def test_element_dtype(self, template_str: str, dtype: DataType, pos: str = "A"):
        match = re.search(rf"using\s+Element{pos}\s*=\s*(\S+);", template_str)
        self.t.assertIsNotNone(match, f"Element{pos} not found for dtype {dtype.value}")
        self.t.assertEqual(match.group(1), dtype.value)

    def test_accu_dtype(self, template_str: str, dtype: DataType):
        match = re.search(
            r"Catlass::Epilogue::Fusion::VisitorAccLoad<(\S+)>;", template_str
        )
        self.t.assertIsNotNone(
            match, f"VistorAccLoad not found for dtype {dtype.value}"
        )
        self.t.assertEqual(match.group(1), dtype.value)

    def test_layout(self, template_str: str, layout_tag: str, pos: str = "TagA"):
        self.t.assertIn(pos, ("TagA", "TagB", "TagC", "TagBias"), "invalid layout pos.")
        match = re.search(
            rf"using\s+Layout{pos}\s*=\s*(?:\w+::)*layout::(\S+);", template_str
        )
        self.t.assertIsNotNone(match, f"Layout{pos} not found for layout {layout_tag}")
        self.t.assertEqual(layout_tag, match.group(1))

    def test_kernel(self, template_str: str, kernel_name: str):
        match = re.search(
            r"using\s+GemmKernel\s*=\s*Gemm::Kernel::(\S+)<", template_str
        )
        self.t.assertIsNotNone(match, f"Kernel {kernel_name} not found in template")
        self.t.assertEqual(match.group(1), kernel_name)

    def test_dispatch_policy(self, template_str: str, dispatch_name: str):
        match = re.search(r"using\s+DispatchPolicy\s*=\s*Gemm::(\S+)<", template_str)
        self.t.assertIsNotNone(
            match, f"DispatchPolicy {dispatch_name} not found in template"
        )
        self.t.assertEqual(match.group(1), dispatch_name)

    def test_arch_tag(self, template_str: str, arch: Arch):
        arch_str = arch.value
        match = re.search(r"using\s+ArchTag\s*=\s*(\S+);", template_str)
        self.t.assertIsNotNone(match, f"ArchTag not found for {arch_str}")
        self.t.assertEqual(match.group(1), arch_str)


class TestEvgAssertions(TestAssertions):
    def __init__(self, test_case: unittest.TestCase):
        super().__init__(test_case)

    def test_boardcast(self, template_str: str, node: str, boardcast: BroadcastType):
        match = re.search(
            rf"using\s+{node}\s*=\s*Catlass::Epilogue::Fusion::Visitor(\S+)<",
            template_str,
        )
        self.t.assertIsNotNone(
            match, f"Boardcast {boardcast} not found for node {node}"
        )
        self.t.assertEqual(match.group(1), BroadcastTag[boardcast])

    def test_visitor_compute(
        self, template_str: str, op: Union[EpilogueOp, List[EpilogueOp]]
    ):
        vistor_pattern = r"Catlass::Epilogue::Fusion::VisitorCompute<(\S+),\s*\S+>;"
        op_list = (
            (EpilogueOpTag[x] for x in op)
            if isinstance(op, Iterable)
            else EpilogueOpTag[op]
        )
        for match in re.finditer(vistor_pattern, template_str):
            self.t.assertIsNotNone(match, "VisitorCompute not found")
            self.t.assertIn(match.group(1), op_list)

    def test_tree_visitor(self, template_str: str, vistor_params: List[str]):
        vistor_pattern = r"Catlass::Epilogue::Fusion::TreeVisitor<(\S+)>;"
        for match, vistor_param in zip(
            re.finditer(vistor_pattern, template_str), vistor_params
        ):
            self.t.assertIsNotNone(match, "TreeVisitor not found")
            self.t.assertEqual(match.group(1), vistor_param)
