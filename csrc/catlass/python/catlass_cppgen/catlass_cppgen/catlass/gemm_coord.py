# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from dataclasses import dataclass


@dataclass
class Shape:
    m: int
    n: int
    k: int

    def __str__(self):
        return "Shape<Int<{m}>, Int<{n}>, Int<{k}>>".format(
            m=self.m, n=self.n, k=self.k
        )


@dataclass
class GemmShape:
    m: int
    n: int
    k: int

    def __str__(self):
        return "GemmShape<{m}, {n}, {k}>".format(m=self.m, n=self.n, k=self.k)

    def tla(self):
        return Shape(self.m, self.n, self.k)


@dataclass
class GemmCoord:
    m: int
    n: int
    k: int

    def __str__(self):
        return "GemmCoord"

    def tla(self):
        return NotImplementedError("GemmCoord tla is not implemented")
