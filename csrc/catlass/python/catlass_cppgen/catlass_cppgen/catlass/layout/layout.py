# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from abc import ABC, abstractmethod
from math import prod
from typing import Iterable


class Coord:
    def __init__(self, value: Iterable[int]):
        self.idx = tuple(value)


class Layout(ABC):
    value: str = ""

    def __init__(self, shape: Iterable[int], stride: Iterable[int] = None):
        self.shape = tuple(shape)
        self.stride = tuple(stride) if stride else tuple(1 for _ in range(len(shape)))

    @property
    def capacity(self) -> int:
        return prod(self.shape)

    def get_offset(self, coord: Iterable[int]) -> int:
        pass

    @abstractmethod
    def is_need_padding(self, align: int) -> bool:
        pass


class RowMajor(Layout):
    value = "layout::RowMajor"

    def __init__(self, shape: tuple[int, int]):
        super().__init__(shape, (shape[1], 1))

    def is_need_padding(self, align: int) -> bool:
        if self.stride[0] < 65536:
            return self.stride[0] % align != 0
        else:
            return True

    def get_padding_layout(self, align: int) -> Layout:
        if self.is_need_padding(align):
            return PaddingRowMajor(
                self.shape[0],
                self.shape[1],
                (self.shape[1] + align - 1) // align * align,
            )
        else:
            return self


class ColumnMajor(Layout):
    value = "layout::ColumnMajor"

    def __init__(self, shape: tuple[int, int]):
        super().__init__(shape, (1, shape[0]))

    def is_need_padding(self, align: int) -> bool:
        if self.stride[0] < 65536:
            return self.stride[0] % align != 0
        else:
            return True

    def get_padding_layout(self, align: int) -> Layout:
        if self.is_need_padding(align):
            return PaddingColumnMajor(
                self.shape[0],
                self.shape[1],
                (self.shape[0] + align - 1) // align * align,
            )
        else:
            return self


class PaddingRowMajor(Layout):
    value = "layout::PaddingRowMajor"

    def __init__(self, shape: tuple[int, int], block_shape: tuple[int, int]):
        super().__init__(
            (
                block_shape[0],
                (shape[0] + block_shape[0] - 1) // (block_shape[0]),
                block_shape[1],
                (shape[1] + block_shape[1] - 1) // (block_shape[1]),
            ),
            (
                block_shape[1],
                block_shape[0]
                * block_shape[1]
                * (shape[1] + block_shape[1] - 1)
                // (block_shape[1]),
                1,
                block_shape[0] * block_shape[1],
            ),
        )

    def is_need_padding(self, align: int) -> bool:
        return False


class PaddingColumnMajor(Layout):
    value = "layout::PaddingColumnMajor"

    def __init__(self, shape: tuple[int, int], block_shape: tuple[int, int]):
        super().__init__(
            (
                block_shape[0],
                (shape[0] + block_shape[0] - 1) // (block_shape[0]),
                block_shape[1],
                (shape[1] + block_shape[1] - 1) // (block_shape[1]),
            ),
            (
                1,
                block_shape[0] * block_shape[1],
                block_shape[1],
                block_shape[1]
                * block_shape[0]
                * (shape[0] + block_shape[0] - 1)
                // (block_shape[0]),
            ),
        )

    def is_need_padding(self, align: int) -> bool:
        return False


class VectorLayout(Layout):
    value = "layout::VectorLayout"

    def __init__(self, shape: int, stride: int = 1):
        super().__init__((shape,), (stride,))

    def is_need_padding(self, align: int) -> bool:
        """检查向量布局是否需要 padding"""
        return False


class PrivateLayout(Layout):
    def is_need_padding(self, align: int) -> bool:
        return False


class nZ(PrivateLayout):
    value = "layout::nZ"


class zN(PrivateLayout):
    value = "layout::zN"


class zZ(PrivateLayout):
    value = "layout::zZ"


class nN(PrivateLayout):
    value = "layout::nN"
