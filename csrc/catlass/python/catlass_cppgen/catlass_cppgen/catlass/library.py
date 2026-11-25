# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import enum

from enum import auto as enum_auto


class LayoutType(enum.Enum):
    ColumnMajor = enum_auto()
    RowMajor = enum_auto()
    VectorLayout = enum_auto()
    nZ = enum_auto()
    zN = enum_auto()
    zZ = enum_auto()
    nN = enum_auto()


#
LayoutTag = {
    LayoutType.ColumnMajor: "Catlass::layout::ColumnMajor",
    LayoutType.RowMajor: "Catlass::layout::RowMajor",
    LayoutType.VectorLayout: "Catlass::layout::VectorLayout",
    LayoutType.nZ: "Catlass::layout::nZ",
    LayoutType.zN: "Catlass::layout::zN",
    LayoutType.zZ: "Catlass::layout::zZ",
    LayoutType.nN: "Catlass::layout::nN",
}


class BroadcastType(enum.Enum):
    RowBroadcast = enum_auto()
    ColBroadcast = enum_auto()
    RowColBroadcast = enum_auto()
    NoBroadcast = enum_auto()


BroadcastTag = {
    BroadcastType.RowBroadcast: "RowBroadcast",
    BroadcastType.ColBroadcast: "ColBroadcast",
    BroadcastType.RowColBroadcast: "RowAndColBroadcast",
    BroadcastType.NoBroadcast: "NoBroadcast",
}


class EpilogueOp(enum.Enum):
    # unary op
    Cast = enum_auto()
    Exp = enum_auto()
    Reciprocal = enum_auto()
    Sqrt = enum_auto()

    # binary op
    Add = enum_auto()
    Adds = enum_auto()  # scalar ver
    Div = enum_auto()
    Max = enum_auto()
    Min = enum_auto()
    Mul = enum_auto()
    Muls = enum_auto()  # scalar ver
    Sub = enum_auto()

    # activation op
    LeakyRelu = enum_auto()
    Prelu = enum_auto()
    Relu = enum_auto()
    Rsqrt = enum_auto()
    Sigmoid = enum_auto()
    Silu = enum_auto()

    # Binary-tensor op
    Maxs = enum_auto()
    Mins = enum_auto()
    AddRelu = enum_auto()


EpilogueOpTag = {
    # unary op
    EpilogueOp.Cast: "Catlass::Epilogue::Fusion::Cast",
    EpilogueOp.Exp: "Catlass::Epilogue::Fusion::Exp",
    EpilogueOp.Sqrt: "Catlass::Epilogue::Fusion::Sqrt",
    EpilogueOp.Rsqrt: "Catlass::Epilogue::Fusion::Rsqrt",
    EpilogueOp.Reciprocal: "Catlass::Epilogue::Fusion::Reciprocal",
    # binary op
    EpilogueOp.Add: "Catlass::Epilogue::Fusion::Add",
    EpilogueOp.Adds: "Catlass::Epilogue::Fusion::Adds",
    EpilogueOp.Div: "Catlass::Epilogue::Fusion::Div",
    EpilogueOp.Max: "Catlass::Epilogue::Fusion::Max",
    EpilogueOp.Min: "Catlass::Epilogue::Fusion::Min",
    EpilogueOp.Mul: "Catlass::Epilogue::Fusion::Mul",
    EpilogueOp.Muls: "Catlass::Epilogue::Fusion::Muls",
    EpilogueOp.Sub: "Catlass::Epilogue::Fusion::Sub",
    # activation op
    EpilogueOp.LeakyRelu: "Catlass::Epilogue::Fusion::LeakyRelu",
    EpilogueOp.Prelu: "Catlass::Epilogue::Fusion::Prelu",
    EpilogueOp.Relu: "Catlass::Epilogue::Fusion::Relu",
    EpilogueOp.Sigmoid: "Catlass::Epilogue::Fusion::Sigmoid",
    EpilogueOp.Silu: "Catlass::Epilogue::Fusion::Silu",
    # tensor
    EpilogueOp.Maxs: "Catlass::Epilogue::Fusion::Maxs",
    EpilogueOp.Mins: "Catlass::Epilogue::Fusion::Mins",
    EpilogueOp.AddRelu: "Catlass::Epilogue::Fusion::AddRelu",
}


EpilogueOpVectorToScalar = {
    EpilogueOp.Add: EpilogueOp.Adds,
    EpilogueOp.Mul: EpilogueOp.Muls,
}


EpilogueScalarOp = {
    EpilogueOp.Adds,
    EpilogueOp.Muls,
}


class CastType(enum.Enum):
    NONE = enum_auto()  # When there is precision loss in conversion, it means RINT mode; when there is no precision loss, it means no rounding
    RINT = enum_auto()  # round to nearest even (bankers' rounding)
    FLOOR = enum_auto()  # round towards negative infinity
    CEIL = enum_auto()  # round towards positive infinity
    ROUND = enum_auto()  # round half away from zero
    TRUNC = enum_auto()  # round half away from zero
    ODD = enum_auto()  # Von Neumann rounding, round to nearest odd


CastTypeTag = {
    CastType.NONE: "AscendC::RoundMode::CAST_NONE",
    CastType.RINT: "AscendC::RoundMode::CAST_RINT",
    CastType.FLOOR: "AscendC::RoundMode::CAST_FLOOR",
    CastType.CEIL: "AscendC::RoundMode::CAST_CEIL",
    CastType.ROUND: "AscendC::RoundMode::CAST_ROUND",
    CastType.TRUNC: "AscendC::RoundMode::CAST_TRUNC",
    CastType.ODD: "AscendC::RoundMode::CAST_ODD",
}


class TileDescription:
    def __init__(self, L1TileShape, L0TileShape):
        self.l1_tile_shape = list(L1TileShape)
        self.l0_tile_shape = list(L0TileShape)

    @property
    def l1_m(self):
        return self.l1_tile_shape[0]

    @property
    def l1_n(self):
        return self.l1_tile_shape[1]

    @property
    def l1_k(self):
        return self.l1_tile_shape[2]

    @property
    def l0_m(self):
        return self.l0_tile_shape[0]

    @property
    def l0_n(self):
        return self.l0_tile_shape[1]

    @property
    def l0_k(self):
        return self.l0_tile_shape[2]

    def set_l1_tile(self, new_l1_tile):
        self.l1_tile_shape = list(new_l1_tile)
        self.l0_tile_shape[0] = self.l1_m
        self.l0_tile_shape[1] = self.l1_n
        # the new l1k may be less than l0k
        if self.l0_k > self.l1_k:
            self.l0_tile_shape[2] = self.l1_k

    def procedural_name(self):
        return "l1_{l1m}x{l1n}x{l1k}_l0_{l0m}x{l0n}x{l0k}".format(
            l1m=self.l1_tile_shape[0],
            l1n=self.l1_tile_shape[1],
            l1k=self.l1_tile_shape[2],
            l0m=self.l0_tile_shape[0],
            l0n=self.l0_tile_shape[1],
            l0k=self.l0_tile_shape[2],
        )

    def l1_tile_typename(self, is_tla=False):
        if is_tla:
            tile_fmt = "tla::Shape<Int<{l1m}>, Int<{l1n}>, Int<{l1k}>>"
        else:
            tile_fmt = "GemmShape<{l1m}, {l1n}, {l1k}>"
        return tile_fmt.format(
            l1m=self.l1_tile_shape[0],
            l1n=self.l1_tile_shape[1],
            l1k=self.l1_tile_shape[2],
        )

    def l0_tile_typename(self, is_tla=False):
        if is_tla:
            tile_fmt = "tla::Shape<Int<{l0m}>, Int<{l0n}>, Int<{l0k}>>"
        else:
            tile_fmt = "GemmShape<{l0m}, {l0n}, {l0k}>"
        return tile_fmt.format(
            l0m=self.l0_tile_shape[0],
            l0n=self.l0_tile_shape[1],
            l0k=self.l0_tile_shape[2],
        )
