# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from catlass_cppgen.op import Gemm, OperationBase
from catlass_cppgen.common import DataType, get_default_accumulator
from catlass_cppgen.catlass import (
    Arch,
    GemmShape,
    GemmCoord,
    Shape,
    MmadBase,
    MmadAtlasA2,
    MmadAtlasA2Async,
    MmadAtlasA2Pingpong,
    MmadAtlasA2PingpongSliceKWithPrologue,
    MmadAtlasA2PingPongWithPrologue,
    MmadAtlasA2Preload,
    MmadAtlasA2PreloadAsync,
    MmadAtlasA2PreloadAsyncWithCallback,
    GemmAtlasA2,
    GemvAtlasA2,
    MmadAtlasA2PingpongBias,
    MmadAtlasA2FullLoadA,
    MmadAtlasA2W8A16,
    MmadAtlasA2DynamicCommon,
    MmadAtlasA2Small,
    MmadPingpong,
    MmadPingpongTlaV2,
    MmadPreloadAsyncWithCallback,
    MmadMultiBatch,
)
from catlass_cppgen.catlass.layout import (
    Layout,
    RowMajor,
    ColumnMajor,
    PaddingRowMajor,
    PaddingColumnMajor,
    VectorLayout,
    nZ,
    zN,
    zZ,
    nN,
)

from catlass_cppgen._version import __version__

__all__ = [
    # op
    "Gemm",
    "OperationBase",
    # common
    "DataType",
    "get_default_accumulator",
    # catlass
    "Arch",
    "GemmShape",
    "GemmCoord",
    "Shape",
    "MmadBase",
    "MmadAtlasA2",
    "MmadAtlasA2Async",
    "MmadAtlasA2Pingpong",
    "MmadAtlasA2PingpongSliceKWithPrologue",
    "MmadAtlasA2PingPongWithPrologue",
    "MmadAtlasA2Preload",
    "MmadAtlasA2PreloadAsync",
    "MmadAtlasA2PreloadAsyncWithCallback",
    "GemmAtlasA2",
    "GemvAtlasA2",
    "MmadAtlasA2PingpongBias",
    "MmadAtlasA2FullLoadA",
    "MmadAtlasA2W8A16",
    "MmadAtlasA2DynamicCommon",
    "MmadAtlasA2Small",
    "MmadPingpong",
    "MmadPingpongTlaV2",
    "MmadPreloadAsyncWithCallback",
    "MmadMultiBatch",
    # catlass.layout
    "Layout",
    "RowMajor",
    "ColumnMajor",
    "PaddingRowMajor",
    "PaddingColumnMajor",
    "VectorLayout",
    "nZ",
    "zN",
    "zZ",
    "nN",
    # version
    "__version__",
]
