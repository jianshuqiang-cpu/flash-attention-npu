# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from typing import Any, Dict, List, Optional, Tuple

from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadPingpongTlaV2
from catlass_cppgen.catlass.gemm_coord import GemmCoord, GemmShape
from catlass_cppgen.catlass.layout.layout import ColumnMajor, Layout, RowMajor
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.common.typing import GM_ADDR
from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase


class StridedBatchedMatmulKernel(GemmKernelBase):
    """Generate the strided-batched TLA kernel used by example 45.

    ``transA`` and ``transB`` move the physical batch axis between the two
    logical matrix axes. They do not transpose the individual matrices.
    """

    slice_axis = None
    _KERNEL_NAME_BASE = "StridedBatchedMatmulTla"
    _FEATURES = {
        "is_support_evg": False,
        "is_support_relu": False,
        "slice_axis": None,
        "is_mix": False,
    }

    _INCLUDES = [
        "type_traits",
        "catlass/catlass.hpp",
        "catlass/arch/arch.hpp",
        "catlass/layout/layout.hpp",
        "catlass/status.hpp",
        "catlass/gemm/block/block_mmad.hpp",
        "catlass/gemm/block/block_swizzle.hpp",
        "catlass/gemm/dispatch_policy.hpp",
        "catlass/gemm/gemm_type.hpp",
        "catlass/gemm/device/device_gemm.hpp",
        "catlass/gemm_coord.hpp",
        "catlass/matrix_coord.hpp",
        "tla/layout.hpp",
        "tla/tensor.hpp",
        "catlass/gemm/kernel/strided_batched_matmul_tla.hpp",
    ]
    _KERNEL_NAME = (
        "{arch_name}_{kernel_name}_{dispatch_policy_name}_{transpose_name}_"
        "{swizzle_name}_{l1_tile_shape_str}_{l0_tile_shape_str}"
    )
    _PARAMS_DEVICE = [
        (DataType.UINT32, "batchCount"),
        (GemmCoord, "problemShape"),
        (GM_ADDR, "deviceA"),
        (Layout, "layoutA"),
        (GM_ADDR, "deviceB"),
        (Layout, "layoutB"),
        (GM_ADDR, "deviceC"),
        (Layout, "layoutC"),
    ]
    _DISPATCH_POLICY = """\
    using ArchTag = {arch_tag};
{constexpr_declarations}
    using DispatchPolicy = {dispatch_policy_template};
"""
    _KERNEL_TEMPLATE = """\
    using L1TileShape = {l1_tile_shape_tla};
    using L0TileShape = {l0_tile_shape_tla};

    using ElementA = {element_A};
    using ElementB = {element_B};
    using ElementC = {element_C};
    using LayoutTagA = {layout_A};
    using LayoutTagB = {layout_B};
    using LayoutTagC = layout::RowMajor;

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<{swizzle_offset}, {swizzle_direction}>;
    using GemmKernel = Gemm::Kernel::StridedBatchedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;
"""
    _INPUT_TEMPLATE = """\
    uint32_t m = M;
    uint32_t k = K;
    uint32_t n = N;
"""
    _LAYOUT_TEMPLATE = """\
    uint32_t batchCount = {batchCount};
    constexpr bool transA = {transA};
    constexpr bool transB = {transB};
    GemmCoord problemShape{{m, n, k}};

    auto layoutA = [&]() {{
        if constexpr (std::is_same_v<LayoutTagA, layout::RowMajor>) {{
            int64_t strideBatchA = transA ? static_cast<int64_t>(k) : static_cast<int64_t>(m) * k;
            int64_t strideMA = transA ? static_cast<int64_t>(batchCount) * k : static_cast<int64_t>(k);
            return tla::MakeLayout(
                tla::MakeShape(batchCount, m, k),
                tla::MakeStride(strideBatchA, strideMA, tla::Int<1>{{}}));
        }} else {{
            int64_t strideBatchA = transA ? static_cast<int64_t>(m) : static_cast<int64_t>(m) * k;
            int64_t strideKA = transA ? static_cast<int64_t>(batchCount) * m : static_cast<int64_t>(m);
            return tla::MakeLayout(
                tla::MakeShape(batchCount, m, k),
                tla::MakeStride(strideBatchA, tla::Int<1>{{}}, strideKA));
        }}
    }}();

    auto layoutB = [&]() {{
        if constexpr (std::is_same_v<LayoutTagB, layout::RowMajor>) {{
            int64_t strideBatchB = transB ? static_cast<int64_t>(n) : static_cast<int64_t>(k) * n;
            int64_t strideKB = transB ? static_cast<int64_t>(batchCount) * n : static_cast<int64_t>(n);
            return tla::MakeLayout(
                tla::MakeShape(batchCount, k, n),
                tla::MakeStride(strideBatchB, strideKB, tla::Int<1>{{}}));
        }} else {{
            int64_t strideBatchB = transB ? static_cast<int64_t>(k) : static_cast<int64_t>(k) * n;
            int64_t strideNB = transB ? static_cast<int64_t>(batchCount) * k : static_cast<int64_t>(k);
            return tla::MakeLayout(
                tla::MakeShape(batchCount, k, n),
                tla::MakeStride(strideBatchB, tla::Int<1>{{}}, strideNB));
        }}
    }}();

    int64_t strideC = static_cast<int64_t>(m) * n;
    auto layoutC = tla::MakeLayout(
        tla::MakeShape(batchCount, m, n),
        tla::MakeStride(strideC, static_cast<int64_t>(n), tla::Int<1>{{}}));
"""

    def __init__(
        self,
        batchCount: Optional[int] = None,
        transA: bool = True,
        transB: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batchCount = batchCount if batchCount is not None else 1
        self.transA = transA
        self.transB = transB
        if self.batchCount <= 0:
            raise ValueError("batchCount must be greater than 0")
        if not isinstance(self.layout_A, (RowMajor, ColumnMajor)):
            raise ValueError(
                "StridedBatchedMatmulKernel only supports RowMajor or ColumnMajor layout_A"
            )
        if not isinstance(self.layout_B, (RowMajor, ColumnMajor)):
            raise ValueError(
                "StridedBatchedMatmulKernel only supports RowMajor or ColumnMajor layout_B"
            )

    def get_default_tile_shape(self) -> Tuple[GemmShape, GemmShape]:
        element_max_size = max(
            self.element_A.data_size(),
            self.element_B.data_size(),
            self.element_C.data_size(),
        )
        tile_m = 128 if self.arch_tag == Arch.AtlasA2 else 256
        tile_n = 256
        tile_k = 512 // element_max_size
        l0_tile_k = 128 // element_max_size
        return (
            GemmShape(tile_m, tile_n, tile_k),
            GemmShape(tile_m, tile_n, l0_tile_k),
        )

    def get_default_dispatch_policy_list(self) -> List:
        return [
            MmadPingpongTlaV2(
                arch_tag=self.arch_tag,
                enable_unit_flag=True,
            )
        ]

    def get_render_params(self, use_constexpr: bool = True) -> Dict[str, Any]:
        params = super().get_render_params(use_constexpr)
        params.update(
            {
                "batchCount": self.batchCount,
                "transA": "true" if self.transA else "false",
                "transB": "true" if self.transB else "false",
                "transpose_name": (
                    f"{'T' if self.transA else 'N'}{'T' if self.transB else 'N'}"
                ),
                "swizzle_offset": 3,
                "swizzle_direction": 0 if self.M > self.N else 1,
            }
        )
        params = self._add_kernel_name_params(params, self._KERNEL_NAME_BASE)
        params["swizzle_name"] = (
            f"GemmIdentityBlockSwizzle_{params['swizzle_offset']}_"
            f"{params['swizzle_direction']}"
        )
        return params
