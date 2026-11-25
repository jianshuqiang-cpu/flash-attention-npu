# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from typing import Any, Dict, List, Tuple
from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase
from catlass_cppgen.catlass.gemm_coord import GemmCoord, GemmShape
from catlass_cppgen.catlass.layout.layout import Layout
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.common.typing import GM_ADDR
from catlass_cppgen.catlass.gemm.dispatch_policy import (
    MmadPingpong,
)


class MultiCoreSplitkMatmulKernel(GemmKernelBase):
    _KERNEL_NAME_BASE = "MultiCoreSplitkMatmulTla"
    _FEATURES = {
        "is_support_evg": False,
        "is_support_relu": False,
        "slice_axis": "K",
        "is_mix": True,
    }

    _INCLUDES = [
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
        "catlass/gemm/kernel/multi_core_splitk_matmul_tla.hpp",
    ]
    _KERNEL_NAME = "{arch_name}_{kernel_name}_{dispatch_policy_name}_{swizzle_name}_{l1_tile_shape_str}_{l0_tile_shape_str}"
    _PARAMS_DEVICE = [
        (GemmCoord, "problemShape"),
        (GM_ADDR, "deviceA"),
        (Layout, "layoutA"),
        (GM_ADDR, "deviceB"),
        (Layout, "layoutB"),
        (GM_ADDR, "deviceC"),
        (Layout, "layoutC"),
        (DataType.AUTO, "aicCoreNum"),
        (GM_ADDR, "deviceBias"),
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
    using ElementBias = {element_Bias};
    using ElementBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
    using LayoutTagA = {layout_A};
    using LayoutTagB = {layout_B};
    using LayoutTagC = layout::RowMajor;

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>;
    using GemmKernel = Gemm::Kernel::MultiCoreSplitkMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;
"""
    _INPUT_TEMPLATE = """\
    uint32_t m = M;
    uint32_t k = K;
    uint32_t n = N;
"""
    _LAYOUT_TEMPLATE = """\
    GemmCoord problemShape{{m, n, k}};
    // Define the layout of each matrix
    LayoutTagA tagA{{m, k}};
    LayoutTagB tagB{{k, n}};
    LayoutTagC tagC{{m, n}};
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);
"""

    def get_default_tile_shape(self) -> Tuple[GemmShape, GemmShape]:
        element_max_size = max(  # noqa: F841
            self.element_A.data_size(),
            self.element_B.data_size(),
            self.element_C.data_size(),
        )
        # 根据是否传入bias返回不同的shape
        if self.element_Bias is not None and self.element_Bias != "void":
            return GemmShape(240, 256, 128), GemmShape(240, 256, 32)
        else:
            return GemmShape(256, 256, 128), GemmShape(256, 256, 32)

    def get_default_dispatch_policy_list(self) -> List:
        """获取 MultiCoreSplitkMatmulKernel 的默认 dispatch_policy 列表.

        :return: 包含默认 dispatch_policy 的列表，列表的第一个元素 [0] 是默认策略.
        :rtype: List
        """
        return [MmadPingpong(arch_tag=self.arch_tag, enable_unit_flag=True)]

    def get_render_params(self, use_constexpr: bool = True) -> Dict[str, Any]:
        """获取渲染参数，包括 kernel 名称格式化参数."""
        params = super().get_render_params(use_constexpr)
        return self._add_kernel_name_params(params, self._KERNEL_NAME_BASE)
