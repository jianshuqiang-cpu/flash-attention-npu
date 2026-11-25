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
from catlass_cppgen.common.typing import GM_ADDR
from catlass_cppgen.catlass.gemm.dispatch_policy import (
    MmadPingpong,
)
from catlass_cppgen.catlass.arch.arch import Arch


class BasicMatmulKernel(GemmKernelBase):
    _KERNEL_NAME_BASE = "BasicMatmulTla"
    _FEATURES = {
        "is_support_evg": True,
        "is_support_relu": True,
        "slice_axis": None,
        "is_mix": False,
    }

    def __init__(self, **kwargs):
        """初始化 BasicMatmulKernel.

        relu_enable 现在通过 tune() 方法设置，默认为 False。
        """
        super().__init__(**kwargs)

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
        "catlass/gemm/kernel/basic_matmul_tla.hpp",
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
    using LayoutTagA = {layout_A};
    using LayoutTagB = {layout_B};
    using LayoutTagC = layout::RowMajor;
    using ElementBias = {element_Bias};

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias, {relu_enable}>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using GemmKernel = Gemm::Kernel::BasicMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;
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
        element_max_size = max(
            self.element_A.data_size(),
            self.element_B.data_size(),
            self.element_C.data_size(),
        )
        if self.arch_tag == Arch.AtlasA2:
            l1_m, l1_n, l1_k, l0_k = (
                128,
                256,
                512 // element_max_size,
                128 // element_max_size,
            )
        elif self.arch_tag == Arch.Ascend950:
            l1_m, l1_n, l1_k, l0_k = (
                256,
                256,
                512 // element_max_size,
                128 // element_max_size,
            )
        else:
            l1_m, l1_n, l1_k, l0_k = (
                128,
                128,
                512 // element_max_size,
                128 // element_max_size,
            )
        if self.element_Bias is not None and self.element_Bias != "void":
            l1_m -= 16
        return GemmShape(l1_m, l1_n, l1_k), GemmShape(l1_m, l1_n, l0_k)

    def get_default_dispatch_policy_list(self) -> List[MmadPingpong]:
        """获取 BasicMatmulKernel 的默认 dispatch_policy 列表.

        :return: 包含默认 dispatch_policy 的列表，列表的第一个元素 [0] 是默认策略.
        :rtype: List[MmadPingpong]
        """
        return [MmadPingpong(arch_tag=self.arch_tag, enable_unit_flag=True)]

    def get_render_params(self, use_constexpr: bool = True) -> Dict[str, Any]:
        """获取渲染参数，包括 kernel 名称格式化参数."""
        params = super().get_render_params(use_constexpr)
        params["relu_enable"] = "true" if self.relu_enable else "false"
        return self._add_kernel_name_params(params, self._KERNEL_NAME_BASE)
