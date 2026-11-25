# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from typing import Any, Dict, List, Optional, Tuple
from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase
from catlass_cppgen.catlass.gemm_coord import GemmCoord, GemmShape
from catlass_cppgen.catlass.layout.layout import Layout
from catlass_cppgen.common.typing import GM_ADDR
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.common.utils import get_type_name
from catlass_cppgen.catlass.gemm.dispatch_policy import (
    MmadPingpong,
)
from catlass_cppgen.catlass.arch.arch import Arch


class GroupedMatmulSliceMKernel(GemmKernelBase):
    _KERNEL_NAME_BASE = "GroupedMatmulSliceMTla"
    _FEATURES = {
        "is_support_evg": False,
        "is_support_relu": True,
        "slice_axis": "M",
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
        "catlass/gemm/kernel/grouped_matmul_slice_m_tla.hpp",
        "tla/layout.hpp",
    ]
    _KERNEL_NAME = "{arch_name}_{kernel_name}_{dispatch_policy_name}_{swizzle_name}_{l1_tile_shape_str}_{l0_tile_shape_str}"
    _PARAMS_DEVICE = [
        (GemmCoord, "problemShape"),
        (DataType.UINT32, "problemCount"),
        (GM_ADDR, "deviceGroupList"),
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

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, {relu_enable}>;
    using BlockMmadTla = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using GemmKernel = Gemm::Kernel::GroupedMatmulSliceMTla<BlockMmadTla, BlockEpilogue, BlockScheduler, {groupList_element_type}>;
"""
    _INPUT_TEMPLATE = """\
    uint32_t problemCount = {problemCount};
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

    def __init__(
        self,
        problemCount: Optional[int] = None,
        groupList_element: Optional[DataType] = None,
        **kwargs,
    ):
        """初始化 GroupedMatmulSliceMKernel.

        :param problemCount: 问题数量，如果为 None 则使用默认值 1
        :param groupList_element: groupList 的数据类型，如果为 None 则使用默认值 int64_t
        :param kwargs: 传递给父类的其他参数，包括 M, K, N 等
        """
        super().__init__(**kwargs)
        self.problemCount = problemCount if problemCount is not None else 1
        self.groupList_element = (
            groupList_element if groupList_element is not None else DataType.INT64
        )

    def get_default_tile_shape(self) -> Tuple[GemmShape, GemmShape]:
        """获取默认的 tile shape.

        默认fp16，L1, L0 STAGES均为2：
        L1TileShape = Shape<Int<256>, Int<256>, Int<256>>
        L0TileShape = Shape<Int<256>, Int<256>, Int<64>>
        """
        element_max_size = max(self.element_A.data_size(), self.element_B.data_size())
        if self.arch_tag == Arch.AtlasA2:
            l1m, l1n, l1k, l0k = (
                256,
                128,
                512 // element_max_size,
                128 // element_max_size,
            )
        elif self.arch_tag == Arch.Ascend950:
            l1m, l1n, l1k, l0k = (
                256,
                256,
                512 // element_max_size,
                128 // element_max_size,
            )
        else:
            l1m, l1n, l1k, l0k = (
                128,
                128,
                512 // element_max_size,
                128 // element_max_size,
            )
        return GemmShape(l1m, l1n, l1k), GemmShape(l1m, l1n, l0k)

    def get_default_dispatch_policy_list(self) -> List:
        """获取 GroupedMatmulSliceMKernel 的默认 dispatch_policy 列表.

        :return: 包含默认 dispatch_policy 的列表，列表的第一个元素 [0] 是默认策略.
        :rtype: List
        """
        return [MmadPingpong(arch_tag=self.arch_tag, enable_unit_flag=True)]

    def get_render_params(self, use_constexpr: bool = True) -> Dict[str, Any]:
        """获取渲染参数，包括动态生成的 dispatch_policy C++ 代码.

        :param use_constexpr: 当为 True 时，生成包含常量声明的完整代码块；当为 False 时，只生成 using 语句（使用变量名）
        :return: 渲染参数字典.
        :rtype: Dict[str, Any]
        """
        params = super().get_render_params(use_constexpr)
        params["problemCount"] = self.problemCount
        # 将 groupList_element 转换为 C++ 类型字符串
        params["groupList_element_type"] = get_type_name(self.groupList_element)
        # 添加 relu_enable 参数
        params["relu_enable"] = "true" if self.relu_enable else "false"
        return self._add_kernel_name_params(params, self._KERNEL_NAME_BASE)
