# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from typing import TYPE_CHECKING, Any, Dict, Optional

from catlass_cppgen.catlass.layout.layout import Layout
from catlass_cppgen.kernel.kernel_base import KernelBase
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.evg_extension import evg as generate_evg

if TYPE_CHECKING:
    from catlass_cppgen.kernel.gemm.basic_matmul_tla_visitor import (
        BasicMatmulTlaVisitorKernel,
    )


class GemmKernelBase(KernelBase):
    def __init__(
        self,
        element_accumulator: DataType,
        element_A: DataType,
        element_B: DataType,
        element_C: DataType,
        element_Bias: DataType,
        layout_A: Layout,
        layout_B: Layout,
        arch_tag: Arch,
        layout_Bias: Optional[Layout] = None,
        layout_C: Optional[Layout] = None,
        M: Optional[int] = None,
        K: Optional[int] = None,
        N: Optional[int] = None,
        evg: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        self.element_A = element_A
        self.element_B = element_B
        self.element_Bias = element_Bias
        self.element_C = element_C
        self.layout_A = layout_A
        self.layout_B = layout_B
        self.layout_Bias = layout_Bias
        self.layout_C = layout_C
        self.element_accumulator = element_accumulator
        self.arch_tag = arch_tag
        self.M = M
        self.K = K
        self.N = N
        self.evg = evg

        super().__init__(*args, **kwargs)

    def get_render_params(self, use_constexpr: bool = True) -> Dict[str, Any]:
        """获取渲染参数，包括动态生成的 dispatch_policy C++ 代码.

        :param use_constexpr: 当为 True 时，生成包含常量声明的完整代码块；当为 False 时，只生成 using 语句（使用变量名）
        :return: 渲染参数字典.
        :rtype: Dict[str, Any]
        """
        params = {
            "arch_tag": self.arch_tag,
            "l1_tile_shape": self.l1_tile_shape,
            "l0_tile_shape": self.l0_tile_shape,
            "l1_tile_shape_tla": self.l1_tile_shape.tla(),
            "l0_tile_shape_tla": self.l0_tile_shape.tla(),
            "element_A": self.element_A,
            "element_B": self.element_B,
            "element_Bias": self.element_Bias,
            "element_C": self.element_C,
            "layout_A": self.layout_A,
            "layout_B": self.layout_B,
            "layout_Bias": self.layout_Bias,
            "layout_C": self.layout_C,
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "slice_axis": self.slice_axis,
        }
        if params.get("element_Bias") is None:
            params["element_Bias"] = "void"

        if len(self.dispatch_policy) == 0:
            raise ValueError("dispatch_policy cannot be empty")
        dispatch_policy = self.dispatch_policy[0]
        result = dispatch_policy.to_cpp(const_mode=True)

        if self.evg is not None:
            fn_src = self.evg["fn_src"]
            example_inputs = self.evg["example_inputs"]
            callback_name, evg_args, evg_str, arg_renames = generate_evg(
                fn_src=fn_src,
                example_inputs=example_inputs,
            )
            # 将生成的 EVG 信息添加到字典中，同时保留原先内容
            self.evg.update(
                {
                    "callback_name": callback_name,
                    "evg_args": evg_args,
                    "evg_str": evg_str,
                    "arg_renames": arg_renames,
                }
            )
            params["evg_args"] = evg_args
            params["evg_str"] = evg_str
            params["evg_callback_name"] = callback_name  # 添加 callback_name 到渲染参数
            params["epilogue_str"] = f"""Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy,
        ArchTag,
        Int<computeLength>,
        {callback_name},
        ElementC
    >"""
        else:
            params["evg_args"] = ""
            params["evg_str"] = ""
            params["evg_callback_name"] = ""
            params["epilogue_str"] = "void"

        if isinstance(result, tuple):
            const_decls, template_str = result
            params["constexpr_declarations"] = "\n".join(
                [f"    {decl}" for decl in const_decls]
            )
            params["dispatch_policy_template"] = f"{template_str}"
        else:
            params["constexpr_declarations"] = ""
            params["dispatch_policy_template"] = result

        return params

    def _add_kernel_name_params(
        self, params: Dict[str, Any], kernel_name_base: str
    ) -> Dict[str, Any]:
        """添加用于格式化 kernel 名称的参数.

        :param params: 渲染参数字典.
        :param kernel_name_base: kernel 名称基础（如 "BasicMatmulTla", "GroupedMatmulSliceMTla"）.
        :return: 添加了格式化参数的参数字典.
        """
        params["arch_name"] = self.arch_tag.name
        params["kernel_name"] = kernel_name_base

        # dispatch_policy 名称只使用类名
        params["dispatch_policy_name"] = self.dispatch_policy[0].__class__.__name__
        params["swizzle_name"] = "GemmIdentityBlockSwizzle_3_0"
        params["l1_tile_shape_str"] = (
            f"{self.l1_tile_shape.m}_{self.l1_tile_shape.n}_{self.l1_tile_shape.k}"
        )
        params["l0_tile_shape_str"] = (
            f"{self.l0_tile_shape.m}_{self.l0_tile_shape.n}_{self.l0_tile_shape.k}"
        )

        return params

    def to_evg(
        self, evg_config: Dict[str, Any]
    ) -> Optional["BasicMatmulTlaVisitorKernel"]:
        """将支持 EVG 的 kernel 转换为 EVG 版本.

        :param evg_config: EVG 配置，包含 'fn_src' 和 'example_inputs'
        :type evg_config: Dict[str, Any]
        :return: 如果当前 kernel 支持 EVG，返回 BasicMatmulTlaVisitorKernel 实例；否则返回 None
        :rtype: Optional[BasicMatmulTlaVisitorKernel]
        """
        # 检查是否支持 EVG 特性
        if not self._features.get("is_support_evg", False):
            return None

        # 延迟导入以避免循环导入
        from catlass_cppgen.kernel.gemm.basic_matmul_tla_visitor import (
            BasicMatmulTlaVisitorKernel,
        )

        # 创建 BasicMatmulTlaVisitorKernel 实例，传递相同的参数和 EVG 配置
        evg_kernel = BasicMatmulTlaVisitorKernel(
            element_accumulator=self.element_accumulator,
            element_A=self.element_A,
            element_B=self.element_B,
            element_C=self.element_C,
            element_Bias=self.element_Bias,
            layout_A=self.layout_A,
            layout_B=self.layout_B,
            layout_Bias=self.layout_Bias,
            layout_C=self.layout_C,
            arch_tag=self.arch_tag,
            M=self.M,
            K=self.K,
            N=self.N,
            evg=evg_config,
        )

        # 设置 tile shape 和 dispatch_policy
        evg_kernel.tune(
            l1_tile_shape=self.l1_tile_shape,
            l0_tile_shape=self.l0_tile_shape,
            dispatch_policy=self.dispatch_policy,
        )

        return evg_kernel
