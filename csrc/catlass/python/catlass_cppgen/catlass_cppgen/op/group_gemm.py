# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import warnings
from typing import List, Type, Optional, Dict, Any
import math

from catlass_cppgen.op.op import OperationBase
from catlass_cppgen.common.typing import SupportedDataType, SupportedTensor
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import get_default_accumulator
from catlass_cppgen.catlass.layout.layout import Layout, RowMajor
from catlass_cppgen.common.utils import extract_info
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase
from catlass_cppgen.kernel.group_gemm.grouped_matmul_slice_m import (
    GroupedMatmulSliceMKernel,
)

_CORE_NUM = 8  # Default core num


class GroupGemm(OperationBase):
    # 类属性类型注解
    A: Optional[OpTensor]
    B: Optional[OpTensor]
    Bias: Optional[OpTensor]
    C: Optional[OpTensor]
    M: Optional[int]
    K: Optional[int]
    N: Optional[int]
    problemCount: Optional[int]
    groupList: Optional[OpTensor]

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        element_accumulator: Optional[SupportedDataType] = None,
        element: Optional[SupportedDataType] = None,
        layout: Optional[Type[Layout]] = None,
        element_A: Optional[SupportedDataType] = None,
        element_B: Optional[SupportedDataType] = None,
        element_Bias: Optional[SupportedDataType] = None,
        element_C: Optional[SupportedDataType] = None,
        layout_A: Optional[Type[Layout]] = None,
        layout_B: Optional[Type[Layout]] = None,
        layout_Bias: Optional[Type[Layout]] = None,
        epilogue=None,
        evg_config: Optional[Dict[str, Any]] = None,
        atlas_arch: Optional[Arch] = None,
        core_num: Optional[int] = None,
        A: Optional[OpTensor] = None,
        B: Optional[OpTensor] = None,
        Bias: Optional[OpTensor] = None,
        C: Optional[OpTensor] = None,
        groupList: Optional[OpTensor] = None,
    ):
        # 保存 A、B、Bias、C 和 groupList OpTensor
        self.A = A
        self.B = B
        self.Bias = Bias
        self.C = C
        self.groupList = groupList

        # 必须提供 groupList，从它提取 problemCount
        if groupList is None:
            raise ValueError("groupList must be provided")

        groupList_shape, groupList_element, _, _ = extract_info(groupList, None, None)
        if groupList_shape is None:
            raise ValueError("groupList must have valid shape")

        if len(groupList_shape) != 1:
            raise ValueError("groupList must be 1D tensor")
        # 从 groupList 的长度提取 problemCount
        self.problemCount = groupList_shape[0]

        # 如果传入了 A 和 B OpTensor，从它们中提取信息
        if A is not None and B is not None:
            # 从 OpTensor 中提取 shape、element、layout 信息
            A_shape, element_A_from_tensor, layout_A_from_tensor, _ = extract_info(
                A, element_A or element, layout_A or layout
            )
            B_shape, element_B_from_tensor, layout_B_from_tensor, _ = extract_info(
                B, element_B or element, layout_B or layout
            )

            # 确定最终使用的 element 和 layout（传入的参数优先，否则使用 OpTensor 中的信息）
            element_A = element_A_from_tensor or element_A or element
            element_B = element_B_from_tensor or element_B or element
            layout_A = layout_A_from_tensor or layout_A or layout
            layout_B = layout_B_from_tensor or layout_B or layout

            if len(A_shape) != 2 or len(B_shape) != 3:
                raise ValueError(
                    "A must be 2D tensor (`m, k`) and B must be 3D tensor (`problemCount, k, n`) for group gemm"
                )

            # B 是 3D: [problemCount, k, n]
            # 验证第一维是否等于 problemCount
            if B_shape[0] != self.problemCount:
                raise ValueError(
                    f"B's first dimension ({B_shape[0]}) must equal problemCount ({self.problemCount})"
                )
            # 提取 K, N 从最后两维
            B_k = B_shape[1]
            B_n = B_shape[2]

            # 验证 A 的 K 维和 B 的 K 维是否匹配
            if A_shape[1] != B_k:
                raise ValueError(
                    f"A's K dimension ({A_shape[1]}) must match B's K dimension ({B_k})"
                )
            # 提取 M, K, N
            self.M = A_shape[0]
            self.K = A_shape[1]  # 使用 A 的 K
            self.N = B_n  # 每个 group 的 N 维度
        else:
            if element is None and not all([element_A, element_B, element_C]):
                raise ValueError(
                    "must provide 'element', or specify element_A, element_B, element_C separately"
                )
            if layout is None and not all([layout_A, layout_B]):
                raise ValueError(
                    "must provide 'layout', or specify layout_A, layout_B separately"
                )

            element_A = element_A or element
            element_B = element_B or element
            layout_A = layout_A or layout
            layout_B = layout_B or layout

            self.M = None
            self.K = None
            self.N = None

        # 如果传入了 Bias OpTensor，从它中提取信息
        if Bias is not None:
            _, element_Bias_from_tensor, layout_Bias_from_tensor, _ = extract_info(
                Bias, element_Bias, layout_Bias
            )
            element_Bias = element_Bias_from_tensor or element_Bias
            layout_Bias = layout_Bias_from_tensor or layout_Bias

        # 如果传入了 C OpTensor，从它中提取信息
        if C is not None:
            _, element_C_from_tensor, layout_C_from_tensor, _ = extract_info(
                C, element_C or element, None
            )
            element_C = element_C_from_tensor or element_C or element

        self.element_A = element_A
        self.element_B = element_B
        self.element_Bias = element_Bias
        self.element_C = element_C or element
        if element_accumulator is None and not all([self.element_A, self.element_B]):
            raise ValueError(
                "element_accumulator must be provided, or element_A, element_B should be given both so that accumulator type can be auto-derived"
            )
        self.element_accumulator = element_accumulator or get_default_accumulator(
            self.element_A, self.element_B
        )
        self.layout_A = layout_A
        self.layout_B = layout_B
        # layout_Bias不提供则为 None
        self.layout_Bias = layout_Bias
        # layout_C 固定为 RowMajor，如果 M 和 N 已确定则实例化
        if self.M is not None and self.N is not None:
            self.layout_C = RowMajor((self.M, self.N))
        else:
            # 如果 M 和 N 未确定，先保存类，在 get_kernels 中实例化
            self.layout_C = RowMajor
        self.atlas_arch = atlas_arch
        self.alpha = alpha
        self.beta = beta
        # 如果 core_num 未提供，从 driver 获取设备属性
        if core_num is None:
            _override_hint = "override by passing 'core_num' to GroupGemm()"
            try:
                from triton.runtime.driver import driver

                device = driver.active.get_current_device()
                prop = driver.active.utils.get_device_properties(device)
                core_num = prop["num_aicore"]
            except ModuleNotFoundError:
                warnings.warn(
                    "'triton' is not installed on your environment, cannot obtain driver info."
                    f"core_num defaults to ({_CORE_NUM}). ({_override_hint})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                core_num = _CORE_NUM
            except Exception as e:
                warnings.warn(
                    "An unexpected error occurred; "
                    f"core_num defaults to ({_CORE_NUM}). ({_override_hint})\nError details: {e!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                core_num = _CORE_NUM

        self.core_num = core_num

        # 如果 evg_config 不为 None，必须包含 fn_src 和 example_inputs
        if evg_config is not None:
            if "fn_src" not in evg_config or "example_inputs" not in evg_config:
                raise ValueError(
                    "evg_config must contain 'fn_src' and 'example_inputs'"
                )
        self.evg = evg_config

    def can_implement(self) -> bool:
        return math.isclose(self.alpha, 1.0) and (
            math.isclose(self.beta, 0.0) or math.isclose(self.beta, 1.0)
        )

    def get_kernels(
        self,
        A: Optional[SupportedTensor] = None,
        B: Optional[SupportedTensor] = None,
        Bias: Optional[SupportedTensor] = None,
        C: Optional[SupportedTensor] = None,
        groupList: Optional[SupportedTensor] = None,
    ) -> List[GemmKernelBase]:
        # 处理 groupList：优先使用传入的参数，否则使用 __init__ 中保存的值
        final_groupList = groupList if groupList is not None else self.groupList

        # 必须提供 groupList，从它提取 problemCount
        if final_groupList is None:
            raise ValueError(
                "groupList must be provided, either in __init__ or get_kernels"
            )

        groupList_shape, groupList_element, _, _ = extract_info(
            final_groupList, None, None
        )
        if groupList_shape is None:
            raise ValueError("groupList must have valid shape")

        if len(groupList_shape) != 1:
            raise ValueError("groupList must be 1D tensor")
        # 从 groupList 的长度提取 problemCount
        final_problemCount = groupList_shape[0]

        # 优先使用传入的参数（向后兼容），否则使用 __init__ 中保存的值
        use_init_values = (A is None and B is None) and (
            self.A is not None and self.B is not None
        )

        if use_init_values:
            # 使用 __init__ 中保存的值
            if self.M is None or self.K is None or self.N is None:
                raise ValueError(
                    "无法确定 M, K, N，请确保在 GroupGemm.__init__ 中传入了 A 和 B OpTensor"
                )

            # 如果 layout_C 还是类（未实例化），则实例化它
            if isinstance(self.layout_C, type) and issubclass(self.layout_C, Layout):
                self.layout_C = self.layout_C((self.M, self.N))

            element_A = self.element_A
            element_B = self.element_B
            element_C = self.element_C
            element_Bias = self.element_Bias

            layout_A = self.layout_A
            layout_B = self.layout_B
            layout_C = self.layout_C
            layout_Bias = self.layout_Bias

            M = self.M
            K = self.K
            N = self.N

        else:
            # 处理输入：如果传入 OpTensor，直接使用其信息；如果是 torch.Tensor/np.ndarray，提取信息
            # 使用 OpTensor 时，不需要实例化实际的 tensor 数据
            if A is None or B is None:
                raise ValueError(
                    "A 和 B 必须提供，可以通过 GroupGemm.__init__ 或 get_kernels 参数传入"
                )

            A_shape, element_A, layout_A, A_tensor = extract_info(
                A, self.element_A, self.layout_A
            )
            B_shape, element_B, layout_B, B_tensor = extract_info(
                B, self.element_B, self.layout_B
            )
            Bias_shape, element_Bias, layout_Bias, Bias_tensor = extract_info(
                Bias, self.element_Bias, self.layout_Bias
            )
            C_shape, element_C, layout_C, C_tensor = extract_info(
                C, self.element_C, self.layout_C
            )

            element_A = element_A or self.element_A
            element_B = element_B or self.element_B
            element_C = element_C or self.element_C
            element_Bias = element_Bias or self.element_Bias

            layout_A = layout_A or self.layout_A
            layout_B = layout_B or self.layout_B
            layout_C = layout_C or self.layout_C
            layout_Bias = layout_Bias or self.layout_Bias

            if len(A_shape) != 2 or len(B_shape) != 3:
                raise ValueError(
                    "A must be 2D tensor (`m, k`) and B must be 3D tensor (`problemCount, k, n`) for group gemm"
                )

            # 提取 M, K, N
            M = A_shape[0]
            K = A_shape[1]
            N = B_shape[2]

            # 如果 layout_C 还是类（未实例化），则实例化它
            if isinstance(layout_C, type) and issubclass(layout_C, Layout):
                layout_C = layout_C((M, N))

        params = {
            # 不传递 tensor 对象，只传递元数据（使用 OpTensor 时避免实例化）
            "element_accumulator": self.element_accumulator,
            "element_A": element_A,
            "element_B": element_B,
            "element_Bias": element_Bias,
            "element_C": element_C,
            "layout_A": layout_A,
            "layout_B": layout_B,
            "layout_Bias": layout_Bias,
            "layout_C": layout_C,
            "arch_tag": self.atlas_arch,
            "M": M,
            "K": K,
            "N": N,
            "problemCount": final_problemCount,
            "groupList_element": groupList_element,
        }

        if math.isclose(self.alpha, 1.0) and math.isclose(self.beta, 0.0):
            return [GroupedMatmulSliceMKernel(**params)]
        else:
            warnings.warn(
                f"Only alpha=1.0 and beta=0.0 are supported for grouped gemm, "
                f"got alpha={self.alpha}, beta={self.beta}; returning an empty kernel list.",
                UserWarning,
                stacklevel=2,
            )
            return []
