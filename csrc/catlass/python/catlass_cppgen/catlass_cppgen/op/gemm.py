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
from catlass_cppgen.common.typing import SupportedDataType
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.catlass.layout.layout import Layout, RowMajor
from catlass_cppgen.common.data_type import get_default_accumulator
from catlass_cppgen.common.utils import extract_info
from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase
from catlass_cppgen.kernel.gemm.basic_matmul import BasicMatmulKernel
from catlass_cppgen.kernel.gemm.batched_matmul import BatchedMatmulKernel
from catlass_cppgen.kernel.gemm.strided_batched_matmul import (
    StridedBatchedMatmulKernel,
)
from catlass_cppgen.kernel.gemm.multi_core_splitk_matmul import (
    MultiCoreSplitkMatmulKernel,
)
from catlass_cppgen.kernel.gemm.streamk_matmul import StreamkMatmulKernel

from catlass_cppgen.kernel.gemm.tail_multi_core_splitk_matmul import (
    TailMultiCoreSplitkMatmulKernel,
)
from catlass_cppgen.kernel.gemm.basic_matmul_tla_visitor import (
    BasicMatmulTlaVisitorKernel,
)

_CORE_NUM = 8  # Default core num


class Gemm(OperationBase):
    # 类属性类型注解
    A: Optional[OpTensor]
    B: Optional[OpTensor]
    Bias: Optional[OpTensor]
    C: Optional[OpTensor]
    M: Optional[int]
    K: Optional[int]
    N: Optional[int]
    batch_count: Optional[int]
    is_batched: Optional[bool]

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
        evg_config: Optional[Dict[str, Any]] = None,
        atlas_arch: Optional[Arch] = None,
        core_num: Optional[int] = None,
        A: Optional[OpTensor] = None,
        B: Optional[OpTensor] = None,
        Bias: Optional[OpTensor] = None,
        C: Optional[OpTensor] = None,
        strided_batched: bool = False,
        transA: bool = False,
        transB: bool = False,
    ):
        # 保存 A、B、Bias 和 C OpTensor
        self.A = A
        self.B = B
        self.Bias = Bias
        self.C = C
        self.strided_batched = strided_batched
        self.transA = transA
        self.transB = transB

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

            _fall_back_layout = RowMajor
            layout_A = layout_A_from_tensor or layout_A or layout or _fall_back_layout
            layout_B = layout_B_from_tensor or layout_B or layout or _fall_back_layout

            # 判断是否 batched
            is_batched = len(A_shape) == 3 and len(B_shape) == 3

            # 提取 M, K, N
            if is_batched:
                if A_shape[0] != B_shape[0]:
                    raise ValueError(
                        f"A.shape[0] ({A_shape[0]}) must be equal to B.shape[0] ({B_shape[0]}) for batched matmul"
                    )
                self.batch_count = A_shape[0]
                self.M = A_shape[1]
                self.K = A_shape[2]
                self.N = B_shape[2]
            else:
                self.batch_count = None
                self.M = A_shape[0]
                self.K = A_shape[1]
                self.N = B_shape[1]

            self.is_batched = is_batched
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

            self.batch_count = None
            self.M = None
            self.K = None
            self.N = None
            self.is_batched = None

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
            _override_hint = "override by passing 'core_num' to Gemm()"
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

    def get_kernels(self) -> List[GemmKernelBase]:
        # 必须使用 __init__ 中传入的 A 和 B
        if self.A is None or self.B is None:
            raise ValueError("A 和 B 必须在 Gemm.__init__ 中传入")

        # 判断是否 batched 和提取 M, K, N（使用 __init__ 中保存的值）
        if self.M is None or self.K is None or self.N is None:
            raise ValueError(
                "无法确定 M, K, N，请确保在 Gemm.__init__ 中传入了 A 和 B OpTensor"
            )

        # 如果 layout_C 还是类（未实例化），则实例化它
        if isinstance(self.layout_C, type) and issubclass(self.layout_C, Layout):
            self.layout_C = self.layout_C((self.M, self.N))

        # 只提取 Bias 的 shape（用于后续判断）
        Bias_shape = self.Bias.shape if self.Bias is not None else None

        params = {
            # 不传递 tensor 对象，只传递元数据（使用 OpTensor 时避免实例化）
            "element_accumulator": self.element_accumulator,
            "element_A": self.element_A,
            "element_B": self.element_B,
            "element_Bias": self.element_Bias,
            "element_C": self.element_C,
            "layout_A": self.layout_A,
            "layout_B": self.layout_B,
            "layout_Bias": self.layout_Bias,
            "layout_C": self.layout_C,
            "arch_tag": self.atlas_arch,
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "batchCount": self.batch_count if self.is_batched else None,
            "transA": self.transA,
            "transB": self.transB,
            "evg": self.evg,
        }

        if self.evg is not None:
            return [BasicMatmulTlaVisitorKernel(**params)]

        if self.is_batched:
            # BatchedMatmul不支持Bias，带Bias的话返回空列表
            if self.element_Bias is not None:
                return []
            if self.strided_batched:
                return [StridedBatchedMatmulKernel(**params)]
            return [BatchedMatmulKernel(**params)]
        else:
            if math.isclose(self.alpha, 1.0) and math.isclose(self.beta, 0.0):
                if (
                    self.element_Bias is not None
                    and Bias_shape is not None
                    and len(Bias_shape) > 1
                ):
                    return []
                _threshold1 = 4096
                _threshold2 = 2048
                _default_ksplit_tile = (256, 256, 128, 32)
                res = []
                if self.K > _threshold2:
                    # prefer k-split template
                    num_task = math.ceil(self.M / _default_ksplit_tile[0]) * math.ceil(
                        self.N / _default_ksplit_tile[1]
                    )
                    if num_task <= (0.5 * self.core_num):
                        res.append(MultiCoreSplitkMatmulKernel(**params))
                    elif num_task <= (0.9 * self.core_num) or (
                        num_task % self.core_num
                    ) <= (0.9 * self.core_num):
                        res.append(StreamkMatmulKernel(**params))
                    if num_task > self.core_num and num_task < (1.5 * self.core_num):
                        res.append(TailMultiCoreSplitkMatmulKernel(**params))
                if self.K < _threshold1:
                    res.append(BasicMatmulKernel(**params))
                return res
            else:
                warnings.warn(
                    f"Only alpha=1.0 and beta=0.0 are supported for gemm, "
                    f"got alpha={self.alpha}, beta={self.beta}; returning an empty kernel list.",
                    UserWarning,
                    stacklevel=2,
                )
                return []
