# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import ctypes
from typing import Optional
import torch
import numpy as np

from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import (
    Layout,
    RowMajor,
    ColumnMajor,
    VectorLayout,
)
from catlass_cppgen.common.typing import SupportedTensor


def infer_layout_from_stride(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> Layout | None:
    """从 shape 和 stride 推断 Layout 类型

    参数:
    shape: tensor 的形状
    stride: tensor 的步长

    返回:
    Layout 对象
    """
    if len(shape) == 1:
        # 一维向量
        return VectorLayout(shape[0])
    if len(shape) == 2:
        # 二维输入，目前只支持RowMajor和ColumnMajor
        m, n = shape
        stride_m, stride_n = stride
        if stride_m == n and stride_n == 1:
            return RowMajor((m, n))
        if stride_m == 1 and stride_n == m:
            return ColumnMajor((m, n))
        # 非标准布局，默认使用 RowMajor，但保留原始 stride
        return RowMajor((m, n))
    else:
        # 三维输入
        if len(shape) == 3:
            batch, m, n = shape
            stride_batch, stride_m, stride_n = stride
            if stride_batch != m * n:
                # Transposed batch case
                if stride_n == 1:
                    return RowMajor((m, n))
                elif stride_m == 1:
                    return ColumnMajor((m, n))
                else:
                    return None
            if stride_m == n and stride_n == 1:
                return RowMajor((m, n))
            if stride_m == 1 and stride_n == m:
                return ColumnMajor((m, n))

            return RowMajor((m, n))
        else:
            return RowMajor((shape[-2], shape[-1]))


def get_tensor_data_ptr(tensor: SupportedTensor) -> Optional[ctypes.c_void_p]:
    """从 torch.Tensor 或 np.ndarray 获取数据指针

    参数:
    tensor: torch.Tensor 或 np.ndarray

    返回:
    ctypes.c_void_p 数据指针，如果无法获取则返回 None
    """
    if isinstance(tensor, torch.Tensor):
        return ctypes.c_void_p(tensor.data_ptr())
    elif isinstance(tensor, np.ndarray):
        return ctypes.c_void_p(tensor.ctypes.data)
    return None


def extract_info(tensor, default_element, default_layout):
    """从 OpTensor 或实际 tensor 中提取信息

    参数:
    tensor: OpTensor、torch.Tensor、np.ndarray 或 None
    default_element: 默认的数据类型
    default_layout: 默认的布局类型

    返回:
    tuple: (shape, element, layout, tensor_obj)
        - shape: tensor 的形状
        - element: 数据类型
        - layout: 布局类型
        - tensor_obj: tensor 对象（如果是 OpTensor 则为 None，否则为原 tensor）
    """
    # 延迟导入以避免循环导入
    from catlass_cppgen.common.op_tensor import OpTensor

    if tensor is None:
        return None, None, None, None
    if isinstance(tensor, OpTensor):
        # 使用 OpTensor 的信息，避免实例化
        shape = tensor.shape
        element = tensor.dtype
        layout = tensor.layout
        return shape, element, layout, None  # 不传递 tensor 对象
    else:
        # 从实际 tensor 中提取信息（向后兼容）
        shape = tuple(tensor.shape)
        if isinstance(tensor, torch.Tensor):
            element = DataType.from_dtype(tensor.dtype)
        elif isinstance(tensor, np.ndarray):
            element = DataType.from_dtype(tensor.dtype)
        else:
            element = default_element
        # 对于实际 tensor，layout 使用默认值
        layout = default_layout
        return shape, element, layout, tensor  # 传递 tensor 对象以保持兼容


def _get_cpp_value(value: any) -> str:
    """将 Python 值转换为 C++ 代码字符串."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _snake_to_camel(snake_str: str) -> str:
    """将蛇形命名转换为驼峰命名."""
    components = snake_str.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])


def _get_cpp_type(value: any) -> str:
    """根据 Python 值推断对应的 C++ 类型."""
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "uint32_t"
    else:
        return "auto"


def get_type_name(type_str) -> str:
    """获取类型的名称字符串.

    对于 DataType 枚举，返回其 value；对于字符串，直接返回；对于其他类型，返回其 __name__.

    :param type_str: 类型对象，可以是 DataType 枚举、字符串或其他类型.
    :return: 类型的名称字符串.
    :rtype: str
    """
    if isinstance(type_str, DataType):
        return type_str.value
    elif isinstance(type_str, str):
        return type_str
    else:
        return type_str.__name__
