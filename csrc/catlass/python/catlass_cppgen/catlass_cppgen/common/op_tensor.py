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
from catlass_cppgen.catlass.layout.layout import Layout
from catlass_cppgen.common.typing import SupportedTensor
from catlass_cppgen.common.utils import infer_layout_from_stride, get_tensor_data_ptr


class OpTensor:
    """OpTensor类是op的输入输出tensor的抽象，用于表示op的输入输出tensor

    该类提供了统一的tensor抽象，可以从torch.Tensor或np.ndarray创建，
    或直接通过shape和stride创建（避免实例化），并自动推断数据类型和布局信息。

    参数:
    dtype: DataType, tensor的数据类型
    layout: Layout, tensor的布局
    shape: tuple[int, ...], tensor的完整形状
    data_ptr: Optional[ctypes.c_void_p], tensor的数据指针
    """

    def __init__(
        self,
        dtype: DataType,
        layout: Layout,
        shape: Optional[tuple[int, ...]] = None,
        data_ptr: Optional[ctypes.c_void_p] = None,
    ):
        self.dtype = dtype
        self.layout = layout
        self._shape = shape if shape is not None else layout.shape
        self.data_ptr = data_ptr

    @classmethod
    def from_shape_stride(
        cls,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: DataType,
    ) -> "OpTensor":
        """直接从 shape 和 stride 创建 OpTensor（避免实例化 tensor）

        参数:
        shape: tensor 的形状（可以是 2D 或 3D，支持 batched）
        stride: tensor 的步长
        dtype: tensor 的数据类型

        返回:
        OpTensor 对象
        """
        # 从 shape 和 stride 推断 Layout（对于 batched，只推断内层矩阵的布局）
        layout = infer_layout_from_stride(shape, stride)
        # 保存完整的 shape（包括 batch 维度）
        return cls(dtype=dtype, layout=layout, shape=shape, data_ptr=None)

    @classmethod
    def from_tensor(
        cls,
        tensor: SupportedTensor,
        layout: Optional[Layout] = None,
        dtype: Optional[DataType] = None,
    ) -> "OpTensor":
        """从 torch.Tensor 或 np.ndarray 创建 OpTensor

        参数:
        tensor: torch.Tensor 或 np.ndarray
        layout: 可选的 Layout，如果不提供则从 tensor 的 stride 推断
        dtype: 可选的 DataType，如果不提供则从 tensor 的 dtype 推断

        返回:
        OpTensor 对象
        """
        # 获取 shape 和 stride
        if isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
            stride = tuple(tensor.stride())
            raw_dtype = tensor.dtype
        elif isinstance(tensor, np.ndarray):
            shape = tuple(tensor.shape)
            stride = tuple(tensor.strides)
            if tensor.dtype.itemsize > 0:
                stride = tuple(s // tensor.dtype.itemsize for s in stride)
            raw_dtype = tensor.dtype
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")
        if dtype is None:
            dtype = DataType.from_dtype(raw_dtype)
        if layout is None:
            layout = infer_layout_from_stride(shape, stride)
        data_ptr = get_tensor_data_ptr(tensor)
        return cls(dtype=dtype, layout=layout, shape=shape, data_ptr=data_ptr)

    @property
    def shape(self) -> tuple[int, ...]:
        """返回 tensor 的完整形状"""
        return self._shape

    @property
    def stride(self) -> tuple[int, ...]:
        return self.layout.stride

    @property
    def capacity(self) -> int:
        return self.layout.capacity
