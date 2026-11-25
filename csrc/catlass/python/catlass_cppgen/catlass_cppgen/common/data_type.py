# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import warnings
from enum import Enum
from functools import lru_cache

from catlass_cppgen.common.typing import SupportedDataType
import torch


class DataType(Enum):
    """数据类型枚举，仅允许通过 from_dtype 构造"""

    AUTO = "auto"
    UNDEFINED = "void"
    FLOAT = "float"
    FLOAT16 = "half"
    INT8 = "int8_t"
    INT32 = "int32_t"
    UINT8 = "uint8_t"
    INT16 = "int16_t"
    UINT16 = "uint16_t"
    UINT32 = "uint32_t"
    INT64 = "int64_t"
    UINT64 = "uint64_t"
    DOUBLE = "double"
    BOOL = "bool"
    STRING = "string"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    BF16 = "bfloat16_t"
    INT4 = "AscendC::int4_t"
    UINT1 = "uint1"
    COMPLEX32 = "complex32"
    HIFLOAT8 = "hi_float8"
    FLOAT8_E5M2 = "float8_e5m2_t"
    FLOAT8_E4M3FN = "float8_e4m3_t"
    FLOAT8_E8M0 = "float8_e8m0_t"
    FLOAT6_E3M2 = "float6_e3m2"
    FLOAT6_E2M3 = "float6_e2m3"
    FLOAT4_E2M1 = "float4_e2m1x2_t"
    FLOAT4_E1M2 = "float4_e1m2x2_t"

    @classmethod
    @lru_cache(maxsize=10)
    def from_dtype(cls, raw_dtype: SupportedDataType) -> "DataType":
        """仅通过from_dtype接口进行构造, 其余转换接口全部移除"""
        # 定义映射（不保留为类字段，简化为这里局部）
        torch_map = {
            torch.float32: cls.FLOAT,
            torch.float: cls.FLOAT,
            torch.float16: cls.FLOAT16,
            torch.half: cls.FLOAT16,
            torch.int8: cls.INT8,
            torch.int32: cls.INT32,
            torch.int: cls.INT32,
            torch.uint8: cls.UINT8,
            torch.int16: cls.INT16,
            torch.short: cls.INT16,
            torch.int64: cls.INT64,
            torch.long: cls.INT64,
            torch.float64: cls.DOUBLE,
            torch.double: cls.DOUBLE,
            torch.bool: cls.BOOL,
            torch.complex64: cls.COMPLEX64,
            torch.complex128: cls.COMPLEX128,
        }

        torch_map_optional = {
            "bfloat16": cls.BF16,
            "float8_e5m2": cls.FLOAT8_E5M2,
            "float8_e4m3fn": cls.FLOAT8_E4M3FN,
            "float8_e8m0fnu": cls.FLOAT8_E8M0,
            "float4_e2m1fn_x2": cls.FLOAT4_E2M1,
            "float4_e1m2fn_x2": cls.FLOAT4_E1M2,
        }
        for dtype_name, dtype in torch_map_optional.items():
            if hasattr(torch, dtype_name):
                torch_map[getattr(torch, dtype_name)] = dtype

        # direct torch match
        if isinstance(raw_dtype, torch.dtype):
            if raw_dtype in torch_map:
                return torch_map[raw_dtype]
            if str(raw_dtype) == "torch.float8_e5m2":
                return cls.FLOAT8_E5M2
            if str(raw_dtype) == "torch.float8_e4m3fn":
                return cls.FLOAT8_E4M3FN
            # 如果传入的是 torch.dtype 但不在映射中，返回 UNDEFINED
            return cls.UNDEFINED

        # 如果传入的是其他类型（非 torch.dtype），返回 UNDEFINED
        return cls.UNDEFINED

    def data_size(self) -> int:
        """获取数据类型的字节大小

        Returns:
            int: 数据类型的字节大小

        Raises:
            ValueError: 如果数据类型的大小未定义
        """
        size_map = {
            self.FLOAT: 4,
            self.FLOAT16: 2,
            self.BF16: 2,
            self.INT8: 1,
            self.INT32: 4,
            self.INT64: 8,
        }
        if self not in size_map:
            raise ValueError(f"Data size not defined for {self}")
        return size_map[self]


def get_default_accumulator(data_type_A: DataType, data_type_B: DataType) -> DataType:
    """获取默认的累加器数据类型"""
    if data_type_A == DataType.UNDEFINED or data_type_B == DataType.UNDEFINED:
        raise ValueError(
            "accumulator dtype cannot be derived when A or B is DataType.UNDEFINED"
        )
    if data_type_A == DataType.AUTO and data_type_B == DataType.AUTO:
        raise ValueError(
            "accumulator dtype cannot be derived when A and B are both DataType.AUTO"
        )

    if data_type_A == DataType.AUTO:
        warnings.warn(
            "The dtype of A is auto-derived from B since A is DataType.AUTO",
            UserWarning,
            stacklevel=2,
        )
        data_type_A = data_type_B
    if data_type_B == DataType.AUTO:
        warnings.warn(
            "The dtype of B is auto-derived from A since B is DataType.AUTO",
            UserWarning,
            stacklevel=2,
        )
        data_type_B = data_type_A

    if data_type_A != data_type_B:
        raise ValueError(
            "Accumulator type cannot be derived when the dtype of A and B are not the same"
        )

    accumulator_map = {
        (DataType.FLOAT16, DataType.FLOAT16): DataType.FLOAT,
        (DataType.FLOAT, DataType.FLOAT16): DataType.FLOAT,
        (DataType.BF16, DataType.BF16): DataType.FLOAT,
        (DataType.INT8, DataType.INT8): DataType.INT32,
        (DataType.INT4, DataType.INT4): DataType.INT32,
    }

    return accumulator_map.get((data_type_A, data_type_B), data_type_A)
