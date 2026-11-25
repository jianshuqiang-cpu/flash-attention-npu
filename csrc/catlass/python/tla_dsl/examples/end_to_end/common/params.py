# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import catlass.tla as tla


@dataclass(frozen=True)
class TilingParams:
    l1_tm: tla.Constexpr[int] = 256
    l1_tn: tla.Constexpr[int] = 256
    l1_tk: tla.Constexpr[int] = 128
    l0_tm: tla.Constexpr[int] = 256
    l0_tn: tla.Constexpr[int] = 256
    l0_tk: tla.Constexpr[int] = 32


@dataclass(frozen=True)
class SwizzleParams:
    SWIZZLE_DIRECTION: tla.Constexpr[int] = 0
    SWIZZLE_OFFSET: tla.Constexpr[int] = 3
