/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_COMMON_COMMON_HPP
#define SHARED_LIB_COMMON_COMMON_HPP

#include <acl/acl.h>

#include "catlass/detail/dependent_false.hpp"
#include "catlass/layout/layout.hpp"

namespace CatlassKernel {
using namespace Catlass;

/**
 * @brief 判断 RowMajor 布局是否需要 padding。
 *
 * 行主序布局的 padding 判定：如果 stride[0]（单行字节数）>= 65536，
 * 总是需要 padding 以减少 stride；否则 stride[0] 必须能被 align 整除。
 *
 * 用于 GEMM 算子中判断是否需要为输入/输出矩阵插入 padding 对齐。
 *
 * @param layout RowMajor 布局对象（包含 stride 信息）。
 * @param align  对齐边界（字节对齐数，通常为 32/64 等）。
 * @return true  需要 padding。
 * @return false 不需要 padding。
 */
inline bool IsNeedPadding(layout::RowMajor layout, uint32_t align) {
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

/**
 * @brief 判断 ColumnMajor 布局是否需要 padding。
 *
 * 列主序布局：使用 stride[1]（单列字节数）替代 RowMajor 的 stride[0]。
 * 逻辑同 IsNeedPadding(RowMajor)。
 *
 * @param layout ColumnMajor 布局对象。
 * @param align  对齐边界。
 * @return true  需要 padding。
 * @return false 不需要 padding。
 */
inline bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align) {
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

/**
 * @brief zN（分块 NZ->标准）布局永不需 padding。
 *
 * zN 布局将分块数据重新组织为标准行列布局，内部已由数据搬运层
 * 完成对齐处理，因此直接返回 false。
 *
 * @param layout zN 布局对象（未使用，仅用于重载决议）。
 * @param align  对齐边界（未使用）。
 * @return false 始终不需要 padding。
 */
inline bool IsNeedPadding(layout::zN layout, uint32_t align) {
    (void)layout;
    (void)align;
    return false;
}

/**
 * @brief nZ（标准->分块 NZ）布局永不需 padding。
 *
 * 同 zN，分块布局的数据搬运层已处理对齐，不对外暴露 padding 需求。
 *
 * @param layout nZ 布局对象（未使用）。
 * @param align  对齐边界（未使用）。
 * @return false 始终不需要 padding。
 */
inline bool IsNeedPadding(layout::nZ layout, uint32_t align) {
    (void)layout;
    (void)align;
    return false;
}

} // namespace CatlassKernel
#endif
