/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNITTEST_TILE_HELPER_HPP
#define UNITTEST_TILE_HELPER_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Test::Helper {

// Pre-defined constants
constexpr uint32_t _1 = 1U;
constexpr uint32_t _0 = 0U;

////////////// HELPER FUNCTIONS
template <class Element>
inline constexpr uint32_t GetEleNumPerC0() {
    return BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
}

template <class Element, class ElementDst = Element, class LayoutSrc, class LayoutDst>
inline void setLayout(const uint32_t row, const uint32_t col, LayoutSrc& layoutSrc, LayoutDst& layoutDst) {
    layoutSrc = LayoutSrc::template MakeLayout<Element>(row, col);
    layoutDst = LayoutDst::template MakeLayout<ElementDst>(row, col);
}

// Determine whether current Layout is contiguous
template <class Layout>
bool isContiguous(Layout layout) {
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
        return (layout.stride(0) == layout.shape(1) && layout.stride(1) == 1L);
    } else if constexpr (std::is_same_v<Layout, layout::ColumnMajor>) {
        return (layout.stride(1) == layout.shape(0) && layout.stride(0) == 1L);
    } else if constexpr (std::is_same_v<Layout, layout::zN>) {
        return (layout.stride(0) == layout.shape(2) && 
        layout.stride(1) == layout.shape(0) * layout.shape(2) && 
        layout.stride(3) == layout.shape(0) * layout.shape(1) * layout.shape(2) &&
        layout.stride(2) == 1L);
    } else if constexpr (std::is_same_v<Layout, layout::nZ>) {
        return (layout.stride(0) == 1L && 
            layout.stride(1) == layout.shape(0) * layout.shape(2) * layout.shape(3) &&
            layout.stride(2) == layout.shape(0) && 
            layout.stride(3) == layout.shape(0) * layout.shape(2));
    } else if constexpr (std::is_same_v<Layout, layout::zZ>) {
        return (layout.stride(2) == 1 && 
            layout.stride(0) == layout.shape(2) && 
            layout.stride(1) == layout.shape(0) * layout.shape(2) * layout.shape(3) && 
            layout.stride(3) == layout.shape(0) * layout.shape(2));
    } else if constexpr (std::is_same_v<Layout, layout::nN>) {
        return (layout.stride(0) == 1L && 
            layout.stride(1) == layout.shape(0) * layout.shape(2) &&
            layout.stride(2) == layout.shape(0) && 
            layout.stride(3) == layout.shape(0) * layout.shape(1) * layout.shape(2));
    } else if constexpr (std::is_same_v<Layout, layout::VectorLayout>) {
        return layout.stride(0) == 1L;
    } else {
        return false; // currently not support other Layout
    }
}

// Helper struct for determine if the caculation should consider trans
template <class Layout>
struct isTrans {
    static_assert(DEPENDENT_FALSE<Layout>, "trans auto-derivation not supported");
};

template <>
struct isTrans<layout::RowMajor> {
    static constexpr bool value = false;
};

template <>
struct isTrans<layout::ColumnMajor> {
    static constexpr bool value = true;
};

template <>
struct isTrans<layout::zN> {
    static constexpr bool value = false;
};

template <>
struct isTrans<layout::zZ> {
    static constexpr bool value = false;
};

template <>
struct isTrans<layout::nN> {
    static constexpr bool value = true;
};

template <>
struct isTrans<layout::nZ> {
    static constexpr bool value = true;
};

template <>
struct isTrans<layout::PaddingRowMajor> {
    static constexpr bool value = false;
};

template <>
struct isTrans<layout::PaddingColumnMajor> {
    static constexpr bool value = true;
};

template <class Layout>
static constexpr bool isTrans_v = isTrans<Layout>::value;

}


#endif // UNITTEST_TILE_HELPER_HPP
