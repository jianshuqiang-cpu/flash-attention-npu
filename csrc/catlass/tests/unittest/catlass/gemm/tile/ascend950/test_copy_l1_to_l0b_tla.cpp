/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/tensor.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyL1ToL0BTlaAscend950Test : public TileCopyTlaTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isSrcTrans = false, bool isDstTrans = true>
    void setShape()
    {
        const auto& param = GetParam();
        _row = param.row;
        _col = param.col;
        _dst_row = param.row;
        _dst_col = param.col;
        TileCopyTlaTest::_setShape<Element, isSrcTrans, isDstTrans>(_row, _col);
    }

    template <class Element>
    void BaseCheck(const AscendCCallLog& logTileCopy)
    {
        ASSERT_EQ(logTileCopy.name, "LoadData");
        ASSERT_EQ(logTileCopy.args.size(), 3);
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));
    }
};

using L1ToL0B950CoordZero = tla::Coord<uint32_t, uint32_t>;
template <class Element, class LayoutSrc>
using L1ToL0B950TensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, L1ToL0B950CoordZero,
    AscendC::TPosition::A1>;
template <class Element, class LayoutDst>
using L1ToL0B950TensorDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, L1ToL0B950CoordZero,
    AscendC::TPosition::B2>;

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN (L1) → nZ (L0B)
// Element-type: half
// Speciality: Half (TLA single LoadData with ifTranspose enabled)
TEST_P(TileCopyL1ToL0BTlaAscend950Test, zNTonZTestHalf)
{
    using Element = half;
    using LayoutSrcTag = layout::zN;
    using LayoutDstTag = layout::nZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0bDst;

    setShape<Element, false, true>();
    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    L1ToL0B950TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc, tla::MakeCoord(0U, 0U));
    L1ToL0B950TensorDst<Element, LayoutDst> tensorL0B(l0bDst, layoutDst, tla::MakeCoord(0U, 0U));

    TileCopyTla<Arch::Ascend950, decltype(tensorL1), decltype(tensorL0B)> copyL1ToL0B;
    copyL1ToL0B(tensorL0B, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    const auto& logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), l0bDst.GetAddr());
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), l1Src.GetAddr());

    const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, _dst_row_by_fractal);
    ASSERT_EQ(loadDataArg->kStep, _dst_cols_by_fractal);
    ASSERT_EQ(loadDataArg->srcStride, _row_round * GetEleNumPerC0<Element>() / ELE_NUM_PER_FRACTAL);
    ASSERT_EQ(loadDataArg->dstStride, _dst_col_round * C0_NUM_PER_FRACTAL / ELE_NUM_PER_FRACTAL);
    ASSERT_EQ(loadDataArg->ifTranspose, true);
    ASSERT_EQ(loadDataArg->sid, _0);
}

// Data-path: zN (L1) → nZ (L0B)
// Element-type: no-except (float)
// Speciality: Float (TLA single LoadData transpose, float kStep rounded up to even)
TEST_P(TileCopyL1ToL0BTlaAscend950Test, zNTonZTestFloat)
{
    using Element = float;
    using LayoutSrcTag = layout::zN;
    using LayoutDstTag = layout::nZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0bDst;

    setShape<Element, false, true>();
    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    L1ToL0B950TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc, tla::MakeCoord(0U, 0U));
    L1ToL0B950TensorDst<Element, LayoutDst> tensorL0B(l0bDst, layoutDst, tla::MakeCoord(0U, 0U));

    TileCopyTla<Arch::Ascend950, decltype(tensorL1), decltype(tensorL0B)> copyL1ToL0B;
    copyL1ToL0B(tensorL0B, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    const auto& logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), l0bDst.GetAddr());
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), l1Src.GetAddr());

    const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, _rows_by_fractal);
    ASSERT_EQ(loadDataArg->kStep, RoundUp<2>(_cols_by_fractal));
    ASSERT_EQ(loadDataArg->srcStride, _rows_by_fractal);
    ASSERT_EQ(loadDataArg->dstStride, _dst_row_by_fractal);
    ASSERT_EQ(loadDataArg->ifTranspose, true);
    ASSERT_EQ(loadDataArg->sid, _0);
}

// Data-path: zN (L1) → nZ (L0B)
// Element-type: int8
// Speciality: Int8 (TLA single LoadData transpose, aligned N)
TEST_P(TileCopyL1ToL0BTlaAscend950Test, zNTonZTestInt8)
{
    using Element = int8_t;
    using LayoutSrcTag = layout::zN;
    using LayoutDstTag = layout::nZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0bDst;

    setShape<Element, false, true>();
    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    L1ToL0B950TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc, tla::MakeCoord(0U, 0U));
    L1ToL0B950TensorDst<Element, LayoutDst> tensorL0B(l0bDst, layoutDst, tla::MakeCoord(0U, 0U));

    TileCopyTla<Arch::Ascend950, decltype(tensorL1), decltype(tensorL0B)> copyL1ToL0B;
    copyL1ToL0B(tensorL0B, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    const auto& logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), l0bDst.GetAddr());
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), l1Src.GetAddr());

    const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, RoundUp<2>(_rows_by_fractal));
    ASSERT_EQ(loadDataArg->kStep, _cols_by_fractal);
    ASSERT_EQ(loadDataArg->srcStride, _cols_by_fractal);
    ASSERT_EQ(loadDataArg->dstStride, _dst_cols_by_fractal);
    ASSERT_EQ(loadDataArg->ifTranspose, true);
    ASSERT_EQ(loadDataArg->sid, _0);
}

// Data-path: zN (L1) → nZ (L0B)
// Element-type: int8
// Speciality: Int8NonAlignedN (RoundUp(N) % 32 != 0, per-k multi-LoadData split)
TEST_P(TileCopyL1ToL0BTlaAscend950Test, zNTonZTestInt8NonAlignedN)
{
    using Element = int8_t;
    using LayoutSrcTag = layout::zN;
    using LayoutDstTag = layout::nZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0bDst;

    setShape<Element, false, true>();
    // Override _dst_col to trigger RoundUp<16>(N) % 32 != 0 for int8
    uint32_t nonAlignedN = 112;
    _dst_col = nonAlignedN;
    _dst_col_round = RoundUp<C0_NUM_PER_FRACTAL>(_dst_col);
    _dst_cols_by_fractal = CeilDiv<C0_NUM_PER_FRACTAL>(_dst_col);

    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    L1ToL0B950TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc, tla::MakeCoord(0U, 0U));
    L1ToL0B950TensorDst<Element, LayoutDst> tensorL0B(l0bDst, layoutDst, tla::MakeCoord(0U, 0U));

    TileCopyTla<Arch::Ascend950, decltype(tensorL1), decltype(tensorL0B)> copyL1ToL0B;
    copyL1ToL0B(tensorL0B, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t kLoops = CeilDiv<ELE_NUM_PER_C0>(_dst_row);
    uint32_t srcOuterStrideCol = _row_round * ELE_NUM_PER_C0;
    uint32_t dstOuterStrideRow = _dst_col_round * ELE_NUM_PER_C0;
    ASSERT_EQ(logs.size(), kLoops);

    for (uint32_t kIdx = 0; kIdx < kLoops; ++kIdx) {
        const auto& logTileCopy = logs[kIdx];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), (kIdx * dstOuterStrideRow) * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);

        const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
        ASSERT_EQ(loadDataArg->mStartPosition, kIdx * 2);
        ASSERT_EQ(loadDataArg->kStartPosition, _0);
        ASSERT_EQ(loadDataArg->mStep, 2);
        ASSERT_EQ(loadDataArg->kStep, CeilDiv<ELE_NUM_PER_C0>(_dst_col));
        ASSERT_EQ(loadDataArg->srcStride, CeilDiv<ELE_NUM_PER_FRACTAL>(srcOuterStrideCol));
        ASSERT_EQ(loadDataArg->dstStride, CeilDiv<ELE_NUM_PER_FRACTAL>(dstOuterStrideRow));
        ASSERT_EQ(loadDataArg->ifTranspose, true);
        ASSERT_EQ(loadDataArg->sid, _0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CopyL1ToL0BTlaAscend950,
    TileCopyL1ToL0BTlaAscend950Test,
    ::testing::Values(
        TestMatrixShape{128U, 256U},
        TestMatrixShape{64U, 128U}
    )
);

#endif // CATLASS_ARCH == 3510
