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

#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyL0CToGmTlaTest : public TileCopyTlaTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element>
    void setShape()
    {
        const auto& param = GetParam();
        _row = param.row;
        _col = param.col;
        _dst_row = param.row;
        _dst_col = param.col;
        TileCopyTlaTest::_setShape<Element, false, false>(_row, _col);
    }

    template <class ElementDst, class ElementSrc>
    void BaseCheck(const AscendCCallLog& logTileCopy)
    {
        ASSERT_EQ(logTileCopy.name, "Fixpipe");
        ASSERT_EQ(logTileCopy.args.size(), 3);
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(ElementDst));
        ASSERT_EQ(logTileCopy.GetArgsTAt(1).Type(), typeid(ElementSrc));
    }
};

using L0CToGmCoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class LayoutSrc>
using L0CToGmTensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, L0CToGmCoordZero,
    AscendC::TPosition::CO1>;
template <class Element, class LayoutDst>
using L0CToGmTensorDst = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutDst, L0CToGmCoordZero,
    AscendC::TPosition::GM>;

// ============================================================================
// Testsuite from **L0C**
// ============================================================================

// Data-path: L0C → RowMajor
// Element-type: no-except (float)
// Speciality: basic (TLA NO_QUANT Fixpipe, single call)
TEST_P(TileCopyL0CToGmTlaTest, L0CToRowMajorTestBasic)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::RowMajor;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr auto quantPre = CopyL0CToGmQuantMode<Arch::AtlasA2, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::GlobalTensor<ElementDst> gmDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToGmTensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToGmTensorDst<ElementDst, LayoutDst> tensorGm(gmDst, layoutDst);

    CopyL0CToGmTla<Arch::AtlasA2, decltype(tensorL0C), decltype(tensorGm)> copyL0CToGm;
    const uint8_t unitFlag = 0;
    copyL0CToGm(tensorGm, tensorL0C, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* intriParams = logTileCopy.GetArgsAt(2).Value<AscendC::FixpipeParamsV220>();
    ASSERT_EQ(intriParams->nSize, _col);
    ASSERT_EQ(intriParams->mSize, _row);
    ASSERT_EQ(intriParams->srcStride, _row_round);
    ASSERT_EQ(intriParams->dstStride, _col);
    ASSERT_EQ(intriParams->quantPre, quantPre);
    ASSERT_EQ(intriParams->reluEn, _0);
    ASSERT_EQ(intriParams->unitFlag, unitFlag);
}

// Data-path: L0C → zN
// Element-type: no-except (float)
// Speciality: basic (TLA NO_QUANT zN output, channelSplit enabled)
TEST_P(TileCopyL0CToGmTlaTest, L0CTozNTestBasic)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::zN;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr auto quantPre = CopyL0CToGmQuantMode<Arch::AtlasA2, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::GlobalTensor<ElementDst> gmDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToGmTensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToGmTensorDst<ElementDst, LayoutDst> tensorGm(gmDst, layoutDst);

    CopyL0CToGmTla<Arch::AtlasA2, decltype(tensorL0C), decltype(tensorGm)> copyL0CToGm;
    const uint8_t unitFlag = 0;
    copyL0CToGm(tensorGm, tensorL0C, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* intriParams = logTileCopy.GetArgsAt(2).Value<AscendC::FixpipeParamsV220>();
    ASSERT_EQ(intriParams->nSize, _col);
    ASSERT_EQ(intriParams->mSize, _row);
    ASSERT_EQ(intriParams->srcStride, _row_round);
    ASSERT_EQ(intriParams->dstStride, _dst_row_round);
    ASSERT_EQ(intriParams->quantPre, quantPre);
    ASSERT_EQ(intriParams->reluEn, _0);
    ASSERT_EQ(intriParams->unitFlag, unitFlag);
    ASSERT_TRUE(intriParams->isChannelSplit);
}

// Data-path: L0C → RowMajor
// Element-type: half
// Speciality: half (TLA NO_QUANT type-cast fp32→fp16)
TEST_P(TileCopyL0CToGmTlaTest, L0CToRowMajorTestHalf)
{
    using ElementAccumulator = float;
    using ElementDst = half;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::RowMajor;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr auto quantPre = CopyL0CToGmQuantMode<Arch::AtlasA2, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::GlobalTensor<ElementDst> gmDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToGmTensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToGmTensorDst<ElementDst, LayoutDst> tensorGm(gmDst, layoutDst);

    CopyL0CToGmTla<Arch::AtlasA2, decltype(tensorL0C), decltype(tensorGm)> copyL0CToGm;
    const uint8_t unitFlag = 0;
    copyL0CToGm(tensorGm, tensorL0C, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logTileCopy);

    const auto* intriParams = logTileCopy.GetArgsAt(2).Value<AscendC::FixpipeParamsV220>();
    ASSERT_EQ(intriParams->nSize, _col);
    ASSERT_EQ(intriParams->mSize, _row);
    ASSERT_EQ(intriParams->srcStride, _row_round);
    ASSERT_EQ(intriParams->dstStride, _col);
    ASSERT_EQ(intriParams->quantPre, quantPre);
    ASSERT_EQ(intriParams->reluEn, _0);
    ASSERT_EQ(intriParams->unitFlag, unitFlag);
}

INSTANTIATE_TEST_SUITE_P(
    CopyL0CToGmTla,
    TileCopyL0CToGmTlaTest,
    ::testing::Values(
        TestMatrixShape{128U, 256U},
        TestMatrixShape{64U, 128U}
    )
);

#endif // CATLASS_ARCH == 2201
