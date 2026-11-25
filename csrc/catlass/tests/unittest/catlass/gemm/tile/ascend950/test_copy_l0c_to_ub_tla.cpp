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

#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyL0CToUbTlaAscend950Test : public TileCopyTlaTest, public testing::WithParamInterface<TestMatrixShape> {
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
    void BaseCheck(const AscendCCallLog& logFixpipe)
    {
        ASSERT_EQ(logFixpipe.name, "Fixpipe");
        ASSERT_EQ(logFixpipe.args.size(), 3);
        ASSERT_EQ(logFixpipe.GetArgsTAt(0).Type(), typeid(ElementDst));
        ASSERT_EQ(logFixpipe.GetArgsTAt(1).Type(), typeid(ElementSrc));
    }
};

using L0CToUb950CoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class LayoutSrc>
using L0CToUb950TensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, L0CToUb950CoordZero,
    AscendC::TPosition::CO1>;
template <class Element, class LayoutDst>
using L0CToUb950TensorDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, L0CToUb950CoordZero,
    AscendC::TPosition::VECCALC>;

// ============================================================================
// Testsuite from **L0C**
// ============================================================================

// Data-path: L0C → RowMajor (UB)
// Element-type: no-except (float → float)
// Speciality: NoSplit (single-dst Fixpipe, dualDstCtl = 0)
TEST_P(TileCopyL0CToUbTlaAscend950Test, L0CToRowMajorTestNoSplit)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::RowMajor;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementDst);
    constexpr auto quantPre = CopyL0CToDstQuantMode<Arch::Ascend950, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::LocalTensor<ElementDst> ubDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToUb950TensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToUb950TensorDst<ElementDst, LayoutDst> tensorUb(ubDst, layoutDst);

    CopyL0CToUBTla<Arch::Ascend950, decltype(tensorL0C), decltype(tensorUb)> copyL0CToUb;
    const uint8_t unitFlag = 0;
    copyL0CToUb(tensorUb, tensorL0C, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logFixpipe = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logFixpipe);

    ASSERT_EQ(logFixpipe.GetArgsAt(0).GetInstAddr(), _0);
    ASSERT_EQ(logFixpipe.GetArgsAt(1).GetInstAddr(), _0);
    
    const auto* fixpipeArg = logFixpipe.GetArgsAt(2).Value<AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>>();
    ASSERT_EQ(fixpipeArg->nSize, RoundUp(_col, ELE_NUM_PER_BLK));
    ASSERT_EQ(fixpipeArg->mSize, _row);
    ASSERT_EQ(fixpipeArg->srcStride, _row_round);
    ASSERT_EQ(fixpipeArg->dstStride, _col);
    ASSERT_EQ(fixpipeArg->quantPre, quantPre);
    ASSERT_EQ(fixpipeArg->reluEn, _0);
    ASSERT_EQ(fixpipeArg->unitFlag, unitFlag);
    ASSERT_EQ(fixpipeArg->dualDstCtl, _0);
}

// Data-path: L0C → RowMajor (UB)
// Element-type: no-except (float → float)
// Speciality: SplitM (M-split dual-dst Fixpipe, dualDstCtl = 1)
TEST_P(TileCopyL0CToUbTlaAscend950Test, L0CToRowMajorTestSplitM)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::RowMajor;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementDst);

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::LocalTensor<ElementDst> ubDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToUb950TensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToUb950TensorDst<ElementDst, LayoutDst> tensorUb(ubDst, layoutDst);

    CopyL0CToUBTla<Arch::Ascend950, decltype(tensorL0C), decltype(tensorUb),
        CopyL0CToUBMode::SPLIT_M> copyL0CToUb;
    copyL0CToUb(tensorUb, tensorL0C);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logFixpipe = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logFixpipe);
    const auto* fixpipeArg = logFixpipe.GetArgsAt(2).Value<AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>>();
    ASSERT_EQ(fixpipeArg->nSize, RoundUp(_col, ELE_NUM_PER_BLK));
    ASSERT_EQ(fixpipeArg->mSize, RoundUp(_row, 2));
    ASSERT_EQ(fixpipeArg->srcStride, _row_round);
    ASSERT_EQ(fixpipeArg->dstStride, _col);
    ASSERT_EQ(fixpipeArg->dualDstCtl, 1);
}

// Data-path: L0C → RowMajor (UB)
// Element-type: no-except (float → float)
// Speciality: SplitN (N-split dual-dst Fixpipe, dualDstCtl = 2)
TEST_P(TileCopyL0CToUbTlaAscend950Test, L0CToRowMajorTestSplitN)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::RowMajor;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::LocalTensor<ElementDst> ubDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToUb950TensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToUb950TensorDst<ElementDst, LayoutDst> tensorUb(ubDst, layoutDst);

    CopyL0CToUBTla<Arch::Ascend950, decltype(tensorL0C), decltype(tensorUb),
        CopyL0CToUBMode::SPLIT_N> copyL0CToUb;
    copyL0CToUb(tensorUb, tensorL0C);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logFixpipe = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logFixpipe);
    const auto* fixpipeArg = logFixpipe.GetArgsAt(2).Value<AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>>();
    ASSERT_EQ(fixpipeArg->nSize, RoundUp(_col, 32));
    ASSERT_EQ(fixpipeArg->mSize, _row);
    ASSERT_EQ(fixpipeArg->srcStride, _row_round);
    ASSERT_EQ(fixpipeArg->dstStride, _col);
    ASSERT_EQ(fixpipeArg->dualDstCtl, 2);
}

INSTANTIATE_TEST_SUITE_P(
    CopyL0CToUbTlaAscend950,
    TileCopyL0CToUbTlaAscend950Test,
    ::testing::Values(
        TestMatrixShape{128U, 256U},
        TestMatrixShape{65U, 127U}
    )
);

#endif // CATLASS_ARCH == 3510
