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

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyL0CToGmTlaAscend950Test : public TileCopyTlaTest, public testing::WithParamInterface<TestMatrixShape> {
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
};

using L0CToGm950CoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class LayoutSrc>
using L0CToGm950TensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, L0CToGm950CoordZero,
    AscendC::TPosition::CO1>;
template <class Element, class LayoutDst>
using L0CToGm950TensorDst = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutDst, L0CToGm950CoordZero,
    AscendC::TPosition::GM>;

// Data-path: L0C → RowMajor (GM)
// Element-type: no-except (float → float)
// Speciality: NoQuant (TLA no-quant DataCopy, nz2nd enabled)
TEST_P(TileCopyL0CToGmTlaAscend950Test, L0CToRowMajorTestNoQuant)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::RowMajor;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr auto quantPre = CopyL0CToDstQuantMode<Arch::Ascend950, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::GlobalTensor<ElementDst> gmDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToGm950TensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToGm950TensorDst<ElementDst, LayoutDst> tensorGm(gmDst, layoutDst);

    CopyL0CToGmTla<Arch::Ascend950, decltype(tensorL0C), decltype(tensorGm)> copyL0CToGm;
    const uint8_t unitFlag = 0;
    copyL0CToGm(tensorGm, tensorL0C, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 2);

    const auto& logFixpipeCfg = logs[0];
    ASSERT_EQ(logFixpipeCfg.name, "SetFixpipeNz2ndFlag");
    ASSERT_EQ(*logFixpipeCfg.GetArgsAt(0).Value<uint16_t>(), _1);  // ndNum
    ASSERT_EQ(*logFixpipeCfg.GetArgsAt(1).Value<uint16_t>(), _1);  // srcNdStride
    ASSERT_EQ(*logFixpipeCfg.GetArgsAt(2).Value<uint32_t>(), _1);  // dstNdStride

    const auto& logTileCopy = logs[1];
    ASSERT_EQ(logTileCopy.name, "DataCopy");
    ASSERT_EQ(logTileCopy.args.size(), 3);
    ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(ElementDst));
    ASSERT_EQ(logTileCopy.GetArgsTAt(1).Type(), typeid(ElementAccumulator));
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyCO12DstParams>();
    ASSERT_EQ(dataCopyArg->nSize, _col);
    ASSERT_EQ(dataCopyArg->mSize, _row);
    ASSERT_EQ(dataCopyArg->dstStride, _col);
    ASSERT_EQ(dataCopyArg->srcStride, _row_round);
    ASSERT_EQ(dataCopyArg->quantPre, quantPre);
    ASSERT_TRUE(dataCopyArg->nz2ndEn);
    ASSERT_EQ(dataCopyArg->reluPre, _0);
    ASSERT_EQ(dataCopyArg->unitFlag, unitFlag);
}

// Data-path: L0C → zN (GM)
// Element-type: no-except (float → float)
// Speciality: NoQuant (TLA no-quant DataCopy, channelSplit enabled, nz2nd off)
TEST_P(TileCopyL0CToGmTlaAscend950Test, L0CTozNTestNoQuant)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::zN;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr auto quantPre = CopyL0CToDstQuantMode<Arch::Ascend950, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::GlobalTensor<ElementDst> gmDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToGm950TensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToGm950TensorDst<ElementDst, LayoutDst> tensorGm(gmDst, layoutDst);

    CopyL0CToGmTla<Arch::Ascend950, decltype(tensorL0C), decltype(tensorGm)> copyL0CToGm;
    const uint8_t unitFlag = 0;
    copyL0CToGm(tensorGm, tensorL0C, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logTileCopy = logs[0];
    ASSERT_EQ(logTileCopy.name, "DataCopy");
    ASSERT_EQ(logTileCopy.args.size(), 3);
    ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(ElementDst));
    ASSERT_EQ(logTileCopy.GetArgsTAt(1).Type(), typeid(ElementAccumulator));

    const auto* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyCO12DstParams>();
    ASSERT_EQ(dataCopyArg->nSize, _col);
    ASSERT_EQ(dataCopyArg->mSize, _row);
    ASSERT_EQ(dataCopyArg->dstStride, _dst_row_round);
    ASSERT_EQ(dataCopyArg->srcStride, _row_round);
    ASSERT_EQ(dataCopyArg->quantPre, quantPre);
    ASSERT_FALSE(dataCopyArg->nz2ndEn);
    ASSERT_TRUE(dataCopyArg->channelSplit);
    ASSERT_EQ(dataCopyArg->unitFlag, unitFlag);
}

// Data-path: L0C → RowMajor (GM)
// Element-type: float → half
// Speciality: PerTensor (TLA per-tensor quant via Fixpipe, deqScalar applied)
TEST_P(TileCopyL0CToGmTlaAscend950Test, L0CToRowMajorTestPerTensor)
{
    using ElementAccumulator = float;
    using ElementDst = half;
    using LayoutSrc = detail::LayoutL0C;
    using LayoutDstTag = layout::RowMajor;
    using LayoutDst = detail::TagToLayout_t<ElementDst, LayoutDstTag>;
    constexpr auto quantPre = CopyL0CToDstQuantMode<Arch::Ascend950, ElementAccumulator, ElementDst,
        ScaleGranularity::PER_TENSOR>::VALUE;

    AscendC::LocalTensor<ElementAccumulator> l0cSrc;
    AscendC::GlobalTensor<ElementDst> gmDst;

    setShape<ElementDst>();
    auto layoutSrc = tla::MakeLayoutL0C(_row, _col);
    auto layoutDst = tla::MakeLayout<ElementDst, LayoutDstTag>(_dst_row, _dst_col);
    L0CToGm950TensorSrc<ElementAccumulator, LayoutSrc> tensorL0C(l0cSrc, layoutSrc);
    L0CToGm950TensorDst<ElementDst, LayoutDst> tensorGm(gmDst, layoutDst);

    using CopyL0CToGm = CopyL0CToGmTla<Arch::Ascend950, decltype(tensorL0C), decltype(tensorGm),
        ScaleGranularity::PER_TENSOR>;
    CopyL0CToGm copyL0CToGm(typename CopyL0CToGm::Params{2.0f});
    const uint8_t unitFlag = 0;
    copyL0CToGm(tensorGm, tensorL0C, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logFixpipe = logs[0];
    ASSERT_EQ(logFixpipe.name, "Fixpipe");
    ASSERT_EQ(logFixpipe.args.size(), 3);
    ASSERT_EQ(logFixpipe.GetArgsTAt(0).Type(), typeid(ElementDst));
    ASSERT_EQ(logFixpipe.GetArgsTAt(1).Type(), typeid(ElementAccumulator));

    const auto* fixpipeArg = logFixpipe.GetArgsAt(2).Value<AscendC::FixpipeParamsC310<>>();
    ASSERT_EQ(fixpipeArg->nSize, _col);
    ASSERT_EQ(fixpipeArg->mSize, _row);
    ASSERT_EQ(fixpipeArg->srcStride, _row_round);
    ASSERT_EQ(fixpipeArg->dstStride, _col);
    ASSERT_EQ(fixpipeArg->quantPre, quantPre);
    ASSERT_NE(fixpipeArg->deqScalar, static_cast<uint64_t>(0));
    ASSERT_EQ(fixpipeArg->reluEn, _0);
    ASSERT_EQ(fixpipeArg->unitFlag, unitFlag);
}

INSTANTIATE_TEST_SUITE_P(
    CopyL0CToGmTlaAscend950,
    TileCopyL0CToGmTlaAscend950Test,
    ::testing::Values(
        TestMatrixShape{128U, 256U},
        TestMatrixShape{64U, 128U}
    )
);

#endif // CATLASS_ARCH == 3510
