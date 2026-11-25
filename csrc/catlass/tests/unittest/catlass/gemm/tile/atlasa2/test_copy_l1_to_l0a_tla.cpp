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

#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyL1ToL0ATlaTest : public TileCopyTlaTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isSrcTrans = false, bool isDstTrans = false>
    void setShape()
    {
        const auto& param = GetParam();
        _row = param.row;
        _col = param.col;
        _dst_row = param.row;
        _dst_col = param.col;
        TileCopyTlaTest::_setShape<Element, isSrcTrans, isDstTrans>(_row, _col);
    }
};

using CoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class LayoutSrc>
using TensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, CoordZero, AscendC::TPosition::A1>;
template <class Element, class LayoutDst>
using TensorDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZero, AscendC::TPosition::A2>;

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN → zZ
// Element-type: half
// Speciality: half (non-float per-fractal LoadData, no transpose)
TEST_P(TileCopyL1ToL0ATlaTest, zNTozZTestHalf)
{
    using Element = half;
    using LayoutSrcTag = layout::zN;
    using LayoutDstTag = layout::zZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0aDst;

    setShape<Element>();

    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc);
    TensorDst<Element, LayoutDst> tensorL0A(l0aDst, layoutDst);

    using CopyL1ToL0A = TileCopyTla<Arch::AtlasA2, TensorSrc<Element, LayoutSrc>, TensorDst<Element, LayoutDst>>;
    CopyL1ToL0A copyL1ToL0A;
    copyL1ToL0A(tensorL0A, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();

    ASSERT_EQ(logs.size(), _dst_row_by_fractal);
    for (uint32_t i = 0; i < _dst_row_by_fractal; ++i) {
        const auto& logTileCopy = logs[i];
        ASSERT_EQ(logTileCopy.name, "LoadData");
        ASSERT_EQ(logTileCopy.args.size(), 3);
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), (i * _col_round * C0_NUM_PER_FRACTAL) * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * BYTE_PER_FRACTAL);

        const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(loadDataArg->startIndex, _0);
        ASSERT_EQ(loadDataArg->repeatTimes, _dst_cols_by_fractal);
        ASSERT_EQ(loadDataArg->srcStride, _row_round * ELE_NUM_PER_C0 / ELE_NUM_PER_FRACTAL);
        ASSERT_EQ(loadDataArg->sid, _0);
        ASSERT_EQ(loadDataArg->dstGap, _0);
        ASSERT_EQ(loadDataArg->ifTranspose, _0);
        ASSERT_EQ(loadDataArg->addrMode, _0);
    }
}

// Data-path: zN → zZ
// Element-type: no-except (float)
// Speciality: float (SetFmatrix + LoadData3DParamsV2)
TEST_P(TileCopyL1ToL0ATlaTest, zNTozZTestFloat)
{
    using Element = float;
    using LayoutSrcTag = layout::zN;
    using LayoutDstTag = layout::zZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0aDst;

    setShape<Element>();

    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc);
    TensorDst<Element, LayoutDst> tensorL0A(l0aDst, layoutDst);

    using CopyL1ToL0A = TileCopyTla<Arch::AtlasA2, TensorSrc<Element, LayoutSrc>, TensorDst<Element, LayoutDst>>;
    CopyL1ToL0A copyL1ToL0A;
    copyL1ToL0A(tensorL0A, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();

    ASSERT_EQ(logs.size(), 2);

    const auto& logSetFmatrix = logs[0];
    ASSERT_EQ(logSetFmatrix.name, "SetFmatrix");
    ASSERT_EQ(logSetFmatrix.args[0].Value<uint16_t>()[0], 1);
    ASSERT_EQ(logSetFmatrix.args[1].Value<uint16_t>()[0], _row_round);

    const auto& logLoadData = logs[1];
    ASSERT_EQ(logLoadData.name, "LoadData");
    ASSERT_EQ(logLoadData.args.size(), 3);
    ASSERT_EQ(logLoadData.GetArgsTAt(0).Type(), typeid(Element));
    ASSERT_EQ(logLoadData.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logLoadData.GetArgsAt(1).GetInstAddr(), 0);

    const auto* loadDataArg = logLoadData.GetArgsAt(2).Value<AscendC::LoadData3DParamsV2<Element>>();
    ASSERT_EQ(loadDataArg->kExtension, _dst_col_round);
    ASSERT_EQ(loadDataArg->mExtension, _dst_row_round);
    ASSERT_EQ(loadDataArg->channelSize, _col_round);
}

// ============================================================================
// Testsuite from **nZ**
// ============================================================================

// Data-path: nZ → zZ
// Element-type: half
// Speciality: half (non-float/non-int8 LoadData with ifTranspose)
TEST_P(TileCopyL1ToL0ATlaTest, nZTozZTestHalf)
{
    using Element = half;
    using LayoutSrcTag = layout::nZ;
    using LayoutDstTag = layout::zZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0aDst;

    setShape<Element, true>();

    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc);
    TensorDst<Element, LayoutDst> tensorL0A(l0aDst, layoutDst);

    using CopyL1ToL0A = TileCopyTla<Arch::AtlasA2, TensorSrc<Element, LayoutSrc>, TensorDst<Element, LayoutDst>>;
    CopyL1ToL0A copyL1ToL0A;
    copyL1ToL0A(tensorL0A, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();

    ASSERT_EQ(logs.size(), _dst_row_by_fractal);
    for (uint32_t i = 0; i < _dst_row_by_fractal; ++i) {
        const auto& logTileCopy = logs[i];
        ASSERT_EQ(logTileCopy.name, "LoadData");
        ASSERT_EQ(logTileCopy.args.size(), 3);
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), (i * _dst_col_round * C0_NUM_PER_FRACTAL) * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), (i * _col_round * C0_NUM_PER_FRACTAL) * sizeof(Element));

        const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(loadDataArg->startIndex, _0);
        ASSERT_EQ(loadDataArg->repeatTimes, _dst_cols_by_fractal);
        ASSERT_EQ(loadDataArg->srcStride, _1);
        ASSERT_EQ(loadDataArg->sid, _0);
        ASSERT_EQ(loadDataArg->dstGap, _0);
        ASSERT_EQ(loadDataArg->ifTranspose, true);
        ASSERT_EQ(loadDataArg->addrMode, _0);
    }
}

// Data-path: nZ → zZ
// Element-type: no-except (float)
// Speciality: float (SetFmatrix + LoadData3DParamsV2 enTranspose)
TEST_P(TileCopyL1ToL0ATlaTest, nZTozZTestFloat)
{
    using Element = float;
    using LayoutSrcTag = layout::nZ;
    using LayoutDstTag = layout::zZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0aDst;

    setShape<Element, true>();

    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc);
    TensorDst<Element, LayoutDst> tensorL0A(l0aDst, layoutDst);

    using CopyL1ToL0A = TileCopyTla<Arch::AtlasA2, TensorSrc<Element, LayoutSrc>, TensorDst<Element, LayoutDst>>;
    CopyL1ToL0A copyL1ToL0A;
    copyL1ToL0A(tensorL0A, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();

    ASSERT_EQ(logs.size(), 2);

    const auto& logSetFmatrix = logs[0];
    ASSERT_EQ(logSetFmatrix.name, "SetFmatrix");
    ASSERT_EQ(logSetFmatrix.args[0].Value<uint16_t>()[0], 1);
    ASSERT_EQ(logSetFmatrix.args[1].Value<uint16_t>()[0], _col_round);

    const auto& logLoadData = logs[1];
    ASSERT_EQ(logLoadData.name, "LoadData");
    ASSERT_EQ(logLoadData.args.size(), 3);
    ASSERT_EQ(logLoadData.GetArgsTAt(0).Type(), typeid(Element));
    ASSERT_EQ(logLoadData.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logLoadData.GetArgsAt(1).GetInstAddr(), 0);

    const auto* loadDataArg = logLoadData.GetArgsAt(2).Value<AscendC::LoadData3DParamsV2<Element>>();
    ASSERT_EQ(loadDataArg->kExtension, _dst_row_round);
    ASSERT_EQ(loadDataArg->mExtension, _dst_col_round);
    ASSERT_EQ(loadDataArg->enTranspose, true);
    ASSERT_EQ(loadDataArg->channelSize, _row_round);
}

// Data-path: nZ → zZ
// Element-type: int8
// Speciality: int8 (LoadDataWithTranspose, LoadData2dTransposeParams)
TEST_P(TileCopyL1ToL0ATlaTest, nZTozZTestInt8)
{
    using Element = int8_t;
    using LayoutSrcTag = layout::nZ;
    using LayoutDstTag = layout::zZ;
    using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
    using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;

    AscendC::LocalTensor<Element> l1Src;
    AscendC::LocalTensor<Element> l0aDst;

    setShape<Element, true>();

    auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
    auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);
    TensorSrc<Element, LayoutSrc> tensorL1(l1Src, layoutSrc);
    TensorDst<Element, LayoutDst> tensorL0A(l0aDst, layoutDst);

    using CopyL1ToL0A = TileCopyTla<Arch::AtlasA2, TensorSrc<Element, LayoutSrc>, TensorDst<Element, LayoutDst>>;
    CopyL1ToL0A copyL1ToL0A;
    copyL1ToL0A(tensorL0A, tensorL1);

    auto logs = AscendCCallLogger::Instance().GetLogs();

    ASSERT_EQ(logs.size(), _rows_by_fractal);
    for (uint32_t i = 0; i < _rows_by_fractal; ++i) {
        const auto& logTileCopy = logs[i];
        ASSERT_EQ(logTileCopy.name, "LoadDataWithTranspose");
        ASSERT_EQ(logTileCopy.args.size(), 3);
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));

        const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2dTransposeParams>();
        ASSERT_EQ(loadDataArg->startIndex, _0);
        ASSERT_EQ(loadDataArg->repeatTimes, _dst_cols_by_fractal);
        ASSERT_EQ(loadDataArg->srcStride, _1);
        ASSERT_EQ(loadDataArg->dstGap, _0);
        ASSERT_EQ(loadDataArg->dstFracGap, _dst_cols_by_fractal - 1);
    }
}

INSTANTIATE_TEST_SUITE_P(CopyL1ToL0ATla, TileCopyL1ToL0ATlaTest, ::testing::Values(TestMatrixShape{128U, 256U}));

#endif // CATLASS_ARCH == 2201
