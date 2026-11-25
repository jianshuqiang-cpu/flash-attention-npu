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

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

using CoordZeroTla = tla::Coord<tla::Int<0>, tla::Int<0>>;

// TestCase for GM->L1 TileCopy Utilities
class TileCopyGmToL1TestAtlasA2 : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isTrans = false>
    void setShape() {
        uint32_t row = GetParam().row;
        uint32_t col = GetParam().col;
        _setShape<Element, isTrans>(row, col);
    }

    template <class Element>
    void BaseCheck(AscendCCallLog const &logTileCopy){
        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }
};

// ============================================================================
// Testsuite from **RowMajor**
// ============================================================================

// Data-path: RowMajor → zN
// Element-type: no-except (float)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestBasicNd2Nz)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    if (layoutSrc.stride(0) >= STRIDE_LIMIT) {
        GTEST_SKIP() << "RowMajor stride(0) >= STRIDE_LIMIT, skip test";
    }

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: basic-B1 (3-template GmTypeDst at TPosition::B1, contiguous Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestBasicB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using GmTypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, GmTypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: basic-A1 (3-template L1Type at TPosition::A1, single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestBasicA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(0));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → Vector
// Element-type: no-except (float)
// Speciality: basic (dst VectorLayout A1, single DataCopy by fractal cols)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorToVectorTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::VectorLayout;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::GM>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_row, _col};
    LayoutDst layoutDst{_col};

    // Call this datacopy
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(repeatParams->blockCount, 1);
    ASSERT_EQ(repeatParams->blockLen, _cols_by_fractal);
    ASSERT_EQ(repeatParams->srcStride, 0);
    ASSERT_EQ(repeatParams->dstStride, 0);
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: contiguous-A1 (shape == stride, single whole-tile DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorToRowMajorTestContiguousA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_row, _col};
    LayoutDst layoutDst{_row, _col};

    ASSERT_EQ(layoutSrc.shape(1), layoutSrc.stride(0));
    ASSERT_EQ(layoutDst.shape(1), layoutDst.stride(0));

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    ASSERT_EQ(logs[0].name, "DataCopy");
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: long-stride (src stride ≥ STRIDE_LIMIT, row-by-row Nd2Nz fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    static_assert(std::is_same_v<LayoutSrc, layout::RowMajor> && std::is_same_v<LayoutDst, layout::zN>,
        "This testsuite is RowMajor->zN");
    static_assert(!std::is_same_v<Element, AscendC::int4b_t>, "It is not covered where datatype is `int4b_t`");

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_FALSE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    ASSERT_GE(_very_long_stride, STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();

    ASSERT_EQ(logs.size(), _row);

    for (int i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_C0);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: long-stride (3-template B1 overload, row-by-row Nd2Nz fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestLongStrideTripleArgsB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    static_assert(std::is_same_v<LayoutSrc, layout::RowMajor> && std::is_same_v<LayoutDst, layout::zN>,
        "This testsuite is RowMajor->zN");
    static_assert(!std::is_same_v<Element, AscendC::int4b_t>, "It is not covered where datatype is `int4b_t`");

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_FALSE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    ASSERT_GE(_very_long_stride, STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();

    ASSERT_EQ(logs.size(), _row);

    for (int i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_C0);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: long-stride-A1 (3-template A1 overload, per-row Nd2Nz fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestLongStrideA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;
    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);

    for (uint32_t i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: PaddingRowMajor → zN
// Element-type: no-except (float)
// Speciality: basic (padded RowMajor, single Nd2Nz using orgShape)
TEST_P(TileCopyGmToL1TestAtlasA2, PaddingRowMajorTozNTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::PaddingRowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_row, _col, 16U, 16U};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->dValue, layoutSrc.orgShape(1));
    ASSERT_EQ(nd2nzArg->nValue, layoutSrc.orgShape(0));
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(0));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: PaddingRowMajor → zN
// Element-type: half
// Speciality: interval-data-copy (padded src, per-row DataCopy over orgShape)
TEST_P(TileCopyGmToL1TestAtlasA2, PaddingRowMajorTozNTestIntervalDataCopyHalf)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::PaddingRowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_row, _col, 16U, 16U};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), layoutSrc.orgShape(0));

    for (uint32_t i = 0; i < layoutSrc.orgShape(0); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(0) * sizeof(Element));

        const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(repeatParams->blockCount, CeilDiv(layoutSrc.orgShape(1), ELE_NUM_PER_C0));
        ASSERT_EQ(repeatParams->srcStride, _0);
    }
}

// Data-path: RowMajor → zN
// Element-type: half
// Speciality: interval-data-copy (CopyGmToL1IntervalDataCopy, per-row DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestIntervalDataCopyHalf)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);

    for (uint32_t i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(0) * sizeof(Element));

        const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(repeatParams->blockCount, CeilDiv(_col, ELE_NUM_PER_C0));
        ASSERT_EQ(repeatParams->srcStride, _0);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: multi-param-short-stride (explicit ndNum params, short srcNdMatrixStride single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestMultiParamShortStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    uint32_t ndNum = 2;
    uint32_t srcNdMatrixStride = 1024;
    uint32_t dstNzNStride = 8;
    uint32_t dstNzMatrixStride = 128;
    uint32_t dstNzC0Stride = 16;

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc,
               ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, ndNum);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdMatrixStride);
    ASSERT_EQ(nd2nzArg->dstNzNStride, dstNzNStride);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, dstNzMatrixStride);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: multi-param-long-stride (explicit ndNum params, long srcNdMatrixStride per-nd split)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestMultiParamLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    uint32_t ndNum = 3;
    uint32_t srcNdMatrixStride = STRIDE_LIMIT + 100;
    uint32_t dstNzNStride = 8;
    uint32_t dstNzMatrixStride = 128;
    uint32_t dstNzC0Stride = 16;

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc,
               ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), ndNum);

    for (uint32_t i = 0; i < ndNum; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * srcNdMatrixStride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _row);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcDValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzNStride, dstNzNStride);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: one-row-GMMPTD (row == 1, CopyGmToL1GMMPTD single DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestOneRowGMMPTD)
{
    if (GetParam().row != 1U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    ASSERT_EQ(layoutSrc.shape(0), 1);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    ASSERT_EQ(logs.size(), 1);

    auto logTileCopyGmTensor = logTileCopy.GetArgsAt(1).RawValue();
    auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(0).RawValue();
    ASSERT_EQ(logTileCopyGmTensor, &gmTensor);
    ASSERT_EQ(logTileCopyL1Tensor, &l1Tensor);

    const AscendC::DataCopyParams* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(dataCopyArg->blockCount, CeilDiv(_col, ELE_NUM_PER_C0));
    ASSERT_EQ(dataCopyArg->blockLen, _1);
    ASSERT_EQ(dataCopyArg->srcStride, _0);
    ASSERT_EQ(dataCopyArg->dstStride, C0_NUM_PER_FRACTAL - _1);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: multi-row-GMMPTD (row > 1, CopyGmToL1GMMPTD single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestMultiRowGMMPTD)
{
    if (GetParam().row <= 1U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    if (GetParam().col == ELE_NUM_PER_C0) { GTEST_SKIP(); }

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_GT(layoutSrc.shape(0), 1);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: GMMPTD-long-stride (src stride ≥ STRIDE_LIMIT, per-row Nd2Nz fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestGMMPTDLongStride)
{
    if (GetParam().row <= 1U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_GT(layoutSrc.shape(0), 1);
    ASSERT_GE(layoutSrc.stride(0), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);

    for (uint32_t i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round); // per 32B
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: GMMPTD-multi-param-short (explicit ndNum params, short srcNdMatrixStride single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestGMMPTDMultiParamShort)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    uint32_t ndNum = 2;
    uint32_t srcNdMatrixStride = 512;
    uint32_t dstNzNStride = 4;
    uint32_t dstNzMatrixStride = 64;
    uint32_t dstNzC0Stride = 16;

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc,
               ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, ndNum);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdMatrixStride);
    ASSERT_EQ(nd2nzArg->dstNzNStride, dstNzNStride);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, dstNzMatrixStride);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: GMMPTD-multi-param-long (explicit ndNum params, long srcNdMatrixStride per-nd split)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestGMMPTDMultiParamLong)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    uint32_t ndNum = 3;
    uint32_t srcNdMatrixStride = STRIDE_LIMIT + 200;
    uint32_t dstNzNStride = 4;
    uint32_t dstNzMatrixStride = 64;
    uint32_t dstNzC0Stride = 16;

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc,
               ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), ndNum);

    for (uint32_t i = 0; i < ndNum; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * srcNdMatrixStride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _row);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcDValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzNStride, dstNzNStride);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: GMMPTD-contiguous (C0-aligned col == ELE_NUM_PER_C0, single DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestGMMPTDContiguous)
{
    if (GetParam().row <= 1U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1GMMPTD<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    LayoutSrc layoutSrc{_row, ELE_NUM_PER_C0, ELE_NUM_PER_C0};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, ELE_NUM_PER_C0);
    _setShape<Element>(_row, ELE_NUM_PER_C0);

    ASSERT_EQ(layoutSrc.shape(1), ELE_NUM_PER_C0);
    ASSERT_EQ(layoutSrc.stride(0), ELE_NUM_PER_C0);
    ASSERT_GT(layoutSrc.shape(0), 1);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    ASSERT_EQ(logTileCopy.name, "DataCopy");
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: dynamic-optimized-few-rows (row ≤ 16, per-row DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestDynamicOptimizedFewRows)
{
    if (GetParam().row > 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_LE(layoutSrc.shape(0), 16);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);

    for (uint32_t i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(0) * sizeof(Element));

        const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(repeatParams->blockCount, CeilDiv(_col, ELE_NUM_PER_C0));
        ASSERT_EQ(repeatParams->srcStride, _0);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: dynamic-optimized-nd2nz (row > 16, short stride, single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestDynamicOptimizedNd2Nz)
{
    if (GetParam().row <= 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    if (layoutSrc.shape(1) == ELE_NUM_PER_C0 && layoutSrc.stride(0) == ELE_NUM_PER_C0) { GTEST_SKIP(); }

    ASSERT_GT(layoutSrc.shape(0), 16);
    ASSERT_LT(layoutSrc.stride(0), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(0));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: dynamic-optimized-contiguous (row > 16, C0-aligned, single DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestDynamicOptimizedContiguous)
{
    if (GetParam().row <= 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    LayoutSrc layoutSrc{_row, ELE_NUM_PER_C0, ELE_NUM_PER_C0};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, ELE_NUM_PER_C0);
    _setShape<Element>(_row, ELE_NUM_PER_C0);

    ASSERT_GT(layoutSrc.shape(0), 16);
    ASSERT_EQ(layoutSrc.shape(1), ELE_NUM_PER_C0);
    ASSERT_EQ(layoutSrc.stride(0), ELE_NUM_PER_C0);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    ASSERT_EQ(logTileCopy.name, "DataCopy");
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: dynamic-optimized-long-stride (row > 16, stride ≥ STRIDE_LIMIT, per-row Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestDynamicOptimizedLongStride)
{
    if (GetParam().row <= 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_GT(layoutSrc.shape(0), 16);
    ASSERT_GE(layoutSrc.stride(0), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);

    for (uint32_t i = 0; i < _row; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: datacopy-params-A1 (ldm != cols, per-block DataCopyParams with gaps)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorToRowMajorTestDataCopyParamsA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    uint32_t srcLdm = _col + ELE_NUM_PER_BLK;
    uint32_t dstLdm = _col + ELE_NUM_PER_BLK;
    LayoutSrc layoutSrc{_row, _col, srcLdm};
    LayoutDst layoutDst{_row, _col, dstLdm};

    ASSERT_NE(layoutSrc.shape(1), layoutSrc.stride(0));

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_GE(logs.size(), 1);
    ASSERT_EQ(logs[0].name, "DataCopy");

    const auto* repeatParams = logs[0].GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(repeatParams->blockLen, _col / ELE_NUM_PER_BLK);
    ASSERT_EQ(repeatParams->srcStride, (srcLdm - _col) / ELE_NUM_PER_BLK);
    ASSERT_EQ(repeatParams->dstStride, (dstLdm - _col) / ELE_NUM_PER_BLK);
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: per-row-fallback-A1 (huge src stride, per-row DataCopy fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorToRowMajorTestPerRowFallbackA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;

    setShape<Element>();
    uint32_t hugeStride = STRIDE_LIMIT * 16;
    LayoutSrc layoutSrc{_row, _col, hugeStride};
    LayoutDst layoutDst{_row, _col, _col};
    ASSERT_NE(layoutSrc.shape(1), layoutSrc.stride(0));

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);
    for (uint32_t i = 0; i < _row; ++i) {
        ASSERT_EQ(logs[i].name, "DataCopy");
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * hugeStride * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * layoutDst.stride(0) * sizeof(Element));
        ASSERT_EQ(*logs[i].GetArgsAt(2).Value<uint32_t>(), _col);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: tla-long-stride (TileCopyTla, src stride ≥ STRIDE_LIMIT, per-row split)
TEST_P(TileCopyGmToL1TestAtlasA2, RowMajorTozNTestTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    int64_t bigStride = STRIDE_LIMIT + 1;
    auto baseSrc = tla::MakeLayout<Element, layout::RowMajor>(_row, _col);
    auto layoutSrc = tla::MakeLayout(baseSrc.shape(),
        tla::MakeStride(bigStride, tla::Int<1>{}), baseSrc.originShape());
    auto layoutDst = tla::MakeLayout<Element, layout::zN>(_row, _col);
    using LayoutSrc = decltype(layoutSrc);
    using LayoutDst = decltype(layoutDst);

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZeroTla, AscendC::TPosition::GM> tensorGm(gmTensor, layoutSrc);
    tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZeroTla, AscendC::TPosition::A1> tensorL1(l1Tensor, layoutDst);

    TileCopyTla<ArchTag, decltype(tensorGm), decltype(tensorL1)> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t dstNzC0Stride = tla::get<1, 1>(layoutDst.stride()) / ELE_NUM_PER_C0;
    ASSERT_EQ(logs.size(), _row);
    for (uint32_t i = 0; i < _row; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * ELE_NUM_PER_C0 * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * (uint32_t)bigStride * sizeof(Element));
        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// ============================================================================
// Testsuite from **ColumnMajor**
// ============================================================================

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: basic (short stride, single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_LT(layoutSrc.stride(1), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _col);
    ASSERT_EQ(nd2nzArg->dValue, _row);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(1));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: basic-B1 (3-template L1Type at TPosition::B1, single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestBasicB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _col);
    ASSERT_EQ(nd2nzArg->dValue, _row);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(1));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: basic-A1 (3-template L1Type at TPosition::A1, single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestBasicA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _col);
    ASSERT_EQ(nd2nzArg->dValue, _row);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(1));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: long-stride (src stride ≥ STRIDE_LIMIT, per-col Nd2Nz fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_GE(layoutSrc.stride(1), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _col);

    for (uint32_t i = 0; i < _col; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: long-stride (3-template B1 overload, per-col Nd2Nz fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestLongStrideTripleArgsB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_GE(layoutSrc.stride(1), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _col);

    for (uint32_t i = 0; i < _col; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: long-stride (3-template A1 overload, per-col Nd2Nz fallback)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestLongStrideTripleArgsA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_GE(layoutSrc.stride(1), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _col);

    for (uint32_t i = 0; i < _col; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * _very_long_stride * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: PaddingColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: basic (padded ColumnMajor, single Nd2Nz using orgShape)
TEST_P(TileCopyGmToL1TestAtlasA2, PaddingColumnMajorTonZTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::PaddingColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc{_row, _col, 16U, 16U};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->dValue, layoutSrc.orgShape(0));
    ASSERT_EQ(nd2nzArg->nValue, layoutSrc.orgShape(1));
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(2));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: PaddingColumnMajor → nZ
// Element-type: half
// Speciality: interval-data-copy (padded src, per-col DataCopy over orgShape)
TEST_P(TileCopyGmToL1TestAtlasA2, PaddingColumnMajorTonZTestIntervalDataCopyHalf)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::PaddingColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc{_row, _col, 16U, 16U};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), layoutSrc.orgShape(1));

    for (uint32_t i = 0; i < layoutSrc.orgShape(1); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(2) * sizeof(Element));

        const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(repeatParams->blockCount, CeilDiv(layoutSrc.orgShape(0), ELE_NUM_PER_C0));
        ASSERT_EQ(repeatParams->srcStride, _0);
    }
}

// Data-path: ColumnMajor → nZ
// Element-type: half
// Speciality: interval-data-copy (CopyGmToL1IntervalDataCopy, per-col DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestIntervalDataCopyHalf)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _col);

    for (uint32_t i = 0; i < _col; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(1) * sizeof(Element));

        const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(repeatParams->blockCount, CeilDiv(_row, ELE_NUM_PER_C0));
        ASSERT_EQ(repeatParams->srcStride, _0);
    }
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: dynamic-optimized-few-cols (col ≤ 16, per-col DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestDynamicOptimizedFewCols)
{
    if (GetParam().col > 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_LE(layoutSrc.shape(1), 16);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _col);

    for (uint32_t i = 0; i < _col; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(repeatParams->blockCount, CeilDiv(_row, ELE_NUM_PER_C0));
        ASSERT_EQ(repeatParams->srcStride, _0);
    }
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: dynamic-optimized-one-fractal-col (row == ELE_NUM_PER_C0, single whole-tile DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestDynamicOptimizedOneFracCol)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    if (GetParam().row != ELE_NUM_PER_C0) {GTEST_SKIP(); }

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    ASSERT_EQ(layoutSrc.shape(0), ELE_NUM_PER_C0); // To fit current situation

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    auto logTileCopy = logs[0];

    ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);
    const uint32_t* dataCount = logTileCopy.GetArgsAt(2).Value<uint32_t>();
    ASSERT_EQ(*dataCount, _row * _col);
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: dynamic-optimized-nd2nz (col > 16, short stride, single Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestDynamicOptimizedNd2Nz)
{
    if (GetParam().col <= 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);

    if (layoutSrc.shape(0) == ELE_NUM_PER_C0 && layoutSrc.stride(1) == ELE_NUM_PER_C0) { GTEST_SKIP(); }

    ASSERT_GT(layoutSrc.shape(1), 16);
    ASSERT_LT(layoutSrc.stride(1), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _col);
    ASSERT_EQ(nd2nzArg->dValue, _row);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(1));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: dynamic-optimized-long-stride (col > 16, stride ≥ STRIDE_LIMIT, per-col Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestDynamicOptimizedLongStride)
{
    if (GetParam().col <= 16U) { GTEST_SKIP(); }

    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_GT(layoutSrc.shape(1), 16);
    ASSERT_GE(layoutSrc.stride(1), STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _col);

    for (uint32_t i = 0; i < _col; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _col_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: tla-long-stride (TileCopyTla, src stride ≥ STRIDE_LIMIT, per-col split)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    int64_t bigStride = STRIDE_LIMIT + 1;
    auto baseSrc = tla::MakeLayout<Element, layout::ColumnMajor>(_row, _col);
    auto layoutSrc = tla::MakeLayout(baseSrc.shape(),
        tla::MakeStride(tla::Int<1>{}, bigStride), baseSrc.originShape());
    auto layoutDst = tla::MakeLayout<Element, layout::nZ>(_row, _col);
    using LayoutSrc = decltype(layoutSrc);
    using LayoutDst = decltype(layoutDst);

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZeroTla, AscendC::TPosition::GM> tensorGm(gmTensor, layoutSrc);
    tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZeroTla, AscendC::TPosition::A1> tensorL1(l1Tensor, layoutDst);

    TileCopyTla<ArchTag, decltype(tensorGm), decltype(tensorL1)> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t dstNzC0Stride = tla::get<0, 1>(layoutDst.stride()) / ELE_NUM_PER_C0;
    ASSERT_EQ(logs.size(), _col);
    for (uint32_t i = 0; i < _col; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * ELE_NUM_PER_C0 * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * (uint32_t)bigStride * sizeof(Element));
        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: sparse-tla-long-stride (TileCopySparseTla, src stride ≥ STRIDE_LIMIT, per-col split)
TEST_P(TileCopyGmToL1TestAtlasA2, ColumnMajorTonZTestSparseTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    int64_t bigStride = STRIDE_LIMIT + 1;
    auto baseSrc = tla::MakeLayout<Element, layout::ColumnMajor>(_row, _col);
    auto layoutSrc = tla::MakeLayout(baseSrc.shape(),
        tla::MakeStride(tla::Int<1>{}, bigStride), baseSrc.originShape());
    auto layoutDst = tla::MakeLayout<Element, layout::nZ>(_row, _col);
    using LayoutSrc = decltype(layoutSrc);
    using LayoutDst = decltype(layoutDst);

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZeroTla, AscendC::TPosition::GM> tensorGm(gmTensor, layoutSrc);
    tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZeroTla, AscendC::TPosition::A1> tensorL1(l1Tensor, layoutDst);

    TileCopySparseTla<ArchTag, decltype(tensorGm), decltype(tensorL1)> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t dstNzC0Stride = tla::get<0, 1>(layoutDst.stride()) / ELE_NUM_PER_C0;
    ASSERT_EQ(logs.size(), _col);
    for (uint32_t i = 0; i < _col; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * ELE_NUM_PER_C0 * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * (uint32_t)bigStride * sizeof(Element));
        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, dstNzC0Stride);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// ============================================================================
// Testsuite from **Vector**
// ============================================================================

// Data-path: Vector → zN
// Element-type: no-except (float)
// Speciality: basic-A1 (VectorLayout src to zN A1, single-row Nd2Nz)
TEST_P(TileCopyGmToL1TestAtlasA2, VectorTozNTestBasicA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::zN;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_col};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(1U, _col);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _1);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: Vector → Vector
// Element-type: no-except (float)
// Speciality: basic (VectorLayout GM to A1, single DataCopy by fractal cols)
TEST_P(TileCopyGmToL1TestAtlasA2, VectorToVectorTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    using GmTypeSrc = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::GM>;
    using L1TypeDst = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmTypeSrc, L1TypeDst> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc{_col};
    LayoutDst layoutDst{_col};

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(repeatParams->blockCount, 1);
    ASSERT_EQ(repeatParams->blockLen, _col / ELE_NUM_PER_C0);
    ASSERT_EQ(repeatParams->srcStride, 0);
    ASSERT_EQ(repeatParams->dstStride, 0);
}

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN → zN
// Element-type: no-except (float)
// Speciality: short-stride (fractal-aligned, single DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, zNTozNTestShortStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_LT(layoutSrc.stride(3) / ELE_NUM_PER_C0, STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
    uint32_t blockLen = layoutSrc.orgShape(0);

    const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(repeatParams->blockCount, blockCount);
    ASSERT_EQ(repeatParams->blockLen, blockLen);
    ASSERT_EQ(repeatParams->srcStride, layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen);
    ASSERT_EQ(repeatParams->dstStride, layoutDst.stride(3) / ELE_NUM_PER_C0 - blockLen);
}

// Data-path: zN → zN
// Element-type: no-except (float)
// Speciality: no-tla-long-stride (outer-col stride ≥ STRIDE_LIMIT, per-fractal-col split)
TEST_P(TileCopyGmToL1TestAtlasA2, zNTozNTestNoTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    setShape<Element>();
    int64_t longColByFractal = (int64_t)(STRIDE_LIMIT + 1) * ELE_NUM_PER_C0;
    LayoutSrc layoutSrc(_row, _col, C0_NUM_PER_FRACTAL, _row_round / C0_NUM_PER_FRACTAL,
                        ELE_NUM_PER_C0, _col_round / ELE_NUM_PER_C0,
                        ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL, 1, longColByFractal);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_GE(layoutSrc.stride(3) / ELE_NUM_PER_C0, STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
    uint32_t blockLen = layoutSrc.orgShape(0);
    ASSERT_EQ(logs.size(), blockCount);
    for (uint32_t i = 0; i < blockCount; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * layoutDst.stride(3) * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * layoutSrc.stride(3) * sizeof(Element));
        const auto* dataCopyArg = logs[i].GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyArg->blockCount, _1);
        ASSERT_EQ(dataCopyArg->blockLen, blockLen);
        ASSERT_EQ(dataCopyArg->srcStride, _0);
        ASSERT_EQ(dataCopyArg->dstStride, _0);
    }
}

// Data-path: zN → zN
// Element-type: no-except (float)
// Speciality: tla-long-stride (TileCopyTla, long outer-col stride, per-fractal-col split)
TEST_P(TileCopyGmToL1TestAtlasA2, zNTozNTestTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    setShape<Element>();
    int64_t bigOuterCol = (int64_t)(STRIDE_LIMIT + 1) * ELE_NUM_PER_C0;
    auto baseSrc = tla::MakeLayout<Element, layout::zN>(_row, _col);
    auto layoutSrc = tla::MakeLayout(baseSrc.shape(),
        tla::MakeStride(tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_FRACTAL>{}),
                        tla::MakeStride(tla::Int<1>{}, bigOuterCol)),
        baseSrc.originShape());
    auto layoutDst = tla::MakeLayout<Element, layout::zN>(_row, _col);
    using LayoutSrc = decltype(layoutSrc);
    using LayoutDst = decltype(layoutDst);

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZeroTla, AscendC::TPosition::GM> tensorGm(gmTensor, layoutSrc);
    tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZeroTla, AscendC::TPosition::A1> tensorL1(l1Tensor, layoutDst);

    TileCopyTla<ArchTag, decltype(tensorGm), decltype(tensorL1)> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(_col);
    uint32_t blockLen = _row;
    uint32_t srcOuterStrideCol = static_cast<uint32_t>(bigOuterCol);
    uint32_t dstOuterStrideCol = tla::get<1, 1>(layoutDst.stride());
    ASSERT_EQ(logs.size(), blockCount);
    for (uint32_t i = 0; i < blockCount; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * dstOuterStrideCol * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * srcOuterStrideCol * sizeof(Element));
        const auto* dataCopyArg = logs[i].GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyArg->blockCount, _1);
        ASSERT_EQ(dataCopyArg->blockLen, blockLen);
        ASSERT_EQ(dataCopyArg->srcStride, _0);
        ASSERT_EQ(dataCopyArg->dstStride, _0);
    }
}

// Data-path: zN → zN
// Element-type: no-except (float)
// Speciality: tla-ext-long-stride (TileCopyTlaExt with actualShape, per-fractal-col split)
TEST_P(TileCopyGmToL1TestAtlasA2, zNTozNTestTlaExtLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    setShape<Element>();
    int64_t bigOuterCol = (int64_t)(STRIDE_LIMIT + 1) * ELE_NUM_PER_C0;
    auto baseSrc = tla::MakeLayout<Element, layout::zN>(_row, _col);
    auto layoutSrc = tla::MakeLayout(baseSrc.shape(),
        tla::MakeStride(tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_FRACTAL>{}),
                        tla::MakeStride(tla::Int<1>{}, bigOuterCol)),
        baseSrc.originShape());
    auto layoutDst = tla::MakeLayout<Element, layout::zN>(_row, _col);
    using LayoutSrc = decltype(layoutSrc);
    using LayoutDst = decltype(layoutDst);

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZeroTla, AscendC::TPosition::GM> tensorGm(gmTensor, layoutSrc);
    tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZeroTla, AscendC::TPosition::A1> tensorL1(l1Tensor, layoutDst);

    tla::Shape<uint32_t, uint32_t> actualShape{_row, _col};
    TileCopyTlaExt<ArchTag, decltype(tensorGm), decltype(tensorL1), layout::zN, layout::zN> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm, actualShape);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(_col);
    uint32_t blockLen = _row;
    uint32_t srcOuterStrideCol = static_cast<uint32_t>(bigOuterCol);
    uint32_t dstOuterStrideCol = tla::get<1, 1>(layoutDst.stride());
    ASSERT_EQ(logs.size(), blockCount);
    for (uint32_t i = 0; i < blockCount; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * dstOuterStrideCol * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * srcOuterStrideCol * sizeof(Element));
        const auto* dataCopyArg = logs[i].GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyArg->blockCount, _1);
        ASSERT_EQ(dataCopyArg->blockLen, blockLen);
        ASSERT_EQ(dataCopyArg->srcStride, _0);
        ASSERT_EQ(dataCopyArg->dstStride, _0);
    }
}

// ============================================================================
// Testsuite from **nZ**
// ============================================================================

// Data-path: nZ → nZ
// Element-type: no-except (float)
// Speciality: short-stride (fractal-aligned, single DataCopy)
TEST_P(TileCopyGmToL1TestAtlasA2, nZTonZTestShortStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    ASSERT_LT(layoutSrc.stride(1) / ELE_NUM_PER_C0, STRIDE_LIMIT);

    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);
    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);

    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0));
    uint32_t blockLen = layoutSrc.orgShape(1);

    const auto* repeatParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(repeatParams->blockCount, blockCount);
    ASSERT_EQ(repeatParams->blockLen, blockLen);
    ASSERT_EQ(repeatParams->srcStride, layoutSrc.stride(1) / ELE_NUM_PER_C0 - blockLen);
    ASSERT_EQ(repeatParams->dstStride, layoutDst.stride(1) / ELE_NUM_PER_C0 - blockLen);
}

// Data-path: nZ → nZ
// Element-type: no-except (float)
// Speciality: no-tla-long-stride (outer-row stride ≥ STRIDE_LIMIT, per-fractal-row split)
TEST_P(TileCopyGmToL1TestAtlasA2, nZTonZTestNoTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    setShape<Element, true>();
    int64_t longRowByFractal = (int64_t)(STRIDE_LIMIT + 1) * ELE_NUM_PER_C0;
    LayoutSrc layoutSrc(_row, _col, ELE_NUM_PER_C0, _row_round / ELE_NUM_PER_C0,
                        C0_NUM_PER_FRACTAL, _col_round / C0_NUM_PER_FRACTAL,
                        1, longRowByFractal, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_GE(layoutSrc.stride(1) / ELE_NUM_PER_C0, STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0));
    uint32_t blockLen = layoutSrc.orgShape(1);
    ASSERT_EQ(logs.size(), blockCount);
    for (uint32_t i = 0; i < blockCount; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * layoutDst.stride(1) * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * layoutSrc.stride(1) * sizeof(Element));
        const auto* dataCopyArg = logs[i].GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyArg->blockCount, _1);
        ASSERT_EQ(dataCopyArg->blockLen, blockLen);
        ASSERT_EQ(dataCopyArg->srcStride, _0);
        ASSERT_EQ(dataCopyArg->dstStride, _0);
    }
}

// Data-path: nZ → nZ
// Element-type: no-except (float)
// Speciality: tla-long-stride (TileCopyTla, long outer-row stride, per-fractal-row split)
TEST_P(TileCopyGmToL1TestAtlasA2, nZTonZTestTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    setShape<Element, true>();
    int64_t bigOuterRow = (int64_t)(STRIDE_LIMIT + 1) * ELE_NUM_PER_C0;
    auto baseSrc = tla::MakeLayout<Element, layout::nZ>(_row, _col);
    auto layoutSrc = tla::MakeLayout(baseSrc.shape(),
        tla::MakeStride(tla::MakeStride(tla::Int<1>{}, bigOuterRow),
                        tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_FRACTAL>{})),
        baseSrc.originShape());
    auto layoutDst = tla::MakeLayout<Element, layout::nZ>(_row, _col);
    using LayoutSrc = decltype(layoutSrc);
    using LayoutDst = decltype(layoutDst);

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZeroTla, AscendC::TPosition::GM> tensorGm(gmTensor, layoutSrc);
    tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZeroTla, AscendC::TPosition::A1> tensorL1(l1Tensor, layoutDst);

    TileCopyTla<ArchTag, decltype(tensorGm), decltype(tensorL1)> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(_row);
    uint32_t blockLen = _col;
    uint32_t srcOuterStrideRow = static_cast<uint32_t>(bigOuterRow);
    uint32_t dstOuterStrideRow = tla::get<0, 1>(layoutDst.stride());
    ASSERT_EQ(logs.size(), blockCount);
    for (uint32_t i = 0; i < blockCount; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * dstOuterStrideRow * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * srcOuterStrideRow * sizeof(Element));
        const auto* dataCopyArg = logs[i].GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyArg->blockCount, _1);
        ASSERT_EQ(dataCopyArg->blockLen, blockLen);
        ASSERT_EQ(dataCopyArg->srcStride, _0);
        ASSERT_EQ(dataCopyArg->dstStride, _0);
    }
}

// Data-path: nZ → nZ
// Element-type: no-except (float)
// Speciality: tla-ext-long-stride (TileCopyTlaExt with actualShape, per-fractal-row split)
TEST_P(TileCopyGmToL1TestAtlasA2, nZTonZTestTlaExtLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    setShape<Element, true>();
    int64_t bigOuterRow = (int64_t)(STRIDE_LIMIT + 1) * ELE_NUM_PER_C0;
    auto baseSrc = tla::MakeLayout<Element, layout::nZ>(_row, _col);
    auto layoutSrc = tla::MakeLayout(baseSrc.shape(),
        tla::MakeStride(tla::MakeStride(tla::Int<1>{}, bigOuterRow),
                        tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_FRACTAL>{})),
        baseSrc.originShape());
    auto layoutDst = tla::MakeLayout<Element, layout::nZ>(_row, _col);
    using LayoutSrc = decltype(layoutSrc);
    using LayoutDst = decltype(layoutDst);

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZeroTla, AscendC::TPosition::GM> tensorGm(gmTensor, layoutSrc);
    tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZeroTla, AscendC::TPosition::A1> tensorL1(l1Tensor, layoutDst);

    tla::Shape<uint32_t, uint32_t> actualShape{_row, _col};
    TileCopyTlaExt<ArchTag, decltype(tensorGm), decltype(tensorL1), layout::nZ, layout::nZ> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm, actualShape);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(_row);
    uint32_t blockLen = _col;
    uint32_t srcOuterStrideRow = static_cast<uint32_t>(bigOuterRow);
    uint32_t dstOuterStrideRow = tla::get<0, 1>(layoutDst.stride());
    ASSERT_EQ(logs.size(), blockCount);
    for (uint32_t i = 0; i < blockCount; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * dstOuterStrideRow * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * srcOuterStrideRow * sizeof(Element));
        const auto* dataCopyArg = logs[i].GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyArg->blockCount, _1);
        ASSERT_EQ(dataCopyArg->blockLen, blockLen);
        ASSERT_EQ(dataCopyArg->srcStride, _0);
        ASSERT_EQ(dataCopyArg->dstStride, _0);
    }
}

///////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyGmToL1,
    TileCopyGmToL1TestAtlasA2,
    ::testing::Values(
        TestMatrixShape{128U, 256U},  // aligned
        TestMatrixShape{1U, 256U},    // 1-d
        TestMatrixShape{42U, 256U},   // not aligned
        TestMatrixShape{42U, 283U},   // not aligned (on both sides)
        TestMatrixShape{8U, 128U},    // rows<=16 for DynamicOptimized (also equals ELE_NUM_PER_C0 when B32)
        TestMatrixShape{128U, 8U}     // cols<=16 for ColumnMajor DynamicOptimized
    )
);

#endif // CATLASS_ARCH == 2201
