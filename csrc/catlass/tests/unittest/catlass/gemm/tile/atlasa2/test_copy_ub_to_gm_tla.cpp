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

#include "catlass/gemm/tile/copy_ub_to_gm.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyUbToGmTlaTest : public UBTileCopyTest, public testing::WithParamInterface<TestVectorShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        _setShape(GetParam().blkLen, GetParam().blkCnt);
    }

    template <class Element>
    void BaseCheck(const AscendCCallLog& logVecCopy)
    {
        ASSERT_EQ(logVecCopy.name, "DataCopyPad");
        ASSERT_EQ(logVecCopy.args.size(), 3);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }
};

class TileCopyUbToGmTlaNonContiguousTest : public UBTileCopyTest,
    public testing::WithParamInterface<TestVectorShapeWithStride> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        _setShape(GetParam().blkLen, GetParam().blkCnt);
        _srcStride = GetParam().srcStride;
        _dstStride = GetParam().dstStride;
    }

    template <class Element>
    void BaseCheck(const AscendCCallLog& logVecCopy)
    {
        ASSERT_EQ(logVecCopy.name, "DataCopyPad");
        ASSERT_EQ(logVecCopy.args.size(), 3);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }

    uint32_t _srcStride = 128;
    uint32_t _dstStride = 128;
};

using UbToGmCoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class LayoutSrc>
using UbToGmTensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, LayoutSrc, UbToGmCoordZero,
    AscendC::TPosition::VECCALC>;
template <class Element, class LayoutDst>
using UbToGmTensorDst = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutDst, UbToGmCoordZero,
    AscendC::TPosition::GM>;

// ============================================================================
// Testsuite from **RowMajor**
// ============================================================================

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: basic (TLA contiguous, single DataCopyPad)
TEST_P(TileCopyUbToGmTlaTest, RowMajorToRowMajorTestBasic)
{
    using Element = float;
    using LayoutTag = layout::RowMajor;
    using Layout = detail::TagToLayout_t<Element, LayoutTag>;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::GlobalTensor<Element> gmDst;

    setShape();
    auto layoutSrc = tla::MakeLayout<Element, LayoutTag>(static_cast<uint32_t>(_blkCnt), _blkLen);
    auto layoutDst = tla::MakeLayout<Element, LayoutTag>(static_cast<uint32_t>(_blkCnt), _blkLen);
    UbToGmTensorSrc<Element, Layout> tensorUb(ubSrc, layoutSrc);
    UbToGmTensorDst<Element, Layout> tensorGm(gmDst, layoutDst);

    TileCopyTla<Arch::AtlasA2, decltype(tensorUb), decltype(tensorGm)> copyUbToGm;
    copyUbToGm(tensorGm, tensorUb);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy);
    ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    ASSERT_EQ(dataCopyParams->blockCount, _blkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, _blkLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, _0);
    ASSERT_EQ(dataCopyParams->dstStride, _0);
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: non-contiguous (TLA strided, single DataCopyPad with srcStride/dstStride)
TEST_P(TileCopyUbToGmTlaNonContiguousTest, RowMajorToRowMajorTestNonContiguous)
{
    using Element = float;
    using Layout = detail::TagToLayout_t<Element, layout::RowMajor>;
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::GlobalTensor<Element> gmDst;

    setShape();
    if (_srcStride == _blkLen && _dstStride == _blkLen) {
        GTEST_SKIP();
    }

    auto layoutSrc = tla::MakeLayout(
        tla::MakeShape(static_cast<uint32_t>(_blkCnt), _blkLen),
        tla::MakeStride(static_cast<int64_t>(_srcStride), tla::Int<1>{}),
        tla::MakeShape(static_cast<uint32_t>(_blkCnt), _blkLen));
    auto layoutDst = tla::MakeLayout(
        tla::MakeShape(static_cast<uint32_t>(_blkCnt), _blkLen),
        tla::MakeStride(static_cast<int64_t>(_dstStride), tla::Int<1>{}),
        tla::MakeShape(static_cast<uint32_t>(_blkCnt), _blkLen));
    UbToGmTensorSrc<Element, Layout> tensorUb(ubSrc, layoutSrc);
    UbToGmTensorDst<Element, Layout> tensorGm(gmDst, layoutDst);

    TileCopyTla<Arch::AtlasA2, decltype(tensorUb), decltype(tensorGm)> copyUbToGm;
    copyUbToGm(tensorGm, tensorUb);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy);
    ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    ASSERT_EQ(dataCopyParams->blockCount, _blkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, _blkLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, (_srcStride - _blkLen) / ELE_NUM_PER_C0);
    ASSERT_EQ(dataCopyParams->dstStride, (_dstStride - _blkLen) * sizeof(Element));
}

INSTANTIATE_TEST_SUITE_P(
    CopyUbToGmTla,
    TileCopyUbToGmTlaTest,
    ::testing::Values(
        TestVectorShape{128U, 1U},
        TestVectorShape{256U, 4U},
        TestVectorShape{64U, 8U}
    )
);

INSTANTIATE_TEST_SUITE_P(
    CopyUbToGmTlaNonContiguous,
    TileCopyUbToGmTlaNonContiguousTest,
    ::testing::Values(
        TestVectorShapeWithStride{128U, 4U, 256U, 256U},
        TestVectorShapeWithStride{64U, 8U, 128U, 128U},
        TestVectorShapeWithStride{256U, 2U, 512U, 512U}
    )
);

#endif // CATLASS_ARCH == 2201
