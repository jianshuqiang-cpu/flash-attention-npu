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

#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Epilogue::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;
using ArchTag = Catlass::Arch::Ascend950;

class TileCopyUbToGmTestAscend950 : public UBTileCopyTest, public ::testing::WithParamInterface<TestVectorShape> {
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
    void BaseCheck(const AscendCCallLog& logVecCopy, 
        const AscendC::GlobalTensor<Element>& gmTensor, 
        const AscendC::LocalTensor<Element>& ubTensor)
    {
        ASSERT_EQ(logVecCopy.name, "DataCopyPad");
        ASSERT_EQ(logVecCopy.args.size(), 3);

        auto logGmTensor = logVecCopy.GetArgsAt(0).RawValue();
        auto logUbTensor = logVecCopy.GetArgsAt(1).RawValue();
        ASSERT_EQ(logGmTensor, &gmTensor);
        ASSERT_EQ(logUbTensor, &ubTensor);
        ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), 0);
        ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), 0);

        const std::type_index& T0 = logVecCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }
};

// ========================================================================
// NOTE: CopyUb2GmAligned 仅有 Arch::AtlasA2 特化，Ascend950 下不存在。
// Ascend950 下 CopyUb2Gm / CopyUb2GmAligned 覆盖率已达。
// ========================================================================

// ============================================================================
// Testsuite from **RowMajor**
// ============================================================================

// Data-path: RowMajor (UB) → RowMajor (GM)
// Element-type: no-except (float)
// Speciality: Basic (CopyUb2Gm contiguous, single DataCopyPad)
TEST_P(TileCopyUbToGmTestAscend950, RowMajorToRowMajorTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    CopyUb2Gm<ArchTag, GmType> copyUbToGm;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> ubTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape();
    setLayout<Element>(_blkCnt, _blkLen, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc) && isContiguous(layoutDst));

    copyUbToGm(gmTensor, ubTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy, gmTensor, ubTensor);

    const AscendC::DataCopyExtParams* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    ASSERT_EQ(dataCopyParams->blockCount, _blkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, _blkLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, _0);
    ASSERT_EQ(dataCopyParams->dstStride, _0);
}

// ============================================================================
// Testsuite from **Vector**
// ============================================================================

// Data-path: Vector (UB) → Vector (GM)
// Element-type: no-except (float)
// Speciality: Basic (CopyUb2Gm rank-1, single DataCopyPad with blockCount = 1)
TEST_P(TileCopyUbToGmTestAscend950, VectorToVectorTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    CopyUb2Gm<ArchTag, GmType> copyUbToGm;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> ubTensor;

    setShape();
    LayoutSrc layoutSrc{_totalLen};
    LayoutDst layoutDst{_totalLen};
    ASSERT_TRUE(isContiguous(layoutSrc) && isContiguous(layoutDst));

    copyUbToGm(gmTensor, ubTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy, gmTensor, ubTensor);

    const AscendC::DataCopyExtParams* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    ASSERT_EQ(dataCopyParams->blockCount, _1);
    ASSERT_EQ(dataCopyParams->blockLen, _totalLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, _0);
    ASSERT_EQ(dataCopyParams->dstStride, _0);
}

// ========================================================================
// NOTE: CopyUb2GmAligned 仅有 Arch::AtlasA2 特化，Ascend950 下不存在
// Ascend950 下 CopyUb2Gm 采用 DataCopyPad 通过，覆盖率已达
// ========================================================================

INSTANTIATE_TEST_SUITE_P(
    CopyUbToGm,
    TileCopyUbToGmTestAscend950,
    ::testing::Values(
        TestVectorShape{128U, 1U},
        TestVectorShape{32U, 256U},
        TestVectorShape{64U, 193U}
    )
);

#endif // CATLASS_ARCH == 3510
