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

#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Epilogue::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyGmToUbTest : public UBTileCopyTest, public ::testing::WithParamInterface<TestVectorShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        _setShape(GetParam().blkLen, GetParam().blkCnt);
    }

    template <class Element, bool isPad=true>
    void BaseCheck(const AscendCCallLog& logVecCopy, 
        const AscendC::GlobalTensor<Element>& gmTensor, 
        const AscendC::LocalTensor<Element>& ubTensor)
    {
        if constexpr (isPad) {
            // 验证调用名称：应为"DataCopyPad"(ISASI)
            ASSERT_EQ(logVecCopy.name, "DataCopyPad");
            // 验证参数数量：VecCopy应包含4个参数（dst, src, dataCopyParams, padParams）
            ASSERT_EQ(logVecCopy.args.size(), 4);
        } else {
            // 验证调用名称：应为"DataCopy"(ISASI)
            ASSERT_EQ(logVecCopy.name, "DataCopy");
            // 验证参数数量：VecCopy应包含3个参数（dst, src, dataCopyParams）
            ASSERT_EQ(logVecCopy.args.size(), 3);
        }

        // check if the pointer is correct
        auto logGmTensor = logVecCopy.GetArgsAt(1).RawValue();
        auto logUbTensor = logVecCopy.GetArgsAt(0).RawValue();
        ASSERT_EQ(logGmTensor, &gmTensor);
        ASSERT_EQ(logUbTensor, &ubTensor);
        ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), 0);
        ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), 0);

        // check if the data-type is match
        const std::type_index& T0 = logVecCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }
};

class TileCopyGmToUbNonContiguousTest : public UBTileCopyTest, public ::testing::WithParamInterface<TestVectorShapeWithStride> {
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

    template <class Element, bool isPad = true>
    void BaseCheck(AscendCCallLog& logVecCopy)
    {
        if constexpr (isPad) {
            // 验证调用名称：应为"DataCopyPad"(ISASI)
            ASSERT_EQ(logVecCopy.name, "DataCopyPad");
            // 验证参数数量：VecCopy应包含4个参数（dst, src, dataCopyParams, padParams）
            ASSERT_EQ(logVecCopy.args.size(), 4);
        } else {
            // 验证调用名称：应为"DataCopy"(ISASI)
            ASSERT_EQ(logVecCopy.name, "DataCopy");
            // 验证参数数量：VecCopy应包含3个参数（dst, src, dataCopyParams）
            ASSERT_EQ(logVecCopy.args.size(), 3);
        }

        // check if the data-type is match
        const std::type_index& T0 = logVecCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }

protected:
    uint32_t _srcStride = 128;
    uint32_t _dstStride = 128;

    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;
};

// ============================================================================
// Testsuite from **RowMajor**
// ============================================================================

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: basic (contiguous, single DataCopyPad)
TEST_P(TileCopyGmToUbTest, RowMajorToRowMajorTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    CopyGm2Ub<ArchTag, GmType> copyGmToUb;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> ubTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape();
    setLayout<Element>(_blkCnt, _blkLen, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc) && isContiguous(layoutDst));

    // caller
    copyGmToUb(ubTensor, gmTensor, layoutDst, layoutSrc);

    // Get logs
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy, gmTensor, ubTensor);

    const AscendC::DataCopyExtParams* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    const AscendC::DataCopyPadExtParams<Element>* padParams = logVecCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_EQ(dataCopyParams->blockCount, _blkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, _blkLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, _0);                // 源操作数相邻数据块间的间隔(单位:32B[OnLocal]/Byte[GM])
    ASSERT_EQ(dataCopyParams->dstStride, _0);                // 目的操作数相邻数据块间的间隔(单位:32B[OnLocal]/Byte[UB])

    ASSERT_EQ(padParams->isPad, false);
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: aligned-contiguous (CopyGm2UbAligned, single DataCopy by total count)
TEST_P(TileCopyGmToUbTest, RowMajorToRowMajorTestAligned)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    CopyGm2UbAligned<ArchTag, GmType> copyGmToUb;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> ubTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape();
    setLayout<Element>(_blkCnt, _blkLen, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc) && isContiguous(layoutDst));

    // caller
    copyGmToUb(ubTensor, gmTensor, layoutDst, layoutSrc);

    // Get logs
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logVecCopy = logs[0];
    BaseCheck<Element, false /*not DataCopyPad*/>(logVecCopy, gmTensor, ubTensor);

    const uint32_t* count = logVecCopy.GetArgsAt(2).Value<uint32_t>();
    ASSERT_EQ(*count, _totalLen);
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: aligned-non-contiguous (CopyGm2UbAligned, per-block DataCopyParams)
TEST_P(TileCopyGmToUbNonContiguousTest, RowMajorToRowMajorTestNonContiguous)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;
    constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    CopyGm2UbAligned<ArchTag, GmType> copyGmToUb;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> ubTensor;

    setShape();
    LayoutSrc layoutSrc{_blkCnt, _blkLen, _srcStride};
    LayoutDst layoutDst{_blkCnt, _blkLen, _dstStride};
    // if the given stride matches with shape, skip the test
    if (isContiguous(layoutSrc) && isContiguous(layoutDst)) { GTEST_SKIP(); }
    // for long stride senario, this test case is not supported
    if (_srcStride >= STRIDE_LIMIT || _dstStride >= STRIDE_LIMIT || 
        _blkLen >= ELE_NUM_PER_BLK * BLOCK_LEN_LIMIT) { GTEST_SKIP(); }



    copyGmToUb(ubTensor, gmTensor, layoutDst, layoutSrc);

    // Get logs
    uint32_t datacopyLoops = CeilDiv<MAX_REPEAT>(_blkCnt);
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), datacopyLoops);

    for (int i = 0; i < datacopyLoops; i++) {
        AscendCCallLog logVecCopy = logs[i];
        BaseCheck<Element, false /*not DataCopyPad*/>(logVecCopy);

        ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), i * MAX_REPEAT * _srcStride * sizeof(Element));
        ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), i * MAX_REPEAT * _dstStride * sizeof(Element));

        const AscendC::DataCopyParams* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyParams->blockCount, i==datacopyLoops-1 ? _blkCnt - i*MAX_REPEAT: MAX_REPEAT);
        ASSERT_EQ(dataCopyParams->blockLen, _blkLen / ELE_NUM_PER_BLK);
        ASSERT_EQ(dataCopyParams->srcGap, (_srcStride - _blkLen) / ELE_NUM_PER_BLK); // May fail ?
        ASSERT_EQ(dataCopyParams->dstGap, (_dstStride - _blkLen) / ELE_NUM_PER_BLK); // May fail ?
    }
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: aligned-long-stride (CopyGm2UbAligned, row-by-row DataCopyParams)
TEST_P(TileCopyGmToUbNonContiguousTest, RowMajorToRowMajorTestLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;
    constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    CopyGm2UbAligned<ArchTag, GmType> copyGmToUb;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> ubTensor;

    setShape();
    uint32_t longStride = _blkLen + STRIDE_LIMIT * ELE_NUM_PER_BLK + STRIDE_LIMIT;
    LayoutSrc layoutSrc{_blkCnt, _blkLen, longStride};
    LayoutDst layoutDst{_blkCnt, _blkLen, longStride};

    copyGmToUb(ubTensor, gmTensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), _blkCnt);

    for (uint32_t i = 0; i < _blkCnt; i++) {
        AscendCCallLog logVecCopy = logs[i];
        BaseCheck<Element, false>(logVecCopy);
        ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), i * longStride * sizeof(Element));
        ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), i * longStride * sizeof(Element));
        ASSERT_EQ(*logVecCopy.GetArgsAt(2).Value<uint32_t>(), _blkLen);
    }
}

// ============================================================================
// Testsuite from **Vector**
// ============================================================================

// Data-path: Vector → Vector
// Element-type: no-except (float)
// Speciality: basic (single DataCopyPad, blockCount = 1)
TEST_P(TileCopyGmToUbTest, VectorToVectorTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    CopyGm2Ub<ArchTag, GmType> copyGmToUb;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> ubTensor;
    
    setShape();
    LayoutSrc layoutSrc{_totalLen};
    LayoutDst layoutDst{_totalLen};
    ASSERT_TRUE(isContiguous(layoutSrc) && isContiguous(layoutDst));
    // caller
    copyGmToUb(ubTensor, gmTensor, layoutDst, layoutSrc);

    // Get logs
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy, gmTensor, ubTensor);

    const AscendC::DataCopyExtParams* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    const AscendC::DataCopyPadExtParams<Element>* padParams = logVecCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();  
    ASSERT_EQ(dataCopyParams->blockCount, _1);
    ASSERT_EQ(dataCopyParams->blockLen, _totalLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, _0);                // 源操作数相邻数据块间的间隔(单位:32B[OnLocal]/Byte[GM])
    ASSERT_EQ(dataCopyParams->dstStride, _0);                // 目的操作数相邻数据块间的间隔(单位:32B[OnLocal]/Byte[UB])

    ASSERT_EQ(padParams->isPad, false);
}

INSTANTIATE_TEST_SUITE_P(
    GmToUbTestShapes,
    TileCopyGmToUbTest,
    ::testing::Values(
        TestVectorShape{128U, 1U},
        TestVectorShape{256U, 4U},
        TestVectorShape{64U, 8U}
    )
);

INSTANTIATE_TEST_SUITE_P(
    GmToUbNonContiguousTestShapes,
    TileCopyGmToUbNonContiguousTest,
    ::testing::Values(
        TestVectorShapeWithStride{128U, 4U, 256U, 128U},
        TestVectorShapeWithStride{64U, 8U, 128U, 64U},
        TestVectorShapeWithStride{256U, 2U, 512U, 256U}
    )
);

#endif // CATLASS_ARCH == 2201
