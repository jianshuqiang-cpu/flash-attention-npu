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

#include "catlass/gemm/tile/copy_l1_to_fp.hpp"
#include "stub/ascendc_logger.h"

#include "common/helper.hpp"
#include "common/shape.hpp"

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// NOTE: copy_l1_to_fp.hpp pulls in `using namespace tla;` at global scope, which makes the unqualified
// `_0`/`_1` ambiguous with tla's homonyms; the macro below uses Catlass::Test::Helper::_0 / _1 fully
// qualified so it builds for both generations.

// TestCase for L1->FP TileCopy Utilities (Common test scenarios, parameterized by ArchTag)
template <typename ArchTag>
class TypedTileCopyL1ToFpTest : public AscendCTest, public ::testing::WithParamInterface<TestVectorShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        _blkLen = GetParam().blkLen;
    }

    template <class Element>
    void BaseCheck(const AscendCCallLog& logTileCopy)
    {
        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }

    // Invoke the CopyL1ToFP<ArchTag, ...> component (L1 A1 -> Fixpipe C2PIPE2GM) and return the logs.
    template <class Element>
    auto MakeCall()
    {
        using L1Type = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>;
        using FpType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>;

        AscendC::LocalTensor<Element> l1Tensor;
        AscendC::LocalTensor<Element> fpTensor;

        setShape();
        layout::VectorLayout layoutSrc{_blkLen};
        layout::VectorLayout layoutDst{_blkLen};

        CopyL1ToFP<ArchTag, L1Type, FpType> copyL1ToFP;
        copyL1ToFP(fpTensor, l1Tensor, layoutDst, layoutSrc);

        return AscendCCallLogger::Instance().GetLogs();
    }

protected:
    template <class Element>
    constexpr static uint32_t GetEleNumPerFp()
    {
        return BytesToBits(BYTE_PER_BLK_FP) / SizeOfBits<Element>::value;
    }

    uint32_t _blkLen = 0;
};

// ============================================================================
// Common tests — CopyL1ToFP is specialized generically over ArchTag, so the
// same scenarios cover both AtlasA2 and Ascend950.
// ============================================================================
#define DEFINE_L1TOFP_TESTS(SuiteName)                                                                                 \
    /* Data-path: Vector (L1) → Vector (FP) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: basic (single DataCopy, blockLen = ceil(len / FP-block)) */ \
    TEST_P(SuiteName, VectorToVectorTestBasic)                                                                         \
    {                                                                                                                  \
        using Element = float;                                                                                         \
        constexpr uint32_t ELE_NUM_PER_FP = GetEleNumPerFp<Element>();                                                   \
                                                                                                                       \
        auto logs = this->template MakeCall<Element>();                                                                \
                                                                                                                       \
        ASSERT_EQ(logs.size(), Catlass::Test::Helper::_1);                                                             \
        const auto& logTileCopy = logs[0];                                                                             \
        this->template BaseCheck<Element>(logTileCopy);                                                                \
                                                                                                                       \
        const auto* dataCopyParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                        \
        ASSERT_EQ(dataCopyParams->blockCount, Catlass::Test::Helper::_1);                                              \
        ASSERT_EQ(dataCopyParams->blockLen, CeilDiv<ELE_NUM_PER_FP>(this->_blkLen));                                   \
        ASSERT_EQ(dataCopyParams->srcStride, Catlass::Test::Helper::_0);                                               \
        ASSERT_EQ(dataCopyParams->dstStride, Catlass::Test::Helper::_0);                                               \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: Vector (L1) → Vector (FP) */ \
    /* Element-type: uint64_t */ \
    /* Speciality: basic (wide element, blockLen = ceil(len / FP-block)) */ \
    TEST_P(SuiteName, VectorToVectorTestUint64)                                                                        \
    {                                                                                                                  \
        using Element = uint64_t;                                                                                      \
        constexpr uint32_t ELE_NUM_PER_FP = GetEleNumPerFp<Element>();                                                   \
                                                                                                                       \
        auto logs = this->template MakeCall<Element>();                                                                \
                                                                                                                       \
        ASSERT_EQ(logs.size(), Catlass::Test::Helper::_1);                                                             \
        const auto& logTileCopy = logs[0];                                                                             \
        this->template BaseCheck<Element>(logTileCopy);                                                                \
                                                                                                                       \
        const auto* dataCopyParams = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                        \
        ASSERT_EQ(dataCopyParams->blockCount, Catlass::Test::Helper::_1);                                              \
        ASSERT_EQ(dataCopyParams->blockLen, CeilDiv<ELE_NUM_PER_FP>(_blkLen));                                   \
        ASSERT_EQ(dataCopyParams->srcStride, Catlass::Test::Helper::_0);                                               \
        ASSERT_EQ(dataCopyParams->dstStride, Catlass::Test::Helper::_0);                                               \
    }

// ============================================================================
// Instantiate for Catlass::Arch::AtlasA2
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201
using TypedL1ToFpA2Suite = TypedTileCopyL1ToFpTest<Catlass::Arch::AtlasA2>;

DEFINE_L1TOFP_TESTS(TypedL1ToFpA2Suite)

INSTANTIATE_TEST_SUITE_P(AtlasA2, TypedL1ToFpA2Suite,
    ::testing::Values(
        TestVectorShape{128U, 1U},       // Aligned
        TestVectorShape{67U, 1U},        // Unaligned
        TestVectorShape{256U, 1U}));     // Aligned(sec)
#endif // CATLASS_ARCH == 2201

// ============================================================================
// Instantiate for Catlass::Arch::Ascend950
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using TypedL1ToFpA950Suite = TypedTileCopyL1ToFpTest<Catlass::Arch::Ascend950>;

DEFINE_L1TOFP_TESTS(TypedL1ToFpA950Suite)

INSTANTIATE_TEST_SUITE_P(Ascend950, TypedL1ToFpA950Suite,
    ::testing::Values(
        TestVectorShape{128U, 1U},       // Aligned
        TestVectorShape{67U, 1U},        // Unaligned
        TestVectorShape{256U, 1U}));     // Aligned(sec)
#endif // CATLASS_ARCH == 3510
