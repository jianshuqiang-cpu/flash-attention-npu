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
#include "catlass/gemm/tile/tile_copy_tla.hpp"

#include "stub/ascendc_logger.h"

#include "common/helper.hpp"
#include "common/shape.hpp"

// The two arch headers both define the primary `CopyL1ToBT` template, so the
// includes must be isolated by CATLASS_ARCH to avoid a redefinition.
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201
#include "catlass/gemm/tile/atlasa2/copy_l1_to_bt.hpp"
#endif
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
#include "catlass/gemm/tile/ascend950/copy_l1_to_bt.hpp"
#endif

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// NOTE: atlasa2/copy_l1_to_bt.hpp pulls in `using namespace tla;` at global scope, which makes the
// unqualified `_0`/`_1` ambiguous with tla's homonyms in the AtlasA2 TU. The macros below therefore
// reference Catlass::Test::Helper::_0 / _1 with full qualification so they build for both generations.

// Pre-defined types
using L1ToBTCoordZero = tla::Coord<tla::Int<0>>;
template <class Element, class Layout>
using L1ToBTTensorSrc = tla::Tensor<AscendC::LocalTensor<Element>, Layout, L1ToBTCoordZero, AscendC::TPosition::A1>;
template <class Element, class Layout>
using L1ToBTTensorDst = tla::Tensor<AscendC::LocalTensor<Element>, Layout, L1ToBTCoordZero, AscendC::TPosition::C2>;

template <typename ArchTag>
class TypedTileCopyL1ToBTTest : public UBTileCopyTest, public testing::WithParamInterface<TestVectorShape> {
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
    void BaseCheck(const AscendCCallLog& logTileCopy)
    {
        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }

    template <class Element>
    auto MakeCall()
    {
        using L1Type = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>;
        using BTType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2>;

        AscendC::LocalTensor<Element> l1Tensor;
        AscendC::LocalTensor<Element> btTensor;

        setShape();
        layout::VectorLayout layoutSrc{_totalLen};
        layout::VectorLayout layoutDst{_totalLen};

        CopyL1ToBT<ArchTag, L1Type, BTType> copyL1ToBT;
        copyL1ToBT(btTensor, l1Tensor, layoutDst, layoutSrc);

        return AscendCCallLogger::Instance().GetLogs();
    }

    template <class Element>
    auto MakeCallTla()
    {
        using LayoutTag = layout::VectorLayout;
        using Layout = Catlass::detail::TagToLayout_t<Element, LayoutTag>;

        AscendC::LocalTensor<Element> l1Src;
        AscendC::LocalTensor<Element> btDst;

        setShape();
        auto layoutSrc = tla::MakeLayout(_totalLen);
        auto layoutDst = tla::MakeLayout(_totalLen);
        L1ToBTTensorSrc<Element, Layout> tensorL1(l1Src, layoutSrc);
        L1ToBTTensorDst<Element, Layout> tensorBT(btDst, layoutDst);

        TileCopyTla<ArchTag, decltype(tensorL1), decltype(tensorBT)> copyL1ToBT;
        copyL1ToBT(tensorBT, tensorL1);

        return AscendCCallLogger::Instance().GetLogs();
    }

protected:
    // Expected DataCopyParams::blockLen for copying `len` elements into the bias table.
    //  - Ascend950: ceil(len / C0), rounded up to an even burst for B32.
    //  - AtlasA2  : ceil(len / C2).
    template <class Element>
    uint32_t ExpectedBlockLen(uint32_t len)
    {
        if constexpr (std::is_same_v<ArchTag, Arch::Ascend950>) {
            constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
            uint32_t blockLen = CeilDiv<ELE_NUM_PER_C0>(len);
            if constexpr (sizeof(Element) == 4) {
                blockLen = RoundUp(blockLen, 2);
            }
            return blockLen;
        } else {
            constexpr uint32_t ELE_NUM_PER_C2 = BYTE_PER_C2 / sizeof(Element);
            return CeilDiv<ELE_NUM_PER_C2>(len);
        }
    }
};

// ============================================================================
// Common tests — non-TLA CopyL1ToBT (specialized for both AtlasA2 and Ascend950)
// ============================================================================
#define DEFINE_L1TOBT_TESTS(SuiteName)                                                                                 \
    /* Data-path: Vector (L1) → Vector (BT) */ \
    /* Element-type: half */ \
    /* Speciality: Half (DataCopy, blockLen = ceil(len / C)) */ \
    TEST_P(SuiteName, VectorToVectorTestHalf)                                                                          \
    {                                                                                                                  \
        using Element = half;                                                                                          \
                                                                                                                       \
        auto logs = this->template MakeCall<Element>();                                                                \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        const auto& logTileCopy = logs[0];                                                                             \
        this->template BaseCheck<Element>(logTileCopy);                                                                \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);                                                          \
                                                                                                                       \
        const auto* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                           \
        ASSERT_EQ(dataCopyArg->blockCount, Catlass::Test::Helper::_1);                                                 \
        ASSERT_EQ(dataCopyArg->blockLen, this->template ExpectedBlockLen<Element>(this->_totalLen));                   \
        ASSERT_EQ(dataCopyArg->srcStride, Catlass::Test::Helper::_0);                                                  \
        ASSERT_EQ(dataCopyArg->dstStride, Catlass::Test::Helper::_0);                                                  \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: Vector (L1) → Vector (BT) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Float (DataCopy, B32 burst rounded to even on Ascend950, plain ceil on AtlasA2) */ \
    TEST_P(SuiteName, VectorToVectorTestFloat)                                                                         \
    {                                                                                                                  \
        using Element = float;                                                                                         \
                                                                                                                       \
        auto logs = this->template MakeCall<Element>();                                                                \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        const auto& logTileCopy = logs[0];                                                                             \
        this->template BaseCheck<Element>(logTileCopy);                                                                \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);                                                          \
                                                                                                                       \
        const auto* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                           \
        ASSERT_EQ(dataCopyArg->blockCount, Catlass::Test::Helper::_1);                                                 \
        ASSERT_EQ(dataCopyArg->blockLen, this->template ExpectedBlockLen<Element>(this->_totalLen));                   \
        ASSERT_EQ(dataCopyArg->srcStride, Catlass::Test::Helper::_0);                                                  \
        ASSERT_EQ(dataCopyArg->dstStride, Catlass::Test::Helper::_0);                                                  \
    }

// ============================================================================
// Ascend950-only tests — TileCopyTla L1->BT (AtlasA2 has no *Tla* specialization)
// ============================================================================
#define DEFINE_L1TOBT_TESTS_ASCEND950(SuiteName)                                                                       \
    /* Data-path: Vector (L1) → Vector (BT) */ \
    /* Element-type: half */ \
    /* Speciality: Half (TLA DataCopy, blockLen = ceil(len / C0)) */ \
    TEST_P(SuiteName, VectorToVectorTlaTestHalf)                                                                       \
    {                                                                                                                  \
        using Element = half;                                                                                          \
                                                                                                                       \
        auto logs = this->template MakeCallTla<Element>();                                                             \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        const auto& logTileCopy = logs[0];                                                                             \
        this->template BaseCheck<Element>(logTileCopy);                                                                \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);                                                          \
                                                                                                                       \
        const auto* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                           \
        ASSERT_EQ(dataCopyArg->blockCount, _1);                                                                        \
        ASSERT_EQ(dataCopyArg->blockLen, this->template ExpectedBlockLen<Element>(this->_totalLen));                   \
        ASSERT_EQ(dataCopyArg->srcStride, _0);                                                                         \
        ASSERT_EQ(dataCopyArg->dstStride, _0);                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: Vector (L1) → Vector (BT) */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Float (TLA DataCopy, blockLen rounded up to even burst for B32) */ \
    TEST_P(SuiteName, VectorToVectorTlaTestFloat)                                                                      \
    {                                                                                                                  \
        using Element = float;                                                                                         \
                                                                                                                       \
        auto logs = this->template MakeCallTla<Element>();                                                             \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        const auto& logTileCopy = logs[0];                                                                             \
        this->template BaseCheck<Element>(logTileCopy);                                                                \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);                                                          \
                                                                                                                       \
        const auto* dataCopyArg = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                           \
        ASSERT_EQ(dataCopyArg->blockCount, _1);                                                                        \
        ASSERT_EQ(dataCopyArg->blockLen, this->template ExpectedBlockLen<Element>(this->_totalLen));                   \
        ASSERT_EQ(dataCopyArg->srcStride, _0);                                                                         \
        ASSERT_EQ(dataCopyArg->dstStride, _0);                                                                         \
    }

// ============================================================================
// Instantiate for Catlass::Arch::AtlasA2
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201
using TypedL1ToBTA2Suite = TypedTileCopyL1ToBTTest<Catlass::Arch::AtlasA2>;

DEFINE_L1TOBT_TESTS(TypedL1ToBTA2Suite)

INSTANTIATE_TEST_SUITE_P(AtlasA2, TypedL1ToBTA2Suite,
    ::testing::Values(
        TestVectorShape{128U, 1U},       // Aligned
        TestVectorShape{67U, 1U},        // Unaligned
        TestVectorShape{256U, 1U}));     // Aligned(sec)
#endif // CATLASS_ARCH == 2201

// ============================================================================
// Instantiate for Catlass::Arch::Ascend950
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using TypedL1ToBTA950Suite = TypedTileCopyL1ToBTTest<Catlass::Arch::Ascend950>;

DEFINE_L1TOBT_TESTS(TypedL1ToBTA950Suite)

DEFINE_L1TOBT_TESTS_ASCEND950(TypedL1ToBTA950Suite)

INSTANTIATE_TEST_SUITE_P(Ascend950, TypedL1ToBTA950Suite,
    ::testing::Values(
        TestVectorShape{128U, 1U},       // Aligned
        TestVectorShape{67U, 1U},        // Unaligned
        TestVectorShape{256U, 1U}));     // Aligned(sec)
#endif // CATLASS_ARCH == 3510
