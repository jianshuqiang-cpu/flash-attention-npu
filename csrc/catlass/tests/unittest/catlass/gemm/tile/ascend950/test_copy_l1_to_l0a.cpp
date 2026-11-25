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
#include "catlass/detail/dependent_false.hpp"

#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// TestCase for L1->L0A TileCopy Utilities
class TileCopyL1ToL0ATestAscend950 : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
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
    void BaseCheck(AscendCCallLog const &logTileCopy, AscendC::LocalTensor<Element>& l0aTensor, AscendC::LocalTensor<Element>& l1Tensor) 
    {
        // The API name should be "LoadData"
        ASSERT_EQ(logTileCopy.name, "LoadData");

        // Check the given address
        auto logTileCopyL0ATensor = logTileCopy.GetArgsAt(0).RawValue();
        auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(1).RawValue();
        ASSERT_EQ(logTileCopyL0ATensor, &l0aTensor);
        ASSERT_EQ(logTileCopyL1Tensor, &l1Tensor);
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);

        // Check for the data type
        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }

};

#define CATLASS_TILE_TEST_CONCAT_IMPL(a, b) a##b
#define CATLASS_TILE_TEST_CONCAT(a, b) CATLASS_TILE_TEST_CONCAT_IMPL(a, b)
#define CATLASS_TILE_TEST_SUFFIX_half Half
#define CATLASS_TILE_TEST_SUFFIX_float Float
#define CATLASS_TILE_TEST_SUFFIX(TYPE) CATLASS_TILE_TEST_CONCAT(CATLASS_TILE_TEST_SUFFIX_, TYPE)
#define CATLASS_TILE_TEST_NAME(PREFIX, TYPE) CATLASS_TILE_TEST_CONCAT(PREFIX, CATLASS_TILE_TEST_SUFFIX(TYPE))

#define ADD_TILE_COPY_TEST_L1_TO_L0A_ZN_TO_ZN(TYPE)                                                   \
TEST_P(TileCopyL1ToL0ATestAscend950, CATLASS_TILE_TEST_NAME(zNTozNTripleTest, TYPE))                  \
{                                                                                                      \
    using Element = TYPE;                                                                              \
    using ArchTag = Catlass::Arch::Ascend950;                                                          \
    using LayoutSrc = layout::zN;                                                                      \
    using LayoutDst = layout::zN;                                                                      \
    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;                         \
    using L0AType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A2>;                        \
                                                                                                       \
    setShape<Element>();                                                                               \
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);                         \
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);                         \
                                                                                                       \
    CopyL1ToL0A<ArchTag, L1Type, L0AType> copyL1ToL0A;                                                 \
    AscendC::LocalTensor<Element> l1Tensor;                                                            \
    AscendC::LocalTensor<Element> l0aTensor;                                                           \
    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);                                            \
                                                                                                       \
    auto logs = AscendCCallLogger::Instance().GetLogs();                                               \
    ASSERT_EQ(logs.size(), _1);                                                                        \
    auto logTileCopy = logs[0];                                                                        \
    BaseCheck<Element>(logTileCopy, l0aTensor, l1Tensor);                                              \
                                                                                                       \
    const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();           \
    ASSERT_EQ(loadDataArg->mStartPosition, _0);                                                        \
    ASSERT_EQ(loadDataArg->kStartPosition, _0);                                                        \
    ASSERT_EQ(loadDataArg->mStep, _rows_by_fractal);                                                   \
    ASSERT_EQ(loadDataArg->kStep, _cols_by_fractal);                                                   \
    ASSERT_EQ(loadDataArg->srcStride, _rows_by_fractal);                                               \
    ASSERT_EQ(loadDataArg->dstStride, _rows_by_fractal);                                               \
    ASSERT_EQ(loadDataArg->ifTranspose, false);                                                        \
}

#define ADD_TILE_COPY_TEST_L1_TO_L0A_NN_TO_ZN(TYPE)                                                    \
TEST_P(TileCopyL1ToL0ATestAscend950, CATLASS_TILE_TEST_NAME(nNTozNTest, TYPE))                         \
{                                                                                                      \
    using Element = TYPE;                                                                              \
    using ArchTag = Catlass::Arch::Ascend950;                                                          \
    using LayoutSrc = layout::nN;                                                                      \
    using LayoutDst = layout::zN;                                                                      \
    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;                         \
    using L0AType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A2>;                        \
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();                                     \
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value; \
                                                                                                       \
    setShape<Element>();                                                                               \
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);                         \
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);                         \
    ASSERT_TRUE(isContiguous(layoutSrc));                                                              \
    ASSERT_TRUE(isContiguous(layoutDst));                                                              \
                                                                                                       \
    CopyL1ToL0A<ArchTag, L1Type, L0AType> copyL1ToL0A;                                                 \
    AscendC::LocalTensor<Element> l1Tensor;                                                            \
    AscendC::LocalTensor<Element> l0aTensor;                                                           \
    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);                                            \
                                                                                                       \
    auto logs = AscendCCallLogger::Instance().GetLogs();                                               \
    ASSERT_EQ(logs.size(), CeilDiv<C0_NUM_PER_FRACTAL>(_col_round));                                   \
    uint32_t expectedDstOffsetStep = _row_round * C0_NUM_PER_FRACTAL;                                  \
    uint32_t expectedSrcOffsetStep = RoundUp<ELE_NUM_PER_C0>(_row) * C0_NUM_PER_FRACTAL;               \
    uint32_t expectedKStep = CeilDiv<ELE_NUM_PER_C0>(_row_round);                                      \
    if constexpr (std::is_same_v<Element, float>) {                                                    \
        expectedKStep = RoundUp<2>(expectedKStep);                                                     \
    }                                                                                                  \
    for (uint32_t i = 0; i < logs.size(); ++i) {                                                       \
        auto logTileCopy = logs[i];                                                                    \
        ASSERT_EQ(logTileCopy.name, "LoadData");                                                      \
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));                                  \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * expectedDstOffsetStep * sizeof(Element)); \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * expectedSrcOffsetStep * sizeof(Element)); \
        const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();       \
        ASSERT_EQ(loadDataArg->mStartPosition, _0);                                                    \
        ASSERT_EQ(loadDataArg->kStartPosition, _0);                                                    \
        ASSERT_EQ(loadDataArg->mStep, _1);                                                             \
        ASSERT_EQ(loadDataArg->kStep, expectedKStep);                                                  \
        ASSERT_EQ(loadDataArg->srcStride, _1);                                                         \
        ASSERT_EQ(loadDataArg->dstStride, _rows_by_fractal);                                           \
        ASSERT_EQ(loadDataArg->ifTranspose, true);                                                     \
    }                                                                                                  \
}

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN (L1) → zN (L0A)
// Element-type: no-except (float)
// Speciality: Basic (single LoadData, no transpose)
TEST_P(TileCopyL1ToL0ATestAscend950, zNTozNTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    CopyL1ToL0A<ArchTag, L1Type> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;
    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), _1);

    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy, l0aTensor, l1Tensor);

    const AscendC::LoadData2DParamsV2* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, _rows_by_fractal);  // Unit: 16(element)
    ASSERT_EQ(loadDataArg->kStep, _cols_by_fractal);  // Unit: 32(Byte)
    ASSERT_EQ(loadDataArg->srcStride, _rows_by_fractal); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->dstStride, _rows_by_fractal); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->ifTranspose, false);
}

// zN(A1) -> zN(A2), 3-param Ascend950 specialization.
ADD_TILE_COPY_TEST_L1_TO_L0A_ZN_TO_ZN(float);

// ============================================================================
// Testsuite from **nZ**
// ============================================================================

// Data-path: nZ (L1) → zN (L0A)
// Element-type: no-except (float)
// Speciality: Basic (single LoadData with ifTranspose enabled)
TEST_P(TileCopyL1ToL0ATestAscend950, nZTozNTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::zN;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    CopyL1ToL0A<ArchTag, L1Type> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;
    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), _1);

    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy, l0aTensor, l1Tensor);

    const AscendC::LoadData2DParamsV2* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, CeilDiv<2>(_cols_by_fractal));    // Unit: 16(element) [it is trans]
    ASSERT_EQ(loadDataArg->kStep, 2 * _rows_by_fractal);            // Unit: 32(Byte)    [it is trans]
    ASSERT_EQ(loadDataArg->srcStride, CeilDiv<2>(_cols_by_fractal)); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->dstStride, _rows_by_fractal);            // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->ifTranspose, true);
}

// ============================================================================
// Testsuite from **nN**
// ============================================================================

// nNTozN basic test, from nN->zN, Ascend950.
ADD_TILE_COPY_TEST_L1_TO_L0A_NN_TO_ZN(half);
ADD_TILE_COPY_TEST_L1_TO_L0A_NN_TO_ZN(float);

#undef ADD_TILE_COPY_TEST_L1_TO_L0A_NN_TO_ZN
#undef ADD_TILE_COPY_TEST_L1_TO_L0A_ZN_TO_ZN
#undef CATLASS_TILE_TEST_NAME
#undef CATLASS_TILE_TEST_SUFFIX
#undef CATLASS_TILE_TEST_SUFFIX_float
#undef CATLASS_TILE_TEST_SUFFIX_half
#undef CATLASS_TILE_TEST_CONCAT
#undef CATLASS_TILE_TEST_CONCAT_IMPL

///////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyL1ToL0A,
    TileCopyL1ToL0ATestAscend950,
    ::testing::Values(
        TestMatrixShape{128U, 64U},
        TestMatrixShape{1U, 128U},
        TestMatrixShape{64U, 42U},
        TestMatrixShape{123U, 8U}
    )
);

#endif // CATLASS_ARCH == 3510
