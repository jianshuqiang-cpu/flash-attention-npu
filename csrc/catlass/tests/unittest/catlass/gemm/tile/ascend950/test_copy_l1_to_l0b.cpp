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

#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// TestCase for L1->L0A TileCopy Utilities
class TileCopyL1ToL0BTestAscend950 : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isTrans = false>
    void setShape() 
    {
        uint32_t row = GetParam().row;
        uint32_t col = GetParam().col;
        _setShape<Element, isTrans>(row, col);
    }

    template <class Element>
    void BaseCheck(AscendCCallLog const &logTileCopy, AscendC::LocalTensor<Element>& l0bTensor, AscendC::LocalTensor<Element>& l1Tensor) 
    {
        // The API name should be "LoadData"
        ASSERT_EQ(logTileCopy.name, "LoadData");

        // Check the given address
        auto logTileCopyL0BTensor = logTileCopy.GetArgsAt(0).RawValue();
        auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(1).RawValue();
        ASSERT_EQ(logTileCopyL0BTensor, &l0bTensor);
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
#define CATLASS_TILE_TEST_SUFFIX_int8_t Int8
#define CATLASS_TILE_TEST_SUFFIX(TYPE) CATLASS_TILE_TEST_CONCAT(CATLASS_TILE_TEST_SUFFIX_, TYPE)
#define CATLASS_TILE_TEST_NAME(PREFIX, TYPE) CATLASS_TILE_TEST_CONCAT(PREFIX, CATLASS_TILE_TEST_SUFFIX(TYPE))

#define ADD_TILE_COPY_TEST_L1_TO_L0B_NZ_TO_NZ(TYPE)                                                    \
TEST_P(TileCopyL1ToL0BTestAscend950, CATLASS_TILE_TEST_NAME(nZTonZTripleTest, TYPE))                   \
{                                                                                                      \
    using Element = TYPE;                                                                              \
    using ArchTag = Catlass::Arch::Ascend950;                                                          \
    using LayoutSrc = layout::nZ;                                                                      \
    using LayoutDst = layout::nZ;                                                                      \
    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;                         \
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;                        \
                                                                                                       \
    setShape<Element, true>();                                                                         \
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);                         \
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);                         \
                                                                                                       \
    AscendC::LocalTensor<Element> l1Tensor;                                                            \
    AscendC::LocalTensor<Element> l0bTensor;                                                           \
    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;                                                 \
    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);                                            \
                                                                                                       \
    auto logs = AscendCCallLogger::Instance().GetLogs();                                               \
    ASSERT_EQ(logs.size(), _1);                                                                        \
    auto logTileCopy = logs[0];                                                                        \
    BaseCheck<Element>(logTileCopy, l0bTensor, l1Tensor);                                              \
                                                                                                       \
    const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();           \
    ASSERT_EQ(loadDataArg->mStartPosition, _0);                                                        \
    ASSERT_EQ(loadDataArg->kStartPosition, _0);                                                        \
    ASSERT_EQ(loadDataArg->mStep, _cols_by_fractal);                                                   \
    ASSERT_EQ(loadDataArg->kStep, _rows_by_fractal);                                                   \
    ASSERT_EQ(loadDataArg->srcStride, _cols_by_fractal);                                               \
    ASSERT_EQ(loadDataArg->dstStride, _cols_by_fractal);                                               \
    ASSERT_EQ(loadDataArg->ifTranspose, false);                                                        \
}

#define ADD_TILE_COPY_TEST_L1_TO_L0B_ZZ_TO_NZ(TYPE)                                                    \
TEST_P(TileCopyL1ToL0BTestAscend950, CATLASS_TILE_TEST_NAME(zZTonZTest, TYPE))                         \
{                                                                                                      \
    using Element = TYPE;                                                                              \
    using ArchTag = Catlass::Arch::Ascend950;                                                          \
    using LayoutSrc = layout::zZ;                                                                      \
    using LayoutDst = layout::nZ;                                                                      \
    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;                         \
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;                        \
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();                                     \
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value; \
                                                                                                       \
    setShape<Element, true>();                                                                         \
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);                         \
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);                         \
    ASSERT_TRUE(isContiguous(layoutSrc));                                                              \
    ASSERT_TRUE(isContiguous(layoutDst));                                                              \
                                                                                                       \
    AscendC::LocalTensor<Element> l1Tensor;                                                            \
    AscendC::LocalTensor<Element> l0bTensor;                                                           \
    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;                                                 \
    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);                                            \
                                                                                                       \
    auto logs = AscendCCallLogger::Instance().GetLogs();                                               \
    ASSERT_EQ(logs.size(), CeilDiv<C0_NUM_PER_FRACTAL>(_row));                                         \
    uint32_t expectedSrcOffsetStep = RoundUp<ELE_NUM_PER_C0>(_col) * C0_NUM_PER_FRACTAL;               \
    uint32_t expectedKStep = CeilDiv<ELE_NUM_PER_C0>(_col_round);                                      \
    if constexpr (std::is_same_v<Element, float>) {                                                    \
        expectedKStep = RoundUp<2>(expectedKStep);                                                     \
    }                                                                                                  \
    for (uint32_t i = 0; i < logs.size(); ++i) {                                                       \
        auto logTileCopy = logs[i];                                                                    \
        ASSERT_EQ(logTileCopy.name, "LoadData");                                                      \
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));                                  \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * _col_round * C0_NUM_PER_FRACTAL * sizeof(Element)); \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * expectedSrcOffsetStep * sizeof(Element)); \
        const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();       \
        ASSERT_EQ(loadDataArg->mStartPosition, _0);                                                    \
        ASSERT_EQ(loadDataArg->kStartPosition, _0);                                                    \
        ASSERT_EQ(loadDataArg->mStep, _1);                                                             \
        ASSERT_EQ(loadDataArg->kStep, expectedKStep);                                                  \
        ASSERT_EQ(loadDataArg->srcStride, _1);                                                         \
        ASSERT_EQ(loadDataArg->dstStride, _cols_by_fractal);                                           \
        ASSERT_EQ(loadDataArg->ifTranspose, true);                                                     \
    }                                                                                                  \
}

#define ADD_TILE_COPY_TEST_L1_TO_L0B_ZN_B1_TO_NZ(TYPE)                                                 \
TEST_P(TileCopyL1ToL0BTestAscend950, CATLASS_TILE_TEST_NAME(zNB1TonZTest, TYPE))                       \
{                                                                                                      \
    using Element = TYPE;                                                                              \
    using ArchTag = Catlass::Arch::Ascend950;                                                          \
    using LayoutSrc = layout::zN;                                                                      \
    using LayoutDst = layout::nZ;                                                                      \
    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::B1>;                         \
    using L0BType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B2>;                        \
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();                                     \
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value; \
    constexpr bool IS_B8_B4 = AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, \
        float4_e2m1x2_t, float4_e1m2x2_t>;                                                            \
                                                                                                       \
    setShape<Element, true>();                                                                         \
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);                         \
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);                         \
                                                                                                       \
    AscendC::LocalTensor<Element> l1Tensor;                                                            \
    AscendC::LocalTensor<Element> l0bTensor;                                                           \
    CopyL1ToL0B<ArchTag, L1Type, L0BType> copyL1ToL0B;                                                 \
    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);                                            \
                                                                                                       \
    auto logs = AscendCCallLogger::Instance().GetLogs();                                               \
    bool singleLoad = !IS_B8_B4 || (_col_round % ELE_NUM_PER_C0 == 0);                                 \
    ASSERT_EQ(logs.size(), singleLoad ? 1U : _rows_by_fractal);                                        \
    uint32_t expectedSrcStride = CeilDiv<ELE_NUM_PER_FRACTAL>(RoundUp<C0_NUM_PER_FRACTAL>(_row) * ELE_NUM_PER_C0); \
                                                                                                       \
    for (uint32_t i = 0; i < logs.size(); ++i) {                                                       \
        auto logTileCopy = logs[i];                                                                    \
        ASSERT_EQ(logTileCopy.name, "LoadData");                                                      \
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));                                  \
        const auto* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();       \
        ASSERT_EQ(loadDataArg->kStartPosition, _0);                                                    \
        ASSERT_EQ(loadDataArg->kStep, CeilDiv<ELE_NUM_PER_C0>(_col_round));                            \
        ASSERT_EQ(loadDataArg->srcStride, expectedSrcStride);                                          \
        ASSERT_EQ(loadDataArg->dstStride, _cols_by_fractal);                                           \
        ASSERT_EQ(loadDataArg->ifTranspose, true);                                                     \
        if (singleLoad) {                                                                              \
            ASSERT_EQ(loadDataArg->mStartPosition, _0);                                                \
            ASSERT_EQ(loadDataArg->mStep, CeilDiv<C0_NUM_PER_FRACTAL>(_row_round));                    \
        } else {                                                                                       \
            ASSERT_EQ(loadDataArg->mStartPosition, i * (ELE_NUM_PER_C0 / C0_NUM_PER_FRACTAL));         \
            ASSERT_EQ(loadDataArg->mStep, ELE_NUM_PER_C0 / C0_NUM_PER_FRACTAL);                        \
        }                                                                                              \
    }                                                                                                  \
}

// ============================================================================
// Testsuite from **nZ**
// ============================================================================

// Data-path: nZ (L1) → nZ (L0B)
// Element-type: no-except (float)
// Speciality: Basic (single LoadData, no transpose)
TEST_P(TileCopyL1ToL0BTestAscend950, nZTonZTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::nZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;
    setShape<Element, true/*inner trans*/>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;
    CopyL1ToL0B<ArchTag, L1Type> copyL1ToL0B;
    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), _1);

    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy, l0bTensor, l1Tensor);

    const AscendC::LoadData2DParamsV2* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, _cols_by_fractal);     // Unit: 16(element) [it is trans]
    ASSERT_EQ(loadDataArg->kStep, _rows_by_fractal);     // Unit: 32(Byte)    [it is trans]
    ASSERT_EQ(loadDataArg->srcStride, _cols_by_fractal); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->dstStride, _cols_by_fractal); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->ifTranspose, false);
}

// nZ(B1) -> nZ(B2), 3-param Ascend950 specialization.
ADD_TILE_COPY_TEST_L1_TO_L0B_NZ_TO_NZ(float);

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN (L1) → nZ (L0B)
// Element-type: no-except (float)
// Speciality: Basic (single LoadData with ifTranspose enabled)
TEST_P(TileCopyL1ToL0BTestAscend950, zNTonZTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::nZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0bTensor;
    CopyL1ToL0B<ArchTag, L1Type> copyL1ToL0B;
    copyL1ToL0B(l0bTensor, l1Tensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), _1);

    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy, l0bTensor, l1Tensor);

    const AscendC::LoadData2DParamsV2* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParamsV2>();
    ASSERT_EQ(loadDataArg->mStartPosition, _0);
    ASSERT_EQ(loadDataArg->kStartPosition, _0);
    ASSERT_EQ(loadDataArg->mStep, CeilDiv<2>(_rows_by_fractal));     // Unit: 16(element) [it is trans]
    ASSERT_EQ(loadDataArg->kStep, 2 * _cols_by_fractal);             // Unit: 32(Byte)    [it is trans]
    ASSERT_EQ(loadDataArg->srcStride, CeilDiv<2>(_rows_by_fractal)); // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->dstStride, _cols_by_fractal);             // Unit: 512(Byte)
    ASSERT_EQ(loadDataArg->ifTranspose, true);
}

// zN(B1) -> nZ(B2), Ascend950 3-param transpose specializations.
ADD_TILE_COPY_TEST_L1_TO_L0B_ZN_B1_TO_NZ(half);
ADD_TILE_COPY_TEST_L1_TO_L0B_ZN_B1_TO_NZ(int8_t);

// ============================================================================
// Testsuite from **zZ**
// ============================================================================

// zZTonZ basic test, from zZ->nZ, Ascend950.
ADD_TILE_COPY_TEST_L1_TO_L0B_ZZ_TO_NZ(half);
ADD_TILE_COPY_TEST_L1_TO_L0B_ZZ_TO_NZ(float);

#undef ADD_TILE_COPY_TEST_L1_TO_L0B_ZN_B1_TO_NZ
#undef ADD_TILE_COPY_TEST_L1_TO_L0B_ZZ_TO_NZ
#undef ADD_TILE_COPY_TEST_L1_TO_L0B_NZ_TO_NZ
#undef CATLASS_TILE_TEST_NAME
#undef CATLASS_TILE_TEST_SUFFIX
#undef CATLASS_TILE_TEST_SUFFIX_int8_t
#undef CATLASS_TILE_TEST_SUFFIX_float
#undef CATLASS_TILE_TEST_SUFFIX_half
#undef CATLASS_TILE_TEST_CONCAT
#undef CATLASS_TILE_TEST_CONCAT_IMPL

/////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyL1ToL0B,
    TileCopyL1ToL0BTestAscend950,
    ::testing::Values(
        TestMatrixShape{128U, 64U},   // aligned
        TestMatrixShape{256U, 128U},  // aligned
        TestMatrixShape{64U, 64U},    // aligned
        TestMatrixShape{128U, 128U},  // aligned
        TestMatrixShape{64U, 48U}     // B8/B4 split branch
    )
);

#endif // CATLASS_ARCH == 3510
