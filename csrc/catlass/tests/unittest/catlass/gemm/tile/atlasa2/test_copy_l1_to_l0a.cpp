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

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// TestCase for L1->L0A TileCopy Utilities
class TileCopyL1ToL0ATest : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
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

    template <class Element, bool isTrans=false>
    void BaseCheck(AscendCCallLog const &logTileCopy){
        // 验证调用名称：应为"LodaData" 或 “LoadDataWithTranspose”
        if constexpr(isTrans){
            ASSERT_EQ(logTileCopy.name, "LoadDataWithTranspose");
        } else {
            ASSERT_EQ(logTileCopy.name, "LoadData");
        }
        ASSERT_EQ(logTileCopy.args.size(), 3);

        // 验证数据类型是否一致
        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }
};

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN → zZ
// Element-type: no-except (float)
// Speciality: basic (per-fractal LoadData, no transpose)
TEST_P(TileCopyL1ToL0ATest, zNTozZTestBasic)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;
    using L0AType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A2>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyL1ToL0A<ArchTag, L1Type, L0AType> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    
    // AtlasA2实现下包括多次搬运
    ASSERT_EQ(logs.size(), _rows_by_fractal);

    for (uint32_t i = 0; i < _rows_by_fractal; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);
        
        // 验证输入输出张量的数据偏移是否正确
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * _col_round * C0_NUM_PER_FRACTAL * sizeof(Element));  // 验证L0A张量地址一致
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * BYTE_PER_FRACTAL);  // 验证L1张量地址一致

        // 验证LoadData2DParams参数是否正确
        const AscendC::LoadData2DParams* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();  // 获取LoadData2DParams参数
        ASSERT_EQ(loadDataArg->startIndex, _0);                 // 分型矩阵在源操作数中起始为第几个分形(0)
        ASSERT_EQ(loadDataArg->repeatTimes, _cols_by_fractal);  // 搬运的分型迭代次数（每个迭代处理512B数据)
        ASSERT_EQ(loadDataArg->srcStride, _rows_by_fractal);    // 相邻分形间前一个分形和后一分形起始地址间隔(单位:512B)
        ASSERT_EQ(loadDataArg->sid, _0);                        // 预留参数(AtlasA2), 为0
        ASSERT_EQ(loadDataArg->dstGap, _0);                     // 相邻迭代间目的操作数前一个至后一个分形起始间隔(单位:512B)
        ASSERT_EQ(loadDataArg->ifTranspose, _0);                // 分形内是否启用转置，为False
        ASSERT_EQ(loadDataArg->addrMode, _0);                   // 预留参数(AtlasA2), 为0
    }
}

// Data-path: zN → zZ
// Element-type: half
// Speciality: 2Param (single L1Type, L0A type auto-deduced as zZ)
TEST_P(TileCopyL1ToL0ATest, zNTozZTest2Param)
{
    using Element = half; // need to be float
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;
    // L0Type = void (default) — L0A type auto-deduced as zZ

    CopyL1ToL0A<ArchTag, L1Type> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);
    
    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _rows_by_fractal);

    for (uint32_t i = 0; i < _rows_by_fractal; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_FRACTAL * _cols_by_fractal);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * BYTE_PER_FRACTAL);

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(p->startIndex, _0);
        ASSERT_EQ(p->repeatTimes, _cols_by_fractal);
        ASSERT_EQ(p->srcStride, _rows_by_fractal);
        ASSERT_EQ(p->dstGap, _0);
        ASSERT_EQ(p->ifTranspose, _0);
        ASSERT_EQ(p->addrMode, _0);
    }
}

// ============================================================================
// Testsuite from **nN**
// ============================================================================

// Data-path: nN → zZ
// Element-type: half
// Speciality: basic (LoadData with in-fractal transpose, 2-byte element)
TEST_P(TileCopyL1ToL0ATest, nNTozZTestBasic)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::nN;
    using LayoutDst = layout::zZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;
    using L0AType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A2>;
    // static_assert(sizeof(Element) == 2, "This testcase(nNTozZTestBasic) is not suitable for 4B, 1B or even half-Byte(int4b_t) datatype");
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    // call CopyL1ToL0A
    CopyL1ToL0A<ArchTag, L1Type, L0AType> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    // Ready to execute copyL1ToL0A
    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);
    
    // 日志获取
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();  // 获取所有调用日志
    
    // AtlasA2实现下包括多次搬运
    ASSERT_EQ(logs.size(), _rows_by_fractal);

    for (uint32_t i = 0; i < _rows_by_fractal; i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);

        // 验证输入输出地址是否一致
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * BYTE_PER_FRACTAL * _cols_by_fractal);
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * BYTE_PER_FRACTAL);

        // 验证LoadData2DParams参数是否正确
        const AscendC::LoadData2DParams* loadDataArg = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(loadDataArg->startIndex, _0);                 // 分形矩阵起始(为0)
        ASSERT_EQ(loadDataArg->repeatTimes, _cols_by_fractal);  // 搬运的分型迭代次数（每个迭代处理16*16*2B数据)
        ASSERT_EQ(loadDataArg->srcStride, _rows_by_fractal);    // 相邻分形间前一个分形和后一分形起始地址间隔(单位:512B)
        ASSERT_EQ(loadDataArg->dstGap, _0);                     // 相邻迭代间目的操作数前一个至后一个分形起始间隔(单位:512B)
        ASSERT_EQ(loadDataArg->ifTranspose, _1);                // 分形内是否启用转置，为True(原分形内为ColumnMajor)
        ASSERT_EQ(loadDataArg->addrMode, _0);                   // 预留参数(AtlasA2), 为0
    }
}

// Data-path: nN → zZ
// Element-type: no-except (float)
// Speciality: float (LoadDataWithTranspose, LoadData2dTransposeParams)
TEST_P(TileCopyL1ToL0ATest, nNTozZTestFloat)
{
    using Element = float;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::nN;
    using LayoutDst = layout::zZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;
    using L0AType = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A2>;

    CopyL1ToL0A<ArchTag, L1Type, L0AType> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    // in the tranposed case, small fractal "n"
    uint32_t _cols_by_fractal_with_trans = CeilDiv<C0_NUM_PER_FRACTAL>(_col); 

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _rows_by_fractal);

    for (uint32_t i = 0; i < logs.size(); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element, true>(logTileCopy);

        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * layoutDst.stride(1) * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(1) * 2 * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2dTransposeParams>();
        ASSERT_EQ(p->startIndex, _0);
        ASSERT_EQ(p->repeatTimes, static_cast<uint8_t>(_cols_by_fractal_with_trans));
        // std::cout << "orgshape(1): " << layoutDst.orgShape(1) << ", _cols: " << _col << std::endl;
        ASSERT_EQ(p->srcStride, _rows_by_fractal);
        ASSERT_EQ(p->dstGap, _1);
        ASSERT_EQ(p->dstFracGap, _0);
        ASSERT_EQ(p->addrMode, _0);
    }
}

// ============================================================================
// Testsuite from **nZ**
// ============================================================================

// Data-path: nZ → zZ
// Element-type: half
// Speciality: basic (generic transpose via LoadData, ifTranspose=1)
TEST_P(TileCopyL1ToL0ATest, nZTozZTestBasic)
{
    using Element = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::zZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;

    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyL1ToL0A<ArchTag, L1Type> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element, true>();  // nZ uses trans round
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    // nZ→zZ 循环次数 = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0))
    ASSERT_EQ(logs.size(), CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)));

    for (uint32_t i = 0; i < logs.size(); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element>(logTileCopy);  // API = LoadData (not LoadDataWithTranspose)

        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * layoutDst.stride(1) * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(1) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2DParams>();
        ASSERT_EQ(p->startIndex, _0);
        ASSERT_EQ(p->repeatTimes, CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        ASSERT_EQ(p->srcStride, layoutSrc.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element)));
        ASSERT_EQ(p->dstGap, layoutDst.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element)) - 1);
        ASSERT_EQ(p->ifTranspose, _1);
        ASSERT_EQ(p->addrMode, _0);
    }
}

// Data-path: nZ → zZ
// Element-type: int8
// Speciality: int8 (LoadDataWithTranspose, LoadData2dTransposeParams)
TEST_P(TileCopyL1ToL0ATest, nZTozZTestInt8)
{
    using Element = int8_t;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::nZ;
    using LayoutDst = layout::zZ;

    using L1Type = Gemm::GemmType<Element, LayoutSrc, AscendC::TPosition::A1>;

    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    CopyL1ToL0A<ArchTag, L1Type> copyL1ToL0A;

    AscendC::LocalTensor<Element> l1Tensor;
    AscendC::LocalTensor<Element> l0aTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<Element, true>();
    setLayout<Element>(_row, _col, layoutSrc, layoutDst);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    copyL1ToL0A(l0aTensor, l1Tensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)));

    for (uint32_t i = 0; i < logs.size(); i++) {
        AscendCCallLog logTileCopy = logs[i];
        BaseCheck<Element, true>(logTileCopy);  // LoadDataWithTranspose

        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), i * layoutDst.stride(1) * 2 * sizeof(Element));
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), i * layoutSrc.stride(1) * sizeof(Element));

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::LoadData2dTransposeParams>();
        ASSERT_EQ(p->startIndex, static_cast<uint16_t>(0));
        ASSERT_EQ(p->repeatTimes, static_cast<uint8_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1))));
        ASSERT_EQ(p->srcStride, static_cast<uint16_t>(1));
        ASSERT_EQ(p->dstGap, static_cast<uint16_t>(0));
        ASSERT_EQ(p->dstFracGap, static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1));
        ASSERT_EQ(p->addrMode, static_cast<uint8_t>(0));
    }
}

INSTANTIATE_TEST_SUITE_P(
    CopyL1ToL0A,
    TileCopyL1ToL0ATest,
    ::testing::Values(
        TestMatrixShape{128U, 64U},
        TestMatrixShape{1U, 128U},
        TestMatrixShape{64U, 42U},
        TestMatrixShape{123U, 8U}
    )
);

#endif // !defined(CATLASS_ARCH) || CATLASS_ARCH == 2201
