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

#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;
using namespace AscendC;

// TestCase for L0C->GM TileCopy Utilities
class TileCopyL0CToGmTest : public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    // 重写SetUp方法，在每个测试用例执行前进行初始化操作
    void SetUp() override
    {
        // 调用父类的SetUp方法完成基础初始化
        AscendCTest::SetUp();
    }

    template <class Element>
    void setShape()
    {
        _setShape<Element, false>(GetParam().row, GetParam().col);
        _m = _row;
        _n = _col;
        _m_round = _row_round;
        _n_round = _col_round;
    }

    template <class ElementDst, class ElementSrc = void>
    void BaseCheck(AscendCCallLog logTileCopy)
    {
        // 验证调用名称：应为"Fixpipe"
        ASSERT_EQ(logTileCopy.name, "Fixpipe");
        // 验证参数数量：AscendC::FixpipeParamsV220版本下包含三个参数: (dst, src, intriParams)
        ASSERT_EQ(logTileCopy.args.size(), 3);

        // 验证数据类型：argsT 存的是模板参数类型（MakeArg<T>），args 存的是值类型（LocalTensor对象）
        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        const std::type_index& T1 = logTileCopy.GetArgsTAt(1).Type();
        ASSERT_EQ(T0, typeid(ElementDst));  // 验证dst模板参数类型
        if constexpr (!std::is_void_v<ElementSrc>) {
            ASSERT_EQ(T1, typeid(ElementSrc));  // 验证src模板参数类型
        } else {
            ASSERT_EQ(T1, typeid(ElementDst));  // 验证src模板参数类型
        }
    }

protected:
    uint16_t _m = 32;
    uint16_t _n = 32;
    uint16_t _m_round = 0;
    uint16_t _n_round = 0;
};

// ============================================================================
// Testsuite from **zN**
// ============================================================================

// Data-path: zN → RowMajor
// Element-type: no-except (float)
// Speciality: basic (NO_QUANT Fixpipe, single call)
TEST_P(TileCopyL0CToGmTest, zNToRowMajorTestBasic)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;

    constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    CopyL0CToGm<ArchTag, ElementAccumulator,Gemm::GemmType<ElementDst, LayoutDst>> copyL0CToGm;
    
    // prepare tensor
    AscendC::GlobalTensor<ElementAccumulator> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementDst>();
    setLayout<ElementAccumulator>(_m, _n, layoutSrc, layoutDst);

    // pre-assert
    ASSERT_EQ(quantPre, QuantMode_t::NoQuant);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    
    // call copyL0CToGm
    const uint8_t unitFlag = 0;
    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, unitFlag);

    // Get logs
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    auto logTileCopyGmTensor = logTileCopy.GetArgsAt(0).RawValue();  // 获取日志中GM张量地址
    auto logTileCopyL0cTensor = logTileCopy.GetArgsAt(1).RawValue();  // 获取日志中L0C张量地址
    ASSERT_EQ(logTileCopyGmTensor, &gmTensor);  // 验证GM张量地址一致
    ASSERT_EQ(logTileCopyL0cTensor, &l0cTensor);  // 验证L0C张量地址一致
    ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);

    // 验证参数是否正确
    const AscendC::FixpipeParamsV220* intriParams = logTileCopy.GetArgsAt(2).Value<AscendC::FixpipeParamsV220>();  // 获取FixpipeParams参数
    ASSERT_EQ(intriParams->nSize, _n);  // 源矩阵在n方向上大小
    ASSERT_EQ(intriParams->mSize, _m);  // 源矩阵在m方向上大小
    ASSERT_EQ(intriParams->srcStride, _m);  // 源矩阵相邻Z排布(分形间)的起始地址偏移(单位：16*sizeof(T))
    ASSERT_EQ(intriParams->dstStride, _n);  // 目标ND矩阵中每一行的元素个数
    ASSERT_EQ(intriParams->quantPre, quantPre);  // 量化模式
    ASSERT_EQ(intriParams->reluEn, _0);  // 是否开启ReLU
    ASSERT_EQ(intriParams->unitFlag, unitFlag);  // 单位标志位
}

// Data-path: zN → zN
// Element-type: no-except (float)
// Speciality: basic (NO_QUANT zN output, channelSplit enabled)
TEST_P(TileCopyL0CToGmTest, zNTozNTestBasic)
{
    using ElementAccumulator = float;
    using ElementDst = float;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;

    static_assert(std::is_same_v<ElementAccumulator, ElementDst>,
        "This testcase is for fp32->fp32 no-quant zN output");

    constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>> copyL0CToGm;

    AscendC::GlobalTensor<ElementDst> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementDst>();
    setLayout<ElementAccumulator>(_m, _n, layoutSrc, layoutDst);

    ASSERT_EQ(quantPre, QuantMode_t::NoQuant);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    const uint8_t unitFlag = 0;
    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, unitFlag);

    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logTileCopy);

    const AscendC::FixpipeParamsV220* intriParams = logTileCopy.GetArgsAt(2).Value<AscendC::FixpipeParamsV220>();
    // zN output: nSize = shape(2) * shape(3), mSize = shape(0) * shape(1)
    ASSERT_EQ(intriParams->nSize, layoutDst.shape(2) * layoutDst.shape(3));
    ASSERT_EQ(intriParams->mSize, layoutDst.shape(0) * layoutDst.shape(1));
    // srcStride = stride(3) / shape(2)
    ASSERT_EQ(intriParams->srcStride, layoutSrc.stride(3) / layoutSrc.shape(2));
    // dstStride = stride(3) / (BYTE_PER_C0 / sizeof(ElementDst))
    ASSERT_EQ(intriParams->dstStride, layoutDst.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst)));

    ASSERT_EQ(intriParams->quantPre, quantPre);
    ASSERT_EQ(intriParams->reluEn, _0);
    ASSERT_EQ(intriParams->unitFlag, unitFlag);
    ASSERT_TRUE(intriParams->isChannelSplit);
}

// Data-path: zN → RowMajor
// Element-type: half
// Speciality: half (NO_QUANT type-cast fp32→fp16, F322F16)
TEST_P(TileCopyL0CToGmTest, zNToRowMajorTestHalf)
{
    using ElementAccumulator = float;
    using ElementDst = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;

    static_assert(!std::is_same_v<ElementAccumulator, ElementDst>,
        "This testcase covers type-cast path (fp32→fp16)");

    constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;

    CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>> copyL0CToGm;

    AscendC::GlobalTensor<ElementDst> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementDst>();
    setLayout<ElementDst>(_m, _n, layoutSrc, layoutDst);

    ASSERT_EQ(quantPre, QuantMode_t::F322F16);
    ASSERT_TRUE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));

    const uint8_t unitFlag = 0;
    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::FixpipeParamsV220>();
    ASSERT_EQ(p->nSize, _n);
    ASSERT_EQ(p->mSize, _m);
    ASSERT_EQ(p->srcStride, layoutSrc.stride(3) / layoutSrc.stride(0));
    ASSERT_EQ(p->dstStride, layoutDst.stride(0));
    ASSERT_EQ(p->quantPre, quantPre);
    ASSERT_EQ(p->reluEn, _0);
    ASSERT_EQ(p->unitFlag, unitFlag);
}

// Data-path: zN → RowMajor
// Element-type: half
// Speciality: per-tensor-quant (PER_TENSOR scale, deqScalar set)
TEST_P(TileCopyL0CToGmTest, zNToRowMajorTestPerTensorQuant)
{
    using ElementAccumulator = float;
    using ElementDst = half;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;

    constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementAccumulator, ElementDst,
        ScaleGranularity::PER_TENSOR>::VALUE;

    // PER_TENSOR uses CopyL0CToGm with explicit Params{scale}
    CopyL0CToGm<ArchTag, ElementAccumulator, Gemm::GemmType<ElementDst, LayoutDst>,
        ScaleGranularity::PER_TENSOR> copyL0CToGm(CopyL0CToGm<ArchTag, ElementAccumulator,
            Gemm::GemmType<ElementDst, LayoutDst>, ScaleGranularity::PER_TENSOR>::Params{2.5f});

    AscendC::GlobalTensor<ElementDst> gmTensor;
    AscendC::LocalTensor<ElementAccumulator> l0cTensor;

    LayoutSrc layoutSrc;
    LayoutDst layoutDst;
    setShape<ElementDst>();
    setLayout<ElementDst>(_m, _n, layoutSrc, layoutDst);

    ASSERT_EQ(quantPre, QuantMode_t::F322F16);

    const uint8_t unitFlag = 0;
    copyL0CToGm(gmTensor, l0cTensor, layoutDst, layoutSrc, unitFlag);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    AscendCCallLog logTileCopy = logs[0];
    BaseCheck<ElementDst, ElementAccumulator>(logTileCopy);

    const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::FixpipeParamsV220>();
    ASSERT_EQ(p->nSize, _n);
    ASSERT_EQ(p->mSize, _m);
    ASSERT_EQ(p->srcStride, layoutSrc.stride(3) / layoutSrc.stride(0));
    ASSERT_EQ(p->dstStride, layoutDst.stride(0));
    ASSERT_EQ(p->quantPre, quantPre);
    ASSERT_NE(p->deqScalar, static_cast<uint64_t>(0));  // deqScalar is set from scale
    ASSERT_EQ(p->reluEn, _0);
    ASSERT_EQ(p->unitFlag, unitFlag);
}

INSTANTIATE_TEST_SUITE_P(
    CopyL0CToGm,
    TileCopyL0CToGmTest,
    ::testing::Values(
        TestMatrixShape{32U, 16U},   // aligned
        TestMatrixShape{32U, 32U},   // aligned
        TestMatrixShape{64U, 16U},   // larger rows
        TestMatrixShape{128U, 32U}   // larger
    )
);

#endif // CATLASS_ARCH == 2201
