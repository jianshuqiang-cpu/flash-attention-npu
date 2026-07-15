/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_HIGH_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_HIGH_PREC_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"  // AnyMask: Tile::CopyGm2UbTla for int32 GM->UB preload
// AnyMask wiring fix: this file is a BlockEpilogue partial specialization, so the
// primary template must be declared first. Forward-declare it here (do NOT include
// block_epilogue.hpp -- its includes would compile the catlass rescale_o and steal
// that override). fai_kernel.cpp includes block_epilogue.hpp later; its catlass
// high_prec include is skipped (guard already #defined above), so this local
// specialization wins.

namespace Catlass::Epilogue::Block {
template <class DispatchPolicy, class... Args>
class BlockEpilogue;

template <
    class OutputType_,
    class LayoutS_,
    class MaskType_,
    class TileCopy_>
class BlockEpilogue<
    EpilogueFAOnlineSoftmax,
    OutputType_,
    Gemm::GemmType<float, LayoutS_>,
    MaskType_,
    TileCopy_>
{
public:
    using DispatchPolicy = EpilogueFAOnlineSoftmax;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = float;
    using ElementMask = typename MaskType_::Element;
    using TileCopy = TileCopy_;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;
    using LayoutMask = typename MaskType_::Layout;

    using CopyGmToUbMask = typename TileCopy::CopyGmToUbMask;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_UB_P_ELEM_NUM = 8320;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    static constexpr ElementInput MIN_VALUE = -65504.0f;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  64 * sizeof(float);
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = GL_UB_TENSOR_OFFSET +  64 * sizeof(float);

        subBlockIdx_ = AscendC::GetSubBlockIdx(); 

        scaleValue = static_cast<ElementInput>(scaleValue_);
        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LL_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline
    void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m)
    {
        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - m + 1;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - m;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }
    
    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag qkReadyFlag, Arch::CrossCoreFlag softmaxReadyFlag)
    {   
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound + 1;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastMaxStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxStartAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ float*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (isFirstKvSTile) {
            if (n > 64) {
                ComputeScaleAndMax<ElementInput, ElementOutput, false>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, lastMaxStartAddr, pAddr, lastSumAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            } else {
                ComputeScaleAndMax64<ElementInput, ElementOutput, false>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, lastMaxStartAddr, pAddr, lastSumAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            }
        } else {
            if (n > 64) {
                ComputeScaleAndMax<ElementInput, ElementOutput, true>(
                    sAddr, nowMaxAddr, nowMaxStartAddr, lastMaxStartAddr, pAddr, nowSumAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            } else {
                ComputeScaleAndMax64<ElementInput, ElementOutput, true>(
                    sAddr, nowMaxAddr, nowMaxStartAddr, lastMaxStartAddr, pAddr, nowSumAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
        if (!isFirstKvSTile) {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag qkReadyFlag, Arch::CrossCoreFlag softmaxReadyFlag, bool enableDn)
    {   
        uint32_t nCopyOffset = RoundUp(actualBlockShape.m(), 32) / 2;
        uint32_t n = actualBlockShape.m() < nCopyOffset ? actualBlockShape.m() : nCopyOffset;
        n = subBlockIdx_ == 0 ? n : actualBlockShape.m() - n;
        if (n == 0) {
            WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            return;
        }
        uint32_t m = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound / 2 + 1;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastMaxStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxStartAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ float*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        
        uint32_t mAligendTile = mRound / 4;
        uint32_t mFirstTile = m % mAligendTile;
        uint32_t mAligned16TileNum = m / mAligendTile;
        if (mAligned16TileNum == 4) {
            mFirstTile = mAligendTile;
        }
        if (isFirstKvSTile) {
            if (mAligned16TileNum == 0) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Zero>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 1) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::One>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 2) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Two>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 3) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Three>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Four>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            }
        } else {
            if (mAligned16TileNum == 0) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Zero>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 1) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::One>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 2) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Two>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 3) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Three>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Four>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            }
        }
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * nCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
        
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = nRound / 16; // 分两次搬运
        dataCopyParams.blockLen = mRound / 2;
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride = mRound / 2;
        DataCopy(l1PTensorTla.data()[subBlockIdx_ * mRound * nCopyOffset], 
            lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM], dataCopyParams);
        DataCopy(l1PTensorTla.data()[mRound * 8 + subBlockIdx_ * mRound * nCopyOffset],
            lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM + blockStride * 64], dataCopyParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
    }

    template <class TensorP, class TensorMask>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, TensorMask &gmMaskTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag qkReadyFlag, Arch::CrossCoreFlag softmaxReadyFlag,
         uint32_t triUp,  uint32_t triDown, uint32_t globalWindowSize,  uint32_t localWindowSize,
         uint32_t kvSStartIdx, uint32_t kvSEndIdx, uint32_t maskType)
    {
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound + 1;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;

        // calc mask shift in gm
        uint32_t gmOffsetMaskRow = 0;
        uint32_t gmOffsetMaskColumn = 0;
        uint32_t maskColumn = 0;
        uint32_t addMaskUbOffset = 0;
        if (maskType == 1) {
            if (triUp >= kvSStartIdx) {
                gmOffsetMaskRow = triUp - kvSStartIdx;
                gmOffsetMaskColumn = 0;
                maskColumn = kvSEndIdx - kvSStartIdx;
            } else {
                gmOffsetMaskRow = 0;
                gmOffsetMaskColumn = kvSStartIdx - triUp;
                maskColumn = n;
                addMaskUbOffset = 0;
            }
        } else {
            if (maskType == 2) {
                gmOffsetMaskRow = triDown - actualBlockShape.m();
            } else if (maskType == 3) {
                gmOffsetMaskRow = globalWindowSize + localWindowSize - 1;
            }
            gmOffsetMaskColumn = kvSStartIdx;
            maskColumn = kvSEndIdx - kvSStartIdx;
        }

        uint32_t maskColumnRound = RoundUp(maskColumn, 128);
        auto gMaskTensorTlaTile = GetTile(gmMaskTensorTla,
                        tla::MakeCoord(gmOffsetMaskRow + subBlockIdx_ * mCopyOffset, gmOffsetMaskColumn), tla::MakeShape(m, maskColumnRound));
        CopyGmToUbMask copyGmToUbMask;
        auto ubMaskLayoutTla = tla::MakeLayout<ElementMask, LayoutMask>(m, maskColumnRound);
        auto ubMaskTensorTla = tla::MakeTensor(maskUbTensor, ubMaskLayoutTla, Arch::PositionUB{});
        auto ubMaskTensorTlaTile = GetTile(ubMaskTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, maskColumnRound));

        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastMaxStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxStartAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ float*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementMask *maskUbAddr = (__ubuf__ ElementMask *)maskUbTensor.GetPhyAddr();

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        copyGmToUbMask(ubMaskTensorTlaTile, gMaskTensorTlaTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);

        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (isFirstKvSTile) {
            if (n > 64) {
                ComputeScaleAndMaxMask<ElementInput, ElementOutput, false>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, lastMaxStartAddr, pAddr, lastSumAddr, maskUbAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            } else {
                ComputeScaleAndMaxMask64<ElementInput, ElementOutput, false>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, lastMaxStartAddr, pAddr, lastSumAddr, maskUbAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            }
        } else {
            if (n > 64) {
                ComputeScaleAndMaxMask<ElementInput, ElementOutput, true>(
                    sAddr, nowMaxAddr, nowMaxStartAddr, lastMaxStartAddr, pAddr, nowSumAddr, maskUbAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            } else {
                ComputeScaleAndMaxMask64<ElementInput, ElementOutput, true>(
                    sAddr, nowMaxAddr, nowMaxStartAddr, lastMaxStartAddr, pAddr, nowSumAddr, maskUbAddr, m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
        if (!isFirstKvSTile) {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // AnyMask (v1, design D4 kernel-side vectorized, fused): call the fused
    // ComputeScaleAndAnyMask which generates the mask predicate inline in its row
    // loop (just before Select) from maskr/holel/holes -- no maskUb, no separate
    // gen pass. blockFullyMasked (sparse_compute) fills MIN_VALUE directly.
    // maskRowBase = curBatch*maxQSeqlen + qSOffset; overload adds subBlockIdx_*
    // mCopyOffset. ⚠️ MicroAPI in ComputeScaleAndAnyMask + pipeline flags need 950.
    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla,
        __gm__ int32_t *maskrGm, __gm__ int32_t *holelGm, __gm__ int32_t *holesGm,
        GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag qkReadyFlag, Arch::CrossCoreFlag softmaxReadyFlag,
         uint32_t maskRowBase, uint32_t kvSStartIdx, uint32_t kvSEndIdx,
         uint32_t Hn, uint32_t blockFullyMasked)
    {
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound + 1;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;

        // AnyMask fused (design D4): ComputeScaleAndAnyMask generates the mask
        // predicate inline in its row loop (just before Select) from maskr/holel/
        // holes -- no maskUb buffer, no separate gen pass. qRowBase is the absolute
        // Q row (maskRowBase + subBlockIdx_*mCopyOffset); maskr indexed [qRow],
        // holel/holes indexed [qRow*Hn+h]. blockFullyMasked skips gen (sparse_compute).
        // FIXME(950): maskr/holel/holes GM loads inside ComputeScaleAndAnyMask may
        // be MTE2; the V_MTE2/MTE2_V pair (old maskUb copy) is removed here -- verify
        // flag balance / add LocalMemBar on 950.
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastMaxStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxStartAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ float*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        uint32_t qRowBase = maskRowBase + subBlockIdx_ * mCopyOffset;

        // ---- AnyMask: preload maskr/holel/holes GM->UB (maskUbTensor space) ----
        // UB layout (int32): maskr[m], holel[m*Hn], holes[m*Hn]. Position ramp generated
        // in-register by Reg::Arange inside ComputeScaleAndAnyMask (no UB position buffer).
        // FIXME(950-perf): scalar GM->UB preload (correct, slow) -- replace with vectorized
        // CopyGm2UbTla<int32> once the GlobalTensor-ctor/SetGlobalBuffer/GetTile API is resolved.
        // FIXME(950): UB budget -- m*(1+2*Hn) int32 must fit maskUbTensor; lower HN_MAX if not.
        __ubuf__ int32_t *maskrUbPtr = reinterpret_cast<__ubuf__ int32_t*>(maskUbTensor.GetPhyAddr());
        __ubuf__ int32_t *holelUbPtr = maskrUbPtr + m;
        __ubuf__ int32_t *holesUbPtr = maskrUbPtr + m + m * Hn;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        if (blockFullyMasked == 0) {
            // GM->UB via MTE2 DataCopy (mirrors the dense-mask path at :420-440). Scalar
            // GM dereference does NOT go through the MTE2 pipe, so the V_MTE2/MTE2_V flag
            // pair below does not gate it and the V-pipe LoadAlign in ComputeScaleAndAnyMask
            // reads unsynchronized/stale UB -> wrong maskr/holel/holes -> wrong predicate.
            // Copy the int32 tensors as ElementMask (uint8) bytes via the proven
            // CopyGmToUbMask functor (Tile::CopyGm2UbTla only specializes for ElementMask
            // layouts; m int32 = 4m bytes). 32B alignment holds under seqlen_q%16==0 (host
            // guard): m and qRowBase are 8-aligned -> byte offsets/lengths are 32B.
            AscendC::GlobalTensor<ElementMask> gMaskrAny, gHolelAny, gHolesAny;
            gMaskrAny.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMask*>(maskrGm));
            gHolelAny.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMask*>(holelGm));
            gHolesAny.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMask*>(holesGm));
            auto gmMaskrTla = tla::MakeTensor(gMaskrAny,
                tla::MakeLayout<ElementMask, LayoutMask>(1, (qRowBase + m) * sizeof(int32_t)), Arch::PositionGM{});
            auto gmHolelTla = tla::MakeTensor(gHolelAny,
                tla::MakeLayout<ElementMask, LayoutMask>(1, (qRowBase * Hn + m * Hn) * sizeof(int32_t)), Arch::PositionGM{});
            auto gmHolesTla = tla::MakeTensor(gHolesAny,
                tla::MakeLayout<ElementMask, LayoutMask>(1, (qRowBase * Hn + m * Hn) * sizeof(int32_t)), Arch::PositionGM{});
            auto gmMaskrTile = GetTile(gmMaskrTla, tla::MakeCoord(0, qRowBase * sizeof(int32_t)),
                                       tla::MakeShape(1, m * sizeof(int32_t)));
            auto gmHolelTile = GetTile(gmHolelTla, tla::MakeCoord(0, qRowBase * Hn * sizeof(int32_t)),
                                       tla::MakeShape(1, m * Hn * sizeof(int32_t)));
            auto gmHolesTile = GetTile(gmHolesTla, tla::MakeCoord(0, qRowBase * Hn * sizeof(int32_t)),
                                       tla::MakeShape(1, m * Hn * sizeof(int32_t)));

            auto ubAnyTla = tla::MakeTensor(maskUbTensor,
                tla::MakeLayout<ElementMask, LayoutMask>(1, (m + 2 * m * Hn) * sizeof(int32_t)), Arch::PositionUB{});
            auto ubMaskrTile = GetTile(ubAnyTla, tla::MakeCoord(0, 0),
                                       tla::MakeShape(1, m * sizeof(int32_t)));
            auto ubHolelTile = GetTile(ubAnyTla, tla::MakeCoord(0, m * sizeof(int32_t)),
                                       tla::MakeShape(1, m * Hn * sizeof(int32_t)));
            auto ubHolesTile = GetTile(ubAnyTla, tla::MakeCoord(0, (m + m * Hn) * sizeof(int32_t)),
                                       tla::MakeShape(1, m * Hn * sizeof(int32_t)));

            CopyGmToUbMask copyAny;  // existing uint8 RowMajor functor (same as :422 dense path)
            copyAny(ubMaskrTile, gmMaskrTile);
            copyAny(ubHolelTile, gmHolelTile);
            copyAny(ubHolesTile, gmHolesTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);

        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(qkReadyFlag);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (isFirstKvSTile) {
            if (n > 64) {
                ComputeScaleAndAnyMask<ElementInput, ElementOutput, false>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, lastMaxStartAddr, pAddr, lastSumAddr,
                    m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound,
                    maskrUbPtr, holelUbPtr, holesUbPtr, kvSStartIdx, Hn, blockFullyMasked);
            } else {
                ComputeScaleAndAnyMask64<ElementInput, ElementOutput, false>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, lastMaxStartAddr, pAddr, lastSumAddr,
                    m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound,
                    maskrUbPtr, holelUbPtr, holesUbPtr, kvSStartIdx, Hn, blockFullyMasked);
            }
        } else {
            if (n > 64) {
                ComputeScaleAndAnyMask<ElementInput, ElementOutput, true>(
                    sAddr, nowMaxAddr, nowMaxStartAddr, lastMaxStartAddr, pAddr, nowSumAddr,
                    m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound,
                    maskrUbPtr, holelUbPtr, holesUbPtr, kvSStartIdx, Hn, blockFullyMasked);
            } else {
                ComputeScaleAndAnyMask64<ElementInput, ElementOutput, true>(
                    sAddr, nowMaxAddr, nowMaxStartAddr, lastMaxStartAddr, pAddr, nowSumAddr,
                    m, nLoops, tailN, nPadding, scaleValue, 128, blockStride, nRound,
                    maskrUbPtr, holelUbPtr, holesUbPtr, kvSStartIdx, Hn, blockFullyMasked);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        // Replenish the V_MTE2(4) token consumed by the preload WaitFlag above
        // (mirrors non-AnyMask line ~466). V is done reading maskr/holel/holes UB,
        // so the next tile's preload may overwrite maskUbTensor.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(qkReadyFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_P_ELEM_NUM],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(softmaxReadyFlag);
        if (!isFirstKvSTile) {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    
private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<ElementMask> maskUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    uint32_t subBlockIdx_;

    enum class MAligendTileNum  {
        Zero = 0,
        One = 1,
        Two = 2,
        Three = 3,
        Four = 4
    };

    template <typename ElementS, typename ElementP, bool isUpdate>
    __simd_vf__ inline void ComputeScaleAndMax(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *newMaxUbStart, __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb,
        __ubuf__ ElementS *expSumUb, uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize, uint32_t blockStride, uint32_t repeatStride)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg;
        RegTensor<float> srcVreg;
        RegTensor<float> srcVreg_unroll;
        RegTensor<float> srcVreg_unroll_new;
        RegTensor<float> maxSrcVreg;
        RegTensor<float> maxTmpVreg;
        RegTensor<float> maxBrcVreg;
        RegTensor<float> expEvenVreg;
        RegTensor<float> expOddVreg;
        RegTensor<float> expSumVreg;
        UnalignRegForStore maxUreg;
        UnalignRegForStore expSumUreg;

        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;

        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            LoadAlign(srcVreg_unroll, srcUb + i * S2BaseSize + FLOAT_REP_SIZE);
            Muls(srcVreg, srcVreg, dScale, pregFull);
            Muls(srcVreg_unroll, srcVreg_unroll, dScale, pregTailN);
            Select(srcVreg_unroll_new, srcVreg_unroll, minVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize, srcVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize + FLOAT_REP_SIZE, srcVreg_unroll_new, pregFull);
            Max(maxTmpVreg, srcVreg, srcVreg_unroll_new, pregFull);
            // [0, 1, 2, 4, 5, .., 63] -> reduce -> [63, 0, 0, .., 0]
            Reduce<AscendC::MicroAPI::ReduceType::MAX, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                maxSrcVreg, maxTmpVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxSrcVreg, maxUreg, 1);
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            newMaxUb, maxUreg, 0);
        if constexpr (isUpdate) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign(maxSrcVreg, newMaxUbStart);
            LoadAlign(maxTmpVreg, LastMaxUbStart);
            Max(maxSrcVreg, maxSrcVreg, maxTmpVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(newMaxUbStart, maxSrcVreg, pregFull);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxBrcVreg, newMaxUbStart + i);
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                srcVreg, srcVreg_unroll, srcUb + i * S2BaseSize);
            ExpSub(expEvenVreg, srcVreg, maxBrcVreg, pregFull);
            ExpSub(expOddVreg, srcVreg_unroll, maxBrcVreg, pregFull);
            Add(expSumVreg, expEvenVreg, expOddVreg, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumVreg, expSumUreg, 1);
                
            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(vreg_exp_even_bf16, expEvenVreg, pregFull);
                Cast<bfloat16_t, float, castTraitOne>(vreg_exp_odd_bf16, expOddVreg, pregFull);
                Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ bfloat16_t *&)expUb), vreg_exp_bf16, blockStride, 1, preg_all_b16);
            } else {
                Cast<half, float, castTraitZero>(vreg_exp_even_f16, expEvenVreg, pregFull);
                Cast<half, float, castTraitOne>(vreg_exp_odd_f16, expOddVreg, pregFull);
                Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ half *&)expUb), vreg_exp_f16, blockStride, 1, preg_all_b16);
            }
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumUreg, 0);
    }

    template <typename ElementS, typename ElementP, bool isUpdate>
    __simd_vf__ inline void ComputeScaleAndMax64(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *newMaxUbStart, __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb,
        __ubuf__ ElementS *expSumUb, uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize, uint32_t blockStride, uint32_t repeatStride)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg;
        RegTensor<float> srcVreg;
        RegTensor<float> srcVreg_unroll;
        RegTensor<float> srcVreg_unroll_new;
        RegTensor<float> maxSrcVreg;
        RegTensor<float> maxTmpVreg;
        RegTensor<float> maxBrcVreg;
        RegTensor<float> expEvenVreg;
        RegTensor<float> expOddVreg;
        RegTensor<float> expSumVreg;
        UnalignRegForStore maxUreg;
        UnalignRegForStore expSumUreg;

        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;

        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            Muls(srcVreg, srcVreg, dScale, pregTailN);
            Select(srcVreg_unroll_new, srcVreg, minVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize, srcVreg_unroll_new, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::MAX, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                maxSrcVreg, srcVreg_unroll_new, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxSrcVreg, maxUreg, 1);
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            newMaxUb, maxUreg, 0);
        if constexpr (isUpdate) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign(maxSrcVreg, newMaxUbStart);
            LoadAlign(maxTmpVreg, LastMaxUbStart);
            Max(maxSrcVreg, maxSrcVreg, maxTmpVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(newMaxUbStart, maxSrcVreg, pregFull);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxBrcVreg, newMaxUbStart + i);
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            ExpSub(expEvenVreg, srcVreg, maxBrcVreg, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            expSumVreg, expEvenVreg, pregFull);
            StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumVreg, expSumUreg, 1);
                
            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(vreg_exp_bf16, expEvenVreg, pregFull);
                DeInterleave(vreg_exp_even_bf16, vreg_exp_odd_bf16, vreg_exp_bf16, vreg_exp_bf16);
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ bfloat16_t *&)expUb), vreg_exp_even_bf16, blockStride, 1, preg_all_b16);
            } else {
                Cast<half, float, castTraitZero>(vreg_exp_f16, expEvenVreg, pregFull);
                DeInterleave(vreg_exp_even_f16, vreg_exp_odd_f16, vreg_exp_f16, vreg_exp_f16);
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ half *&)expUb), vreg_exp_even_f16, blockStride, 1, preg_all_b16);
            }
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumUreg, 0);
    }

    // AnyMask fused (design D4): generate the per-element mask predicate INLINE
    // in the row loop (just before Select) from maskr/holel/holes -- one m-row
    // pass, no separate gen pass and no maskUb buffer. pregMasked =
    // (idx >= maskr) OR (any hole holel<=idx<holel+holes); Select swaps to
    // (minVreg, srcVreg, pregMasked) so masked -> MIN_VALUE. blockFullyMasked
    // (sparse_compute) skips gen and fills MIN_VALUE directly.
    // ⚠️ UNVERIFIED BLIND: Arange / Compares cmp-op (CMPGE,CMPLT) / MaskReg And,Or
    // / int32 Add / GM scalar-broadcast load (DIST_BRC_B32 for int32) -- confirm
    // on 950. Only the mask section differs from ComputeScaleAndMaxMask.
    template <typename ElementS, typename ElementP, bool isUpdate>
    __simd_vf__ inline void ComputeScaleAndAnyMask(
        __ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *newMaxUbStart,
        __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb, __ubuf__ ElementS *expSumUb,
        uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale,
        uint16_t S2BaseSize, uint32_t blockStride, uint32_t repeatStride,
        __ubuf__ int32_t *maskrUb, __ubuf__ int32_t *holelUb, __ubuf__ int32_t *holesUb,
        uint32_t kvSStartIdx, uint32_t Hn, uint32_t blockFullyMasked)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg, srcVreg, srcVreg_unroll, srcVreg_mask, srcVreg_mask_unroll, srcVreg_unroll_new;
        RegTensor<float> maxSrcVreg, maxTmpVreg, maxBrcVreg, expEvenVreg, expOddVreg, expSumVreg;
        UnalignRegForStore maxUreg, expSumUreg;
        RegTensor<int32_t> posVreg0, posVreg1, maskrVreg, holelVreg, holesVreg, holerVreg;
        RegTensor<bfloat16_t> vreg_exp_even_bf16, vreg_exp_odd_bf16, vreg_exp_bf16;
        RegTensor<half> vreg_exp_even_f16, vreg_exp_odd_f16, vreg_exp_f16;
        MaskReg pregMasked0, pregMasked1, pregMr, pregLE, pregLT, pregHole;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        MaskReg preg_all_b16 = CreateMask<half, MaskPattern::ALL>();
        constexpr static CastTrait castTraitZero = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitOne  = {RegLayout::ONE,  SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};

        Duplicate(minVreg, MIN_VALUE);
        // position ramp [kvSStartIdx .. +128) generated in-register (Reg::Arange), 2 chunks x 64.
        AscendC::Reg::Arange<int32_t>(posVreg0, static_cast<int32_t>(kvSStartIdx));
        AscendC::Reg::Arange<int32_t>(posVreg1, static_cast<int32_t>(kvSStartIdx + 64));
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            LoadAlign(srcVreg_unroll, srcUb + i * S2BaseSize + FLOAT_REP_SIZE);
            Muls(srcVreg, srcVreg, dScale, pregFull);
            Muls(srcVreg_unroll, srcVreg_unroll, dScale, pregTailN);

            // ---- fused AnyMask predicate (maskr/holel/holes preloaded to UB) ----
            if (blockFullyMasked != 0) {
                Duplicate(srcVreg_mask, MIN_VALUE);
                Duplicate(srcVreg_mask_unroll, MIN_VALUE);
            } else {
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(maskrVreg, maskrUb + i);          // broadcast maskr[i]
                Compare<int32_t, AscendC::CMPMODE::GE>(pregMr, posVreg0, maskrVreg, pregFull);          // pos >= maskr
                pregMasked0 = pregMr;
                Compare<int32_t, AscendC::CMPMODE::GE>(pregMr, posVreg1, maskrVreg, pregFull);
                pregMasked1 = pregMr;
                for (uint32_t h = 0; h < Hn; ++h) {
                    LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(holelVreg, holelUb + i * Hn + h);
                    LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(holesVreg, holesUb + i * Hn + h);
                    Add(holerVreg, holelVreg, holesVreg, pregFull);                             // holer = holel + holes
                    Compare<int32_t, AscendC::CMPMODE::GE>(pregLE, posVreg0, holelVreg, pregFull);      // pos >= holel
                    Compare<int32_t, AscendC::CMPMODE::LT>(pregLT, posVreg0, holerVreg, pregFull);      // pos < holer
                    And(pregHole, pregLE, pregLT, pregFull);
                    Or(pregMasked0, pregMasked0, pregHole, pregFull);
                    Compare<int32_t, AscendC::CMPMODE::GE>(pregLE, posVreg1, holelVreg, pregFull);
                    Compare<int32_t, AscendC::CMPMODE::LT>(pregLT, posVreg1, holerVreg, pregFull);
                    And(pregHole, pregLE, pregLT, pregFull);
                    Or(pregMasked1, pregMasked1, pregHole, pregFull);
                }
                // masked -> MIN_VALUE (Select src0/src1 swapped)
                Select(srcVreg_mask, minVreg, srcVreg, pregMasked0);
                Select(srcVreg_mask_unroll, minVreg, srcVreg_unroll, pregMasked1);
            }

            Select(srcVreg_unroll_new, srcVreg_mask_unroll, minVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg_mask, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize + FLOAT_REP_SIZE, srcVreg_unroll_new, pregFull);
            Max(maxTmpVreg, srcVreg_mask, srcVreg_unroll_new, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::MAX, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(maxSrcVreg, maxTmpVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxSrcVreg, maxUreg, 1);
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxUreg, 0);
        if constexpr (isUpdate) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign(maxSrcVreg, newMaxUbStart);
            LoadAlign(maxTmpVreg, LastMaxUbStart);
            Max(maxSrcVreg, maxSrcVreg, maxTmpVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(newMaxUbStart, maxSrcVreg, pregFull);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxBrcVreg, newMaxUbStart + i);
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(srcVreg, srcVreg_unroll, srcUb + i * S2BaseSize);
            ExpSub(expEvenVreg, srcVreg, maxBrcVreg, pregFull);
            ExpSub(expOddVreg, srcVreg_unroll, maxBrcVreg, pregFull);
            Add(expSumVreg, expEvenVreg, expOddVreg, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)expSumUb), expSumVreg, expSumUreg, 1);
            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(vreg_exp_even_bf16, expEvenVreg, pregFull);
                Cast<bfloat16_t, float, castTraitOne>(vreg_exp_odd_bf16, expOddVreg, pregFull);
                Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16, (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ bfloat16_t *&)expUb), vreg_exp_bf16, blockStride, 1, preg_all_b16);
            } else {
                Cast<half, float, castTraitZero>(vreg_exp_even_f16, expEvenVreg, pregFull);
                Cast<half, float, castTraitOne>(vreg_exp_odd_f16, expOddVreg, pregFull);
                Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)expUb), vreg_exp_f16, blockStride, 1, preg_all_b16);
            }
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)expSumUb), expSumUreg, 0);
    }

    // AnyMask fused, n<=64 (1 chunk of 64). Mirrors ComputeScaleAndMaxMask64;
    // only the mask section is replaced by inline predicate gen.
    template <typename ElementS, typename ElementP, bool isUpdate>
    __simd_vf__ inline void ComputeScaleAndAnyMask64(
        __ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *newMaxUbStart,
        __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb, __ubuf__ ElementS *expSumUb,
        uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale,
        uint16_t S2BaseSize, uint32_t blockStride, uint32_t repeatStride,
        __ubuf__ int32_t *maskrUb, __ubuf__ int32_t *holelUb, __ubuf__ int32_t *holesUb,
        uint32_t kvSStartIdx, uint32_t Hn, uint32_t blockFullyMasked)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg, srcVreg, srcVreg_unroll_new, maxSrcVreg, maxTmpVreg, maxBrcVreg, expEvenVreg, expOddVreg, expSumVreg;
        UnalignRegForStore maxUreg, expSumUreg;
        RegTensor<int32_t> posVreg0, maskrVreg, holelVreg, holesVreg, holerVreg;
        RegTensor<bfloat16_t> vreg_exp_even_bf16, vreg_exp_odd_bf16, vreg_exp_bf16;
        RegTensor<half> vreg_exp_even_f16, vreg_exp_odd_f16, vreg_exp_f16;
        MaskReg pregMasked0, pregMr, pregLE, pregLT, pregHole;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        constexpr static CastTrait castTraitZero = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitOne  = {RegLayout::ONE,  SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};

        Duplicate(minVreg, MIN_VALUE);
        AscendC::Reg::Arange<int32_t>(posVreg0, static_cast<int32_t>(kvSStartIdx));  // position ramp in-register
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            Muls(srcVreg, srcVreg, dScale, pregTailN);
            if (blockFullyMasked != 0) {
                Duplicate(srcVreg_unroll_new, MIN_VALUE);
            } else {
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(maskrVreg, maskrUb + i);
                Compare<int32_t, AscendC::CMPMODE::GE>(pregMr, posVreg0, maskrVreg, pregFull);
                pregMasked0 = pregMr;
                for (uint32_t h = 0; h < Hn; ++h) {
                    LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(holelVreg, holelUb + i * Hn + h);
                    LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(holesVreg, holesUb + i * Hn + h);
                    Add(holerVreg, holelVreg, holesVreg, pregFull);
                    Compare<int32_t, AscendC::CMPMODE::GE>(pregLE, posVreg0, holelVreg, pregFull);
                    Compare<int32_t, AscendC::CMPMODE::LT>(pregLT, posVreg0, holerVreg, pregFull);
                    And(pregHole, pregLE, pregLT, pregFull);
                    Or(pregMasked0, pregMasked0, pregHole, pregFull);
                }
                Select(srcVreg_unroll_new, minVreg, srcVreg, pregMasked0);  // masked -> MIN_VALUE
            }
            Select(srcVreg_unroll_new, srcVreg_unroll_new, minVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg_unroll_new, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::MAX, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(maxSrcVreg, srcVreg_unroll_new, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxSrcVreg, maxUreg, 1);
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxUreg, 0);
        if constexpr (isUpdate) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign(maxSrcVreg, newMaxUbStart);
            LoadAlign(maxTmpVreg, LastMaxUbStart);
            Max(maxSrcVreg, maxSrcVreg, maxTmpVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(newMaxUbStart, maxSrcVreg, pregFull);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxBrcVreg, newMaxUbStart + i);
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            ExpSub(expEvenVreg, srcVreg, maxBrcVreg, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(expSumVreg, expEvenVreg, pregFull);
            StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)expSumUb), expSumVreg, expSumUreg, 1);
            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(vreg_exp_bf16, expEvenVreg, pregFull);
                DeInterleave(vreg_exp_even_bf16, vreg_exp_odd_bf16, vreg_exp_bf16, vreg_exp_bf16);
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ bfloat16_t *&)expUb), vreg_exp_even_bf16, blockStride, 1, preg_all_b16);
            } else {
                Cast<half, float, castTraitZero>(vreg_exp_f16, expEvenVreg, pregFull);
                DeInterleave(vreg_exp_even_f16, vreg_exp_odd_f16, vreg_exp_f16, vreg_exp_f16);
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)expUb), vreg_exp_even_f16, blockStride, 1, preg_all_b16);
            }
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)expSumUb), expSumUreg, 0);
    }

    template <typename ElementS, typename ElementP, bool isUpdate>
    __simd_vf__ inline void ComputeScaleAndMaxMask(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *newMaxUbStart, __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb,
        __ubuf__ ElementS *expSumUb, __ubuf__ ElementMask *maskUb, uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize, uint32_t blockStride, uint32_t repeatStride)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg;
        RegTensor<float> srcVreg;
        RegTensor<float> srcVreg_unroll;
        RegTensor<float> srcVreg_mask;
        RegTensor<float> srcVreg_mask_unroll;
        RegTensor<float> srcVreg_unroll_new;
        RegTensor<float> maxSrcVreg;
        RegTensor<float> maxTmpVreg;
        RegTensor<float> maxBrcVreg;
        RegTensor<float> expEvenVreg;
        RegTensor<float> expOddVreg;
        RegTensor<float> expSumVreg;
        UnalignRegForStore maxUreg;
        UnalignRegForStore expSumUreg;

        RegTensor<uint8_t> maskVreg;
        RegTensor<uint8_t> maskVreg1;
        RegTensor<half> maskVregb16;
        RegTensor<half> maskVregb16_new;
        RegTensor<half> maskVregb16_unroll_new;
        RegTensor<float> maskVregb32;
        RegTensor<float> maskVregb32_unroll;

        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;

        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        
        MaskReg pregCompare;
        MaskReg pregCompare_unroll;
        MaskReg pregFull1 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        // MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<half, MaskPattern::ALL>();
        
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            LoadAlign(srcVreg_unroll, srcUb + i * S2BaseSize + FLOAT_REP_SIZE);
            
            Muls(srcVreg, srcVreg, dScale, pregFull);
            Muls(srcVreg_unroll, srcVreg_unroll, dScale, pregTailN);
            // mask
            // 1. 数据下采样重复搬2次 load 256 个 uint8_t  vl
            // 2. cast to uint16_t  128个完整mask   vl/2
            // 3. interleave  uint16_t 将128个数切分成前一半和后一半
            // 4. 分别 cast 成 64个元素的 fp32 mask 和 源数据对应
            LoadAlign<ElementMask, LoadDist::DIST_US_B8>(maskVreg, maskUb + i * 128);
            
            Cast<half, ElementMask, castTraitZero>(maskVregb16, maskVreg, preg_all_b16);
            Interleave(maskVregb16_new, maskVregb16_unroll_new, maskVregb16, maskVregb16);
            Cast<float, half, castTraitZero>(maskVregb32, maskVregb16_new, pregFull);
            Cast<float, half, castTraitZero>(maskVregb32_unroll, maskVregb16_unroll_new, pregFull);
            Compares(pregCompare, maskVregb32, static_cast<float>(0), pregFull);
            Compares(pregCompare_unroll, maskVregb32_unroll, static_cast<float>(0), pregFull);
            Select(srcVreg_mask, srcVreg, minVreg, pregCompare);
            Select(srcVreg_mask_unroll, srcVreg_unroll, minVreg, pregCompare_unroll);

            Select(srcVreg_unroll_new, srcVreg_mask_unroll, minVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize, srcVreg_mask, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize + FLOAT_REP_SIZE, srcVreg_unroll_new, pregFull);
            Max(maxTmpVreg, srcVreg_mask, srcVreg_unroll_new, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::MAX, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                maxSrcVreg, maxTmpVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxSrcVreg, maxUreg, 1);
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            newMaxUb, maxUreg, 0);
        if constexpr (isUpdate) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign(maxSrcVreg, newMaxUbStart);
            LoadAlign(maxTmpVreg, LastMaxUbStart);
            Max(maxSrcVreg, maxSrcVreg, maxTmpVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(newMaxUbStart, maxSrcVreg, pregFull);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxBrcVreg, newMaxUbStart + i);
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                srcVreg, srcVreg_unroll, srcUb + i * S2BaseSize);
            ExpSub(expEvenVreg, srcVreg, maxBrcVreg, pregFull);
            ExpSub(expOddVreg, srcVreg_unroll, maxBrcVreg, pregFull);

            Add(expSumVreg, expEvenVreg, expOddVreg, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumVreg, expSumUreg, 1);
                
            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(vreg_exp_even_bf16, expEvenVreg, pregFull);
                Cast<bfloat16_t, float, castTraitOne>(vreg_exp_odd_bf16, expOddVreg, pregFull);
                Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ bfloat16_t *&)expUb), vreg_exp_bf16, blockStride, 1, preg_all_b16);
            } else {
                Cast<half, float, castTraitZero>(vreg_exp_even_f16, expEvenVreg, pregFull);
                Cast<half, float, castTraitOne>(vreg_exp_odd_f16, expOddVreg, pregFull);
                Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ half *&)expUb), vreg_exp_f16, blockStride, 1, preg_all_b16);
            }
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumUreg, 0);
    }

    template <typename ElementS, typename ElementP, bool isUpdate>
    __simd_vf__ inline void ComputeScaleAndMaxMask64(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *newMaxUbStart, __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb,
        __ubuf__ ElementS *expSumUb, __ubuf__ ElementMask *maskUb, uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize, uint32_t blockStride, uint32_t repeatStride)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> minVreg;
        RegTensor<float> srcVreg;
        RegTensor<float> srcVreg_unroll;
        RegTensor<float> srcVreg_unroll_new;
        RegTensor<float> maxSrcVreg;
        RegTensor<float> maxTmpVreg;
        RegTensor<float> maxBrcVreg;
        RegTensor<float> expEvenVreg;
        RegTensor<float> expOddVreg;
        RegTensor<float> expSumVreg;
        UnalignRegForStore maxUreg;
        UnalignRegForStore expSumUreg;

        RegTensor<uint8_t> maskVreg;
        RegTensor<half> maskVregb16;
        RegTensor<float> maskVregb32;
        RegTensor<float> srcVreg_mask;

        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;

        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            Muls(srcVreg, srcVreg, dScale, pregTailN);
            LoadAlign<ElementMask, LoadDist::DIST_UNPACK4_B8>(maskVreg, maskUb + i * 128);
            Cast<half, ElementMask, castTraitZero>(maskVregb16, maskVreg, preg_all_b16);
            Cast<float, half, castTraitZero>(maskVregb32, maskVregb16, pregFull);
            Compares(pregCompare, maskVregb32, static_cast<float>(0), pregFull);
            Select(srcVreg_mask, srcVreg, minVreg, pregCompare);
            Select(srcVreg_unroll_new, srcVreg_mask, minVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize, srcVreg_unroll_new, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::MAX, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                maxSrcVreg, srcVreg_unroll_new, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxSrcVreg, maxUreg, 1);
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            newMaxUb, maxUreg, 0);
        if constexpr (isUpdate) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign(maxSrcVreg, newMaxUbStart);
            LoadAlign(maxTmpVreg, LastMaxUbStart);
            Max(maxSrcVreg, maxSrcVreg, maxTmpVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(newMaxUbStart, maxSrcVreg, pregFull);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxBrcVreg, newMaxUbStart + i);
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            ExpSub(expEvenVreg, srcVreg, maxBrcVreg, pregFull);
            Reduce<AscendC::MicroAPI::ReduceType::SUM, float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            expSumVreg, expEvenVreg, pregFull);
            StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumVreg, expSumUreg, 1);
                
            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(vreg_exp_bf16, expEvenVreg, pregFull);
                DeInterleave(vreg_exp_even_bf16, vreg_exp_odd_bf16, vreg_exp_bf16, vreg_exp_bf16);
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ bfloat16_t *&)expUb), vreg_exp_even_bf16, blockStride, 1, preg_all_b16);
            } else {
                Cast<half, float, castTraitZero>(vreg_exp_f16, expEvenVreg, pregFull);
                DeInterleave(vreg_exp_even_f16, vreg_exp_odd_f16, vreg_exp_f16, vreg_exp_f16);
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ half *&)expUb), vreg_exp_even_f16, blockStride, 1, preg_all_b16);
            }
        }
        StoreUnAlignPost<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ float *&)expSumUb), expSumUreg, 0);
    }

    template <typename ElementS, typename ElementP, bool isUpdate, MAligendTileNum mTileNum>
    __simd_vf__ inline void ComputeScaleAndMaxDn(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb,
        __ubuf__ ElementS *expSumUb, uint16_t mRound, uint16_t m, uint32_t tailN, uint32_t mFirstTile, ElementInput dScale, uint16_t S2BaseSize, 
        uint32_t blockStride, uint32_t repeatStride, __ubuf__ float *expMaxUb, __ubuf__ ElementS *lastExpSumUb)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> src0Vreg;
        RegTensor<float> src1Vreg;
        RegTensor<float> src2Vreg;
        RegTensor<float> src3Vreg;
        RegTensor<float> src4Vreg;

        RegTensor<float> src0Fp32Vreg;
        RegTensor<float> src1Fp32Vreg;
        RegTensor<float> src2Fp32Vreg;
        RegTensor<float> src3Fp32Vreg;
        RegTensor<float> src4Fp32Vreg;

        RegTensor<float> exp0Fp32Vreg;
        RegTensor<float> exp1Fp32Vreg;
        RegTensor<float> exp2Fp32Vreg;
        RegTensor<float> exp3Fp32Vreg;
        RegTensor<float> exp4Fp32Vreg;
        
        RegTensor<float> max0Vreg;
        RegTensor<float> max1Vreg;
        RegTensor<float> max2Vreg;
        RegTensor<float> max3Vreg;

        RegTensor<float> expMaxVreg;
        RegTensor<float> updateExpSumVreg;

        RegTensor<float> sum0Vreg;
        RegTensor<float> sum1Vreg;
        RegTensor<float> sum2Vreg;
        RegTensor<float> sum3Vreg;
        RegTensor<float> lastExpSumVreg;

        RegTensor<half> vreg_x_exp_even_f16;
        RegTensor<half> vreg_x_exp_odd_f16;
        RegTensor<half> vreg_x_exp_f16_packa;
        RegTensor<half> vreg_x_exp_f16_pack;
        RegTensor<half> vreg_x_exp_even_f16_1;
        RegTensor<half> vreg_x_exp_odd_f16_1;
        RegTensor<half> vreg_x_exp_f16_1_pack;
        RegTensor<half> vreg_x_exp_f16_1_packa;

        RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        uint32_t sreg_92 = (uint32_t)128ULL;
        MaskReg preg_136 = UpdateMask<uint16_t>(sreg_92);
        std::conditional_t<AscendC::IsSameType<ElementP, bfloat16_t>::value, 
                           __ubuf__ bfloat16_t*, 
                           __ubuf__ half*> x_exp_1 = expUb + (mRound * 4);
        __ubuf__ float *srcUb0 = srcUb;
        __ubuf__ float *srcUb1 = srcUb0 + S2BaseSize;
        __ubuf__ float *srcUb2 = srcUb0 + S2BaseSize * 2;
        __ubuf__ float *srcUb3 = srcUb0 + S2BaseSize * 3;
        __ubuf__ float *srcUb4 = srcUb0 + S2BaseSize * (m / 4) * 4;
        Duplicate(max0Vreg, MIN_VALUE);
        Duplicate(max1Vreg, MIN_VALUE);
        Duplicate(max2Vreg, MIN_VALUE);
        Duplicate(max3Vreg, MIN_VALUE);
        for (int32_t i = 0; i < int32_t(m / 4); i++) {
            LoadAlign(src0Vreg, srcUb0 + i * S2BaseSize * 4);
            LoadAlign(src1Vreg, srcUb1 + i * S2BaseSize * 4);
            LoadAlign(src2Vreg, srcUb2 + i * S2BaseSize * 4);
            LoadAlign(src3Vreg, srcUb3 + i * S2BaseSize * 4);
            Max(max0Vreg, max0Vreg, src0Vreg, pregTailN);
            Max(max1Vreg, max1Vreg, src1Vreg, pregTailN);
            Max(max2Vreg, max2Vreg, src2Vreg, pregTailN);
            Max(max3Vreg, max3Vreg, src3Vreg, pregTailN);
        }
        Max(max0Vreg, max0Vreg, max2Vreg, pregTailN);
        Max(max1Vreg, max1Vreg, max3Vreg, pregTailN);
        Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);
        for (int32_t i = 0; i < int32_t(m % 4); i++) {
            LoadAlign(src4Vreg, srcUb4 + i * S2BaseSize);
            Max(max0Vreg, max0Vreg, src4Vreg, pregTailN);
        }
        Muls(max0Vreg, max0Vreg, dScale, pregTailN);

        if constexpr (isUpdate) {
            LoadAlign(max1Vreg, LastMaxUbStart);
            LoadAlign(lastExpSumVreg, lastExpSumUb);
            Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);
            FusedExpSub(expMaxVreg, max1Vreg, max0Vreg, pregTailN);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(LastMaxUbStart, max0Vreg, pregTailN);
        }

        StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)newMaxUb, max0Vreg, pregFull);

        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum0Vreg, 0, pregFull);
        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum1Vreg, 0, pregFull);
        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum2Vreg, 0, pregFull);
        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum3Vreg, 0, pregFull);

        for (int32_t i = 0; i < mFirstTile; i++) {
            LoadAlign(src0Fp32Vreg, srcUb + i * S2BaseSize);
            LoadAlign(src1Fp32Vreg, srcUb + mRound * S2BaseSize / 4 + i * S2BaseSize);
            LoadAlign(src2Fp32Vreg, srcUb + mRound * S2BaseSize / 2 + i * S2BaseSize);
            LoadAlign(src3Fp32Vreg, srcUb + mRound * S2BaseSize / 2 + mRound * S2BaseSize / 4 + i * S2BaseSize);

            Muls(src0Fp32Vreg, src0Fp32Vreg, dScale, pregTailN);
            Muls(src1Fp32Vreg, src1Fp32Vreg, dScale, pregTailN);
            Muls(src2Fp32Vreg, src2Fp32Vreg, dScale, pregTailN);
            Muls(src3Fp32Vreg, src3Fp32Vreg, dScale, pregTailN);

            FusedExpSub(exp0Fp32Vreg, src0Fp32Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp1Fp32Vreg, src1Fp32Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp2Fp32Vreg, src2Fp32Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp3Fp32Vreg, src3Fp32Vreg, max0Vreg, pregTailN);

            if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_even_bf16, exp0Fp32Vreg, pregFull);
                Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_odd_bf16, exp2Fp32Vreg, pregFull);
                DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);
                Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_even_bf16_1, exp1Fp32Vreg, pregFull);
                Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_odd_bf16_1, exp3Fp32Vreg, pregFull);
                DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ bfloat16_t *&)expUb), vreg_x_exp_bf16_pack, blockStride, 1, preg_136); 

                if constexpr (mTileNum >= MAligendTileNum::Zero) {
                    Add(sum0Vreg, exp0Fp32Vreg, sum0Vreg, pregTailN);
                }
                if constexpr (mTileNum >= MAligendTileNum::Two) {
                    Add(sum2Vreg, exp2Fp32Vreg, sum2Vreg, pregTailN);
                }
                StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ bfloat16_t *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, 1, preg_136);
                if constexpr (mTileNum >= MAligendTileNum::One) {
                    Add(sum1Vreg, exp1Fp32Vreg, sum1Vreg, pregTailN);
                }
                if constexpr (mTileNum >= MAligendTileNum::Three) {
                    Add(sum3Vreg, exp3Fp32Vreg, sum3Vreg, pregTailN);
                }
            } else {
                Cast<half, float, castTraitZero>(vreg_x_exp_even_f16, exp0Fp32Vreg, pregFull);
                Cast<half, float, castTraitZero>(vreg_x_exp_odd_f16, exp2Fp32Vreg, pregFull);
                DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
                Cast<half, float, castTraitZero>(vreg_x_exp_even_f16_1, exp1Fp32Vreg, pregFull);
                Cast<half, float, castTraitZero>(vreg_x_exp_odd_f16_1, exp3Fp32Vreg, pregFull);
                DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ half *&)expUb), vreg_x_exp_f16_pack, blockStride, 1, preg_136); 
                
                if constexpr (mTileNum >= MAligendTileNum::Zero) {
                    Add(sum0Vreg, exp0Fp32Vreg, sum0Vreg, pregTailN);
                }
                if constexpr (mTileNum >= MAligendTileNum::Two) {
                    Add(sum2Vreg, exp2Fp32Vreg, sum2Vreg, pregTailN);
                }
                StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ half *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, 1, preg_136);
                if constexpr (mTileNum >= MAligendTileNum::One) {
                    Add(sum1Vreg, exp1Fp32Vreg, sum1Vreg, pregTailN);
                }
                if constexpr (mTileNum >= MAligendTileNum::Three) {
                    Add(sum3Vreg, exp3Fp32Vreg, sum3Vreg, pregTailN);
                }
            }
        }
        if constexpr (mTileNum < MAligendTileNum::Four) {
            for (int32_t i = mFirstTile; i < int32_t(mRound / 4); i++) {
                LoadAlign(src0Fp32Vreg, srcUb + i * S2BaseSize);
                LoadAlign(src1Fp32Vreg, srcUb + mRound * S2BaseSize / 4 + i * S2BaseSize);
                LoadAlign(src2Fp32Vreg, srcUb + mRound * S2BaseSize / 2 + i * S2BaseSize);
                LoadAlign(src3Fp32Vreg, srcUb + mRound * S2BaseSize / 2 + mRound * S2BaseSize / 4 + i * S2BaseSize);

                Muls(src0Fp32Vreg, src0Fp32Vreg, dScale, pregTailN);
                Muls(src1Fp32Vreg, src1Fp32Vreg, dScale, pregTailN);
                Muls(src2Fp32Vreg, src2Fp32Vreg, dScale, pregTailN);
                Muls(src3Fp32Vreg, src3Fp32Vreg, dScale, pregTailN);

                FusedExpSub(exp0Fp32Vreg, src0Fp32Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp1Fp32Vreg, src1Fp32Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp2Fp32Vreg, src2Fp32Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp3Fp32Vreg, src3Fp32Vreg, max0Vreg, pregTailN);
                
                if constexpr (AscendC::IsSameType<ElementP, bfloat16_t>::value) {
                    Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_even_bf16, exp0Fp32Vreg, pregFull);
                    Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_odd_bf16, exp2Fp32Vreg, pregFull);
                    DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);
                    Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_even_bf16_1, exp1Fp32Vreg, pregFull);
                    Cast<bfloat16_t, float, castTraitZero>(vreg_x_exp_odd_bf16_1, exp3Fp32Vreg, pregFull);
                    DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
                    StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ bfloat16_t *&)expUb), vreg_x_exp_bf16_pack, blockStride, 1, preg_136); 

                    if constexpr (mTileNum > MAligendTileNum::Zero) {
                        Add(sum0Vreg, exp0Fp32Vreg, sum0Vreg, pregTailN);
                    }
                    if constexpr (mTileNum > MAligendTileNum::Two) {
                        Add(sum2Vreg, exp2Fp32Vreg, sum2Vreg, pregTailN);
                    }
                    StoreAlign<bfloat16_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ bfloat16_t *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, 1, preg_136);
                    if constexpr (mTileNum > MAligendTileNum::One) {
                        Add(sum1Vreg, exp1Fp32Vreg, sum1Vreg, pregTailN);
                    }
                } else {
                    Cast<half, float, castTraitZero>(vreg_x_exp_even_f16, exp0Fp32Vreg, pregFull);
                    Cast<half, float, castTraitZero>(vreg_x_exp_odd_f16, exp2Fp32Vreg, pregFull);
                    DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);
                    Cast<half, float, castTraitZero>(vreg_x_exp_even_f16_1, exp1Fp32Vreg, pregFull);
                    Cast<half, float, castTraitZero>(vreg_x_exp_odd_f16_1, exp3Fp32Vreg, pregFull);
                    DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
                    StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ half *&)expUb), vreg_x_exp_f16_pack, blockStride, 1, preg_136); 

                    if constexpr (mTileNum > MAligendTileNum::Zero) {
                        Add(sum0Vreg, exp0Fp32Vreg, sum0Vreg, pregTailN);
                    }
                    if constexpr (mTileNum > MAligendTileNum::Two) {
                        Add(sum2Vreg, exp2Fp32Vreg, sum2Vreg, pregTailN);
                    }
                    StoreAlign<half, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ half *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, 1, preg_136);
                    if constexpr (mTileNum > MAligendTileNum::One) {
                        Add(sum1Vreg, exp1Fp32Vreg, sum1Vreg, pregTailN);
                    }
                }
            }
        }
        Add(sum0Vreg, sum0Vreg, sum2Vreg, pregFull);
        Add(sum1Vreg, sum1Vreg, sum3Vreg, pregFull);
        Add(sum0Vreg, sum0Vreg, sum1Vreg, pregFull);
        if constexpr (isUpdate) {
            Add(updateExpSumVreg, updateExpSumVreg, sum0Vreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(lastExpSumUb, updateExpSumVreg, pregFull);
        }
        StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ float *&)expSumUb, sum0Vreg, pregFull);
    }

    template <typename ElementS>
    __simd_vf__ inline void CastExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *maxUb,
        __ubuf__ ElementS *expSumUb, __ubuf__ ElementS *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementS> nowMaxVreg0;
        RegTensor<ElementS> nowMaxVreg1;
        RegTensor<ElementS> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<ElementS> brcExpSumVreg0;
        RegTensor<ElementS> brcExpSumVreg1;
        RegTensor<ElementS> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementS, MaskPattern::ALL>();
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementS, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(brcExpSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
            Cast<float, ElementS, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregFull);
        }
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementS, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(brcExpSumTmpVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
        Cast<float, ElementS, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregTailM);
    }

    template <typename ElementS>
    __simd_vf__ inline void UpdateExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *expMaxUb,
        __ubuf__ float *maxUb, __ubuf__ ElementS *expSumUb, __ubuf__ ElementS *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementS> nowMaxVreg0;
        RegTensor<ElementS> nowMaxVreg1;
        RegTensor<ElementS> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<ElementS> brcExpSumVreg0;
        RegTensor<ElementS> brcExpSumVreg1;
        RegTensor<ElementS> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        RegTensor<float> updateExpSumVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementS, MaskPattern::ALL>();
        LoadAlign(lastMaxVreg, maxUb);
        LoadAlign(nowMaxTmpVreg, nowMaxUb);
        FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxTmpVreg, pregFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, pregFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb, nowMaxTmpVreg, pregFull);

        LoadAlign(lastExpSumVreg, sumUb);
        LoadAlign(brcExpSumTmpVreg, expSumUb);
        Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
        Add(updateExpSumVreg, updateExpSumVreg, brcExpSumTmpVreg, pregFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb, updateExpSumVreg, pregFull);
    }
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_HIGH_PREC_HPP