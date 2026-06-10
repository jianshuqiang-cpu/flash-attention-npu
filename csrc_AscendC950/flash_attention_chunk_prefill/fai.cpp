/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

// Helper methods to check for errors
#include "fai_kernel.cpp"
#include "fai_tiling.cpp"
#include "golden.hpp"
#include "helper.hpp"
#include "kernel_common.hpp"

using namespace std;
using namespace optiling;

typedef void (*LaunchFunc)(uint32_t blockDim, aclrtStream stream,
                           uint8_t* qDevice, uint8_t* kDevice, uint8_t* vDevice,
                           uint8_t* maskDevice, uint8_t* blockTableDevice,
                           uint8_t* oDevice, uint8_t* lseDevice,
                           uint8_t* qSeqDevice, uint8_t* kvSeqDevice,
                           uint8_t* workspaceDevice, uint8_t* tilingDevice, bool enableDN);

template<typename T, typename AccT,
         Format QF, Format KVF,
         CacheMode CM, PageShape PS,
         MaskCategory MC, CacheLayout CL>
void LaunchWrapper(uint32_t blockDim, aclrtStream stream,
                   uint8_t* qDevice, uint8_t* kDevice, uint8_t* vDevice,
                   uint8_t* maskDevice, uint8_t* blockTableDevice,
                   uint8_t* oDevice, uint8_t* lseDevice,
                   uint8_t* qSeqDevice, uint8_t* kvSeqDevice,
                   uint8_t* workspaceDevice, uint8_t* tilingDevice, bool enableDN)
{   
    if (!enableDN) {
        FAInfer<T, AccT, QF, KVF, CM, PS, MC, CL>
            <<<blockDim, nullptr, stream>>>(
                qDevice, kDevice, vDevice, maskDevice, blockTableDevice,
                oDevice, lseDevice, qSeqDevice, kvSeqDevice,
                workspaceDevice, tilingDevice);
    } else {
        FAInferDn<T, AccT, QF, KVF, CM, PS, MC, CL>
            <<<blockDim, nullptr, stream>>>(
                qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
    }
}

// 调度表及其一次性初始化
static LaunchFunc g_dispatcher[256] = { nullptr };
static std::once_flag g_init_flag;

// CL: CacheLayout  (0: nd, 1: nz)
// DT: DataType     (0: half, 1: bf16)
// MT: MaskType     (0: NO_MASK, 1: MASK_CAUSAL)
// SW: SWAMASK      (0: NO_MASK, 1: SWA)
// IP: InnerPrec    (0: float, 1: half; bf16时强制为0)
// LO: Layout       (0: TND, 1: BSND)
// CM: CacheMode    (0: normalCache, 1: pagedCache)
// PS: PageShape    (0: BnBsND, 1: BnNBsD; 仅当CM=1时有效)
// T:   数据类型   (half 或 bfloat16_t)
// AccT: 累加精度 (float 或 half)
// QF:  Q张量格式 (TND 或 BSND)
// KVF: KV张量格式 (TND 或 BSND)

#define KEY(CL, DT, MT, SW, IP, LO, CM, PS) \
    ( ((CL)<<0) | ((DT)<<1) | ((MT)<<2) | (SW)<<3 | ((IP)<<4) | ((LO)<<5) | ((CM)<<6) | ((PS)<<7) )

#define REGISTER(CL, DT, MT, SW, IP, LO, CM, PS, T, AccT, QF, KVF, CachingMode, PageShapeType, MaskCat, CacheLay) \
    g_dispatcher[KEY(CL, DT, MT, SW, IP, LO, CM, PS)] = \
        LaunchWrapper<T, AccT, QF, KVF, CachingMode, PageShapeType, MaskCat, CacheLay>;

static void InitDispatcher() {
    // ============ CacheLayout::nd (CL=0)  ============
    // half (DT=0) IP=0
    REGISTER(0,0,0,0,0,0,0,0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,0,0,0,0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,0,0,0,0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,0,1,0,0, half, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,0,1,0,0, half, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,0,1,0,0, half, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)

    REGISTER(0,0,0,0,0,0,1,0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,0,0,1,0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,0,0,1,0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,0,0,1,1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,0,0,1,1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,0,0,1,1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,0,1,1,0, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,0,1,1,0, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,0,1,1,0, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,0,1,1,1, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,0,1,1,1, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,0,1,1,1, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

    // half IP=1
    REGISTER(0,0,0,0,1,0,0,0, half, half,  Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,1,0,0,0, half, half,  Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,1,0,0,0, half, half,  Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,1,1,0,0, half, half,  Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,1,1,0,0, half, half,  Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,1,1,0,0, half, half,  Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)

    REGISTER(0,0,0,0,1,0,1,0, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,1,0,1,0, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,1,0,1,0, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,1,0,1,1, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,1,0,1,1, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,1,0,1,1, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,1,1,1,0, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,1,1,1,0, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,1,1,1,0, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,0,0,0,1,1,1,1, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,0,1,0,1,1,1,1, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,0,0,1,1,1,1,1, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

    // bf16 (DT=1) IP=0
    REGISTER(0,1,0,0,0,0,0,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,1,1,0,0,0,0,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,1,0,1,0,0,0,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,1,0,0,0,1,0,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,1,1,0,0,1,0,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,1,0,1,0,1,0,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,1,0,0,0,0,1,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,1,1,0,0,0,1,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,1,0,1,0,0,1,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,1,0,0,0,0,1,1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,1,1,0,0,0,1,1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,1,0,1,0,0,1,1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,1,0,0,0,1,1,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,1,1,0,0,1,1,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,1,0,1,0,1,1,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nd)
    REGISTER(0,1,0,0,0,1,1,1, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    REGISTER(0,1,1,0,0,1,1,1, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    REGISTER(0,1,0,1,0,1,1,1, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nd)

    // bf16 IP=1
    // REGISTER(0,1,0,1,0,0,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    // REGISTER(0,1,1,1,0,0,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    // REGISTER(0,1,0,1,1,0,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nd)
    // REGISTER(0,1,1,1,1,0,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    // REGISTER(0,1,0,1,0,1,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    // REGISTER(0,1,1,1,0,1,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    // REGISTER(0,1,0,1,0,1,1, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    // REGISTER(0,1,1,1,0,1,1, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    // REGISTER(0,1,0,1,1,1,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nd)
    // REGISTER(0,1,1,1,1,1,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nd)
    // REGISTER(0,1,0,1,1,1,1, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nd)
    // REGISTER(0,1,1,1,1,1,1, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nd)

    // ============ CacheLayout::nz (CL=1)  ============
    // half IP=0
    REGISTER(1,0,0,0,0,0,0,0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,0,0,0,0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,0,0,0,0, half, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,0,1,0,0, half, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,0,1,0,0, half, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,0,1,0,0, half, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,0,0,1,0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,0,0,1,0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,0,0,1,0, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,0,0,1,1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,0,0,1,1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,0,0,1,1, half, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,0,1,1,0, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,0,1,1,0, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,0,1,1,0, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,0,1,1,1, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,0,1,1,1, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,0,1,1,1, half, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nz)
    // half IP=1
    REGISTER(1,0,0,0,1,0,0,0, half, half,  Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,1,0,0,0, half, half,  Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,1,0,0,0, half, half,  Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,1,1,0,0, half, half,  Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,1,1,0,0, half, half,  Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,1,1,0,0, half, half,  Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,1,0,1,0, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,1,0,1,0, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,1,0,1,0, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,1,0,1,1, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,1,0,1,1, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,1,0,1,1, half, half,  Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,1,1,1,0, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,1,1,1,0, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,1,1,1,0, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,0,0,0,1,1,1,1, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,0,1,0,1,1,1,1, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,0,0,1,1,1,1,1, half, half,  Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nz)
    // bf16 IP=0
    REGISTER(1,1,0,0,0,0,0,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,1,1,0,0,0,0,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,1,0,1,0,0,0,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,1,0,0,0,1,0,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,1,1,0,0,1,0,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,1,0,1,0,1,0,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,1,0,0,0,0,1,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,1,1,0,0,0,1,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,1,0,1,0,0,1,0, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,1,0,0,0,0,1,1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,1,1,0,0,0,1,1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,1,0,1,0,0,1,1, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,1,0,0,0,1,1,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,1,1,0,0,1,1,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,1,0,1,0,1,1,0, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_SWA, CacheLayout::nz)
    REGISTER(1,1,0,0,0,1,1,1, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    REGISTER(1,1,1,0,0,1,1,1, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    REGISTER(1,1,0,1,0,1,1,1, bfloat16_t, float, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_SWA, CacheLayout::nz)
    // bf16 IP=1
    // REGISTER(1,1,0,1,0,0,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    // REGISTER(1,1,1,1,0,0,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    // REGISTER(1,1,0,1,1,0,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::NO_MASK,     CacheLayout::nz)
    // REGISTER(1,1,1,1,1,0,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    // REGISTER(1,1,0,1,0,1,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    // REGISTER(1,1,1,1,0,1,0, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    // REGISTER(1,1,0,1,0,1,1, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    // REGISTER(1,1,1,1,0,1,1, bfloat16_t, half, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    // REGISTER(1,1,0,1,1,1,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::NO_MASK,     CacheLayout::nz)
    // REGISTER(1,1,1,1,1,1,0, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnBsND, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
    // REGISTER(1,1,0,1,1,1,1, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::NO_MASK,     CacheLayout::nz)
    // REGISTER(1,1,1,1,1,1,1, bfloat16_t, half, Format::BSND,Format::BSND, CacheMode::pagedCache, PageShape::BnNBsD, MaskCategory::MASK_CAUSAL, CacheLayout::nz)
}
#undef KEY
#undef REGISTER

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fai batch qSeqlen kvSeqlen numHeads kvHeads qkHeadSize vHeadSize isVariedLen maskType [--dtype DTYPE "
        "--datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 7;

    // Define default value.
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t qkHeadSize{0};
    uint32_t vHeadSize{0};
    uint32_t isVariedLen{0};
    uint32_t maskType{0};
    uint32_t deviceId{0};
    uint32_t blockSize{128};
    uint32_t cacheMode{0};
    uint32_t pageShape{0};
    uint32_t layout{0};
    uint32_t numBlocks{0};
    uint32_t innerPrec{0};
    uint32_t lseFlag{0};
    uint32_t sink_flag{0};
    int32_t sparseMode{0};
    uint32_t globalWindowSize{4};
    uint32_t localWindowSize{0};
    int64_t preToken{0};
    int64_t nextToken{0};
    string dataType = "half";
    string cacheLayout = "nd";
    string dataPath = "../../examples/200_ascend950_flash_attention_chunk_prefill/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char **argv) {
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        qSeqlen = atoi(argv[argIndex++]);
        kvSeqlen = atoi(argv[argIndex++]);
        numHeads = atoi(argv[argIndex++]);
        kvHeads = atoi(argv[argIndex++]);
        qkHeadSize = atoi(argv[argIndex++]);
        vHeadSize = atoi(argv[argIndex++]);
        isVariedLen = atoi(argv[argIndex++]);
        maskType = atoi(argv[argIndex++]);
        cacheMode = atoi(argv[argIndex++]);
        pageShape = atoi(argv[argIndex++]);
        layout = atoi(argv[argIndex++]);
        numBlocks = atoi(argv[argIndex++]);
        innerPrec = atoi(argv[argIndex++]);
        lseFlag = atoi(argv[argIndex++]);
        blockSize = atoi(argv[argIndex++]);
        sink_flag = atoi(argv[argIndex++]);
        preToken = atoi(argv[argIndex++]);
        nextToken = atoi(argv[argIndex++]);
        sparseMode = atoi(argv[argIndex++]);
        globalWindowSize = atoi(argv[argIndex++]);
        localWindowSize = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else if (flag == "--dtype") {
                dataType = string(argv[argIndex++]);
            } else if (flag == "--cache_layout") {
                cacheLayout = string(argv[argIndex++]);
            } else {
                printf(HELPER);
               return -1;
            }
        }
        return 0;
    }
};

static void AllocMem(uint8_t **host, uint8_t **device, size_t size) {
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t *host, uint8_t *device) {
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// CATLASS FAI kernel.
static void Run(const Options &options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // auto aicCoreNum = 2;

    // Parameters initialization.
    int32_t batch = options.batch;
    int32_t qSeqlen = options.qSeqlen;
    int32_t kvSeqlen = options.kvSeqlen;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t qkHeadSize = options.qkHeadSize;
    int32_t vHeadSize = options.vHeadSize;
    int32_t blockSize = options.blockSize;
    int32_t maskType = options.maskType;
    int32_t cacheMode = options.cacheMode;
    int32_t pageShape = options.pageShape;
    int32_t lseFlag = options.lseFlag;
    int32_t numBlocks = options.numBlocks;
    int32_t sink_flag = options.sink_flag;
    int32_t sparseMode = options.sparseMode;
    int32_t globalWindowSize = options.globalWindowSize;
    int32_t localWindowSize = options.localWindowSize;
    int64_t preToken = options.preToken;
    int64_t nextToken = options.nextToken;
    int64_t innerPrec = options.innerPrec;
    string layout = "BSND";
    string cacheLayout = options.cacheLayout;
    if(options.layout == 1) {
        layout = "TND";
    }
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    int32_t maxKvSeqlen = kvSeqlen;
    if (numBlocks == 0) {
        numBlocks = batch * ((maxKvSeqlen + blockSize - 1) / blockSize);
    }

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }
    if ((cacheLayout != "nz") && (cacheLayout != "nd")) {
        cerr << "[ERROR] cacheLayout must be 'nz' or 'nd'." << endl;
        return;
    }
    if (maskType == 4) {
        if (cacheMode != 1 ) {
            cerr << "[ERROR] when enable Swa, cacheMode must be PA." << endl;
            return;
        }
        if (pageShape != 1) {
            cerr << "[ERROR] when enable Swa, pageShape must be BnNBsD." << endl;
            return;
        }
        if (globalWindowSize != 4) {
            cerr << "[ERROR] when enable Swa, globalWindowSize must be 4." << endl;
            return;
        }
        if (localWindowSize != 1024  && localWindowSize != 2048) {
            cerr << "[ERROR] when enable Swa, localWindowSize must be 1024 or 2048." << endl;
            return;
        }
    }
    // read qNtokens num
    void *qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t *>(qNtokens)[0];

    void *kvNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/kv_ntokens.bin", kvNtokens, 1 * sizeof(int32_t));
    int32_t kvNumTokens = static_cast<int32_t *>(kvNtokens)[0];

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t seqArraySizeTND = (batch + 1) * sizeof(int64_t);
    uint64_t qSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)qkHeadSize * sizeof(fp16_t); 
    uint64_t oSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)vHeadSize * sizeof(fp16_t); 
    uint64_t lseSize = (uint64_t)numTokens * (uint64_t)numHeads * sizeof(int32_t);
    uint64_t kSize = 0;
    uint64_t vSize = 0;
    if (cacheMode == 1) {
        kSize = (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)qkHeadSize * sizeof(fp16_t);
        vSize = (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)vHeadSize * sizeof(fp16_t);
    } else {
        kSize = (uint64_t)kvNumTokens * (uint64_t)kvHeads * (uint64_t)qkHeadSize * sizeof(fp16_t);
        vSize = (uint64_t)kvNumTokens * (uint64_t)kvHeads * (uint64_t)vHeadSize * sizeof(fp16_t);
    }
    uint64_t maskSize = 2048 * 2048 * sizeof(fp16_t);
    if (maskType == 4) {
        uint64_t len = 2 * globalWindowSize + localWindowSize + 128 + 128;
        maskSize = len * len * sizeof(fp16_t);
        cout << "maskSize:" << maskSize << endl;
    }
    uint64_t blockTableSize = static_cast<uint64_t>(
        batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t)
    );
    // ?????
    uint32_t tilingSize = sizeof(FAInferTilingData);

    // Allocate matrices in host and device memory.
    uint8_t *qSeqHost;
    uint8_t *qSeqDevice;
    if (options.layout == 1) {
        AllocMem(&qSeqHost, &qSeqDevice, seqArraySizeTND);
        ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySizeTND);
        ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySizeTND, qSeqHost, seqArraySizeTND, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
        ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySize);
        ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    
    // Allocate matrices in host and device memory.
    uint8_t *kvSeqHost;
    uint8_t *kvSeqDevice;
    if (options.layout == 1 && cacheMode != 1) {
        AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySizeTND);
        ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySizeTND);
        ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySizeTND, kvSeqHost, seqArraySizeTND, ACL_MEMCPY_HOST_TO_DEVICE));

    } else {
        AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
        ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySize);
        ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    }
    
    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *qHost;
    uint8_t *qDevice;
    AllocMem(&qHost, &qDevice, qSize);
    ReadFile(dataPath + "/q.bin", qHost, qSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t *kHost;
    uint8_t *kDevice;
    AllocMem(&kHost, &kDevice, kSize);
    if (cacheLayout == "nd") {
        ReadFile(dataPath + "/k.bin", kHost, kSize);
    } else if (cacheLayout == "nz") {
        ReadFile(dataPath + "/k_nz.bin", kHost, kSize);
    }
    ACL_CHECK(aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *vHost;
    uint8_t *vDevice;
    AllocMem(&vHost, &vDevice, vSize);
    if (cacheLayout == "nd") {
        ReadFile(dataPath + "/v.bin", vHost, vSize);
    } else if (cacheLayout == "nz") {
        ReadFile(dataPath + "/v_nz.bin", vHost, vSize);
    }
    ACL_CHECK(aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *maskHost;
    uint8_t *maskDevice;
    if (maskType == 1 || maskType == 4) {
        AllocMem(&maskHost, &maskDevice, maskSize);
        ReadFile(dataPath + "/mask.bin", maskHost, maskSize);
        ACL_CHECK(aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t *blockTableHost;
    uint8_t *blockTableDevice;
    if (cacheMode == 1) {
        AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
        ReadFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize);
        ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    bool enableDN = false;
    // 当前逻辑只支持half
    if (maskType == 0 && qkHeadSize <= 128 && vHeadSize <= 128 && innerPrec == 0) {
        enableDN = true;
    }

    FAInferTilingData faiTilingData;
    FAInferContext faiContext;

    uint64_t sinkSize = (uint64_t)numHeads * sizeof(bfloat16_t);
    uint8_t *sinkHost;
    uint8_t *sinkDevice;

    if (options.sink_flag == 1) {
        AllocMem(&sinkHost, &sinkDevice, sinkSize);
        ReadFile(dataPath + "/sink.bin", sinkHost, sinkSize);
        ACL_CHECK(aclrtMemcpy(sinkDevice, sinkSize, sinkHost, sinkSize, ACL_MEMCPY_HOST_TO_DEVICE));
        faiContext.learnableSinkFlag = true;
    } else {
        sinkDevice = nullptr;
    }

    faiContext.pagedCacheFlag = cacheMode == 1;
    faiContext.pagedShapeFlag = pageShape == 1;
    faiContext.kvcacheNzFlag = cacheLayout == "nz";
    faiContext.numHeads = numHeads;
    faiContext.numBlocks = numBlocks;
    faiContext.blockSize = blockSize;
    faiContext.kvHeads = kvHeads;
    faiContext.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * qkHeadSize));;
    faiContext.layout = layout;
    faiContext.lseFlag = lseFlag;
    if (faiContext.pagedCacheFlag) {
        faiContext.maxNumBlocksPerBatch = (maxKvSeqlen + blockSize - 1) / blockSize;
    }
    faiContext.embeddingSize = qkHeadSize;
    faiContext.embeddingSizeV = vHeadSize;
    
    faiContext.maskType = static_cast<optiling::MaskType>(maskType);
    faiContext.dataType = static_cast<optiling::DataType>(dataType == "bf16");
    faiContext.batch = batch;
    faiContext.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
    faiContext.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHost);
    faiContext.isTilingSink = false;
    faiContext.preToken = preToken;
    faiContext.nextToken = nextToken;
    faiContext.sparseMode = sparseMode;
    faiContext.globalWindowSize = globalWindowSize;
    faiContext.localWindowSize = localWindowSize;
    cout << "preToken " << preToken << endl;
    cout << "nextToken " << nextToken << endl;
    cout << "cacheLayout " << cacheLayout << endl;
    cout << "maskType " << maskType << endl;
    cout << "globalWindowSize " << globalWindowSize << endl;
    cout << "localWindowSize " << localWindowSize << endl;

    // flashDecodeFlag 判断逻辑
    int64_t maxQSeqlenCalc = 0;
    int64_t minQSeqlenCalc = INT64_MAX;
    int64_t minKVSeqlenCalc = INT64_MAX;
    for (int32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
        int64_t qSeqlenVal = *(faiContext.qSeqlenList + batchIdx);
        int64_t kvSeqlenVal = *(faiContext.kvSeqlenList + batchIdx);
        if (faiContext.layout == "TND") {
            if (batchIdx > 0) {
                int64_t prevQSeqlenSum = *(faiContext.qSeqlenList + batchIdx - 1);
                qSeqlenVal = qSeqlenVal - prevQSeqlenSum;
                if (!faiContext.pagedCacheFlag) {
                    int64_t prevKvSeqlenSum = *(faiContext.kvSeqlenList + batchIdx - 1);
                    kvSeqlenVal = kvSeqlenVal - prevKvSeqlenSum;
                }
            }
        }
        if (qSeqlenVal > maxQSeqlenCalc) {
            maxQSeqlenCalc = qSeqlenVal;
        }
        if (qSeqlenVal < minQSeqlenCalc) {
            minQSeqlenCalc = qSeqlenVal;
        }
        if (kvSeqlenVal < minKVSeqlenCalc) {
            minKVSeqlenCalc = kvSeqlenVal;
        }
    }
    faiContext.maxQSeqlen = maxQSeqlenCalc;
    uint32_t numTasks = faiContext.batch * faiContext.kvHeads;
    bool isLongSeq = (numTasks <= 0.8 * aicCoreNum) && (minKVSeqlenCalc >= aicCoreNum * 512);
    bool isShortSeq = (numTasks <= 0.4 * aicCoreNum) && (minKVSeqlenCalc >= 1024);
    if ((!faiContext.lseFlag) && (faiContext.pagedCacheFlag) && !(faiContext.learnableSinkFlag) &&
        (faiContext.embeddingSize <= 128) && (maxQSeqlenCalc * (faiContext.numHeads / faiContext.kvHeads) <= 128) && (maxQSeqlenCalc <= 16) && (minQSeqlenCalc > 0) && 
        (isLongSeq || isShortSeq)) {
        faiContext.flashDecodeFlag = true;
    }
    faiContext.flashDecodeFlag = false;
    cout << "faiContext.flashDecodeFlag " << faiContext.flashDecodeFlag << endl;
    
    FAInferTiling fai_tiling(faiContext);
    fai_tiling.SetCoreNum(aicCoreNum);
    fai_tiling.DoTiling(faiTilingData);
    uint64_t tilingKey = fai_tiling.GetTilingKey();

    uint8_t *workspaceDevice{nullptr};
    faiTilingData.workSpaceSize = 1024 * 1024 * 32 * 4;
    cout << "faiTilingData.workSpaceSize " << faiTilingData.workSpaceSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), faiTilingData.workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    
    uint8_t *oDevice{nullptr};
    cout << "oSize " << oSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&oDevice), oSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *lseDevice{nullptr};
    cout << "lseSize " << lseSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&lseDevice), lseSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *tilingDevice;
    cout << "tilingSize " << tilingSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;
    if (faiContext.flashDecodeFlag) {
        auto needCoreNum = faiTilingData.get_needCoreNum();
        if (needCoreNum != 0) {
            blockDim = needCoreNum;
        }
    }

    // tiling output
    cout << "faiTilingData.numHeads" << faiTilingData.numHeads << endl;
    cout << "faiTilingData.embeddingSize" << faiTilingData.embeddingSize << endl;
    cout << "faiTilingData.embeddingSizeV" << faiTilingData.embeddingSizeV << endl;
    cout << "faiTilingData.numBlocks" << faiTilingData.numBlocks << endl;
    cout << "faiTilingData.blockSize" << faiTilingData.blockSize << endl;
    cout << "faiTilingData.maxQSeqlen" << faiTilingData.maxQSeqlen << endl;
    cout << "faiTilingData.maxKvSeqlen" << faiTilingData.maxKvSeqlen << endl;
    cout << "faiTilingData.kvHeads" << faiTilingData.kvHeads << endl;
    cout << "faiTilingData.batch" << faiTilingData.batch << endl;
    cout << "faiTilingData.maxNumBlocksPerBatch" << faiTilingData.maxNumBlocksPerBatch << endl;
    cout << "faiTilingData.totalTaskNum" << faiTilingData.totalTaskNum << endl;
    cout << "faiTilingData.maskType" << faiTilingData.maskType << endl;
    cout << "faiTilingData.mm1OutSize" << faiTilingData.mm1OutSize << endl;
    cout << "faiTilingData.smOnlineOutSize" << faiTilingData.smOnlineOutSize << endl;
    cout << "faiTilingData.mm2OutSize" << faiTilingData.mm2OutSize << endl;
    cout << "faiTilingData.UpdateSize" << faiTilingData.UpdateSize << endl;
    cout << "faiTilingData.workSpaceSize" << faiTilingData.workSpaceSize << endl;
    cout << "faiTilingData.scaleValue" << faiTilingData.scaleValue << endl;
    cout << "faiTilingData.firstBatchTaskNum" << faiTilingData.firstBatchTaskNum << endl;
    cout << "faiTilingData.globalWindowSize" << faiTilingData.globalWindowSize << endl;
    cout << "faiTilingData.localWindowSize" << faiTilingData.localWindowSize << endl;
    tilingHost = reinterpret_cast<void *>(&faiTilingData);
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    cout << "tilingkey: " << tilingKey << endl;

    // ====================================================================
    // 计算 kernelKey（将运行时参数编码为一个 8 位整数）
    // 位定义：
    //   bit 0: cacheLayout  (0: nd, 1: nz)
    //   bit 1: dataType     (0: half, 1: bf16)
    //   bit 2: maskType     (0: NO_MASK, 1: MASK_CAUSAL)
    //   bit 3: maskType     (0: NO_MASK, 1: SWA)
    //   bit 4: innerPrec    (0: float, 1: half)  [bf16 时强制为 0]
    //   bit 5: layout       (0: TND, 1: BSND)
    //   bit 6: cacheMode    (0: normalCache, 1: pagedCache)
    //   bit 7: pageShape    (0: BnBsND, 1: BnNBsD) [仅当 cacheMode==1 时有效]
    // ====================================================================
    uint32_t kernelKey = 0;
    kernelKey |= (cacheLayout == "nd" ? 0u : 1u) << 0;
    kernelKey |= (dataType == "half" ? 0u : 1u) << 1;
    kernelKey |= (maskType == 1 ? 1u : 0u) << 2;
    kernelKey |= (maskType == 4 ? 1u : 0u) << 3;
    int innerPrecBit = (dataType == "half") ? ((innerPrec == 1) ? 1 : 0) : 0;
    kernelKey |= innerPrecBit << 4;
    kernelKey |= (layout == "TND" ? 0u : 1u) << 5;
    kernelKey |= (cacheMode == 1 ? 1u : 0u) << 6;
    int pageShapeBit = (cacheMode == 1) ? ((pageShape == 0) ? 0 : 1) : 0;
    kernelKey |= pageShapeBit << 7;

    std::call_once(g_init_flag, InitDispatcher);
    LaunchFunc func = g_dispatcher[kernelKey];
    if (func) {
        for (int i = 0; i < 1; i++) {
            func(blockDim, stream,
                 qDevice, kDevice, vDevice, maskDevice, blockTableDevice,
                 oDevice, lseDevice, qSeqDevice, kvSeqDevice,
                 workspaceDevice, tilingDevice, enableDN);
            ACL_CHECK(aclrtSynchronizeStream(stream));

            vector<fp16_t> oHostHalf(oSize / sizeof(fp16_t));
            vector<bfloat16> oHostBf16(oSize / sizeof(bfloat16));
            if (dataType == "half") {
                ACL_CHECK(aclrtMemcpy(oHostHalf.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST));
            } else if (dataType == "bf16") {
                ACL_CHECK(aclrtMemcpy(oHostBf16.data(), oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST));
            }

            void *output = nullptr;
            aclrtMallocHost(&output, 128 * 128 * sizeof(fp16_t));
            aclrtMemcpy(output, sizeof(fp16_t) * 128 * 128, workspaceDevice, sizeof(fp16_t) * 128 * 128, ACL_MEMCPY_DEVICE_TO_HOST);

            vector<float> goldenHost(oSize / sizeof(fp16_t));
            const size_t goldenSize = oSize * 2;
            ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

            vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen)
                                                                : golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
            if (errorIndices.empty()) {
                cout << "Compare success." << endl;
            } else {
                cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
            }
        }
    } else {
        printf("Error: unsupported kernelKey %u\n", kernelKey);
    }

    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHost, kvSeqDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    if (maskType == 1 || maskType == 3) {
        FreeMem(maskHost, maskDevice);
    }
    if (cacheMode == 1) {
        FreeMem(blockTableHost, blockTableDevice);
    }
    if (sink_flag == 1) {
        FreeMem(sinkHost, sinkDevice);
    }
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    // aclrtFree(sDevice);
    // aclrtFree(pDevice);
    // aclrtFree(oTempDevice);
    // aclrtFree(oUpdateDevice);
    aclrtFree(workspaceDevice);
    // aclrtFreeHost(tilingHost);
    aclrtFreeHost(qNtokens);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

/// Entry point to mla example.

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}