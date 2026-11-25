/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/**
 * JIT kernel 模板 — basic_matmul
 * 由运行时 bisheng -D... 编译，所有可变参数通过宏注入
 */

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A half
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B half
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C half
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = Catlass::layout::CATLASS_JIT_LAYOUT_C;

/** ---- 固定编译期参数 ---- */
using ArchTag = Catlass::Arch::AtlasA2;
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;

using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, Catlass::GemmShape<128, 256, 256>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, Catlass::GemmShape<128, 256, 64>>::type;

using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

using MatmulKernel = typename Catlass::Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    Catlass::GemmCoord shape{params->m, params->n, params->k};

    typename MatmulKernel::Arguments arguments{
        shape, params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
