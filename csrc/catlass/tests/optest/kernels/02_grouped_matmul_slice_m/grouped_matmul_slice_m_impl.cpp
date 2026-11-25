#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

#include "../common/common.h"
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
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::AtlasA2;
using GroupListElement = int64_t;

#ifndef CATLASS_JIT_K_GT_N
#define CATLASS_JIT_K_GT_N 0
#endif

#if CATLASS_JIT_K_GT_N
using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<1, 2, 2, 4, 1, true, true>;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<256, 128, 256>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<256, 128, 64>>::type;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
#else
using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<1, 2, 4, 2, 1, true, true>;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<128, 256, 256>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<128, 256, 64>>::type;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
#endif

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;

using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, GroupListElement>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    GemmCoord shape{params->m, params->n, params->k};
    uint32_t problemCount = params->batch;
    auto* deviceGroupList = params->inputAddr[2];

    typename MatmulKernel::Arguments arguments{
        shape, problemCount, deviceGroupList, params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
