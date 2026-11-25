#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/optimized_matmul.hpp"
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
using DispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;

using L1TileShapeBase = std::conditional_t<
    std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
    GemmShape<256, 128, 256>,
    GemmShape<128, 256, 256>>;
using L0TileShapeBase = std::conditional_t<
    std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
    GemmShape<256, 128, 64>,
    GemmShape<128, 256, 64>>;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, L1TileShapeBase>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, L0TileShapeBase>::type;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;
using BlockEpilogue = void;

using TileCopy = Gemm::Tile::ReluTileCopy<ArchTag, AType, BType, CType>;
using BlockMmadX = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
    void, void, BlockMmadX, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    GemmCoord shape{params->m, params->n, params->k};

    typename MatmulKernel::Arguments arguments{
        shape, params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
