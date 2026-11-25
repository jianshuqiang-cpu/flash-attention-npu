#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_silu.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_activation.hpp"
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
#ifndef CATLASS_JIT_ELEMENT_D
#define CATLASS_JIT_ELEMENT_D half
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

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementD = CATLASS_JIT_ELEMENT_D;

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<128, 256, 256>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<128, 256, 64>>::type;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<float, LayoutC>;
using DType = Gemm::GemmType<ElementD, LayoutC>;

using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseNoSource;

constexpr uint32_t computeLength = 16384;
using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseSilu<ArchTag, CType, computeLength>;
using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, DType>;
using BlockEpilogue =
    Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, DType, TileElemWiseEpilogue, EpilogueTileCopy>;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;

using MatmulKernel = Gemm::Kernel::MatmulActivation<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    GemmCoord shape{params->m, params->n, params->k};

    typename MatmulKernel::Arguments arguments{
        shape, sizeof(float), params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
