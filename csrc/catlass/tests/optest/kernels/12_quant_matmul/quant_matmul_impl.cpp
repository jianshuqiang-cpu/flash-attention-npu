#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C int32_t
#endif
#ifndef CATLASS_JIT_ELEMENT_D
#define CATLASS_JIT_ELEMENT_D half
#endif
#ifndef CATLASS_JIT_ELEMENT_SCALE
#define CATLASS_JIT_ELEMENT_SCALE half
#endif
#ifndef CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE
#define CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE half
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementD = CATLASS_JIT_ELEMENT_D;
using ElementScale = CATLASS_JIT_ELEMENT_SCALE;
using ElementPerTokenScale = CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE;

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutD = layout::RowMajor;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallback<1, 2, 2, 2, 1, false, true>;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, int8_t, GemmShape<128, 256, 512>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, int8_t, GemmShape<128, 256, 128>>::type;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, layout::RowMajor>;

using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

constexpr uint32_t ubStages = 2;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
using ScaleType = Gemm::GemmType<ElementScale, layout::VectorLayout>;
using PerTokenScaleType = Gemm::GemmType<ElementPerTokenScale, layout::VectorLayout>;
using DType = Gemm::GemmType<ElementD, layout::RowMajor>;

using EpilogueTileShape = MatrixShape<32, 256>;
using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType, DType,
    TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;

constexpr uint32_t workspaceStages = 2;
using MatmulKernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<
    BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    GemmCoord shape{params->m, params->n, params->k};

    typename MatmulKernel::Arguments arguments{
        shape, blockNum, params->inputAddr[0], params->inputAddr[1],
        params->inputAddr[2], params->inputAddr[3], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
