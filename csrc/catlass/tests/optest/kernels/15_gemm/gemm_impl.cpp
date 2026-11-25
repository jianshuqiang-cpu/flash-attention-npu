#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/gemm.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler.h"
#include "common/workspace_alloc.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
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
using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutX = layout::CATLASS_JIT_LAYOUT_C;
using ArchTag = Arch::AtlasA2;
using GemmBlockDP = Gemm::GemmAtlasA2<true, true, true>;
using EpiBlockDP = Epilogue::EpilogueAtlasA2Gemm;
using BaseL1 = GemmShape<128, 128, 128>;
using BaseL0 = GemmShape<128, 128, 64>;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, float, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, float, BaseL0>::type;
using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutX>;
using XType = Gemm::GemmType<ElementC, LayoutX>;
using DType = XType;
using ComputeType = CType;
using GemmBlock = Gemm::Block::BlockGemm<GemmBlockDP, L1TileShape, L0TileShape, AType, BType, CType>;
constexpr uint32_t cL = L1TileShape::MN / 2;
using TileAdd = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, cL>;
using TileMul = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, cL>;
using TileCast = Epilogue::Tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;
using TileCp = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
using EpiBlock = Epilogue::Block::BlockEpilogue<EpiBlockDP, CType, XType, DType, TileAdd, TileMul, TileCast, TileCp>;
using Kernel = Gemm::Kernel::KernelGemm<GemmBlock, EpiBlock>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::GemmParams* params)
{
    uint32_t m = params->m, n = params->n, k = params->k;
    float alpha = params->alpha, beta = params->beta;

    LayoutX layoutX{m, n};
    typename EpiBlock::Params epilogueParams{alpha, beta, params->outputAddr[0], layoutX, params->outputAddr[0], layoutX};

    uint8_t* gws = g_catlassWorkspaceAlloc((size_t)m * n * sizeof(float));

    typename Kernel::Arguments args{GemmCoord{m,n,k}, 128,
        params->inputAddr[0], params->inputAddr[1],
        gws, params->inputAddr[0], params->inputAddr[1], epilogueParams};

    Catlass::RunKernel<Kernel>(args, stream, blockNum);
}
