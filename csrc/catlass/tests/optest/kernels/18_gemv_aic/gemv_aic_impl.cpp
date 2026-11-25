#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aic.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif

using namespace Catlass;
using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutX = layout::VectorLayout;
using LayoutZ = layout::VectorLayout;
using ArchTag = Arch::AtlasA2;
using LayoutC = layout::RowMajor;
using DispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;
using L1TileShape = GemvShape<32, 512>;
using L0TileShape = GemvShape<32, 256>;
using AType = Gemm::GemmType<float, LayoutA>;
using XType = Gemm::GemmType<float, LayoutX>;
using CType = Gemm::GemmType<float, LayoutC>;
using BiasType = void;
using TileCopy = Gemv::Tile::TileCopyGemvAic<typename DispatchPolicy::ArchTag, AType, XType, CType, BiasType>;
using TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, XType, AType, BiasType>;
using BlockGemv = Gemv::Block::BlockGemv<DispatchPolicy, L1TileShape, L0TileShape, AType, XType, CType, BiasType, TileCopy, TileMmad>;
using EpiBlockDP = Epilogue::EpilogueAtlasA2Gemv;
using YType = Gemm::GemmType<float, LayoutZ>;
using ZType = Gemm::GemmType<float, LayoutZ>;
using AXType = Gemm::GemmType<float, LayoutZ>;
using ComputeType = AXType;
using TileAdd = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, 8192>;
using TileMul = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, 8192>;
using TileCp = Epilogue::Tile::TileCopy<ArchTag, YType, AXType, ZType>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpiBlockDP, AXType, YType, ZType, TileAdd, TileMul, TileCp>;
using GemvKernel = Gemv::Kernel::KernelGemvAic<BlockGemv, BlockEpilogue>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::GemmParams* params)
{
    uint32_t m = params->m, n = params->n;
    typename GemvKernel::Arguments arguments{
        GemvCoord{m, n}, params->alpha, params->beta, sizeof(float),
        params->inputAddr[1], params->inputAddr[0], params->outputAddr[0]};
    Catlass::RunKernel<GemvKernel>(arguments, stream, blockNum);
}
