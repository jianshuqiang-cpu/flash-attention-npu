#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_gelu.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_gelu.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

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

using namespace Catlass;

using ArchTag = Arch::Ascend950;

using ElementA = CATLASS_JIT_ELEMENT_A;
using LayoutTagA = layout::RowMajor;

using ElementB = CATLASS_JIT_ELEMENT_B;
using LayoutTagB = layout::RowMajor;

using ElementO = CATLASS_JIT_ELEMENT_C;
using LayoutTagO = layout::RowMajor;

using ElementMmOut = float32_t;
using LayoutTagMmOut = layout::RowMajor;

static constexpr uint32_t PRELOAD_STAGES = 1;
static constexpr uint32_t L1A_STAGES = 2;
static constexpr uint32_t L1B_STAGES = 2;
static constexpr uint32_t L0A_STAGES = 2;
static constexpr uint32_t L0B_STAGES = 2;
static constexpr uint32_t L0C_STAGES = 1;

static constexpr bool ENABLE_UNIT_FLAG = true;
static constexpr bool ENABLE_SHUFFLE_K = false;
static constexpr bool USE_HF32_MODE = false;
static constexpr bool ENABLE_L1_RESIDENT = true;

using DispatchPolicy = Gemm::MmadPreloadAsyncWithCallbackL0CToUB<
    ArchTag, PRELOAD_STAGES, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG,
    ENABLE_SHUFFLE_K, USE_HF32_MODE, ENABLE_L1_RESIDENT>;

using L1_TILE_M = Int<240>;
using L1_TILE_N = Int<256>;

using L1TileShape = Shape<L1_TILE_M, L1_TILE_N, Int<128>>;
using L0TileShape = Shape<L1_TILE_M, L1_TILE_N, Int<64>>;

using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMmOut, LayoutTagMmOut, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false, Gemm::Tile::ScaleGranularity::NO_QUANT>;
using BlockMmadTla = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementMmOut, void, TileCopy>;

using UBTIleCopy = Shape<L1_TILE_M, L1_TILE_N>;
using MmOutType = Gemm::GemmType<ElementMmOut, LayoutTagMmOut, AscendC::TPosition::VECCALC>;
using OType = Gemm::GemmType<ElementO, LayoutTagO>;
using EpilogueDispatchPolicy = Epilogue::EpilogueElemWiseNoSourceFromUB;
using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseGeluRegBase<ArchTag, ElementO, ElementMmOut>;
using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, MmOutType, OType>;

using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, MmOutType, OType, TileElemWiseEpilogue, EpilogueTileCopy, UBTIleCopy>;

using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMGelu<BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t problemCount = params->batch;

    Catlass::GemmCoord problemShape{m, n, k};

    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};

    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);

    typename MatmulKernel::Arguments arguments{
        problemShape,         problemCount, params->inputAddr[2], params->inputAddr[0], layoutA,
        params->inputAddr[1], layoutB, params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
