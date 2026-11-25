#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "catlass/epilogue/tile/tile_pertoken_dequant.hpp"
#endif

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B int8_t
#endif
#define CATLASS_JIT_ELEMENT_C int32_t
#ifndef CATLASS_JIT_ELEMENT_SCALE
#define CATLASS_JIT_ELEMENT_SCALE float
#endif
#ifndef CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE
#define CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE float
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

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementScale = CATLASS_JIT_ELEMENT_SCALE;
using ElementPerToken = CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE;
using ElementD = CATLASS_JIT_ELEMENT_D;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::RowMajor;
using LayoutTagScale = layout::VectorLayout;
using LayoutTagPerToken = layout::VectorLayout;
using LayoutTagD = layout::RowMajor;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;

using DispatchPolicyMmadTla = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<512>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
using TileCopyMmadTla = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC,
    void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using BlockMmadTla = Gemm::Block::BlockMmadTla<
    DispatchPolicyMmadTla, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopyMmadTla>;

constexpr bool ubDB = false;
constexpr uint32_t UB_STAGES = ubDB ? 2 : 1;
using EpilogueTileShape = MatrixShape<tla::get<0>(L1TileShape{}), tla::get<1>(L1TileShape{})>;
using DispatchPolicyDequant = Epilogue::EpilogueAscend950PerTokenDequantTla<UB_STAGES>;
using TilePerTokenDequant = Epilogue::Tile::TilePerTokenDequant<
    ArchTag, ElementC, ElementScale, ElementPerToken, ElementD, EpilogueTileShape>;
using TileCopyEpilogue = Epilogue::Tile::TileCopyDequantTla<
    ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale, ElementPerToken, LayoutTagPerToken, ElementD, LayoutTagD>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    DispatchPolicyDequant, EpilogueTileShape, ElementC, ElementScale, ElementPerToken, ElementD,
    TilePerTokenDequant, TileCopyEpilogue>;
constexpr bool isGmm = true;
using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape, isGmm>;

using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMPerTokenTla<
    BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t problemCount = params->batch;

    GemmCoord problemShape{m, n, k};

    typename MatmulKernel::Arguments arguments{
        problemShape,
        problemCount,
        params->inputAddr[2],
        params->inputAddr[0],
        params->inputAddr[1],
        params->inputAddr[3],
        params->inputAddr[4],
        params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
