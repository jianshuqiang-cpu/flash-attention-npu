#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/trmm.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

#include <type_traits>

using namespace Catlass;

#ifdef CATLASS_JIT_ELEMENT_A
using ElementA = CATLASS_JIT_ELEMENT_A;
#else
using ElementA = float;
#endif
#ifdef CATLASS_JIT_ELEMENT_B
using ElementB = CATLASS_JIT_ELEMENT_B;
#else
using ElementB = float;
#endif
#ifdef CATLASS_JIT_ELEMENT_C
using ElementC = CATLASS_JIT_ELEMENT_C;
#else
using ElementC = float;
#endif
#ifdef CATLASS_JIT_LAYOUT_A
using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
#else
using LayoutTagA = layout::RowMajor;
#endif
#ifdef CATLASS_JIT_LAYOUT_B
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
#else
using LayoutTagB = layout::RowMajor;
#endif
#ifdef CATLASS_JIT_LAYOUT_C
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;
#else
using LayoutTagC = layout::RowMajor;
#endif

#ifdef CATLASS_JIT_TRMM_TILE_VARIANT
constexpr uint32_t trmmTileVariant = CATLASS_JIT_TRMM_TILE_VARIANT;
#else
constexpr uint32_t trmmTileVariant = 0;
#endif

#ifdef CATLASS_JIT_BLOCK_SCHEDULER
constexpr uint32_t blockSchedulerCode = CATLASS_JIT_BLOCK_SCHEDULER;
#else
constexpr uint32_t blockSchedulerCode = 30;
#endif

using BaseL1 = std::conditional_t<
    (trmmTileVariant == 1 || trmmTileVariant == 2), tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>,
    std::conditional_t<
        (trmmTileVariant == 3 || trmmTileVariant == 7), tla::tuple<tla::C<128>, tla::C<128>, tla::C<128>>,
        tla::tuple<tla::C<128>, tla::C<128>, tla::C<256>>>>;
using BaseL0 = tla::tuple<tla::C<128>, tla::C<128>, tla::C<64>>;

using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, float, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, float, BaseL0>::type;
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<(blockSchedulerCode / 10), (blockSchedulerCode % 10)>;

using ArchTag = Arch::AtlasA2;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
using TileCopy =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmad =
    Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;

using TrmmKernel = Gemm::Kernel::Trmm<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::TrmmParams* params)
{
    if (params == nullptr) {
        return;
    }

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(params->m, params->k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(params->k, params->n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(params->m, params->n);

    typename TrmmKernel::Arguments arguments{
        GemmCoord{params->m, params->n, params->k},
        params->inputAddr[0],
        layoutA,
        params->inputAddr[1],
        layoutB,
        params->outputAddr[0],
        layoutC,
        params->side,
        params->uplo,
        params->trans,
        params->diag,
        params->alpha};

    Catlass::RunKernel<TrmmKernel>(arguments, stream, blockNum);
}
