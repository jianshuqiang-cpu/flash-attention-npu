#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/single_core_slicek_matmul.hpp"
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
#define CATLASS_JIT_LAYOUT_B ColumnMajor
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

using ArchTag = Catlass::Arch::AtlasA2;
using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;
using ElementAccumulator =
    typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

template <class ArchTagX, class ATypeX, class BTypeX, class CTypeX, class BiasTypeX = void>
struct TileCopyDynamicOptimized
    : public Catlass::Gemm::Tile::TileCopy<ArchTagX, ATypeX, BTypeX, CTypeX, BiasTypeX> {
    using CopyGmToL1A = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTagX, ATypeX>;
    using CopyGmToL1B = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTagX, BTypeX>;
};

constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, Catlass::layout::zN> || std::is_same_v<LayoutA, Catlass::layout::nZ>)
                                       ? PaddingTag::NO_PADDING
                                       : PaddingTag::PADDING_NZ;
constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, Catlass::layout::zN> || std::is_same_v<LayoutB, Catlass::layout::nZ>)
                                       ? PaddingTag::NO_PADDING
                                       : PaddingTag::PADDING_NZ;
constexpr PaddingTag paddingTagC = PaddingTag::NO_PADDING;

using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA>;
using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB>;
using PaddingA = typename PaddingBuilderA::Padding;
using PaddingB = typename PaddingBuilderB::Padding;

using AType = Catlass::Gemm::GemmType<ElementA, typename PaddingBuilderA::LayoutAfterPadding>;
using BType = Catlass::Gemm::GemmType<ElementB, typename PaddingBuilderB::LayoutAfterPadding>;
using CType = Catlass::Gemm::GemmType<ElementAccumulator, LayoutC>;

using TileCopy = TileCopyDynamicOptimized<ArchTag, AType, BType, CType>;
using BlockEpilogue = void;

constexpr bool enableUnitFlag = false;
constexpr uint32_t l0CStages = 2;
constexpr uint32_t l1AStages = 2;
constexpr uint32_t l1BStages = 1;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, Catlass::GemmShape<128, 256, 512>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, Catlass::GemmShape<128, 128, 128>>::type;

using DispatchPolicy =
    Catlass::Gemm::MmadAtlasA2SingleCoreSplitk<l1AStages, l1BStages, l0CStages, enableUnitFlag>;
using BlockScheduler =
    typename Catlass::Gemm::Block::SingleCoreSplitkGemmIdentityBlockSwizzle<20, 0>;
using BlockMmad = Catlass::Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;

using RemovePaddingNDAndCast_ = Catlass::Gemm::Kernel::RemovePaddingNDAndCast<
    paddingTagC, ArchTag, ElementAccumulator, ElementC, LayoutC>;
using RemovePaddingNDAndCastC = std::conditional_t<
    paddingTagC == PaddingTag::PADDING_ND || !std::is_same_v<ElementAccumulator, ElementC>,
    RemovePaddingNDAndCast_, void>;

using MatmulKernel = Catlass::Gemm::Kernel::SingleCoreSplitkMatmul<
    PaddingA, PaddingB, BlockMmad, BlockEpilogue, BlockScheduler, RemovePaddingNDAndCastC>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    Catlass::GemmCoord shape{params->m, params->n, params->k};

    typename MatmulKernel::Arguments arguments{
        shape, params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
