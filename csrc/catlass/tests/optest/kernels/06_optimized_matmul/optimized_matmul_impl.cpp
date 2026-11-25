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
#ifndef CATLASS_JIT_NEED_PADDING_A
#define CATLASS_JIT_NEED_PADDING_A 0
#endif
#ifndef CATLASS_JIT_NEED_PADDING_B
#define CATLASS_JIT_NEED_PADDING_B 0
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

template <
    class ArchTagT,
    class ATypeT,
    class BTypeT,
    class CTypeT,
    class BiasTypeT = void>
struct TileCopyOpt : public Gemm::Tile::TileCopy<ArchTagT, ATypeT, BTypeT, CTypeT, BiasTypeT> {
    using Base = Gemm::Tile::TileCopy<ArchTagT, ATypeT, BTypeT, CTypeT, BiasTypeT>;
    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;
    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;
    using CopyL0CToGm = typename Base::CopyL0CToGm;
    using BiasTypeSelector = typename Base::BiasTypeSelector;
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};

using PaddingTag = Gemm::Kernel::PaddingTag;

constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>)
                                       ? PaddingTag::NO_PADDING
                                       : PaddingTag::PADDING_NZ;
constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>)
                                       ? PaddingTag::NO_PADDING
                                       : PaddingTag::PADDING_NZ;

static const uint32_t COMPUTE_LENGTH_A = 48 * 1024 / sizeof(ElementA);
static const uint32_t COMPUTE_LENGTH_B = 48 * 1024 / sizeof(ElementB);

using PaddingBuilderA =
    Gemm::Kernel::PaddingBuilder<paddingTagA, ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
using PaddingBuilderB =
    Gemm::Kernel::PaddingBuilder<paddingTagB, ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

#if CATLASS_JIT_NEED_PADDING_A && CATLASS_JIT_NEED_PADDING_B
using GlobalPaddingA = typename PaddingBuilderA::Padding;
using GlobalPaddingB = typename PaddingBuilderB::Padding;
using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BTypeMmad, CType>;
using BlockMmadX = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BTypeMmad, CType, void, TileCopy>;
using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
    GlobalPaddingA, GlobalPaddingB, BlockMmadX, BlockEpilogue, BlockScheduler>;
#elif CATLASS_JIT_NEED_PADDING_A
using GlobalPaddingA = typename PaddingBuilderA::Padding;
using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BType, CType>;
using BlockMmadX = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BType, CType, void, TileCopy>;
using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
    GlobalPaddingA, void, BlockMmadX, BlockEpilogue, BlockScheduler>;
#elif CATLASS_JIT_NEED_PADDING_B
using GlobalPaddingB = typename PaddingBuilderB::Padding;
using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
using TileCopy = TileCopyOpt<ArchTag, AType, BTypeMmad, CType>;
using BlockMmadX = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BTypeMmad, CType, void, TileCopy>;
using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
    void, GlobalPaddingB, BlockMmadX, BlockEpilogue, BlockScheduler>;
#else
using TileCopy = TileCopyOpt<ArchTag, AType, BType, CType>;
using BlockMmadX = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
    void, void, BlockMmadX, BlockEpilogue, BlockScheduler>;
#endif

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    GemmCoord shape{params->m, params->n, params->k};

    typename MatmulKernel::Arguments arguments{
        shape, params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
