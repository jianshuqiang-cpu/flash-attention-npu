#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"
#include "common/workspace_alloc.h"

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
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;

using BaseL1 = tuple<C<128>, C<256>, C<256>>;
using BaseL0 = tuple<C<128>, C<256>, C<64>>;
using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0>::type;

const uint32_t align = 256;

template <class LayoutTag>
auto GetPaddingLayout(LayoutTag layout, uint32_t blockRows, uint32_t blockCols)
{
    if constexpr (std::is_same_v<LayoutTag, layout::RowMajor>) {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(
                static_cast<int64_t>(blockCols), static_cast<int64_t>(blockRows) * RoundUp(layout.shape(1), blockCols)),
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols));
        return MakeLayout(shape, stride);
    } else {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols),
            MakeStride(
                static_cast<int64_t>(blockRows),
                RoundUp(layout.shape(0), blockRows) * static_cast<int64_t>(blockCols)));
        return MakeLayout(shape, stride);
    }
}

template <class LayoutTag>
size_t GetWorkspaceLen(LayoutTag layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows)
         * RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

#ifndef CATLASS_JIT_NEED_PADDING_A
#define CATLASS_JIT_NEED_PADDING_A false
#endif
#ifndef CATLASS_JIT_NEED_PADDING_B
#define CATLASS_JIT_NEED_PADDING_B false
#endif

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;
using BlockEpilogue = void;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);

    using TensorWA = Tensor<
        AscendC::GlobalTensor<ElementA>, decltype(layoutA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorWB = Tensor<
        AscendC::GlobalTensor<ElementB>, decltype(layoutB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorC = Tensor<
        AscendC::GlobalTensor<ElementC>, decltype(layoutC), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceC = params->outputAddr[0];

#if defined(CATLASS_JIT_NEED_PADDING_A) && CATLASS_JIT_NEED_PADDING_A
    auto layoutWA = GetPaddingLayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
    constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
    using TensorWAA = Tensor<
        AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorWA, TensorWAA, computeLengthA>;
#else
    auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
    using TensorWAA = Tensor<
        AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using PaddingA = void;
#endif
#if defined(CATLASS_JIT_NEED_PADDING_B) && CATLASS_JIT_NEED_PADDING_B
    auto layoutWB = GetPaddingLayout(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{}));
    constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
    using TensorWBB = Tensor<
        AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorWB, TensorWBB, computeLengthB>;
#else
    auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
    using TensorWBB = Tensor<
        AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using PaddingB = void;
#endif

    using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
        ArchTag, TensorWAA, LayoutTagA, TensorWBB, LayoutTagB, TensorC, LayoutTagC, void, void,
        CATLASS_JIT_NEED_PADDING_A, CATLASS_JIT_NEED_PADDING_B>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, TensorWAA, TensorWBB, TensorC, void, TileCopy>;

    using MatmulKernel = Gemm::Kernel::OptimizedMatmulTla<
        BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB>;

    uint8_t* deviceWA = deviceA;
    uint8_t* deviceWB = deviceB;
#if CATLASS_JIT_NEED_PADDING_A
    {
        size_t sizeWA = GetWorkspaceLen(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{})) * sizeof(ElementA);
        deviceWA = g_catlassWorkspaceAlloc(sizeWA);
    }
#endif
#if CATLASS_JIT_NEED_PADDING_B
    {
        size_t sizeWB = GetWorkspaceLen(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{})) * sizeof(ElementB);
        deviceWB = g_catlassWorkspaceAlloc(sizeWB);
    }
#endif

    typename MatmulKernel::Arguments arguments{
        GemmCoord{m, n, k},
        deviceA, layoutA, deviceB, layoutB, deviceC, layoutC,
        deviceWA, layoutWA, deviceWB, layoutWB};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
