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
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_per_tensor_per_channel_dequant_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B int8_t
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
#ifndef CATLASS_JIT_QUANT_MODE
#define CATLASS_JIT_QUANT_MODE 0
#endif

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::RowMajor;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Gemm::MmadDequant<ArchTag, enableUnitFlag, useHF32>;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<64>>;

#if CATLASS_JIT_QUANT_MODE == 0
constexpr Gemm::Tile::ScaleGranularity QuantMode = Gemm::Tile::ScaleGranularity::PER_TENSOR;
#else
constexpr Gemm::Tile::ScaleGranularity QuantMode = Gemm::Tile::ScaleGranularity::PER_CHANNEL;
#endif

using TileCopy = Gemm::Tile::PackedTileCopyTla<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, false, QuantMode>;
using BlockMmadTla = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;

constexpr bool isGmm = true;
using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape, isGmm>;

using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMFixpipeDequantTla<
    BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t problemCount = params->batch;

    GemmCoord problemShape{m, n, k};

    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);

    union { uint8_t* ptr; float val; } scaleUnion;
    scaleUnion.ptr = params->inputAddr[4];
    float scalePerTensor = scaleUnion.val;

    typename MatmulKernel::Arguments arguments{
        problemShape,
        problemCount,
        params->inputAddr[2],
        params->inputAddr[0], layoutA,
        params->inputAddr[1], layoutB,
        params->outputAddr[0], layoutC,
        scalePerTensor,
        params->inputAddr[3]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
