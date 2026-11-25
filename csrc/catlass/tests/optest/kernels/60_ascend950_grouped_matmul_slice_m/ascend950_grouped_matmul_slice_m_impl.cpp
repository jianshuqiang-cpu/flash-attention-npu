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
#include "catlass/gemm/kernel/grouped_matmul_slice_m_tla.hpp"
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

#ifndef CATLASS_JIT_K_GT_N
#define CATLASS_JIT_K_GT_N 0
#endif

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutTagA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = Catlass::layout::RowMajor;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<64>>;

using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmadTla = Catlass::Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;

#if CATLASS_JIT_K_GT_N
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
#else
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
#endif

using MatmulKernel = Catlass::Gemm::Kernel::GroupedMatmulSliceMTla<
    BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t problemCount = params->batch;

    Catlass::GemmCoord problemShape{m, n, k};

    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);

    typename MatmulKernel::Arguments arguments{
        problemShape, problemCount, params->inputAddr[2],
        params->inputAddr[0], layoutA,
        params->inputAddr[1], layoutB,
        params->outputAddr[0], layoutC};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
