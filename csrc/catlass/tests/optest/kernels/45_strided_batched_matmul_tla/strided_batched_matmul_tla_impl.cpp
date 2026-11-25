#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/strided_batched_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
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
#define CATLASS_JIT_LAYOUT_B RowMajor
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
using LayoutTagC = layout::RowMajor;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadPingpongTlaV2<ArchTag, true>;

using L1TileShape = Shape<_128, _256, _256>;
using L0TileShape = Shape<_128, _256, _64>;

using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 31
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;

using MatmulKernel = Gemm::Kernel::StridedBatchedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

template <typename LayoutTag>
auto MakeTlaLayoutA(uint32_t batchCount, uint32_t m, uint32_t k, int64_t strideA, int64_t lda)
{
    if constexpr (std::is_same_v<LayoutTag, layout::RowMajor>) {
        return MakeLayout(MakeShape(batchCount, m, k), MakeStride(strideA, lda, Int<1>{}));
    } else {
        return MakeLayout(MakeShape(batchCount, m, k), MakeStride(strideA, Int<1>{}, lda));
    }
}

template <typename LayoutTag>
auto MakeTlaLayoutB(uint32_t batchCount, uint32_t k, uint32_t n, int64_t strideB, int64_t ldb)
{
    if constexpr (std::is_same_v<LayoutTag, layout::RowMajor>) {
        return MakeLayout(MakeShape(batchCount, k, n), MakeStride(strideB, ldb, Int<1>{}));
    } else {
        return MakeLayout(MakeShape(batchCount, k, n), MakeStride(strideB, Int<1>{}, ldb));
    }
}

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::StridedBatchedMatmulParams* params)
{
    uint32_t batchCount = params->batch;
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    int64_t strideA = params->strideA;
    int64_t strideB = params->strideB;
    int64_t strideC = params->strideC;
    int64_t lda = params->lda;
    int64_t ldb = params->ldb;
    int64_t ldc = params->ldc;

    auto layoutA = MakeTlaLayoutA<LayoutTagA>(batchCount, m, k, strideA, lda);
    auto layoutB = MakeTlaLayoutB<LayoutTagB>(batchCount, k, n, strideB, ldb);
    auto layoutC = MakeLayout(MakeShape(batchCount, m, n), MakeStride(strideC, ldc, Int<1>{}));

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceC = params->outputAddr[0];

    typename MatmulKernel::Arguments arguments{
        batchCount, Catlass::GemmCoord{m, n, k},
        deviceA, layoutA,
        deviceB, layoutB,
        deviceC, layoutC};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
