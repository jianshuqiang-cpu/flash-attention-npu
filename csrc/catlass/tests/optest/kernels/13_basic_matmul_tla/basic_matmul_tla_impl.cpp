#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

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
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;

using BaseL1 = tuple<C<128>, C<256>, C<256>>;
using BaseL0 = tuple<C<128>, C<256>, C<64>>;
using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0>::type;

using TileCopy =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;

using MatmulKernel = Gemm::Kernel::BasicMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayout<ElementC, LayoutTagC>(m, n);

    typename MatmulKernel::Arguments arguments{
        GemmCoord{m, n, k}, params->inputAddr[0], layoutA, params->inputAddr[1], layoutB, params->outputAddr[0], layoutC};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
