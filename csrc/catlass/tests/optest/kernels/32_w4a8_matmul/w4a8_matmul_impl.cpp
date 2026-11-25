#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/w4a8_matmul.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler.h"
#include "common/workspace_alloc.h"

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
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif
#ifndef CATLASS_JIT_SCALAR
#define CATLASS_JIT_SCALAR 1.5
#endif

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementPrologueB = AscendC::int4b_t;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutPrologueB = layout::RowMajor;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::AtlasA2;

constexpr bool enableUnitFlag = false;
using DispatchPolicy = Gemm::MmadAtlasA2PingPongWithPrologue<enableUnitFlag>;

using L1TileShape = std::conditional_t<
    std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
    GemmShape<256, 128, 512>, GemmShape<128, 256, 512>>;
using L0TileShape = std::conditional_t<
    std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
    GemmShape<256, 128, 128>, GemmShape<128, 256, 128>>;

using PrologueSrcType = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
using PrologueDstType = Gemm::GemmType<ElementB, LayoutB>;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = PrologueDstType;
using CType = Gemm::GemmType<ElementC, LayoutC>;

using PrologueA = void;
constexpr uint32_t computeLen = 24 * 1024;
using PrologueB = Gemm::Tile::TileCastInt4ToInt8<ArchTag, PrologueSrcType, PrologueDstType, computeLen>;

using TileCopy = Gemm::Tile::TileCopyWithPrologueDeqPerTensor<ArchTag, AType, BType, CType, PrologueA, PrologueB>;
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 31
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;
using BlockEpilogue = void;

using MatmulKernel = Gemm::Kernel::W4A8Matmul<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    LayoutA layoutA{m, k};
    LayoutPrologueB layoutPrologueB{k, n, (n + 1) / 2 * 2};
    LayoutC layoutC{m, n};

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceC = params->outputAddr[0];

    float scalar = 1.5f;

    typename MatmulKernel::Arguments arguments{
        Catlass::GemmCoord{m, n, k},
        deviceA, layoutA,
        deviceB, layoutPrologueB,
        deviceC, layoutC,
        scalar,
        blockNum};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
