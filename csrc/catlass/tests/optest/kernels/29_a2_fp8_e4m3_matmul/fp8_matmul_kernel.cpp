#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/fp8_matmul.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass_kernel.h"
#include "common/workspace_alloc.h"
#include "common/kernel_runner.h"

using namespace Catlass;

using ArchTag = Arch::AtlasA2;

using ElementA = half;
using ElementPrologueA = int8_t;
using ElementB = half;
using ElementPrologueB = int8_t;
using ElementC = float;

using LayoutA = layout::RowMajor;
using LayoutPrologueA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutPrologueB = layout::RowMajor;
using LayoutC = layout::RowMajor;

using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;

constexpr uint32_t mScalar = 2;
constexpr uint32_t nScalar = 2;
constexpr uint32_t splitkLength = 1024;

constexpr bool ENABLE_UNIT_FLAG = true;
using DispatchPolicy = Gemm::MmadAtlasA2PingpongSliceKWithPrologue<ENABLE_UNIT_FLAG>;

using PrologueSrcTypeA = Gemm::GemmType<ElementPrologueA, LayoutPrologueA>;
using PrologueDstTypeA = Gemm::GemmType<ElementA, LayoutA>;
using PrologueSrcTypeB = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
using PrologueDstTypeB = Gemm::GemmType<ElementB, LayoutB>;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

constexpr uint32_t COMPUTE_LENGTH_A = 16 * 1024 / sizeof(int8_t);
constexpr uint32_t COMPUTE_LENGTH_B = 16 * 1024 / sizeof(int8_t);
using PrologueA = Gemm::Tile::TileCastFp8ToFp16Dequant<ArchTag, PrologueSrcTypeA, PrologueDstTypeA, COMPUTE_LENGTH_A>;
using PrologueB = Gemm::Tile::TileCastFp8ToFp16Dequant<ArchTag, PrologueSrcTypeB, PrologueDstTypeB, COMPUTE_LENGTH_B>;
using TileCopy = Gemm::Tile::TileCopyWithPrologue<ArchTag, AType, BType, CType, PrologueA, PrologueB>;
using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
using BlockEpilogue = void;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

using MatmulKernel = Gemm::Kernel::FP8Matmul<BlockMmadOpt, BlockEpilogue, BlockScheduler, mScalar, nScalar, splitkLength>;

extern "C" void A2Fp8E4M3Matmul(
    const uint32_t blockNum, aclrtStream stream, const CatlassKernel::TParams& tParams,
    const CatlassKernel::MatmulParams& params)
{
    GemmCoord shape{params.m, params.n, params.k};
    half scalar = 1.0;
    half zeroPoint = 0.0;

    typename MatmulKernel::Arguments arguments{
        shape, blockNum, params.inputAddr[0], params.inputAddr[1],
        params.outputAddr[0], scalar, zeroPoint};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
