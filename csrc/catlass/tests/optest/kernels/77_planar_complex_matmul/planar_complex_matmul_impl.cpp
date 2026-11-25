#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_planar_complex_fused_tla.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/planar_complex_gemm_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/detail/kernel_adapter.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/workspace_alloc.h"

#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif
#ifdef CATLASS_JIT_USE_FOUR_PASS
constexpr bool kUseFourPass = CATLASS_JIT_USE_FOUR_PASS;
#else
constexpr bool kUseFourPass = false;
#endif
#ifdef CATLASS_JIT_NEGATE_A
constexpr bool kNegateA = CATLASS_JIT_NEGATE_A;
#else
constexpr bool kNegateA = false;
#endif
#ifdef CATLASS_JIT_SWIZZLE_DIR
constexpr uint32_t kSwizzleDir = CATLASS_JIT_SWIZZLE_DIR;
#else
constexpr uint32_t kSwizzleDir = 1;
#endif

using namespace Catlass;
using namespace tla;

#ifdef CATLASS_JIT_ELEMENT_A
using ElementA = CATLASS_JIT_ELEMENT_A;
#else
using ElementA = half;
#endif
#ifdef CATLASS_JIT_ELEMENT_B
using ElementB = CATLASS_JIT_ELEMENT_B;
#else
using ElementB = half;
#endif
#ifdef CATLASS_JIT_ELEMENT_C
using ElementC = CATLASS_JIT_ELEMENT_C;
#else
using ElementC = float;
#endif

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, /*ENABLE_UNIT_FLAG=*/true>;
using L1TileShape = tuple<Int<128>, Int<256>, Int<256>>;
using L0TileShape = tuple<Int<128>, Int<256>, Int<64>>;

using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;

using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, kSwizzleDir>;
using BlockMmadFourPass = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, /*ElementBias=*/void, TileCopy>;
using BlockMmadFused = Gemm::Block::BlockMmadTla<
    Gemm::MmadPlanarComplexFused<ArchTag, /*ENABLE_SHUFFLE_K=*/true>, L1TileShape, L0TileShape, ElementA, ElementB,
    ElementC, /*ElementBias=*/void, TileCopy>;

using PlanarKernel =
    Gemm::Kernel::PlanarComplexGemm<kUseFourPass, kNegateA, BlockMmadFourPass, BlockMmadFused, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    GemmCoord shape{params->m, params->n, params->k};
    typename PlanarKernel::Arguments arguments{shape,   params->inputAddr[0],  params->inputAddr[1],
                                               nullptr, params->inputAddr[2],  params->inputAddr[3],
                                               nullptr, params->outputAddr[0], params->outputAddr[1]};

    if (!PlanarKernel::CanImplement(arguments)) {
        return;
    }

    size_t workspaceSize = PlanarKernel::GetWorkspaceSize(arguments);
    uint8_t* workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = g_catlassWorkspaceAlloc(workspaceSize);
    }
    auto kernelParams = PlanarKernel::ToUnderlyingArguments(arguments, workspace);

    uint64_t hardwareSyncAddr = 0;
    aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr));
    if (hardwareSyncAddr == 0) {
        KernelAdapter<PlanarKernel><<<blockNum, nullptr, stream>>>(kernelParams);
    } else {
        KernelAdapter<PlanarKernel><<<blockNum, nullptr, stream>>>(kernelParams, hardwareSyncAddr);
    }
}
