#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstddef>
#include <type_traits>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"

// Swizzle: Base -- GemmIdentityBlockSwizzle
//          ASWT -- GemmGroupedAswtTailSplitSwizzle
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_swizzle_grouped_aswt.hpp"
// Block: Base: BlockMmadTla (with dispatch: MmadMx)
//        Preload: BlockMmadMxPreloadTla (with dispatch: MmadMxPreload)
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_mx_preload_tla.hpp"
// Kernel: Base: GroupedMxMatmulSliceMTla
//         ASWT: GroupedMxMatmulSliceMAswtTla
#include "catlass/gemm/kernel/grouped_mx_matmul_slice_m_tla.hpp"
#include "catlass/gemm/kernel/grouped_mx_matmul_slice_m_aswt_tla.hpp"

#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
#endif
#ifndef CATLASS_JIT_ELEMENT_MX_SCALE
#define CATLASS_JIT_ELEMENT_MX_SCALE float8_e8m0_t
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif

/// Advanced optimization flags: MX_GMM_ENABLE_PRELOAD
#ifndef MX_GMM_ENABLE_PRELOAD
#define MX_GMM_ENABLE_PRELOAD 0
#endif
#ifndef MX_GMM_ENABLE_ASWT
#define MX_GMM_ENABLE_ASWT 0
#define MX_GMM_ENABLE_BASE_M 0
#endif
#if MX_GMM_ENABLE_ASWT
#define MX_GMM_ENABLE_BASE_M 1 // baseM strategy is enabled by default (with ASWT)
#endif 

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementMxScale = CATLASS_JIT_ELEMENT_MX_SCALE;
using ElementGroupList = int64_t;

using LayoutTagA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = Catlass::layout::RowMajor;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool isFp4 = std::is_same_v<ElementA, float4_e2m1x2_t>;
using L1TileShape = std::conditional_t<isFp4,
    tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<512>>,
    tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>>;
using L0TileShape = std::conditional_t<isFp4,
    tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>,
    tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>>;

constexpr bool enableUnitFlag = true;
constexpr uint32_t l1ScaleFactorK = 16;
constexpr uint32_t l0cStages = 1;  // 与 enableUnitFlag=true 配套，只能为 1
constexpr uint32_t preloadStages = 1;
constexpr bool enableL1Resident = true;
#if MX_GMM_ENABLE_PRELOAD
using DispatchPolicy = Catlass::Gemm::MmadMxPreload<ArchTag, preloadStages, enableUnitFlag, l1ScaleFactorK, l0cStages, enableL1Resident>;
#else
using DispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag, l1ScaleFactorK, l0cStages, enableL1Resident>;
#endif

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t groupCount = params->batch;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    Catlass::GemmCoord problemShape{m, n, k};

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceGroupList = params->inputAddr[2];
    uint8_t* deviceMxScaleA = params->inputAddr[3];
    uint8_t* deviceMxScaleB = params->inputAddr[4];
    uint8_t* deviceC = params->outputAddr[0];

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;

#if MX_GMM_ENABLE_PRELOAD
    using BlockMmad = Catlass::Gemm::Block::BlockMmadMxPreloadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
#else
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
#endif

    using BlockEpilogue = void;
#if MX_GMM_ENABLE_ASWT
    constexpr bool transB = (std::is_same_v<LayoutTagB, Catlass::layout::ColumnMajor> || std::is_same_v<LayoutTagB, Catlass::layout::nZ>) ? true : false;
    using BlockScheduler = typename Catlass::Gemm::Block::GemmGroupedAswtTailSplitSwizzle<4, false, transB>;
    using MatmulKernel = Catlass::Gemm::Kernel::GroupedMxMatmulSliceMAswtTla<BlockMmad,
        BlockEpilogue, BlockScheduler, ElementGroupList, (MX_GMM_ENABLE_BASE_M != 0)>;
    typename MatmulKernel::Arguments arguments{
        problemShape, groupCount,
        deviceGroupList, deviceA, layoutA, deviceB, layoutB,
        deviceMxScaleA, layoutMxScaleA, deviceMxScaleB, layoutMxScaleB,
        deviceC, layoutC};
    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
#else
    if (m >= n * groupCount) {
        using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using MatmulKernel = Catlass::Gemm::Kernel::GroupedMxMatmulSliceMTla<
            BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;

        typename MatmulKernel::Arguments arguments{
            problemShape, groupCount,
            deviceGroupList, deviceA, layoutA, deviceB, layoutB,
            deviceMxScaleA, layoutMxScaleA, deviceMxScaleB, layoutMxScaleB,
            deviceC, layoutC};

        Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
    } else {
        using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using MatmulKernel = Catlass::Gemm::Kernel::GroupedMxMatmulSliceMTla<
            BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;

        typename MatmulKernel::Arguments arguments{
            problemShape, groupCount,
            deviceGroupList, deviceA, layoutA, deviceB, layoutB,
            deviceMxScaleA, layoutMxScaleA, deviceMxScaleB, layoutMxScaleB,
            deviceC, layoutC};

        Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
    }
   
#endif
    
}
