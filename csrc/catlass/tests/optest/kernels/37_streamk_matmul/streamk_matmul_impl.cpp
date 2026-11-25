#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/streamk_matmul.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

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
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = Catlass::layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Catlass::Arch::AtlasA2;
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Streamk<true, true>;

using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, Catlass::GemmShape<128, 256, 256>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, Catlass::GemmShape<128, 256, 64>>::type;

using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

using ElementAccumulator =
    typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = typename Catlass::Gemm::Block::StreamkGemmIdentityBlockSwizzle<3, 0>;

constexpr uint32_t computeLength = 192 * 1024 / sizeof(ElementAccumulator);
using ReduceAdd =
    Catlass::Gemm::Kernel::StreamkReduceAdd<ArchTag, BlockScheduler, ElementAccumulator, ElementC, computeLength>;

using MatmulKernel = Catlass::Gemm::Kernel::StreamkMatmul<BlockMmad, BlockEpilogue, BlockScheduler, ReduceAdd>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    Catlass::GemmCoord shape{params->m, params->n, params->k};

    typename MatmulKernel::Arguments arguments{
        shape, blockNum, params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
