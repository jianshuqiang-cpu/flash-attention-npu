#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/sparse_matmul_tla.hpp"
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
#define CATLASS_JIT_ELEMENT_A int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C int32_t
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

using L1TileShape = Shape<_128, _256, _128>;
using L0TileShape = Shape<_128, _256, _64>;

using DispatchPolicy = Gemm::SparseMatmulMultiBlockOnKAxis<ArchTag, false>;

using TileCopy = Gemm::Tile::SparseTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmad = Gemm::Block::BlockMmadSparseTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;

struct MatmulShape {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t b;
};

using ProblemShape = MatmulShape;
using BlockScheduler = Gemm::Block::BlockSchedulerIterateK<ProblemShape, L1TileShape, L0TileShape>;

using MatmulKernel = Gemm::Kernel::KernelSparseMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k / 2, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceIdx = params->inputAddr[2];
    uint8_t* deviceC = params->outputAddr[0];
    uint8_t* deviceBias = nullptr;

    MatmulShape shape = {m, n, k, 1};

    typename MatmulKernel::Arguments args = {
        shape,
        {deviceA, deviceB, deviceC, deviceBias, deviceIdx, layoutA, layoutB, layoutC},
        blockNum};

    Catlass::RunKernel<MatmulKernel>(args, stream, blockNum);
}
