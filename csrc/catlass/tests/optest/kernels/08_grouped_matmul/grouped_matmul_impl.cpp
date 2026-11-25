#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler.h"
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
#define CATLASS_JIT_LAYOUT_A ColumnMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using namespace Catlass;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<1, 2, 4, 2, 1, true, true>;

using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<128, 256, 256>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, GemmShape<128, 256, 64>>::type;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

using MatmulKernel = Gemm::Kernel::GroupedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t problemCount = params->batch;
    uint32_t m = params->m;
    uint32_t n = params->n;

    auto* deviceGroupList = params->inputAddr[2];

    std::vector<int64_t> hostGroupList(problemCount);
    aclrtMemcpy(hostGroupList.data(), problemCount * sizeof(int64_t),
                deviceGroupList, problemCount * sizeof(int64_t),
                ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<GemmCoord> hostProblemShapes(problemCount);
    std::vector<LayoutA> hostLayoutA(problemCount);
    std::vector<LayoutB> hostLayoutB(problemCount);
    std::vector<LayoutC> hostLayoutC(problemCount);

    for (uint32_t i = 0; i < problemCount; ++i) {
        uint32_t currentK = (i == 0) ? static_cast<uint32_t>(hostGroupList[0])
                                     : static_cast<uint32_t>(hostGroupList[i] - hostGroupList[i - 1]);
        hostProblemShapes[i] = GemmCoord{m, n, currentK};
        hostLayoutA[i] = LayoutA::template MakeLayout<ElementA>(m, currentK);
        hostLayoutB[i] = LayoutB::template MakeLayout<ElementB>(currentK, n);
        hostLayoutC[i] = LayoutC::template MakeLayout<ElementC>(m, n);
    }

    GemmCoord* problemShapeListDevice = (GemmCoord*)g_catlassWorkspaceAllocFromHost(hostProblemShapes.data(), problemCount * sizeof(GemmCoord));
    LayoutA* layoutAListDevice = (LayoutA*)g_catlassWorkspaceAllocFromHost(hostLayoutA.data(), problemCount * sizeof(LayoutA));
    LayoutB* layoutBListDevice = (LayoutB*)g_catlassWorkspaceAllocFromHost(hostLayoutB.data(), problemCount * sizeof(LayoutB));
    LayoutC* layoutCListDevice = (LayoutC*)g_catlassWorkspaceAllocFromHost(hostLayoutC.data(), problemCount * sizeof(LayoutC));

    typename MatmulKernel::Arguments arguments{
        problemCount, reinterpret_cast<uint8_t*>(problemShapeListDevice), params->inputAddr[0],
        reinterpret_cast<uint8_t*>(layoutAListDevice),
        params->inputAddr[1], reinterpret_cast<uint8_t*>(layoutBListDevice),
        params->outputAddr[0], reinterpret_cast<uint8_t*>(layoutCListDevice)};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);

    aclrtSynchronizeStream(stream);
}
