#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstring>
#include <vector>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/group_gemm.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"

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
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using namespace Catlass;
using ScalarType = float;
using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutX = layout::CATLASS_JIT_LAYOUT_C;
using ArchTag = Arch::AtlasA2;
using GemmBlockDP = Gemm::GemmAtlasA2<true, true, true>;
using EpiBlockDP = Epilogue::EpilogueAtlasA2Gemm;
using BaseL1 = GemmShape<128, 256, 256>;
using BaseL0 = GemmShape<128, 256, 64>;
using L1TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScaler<ElementA, half, BaseL0>::type;
using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutX>;
using XType = Gemm::GemmType<ElementC, LayoutX>;
using DType = XType;
using ComputeType = CType;
using GemmBlock = Gemm::Block::BlockGemm<GemmBlockDP, L1TileShape, L0TileShape, AType, BType, CType>;
constexpr uint32_t cL = L1TileShape::MN / 2;
using EpiBlock = Epilogue::Block::BlockEpilogue<EpiBlockDP, CType, XType, DType,
    Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, cL>,
    Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, cL>,
    Epilogue::Tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>,
    Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>>;
using Kernel = Gemm::Kernel::KernelGroupGemm<GemmBlock, EpiBlock>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::GemmParams* params)
{
    uint32_t G = params->batch, m = params->m, n = params->n;
    auto* dg = params->inputAddr[2];
    std::vector<int64_t> hg(G);
    aclrtMemcpy(hg.data(), G*sizeof(int64_t), dg, G*sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<GemmCoord> hs(G);
    std::vector<LayoutA> hla(G);
    std::vector<LayoutB> hlb(G);
    std::vector<LayoutX> hlc(G);
    for(uint32_t i=0; i<G; ++i) {
        uint32_t K = (i==0)?(uint32_t)hg[0]:(uint32_t)(hg[i]-hg[i-1]);
        hs[i] = GemmCoord{m,n,K}; hla[i] = LayoutA{m,K};
        hlb[i] = LayoutB{K,n}; hlc[i] = LayoutX{m,n};
    }

    std::vector<ScalarType> ha(G, params->alpha), hb(G, params->beta);
    ScalarType* dAlpha = (ScalarType*)g_catlassWorkspaceAllocFromHost(ha.data(), G*sizeof(ScalarType));
    ScalarType* dBeta  = (ScalarType*)g_catlassWorkspaceAllocFromHost(hb.data(), G*sizeof(ScalarType));
    GemmCoord*  dS     = (GemmCoord*) g_catlassWorkspaceAllocFromHost(hs.data(), G*sizeof(GemmCoord));
    LayoutA*    dLA    = (LayoutA*)   g_catlassWorkspaceAllocFromHost(hla.data(), G*sizeof(LayoutA));
    LayoutA*    dLWA   = (LayoutA*)   g_catlassWorkspaceAllocFromHost(hla.data(), G*sizeof(LayoutA));
    LayoutB*    dLB    = (LayoutB*)   g_catlassWorkspaceAllocFromHost(hlb.data(), G*sizeof(LayoutB));
    LayoutB*    dLWB   = (LayoutB*)   g_catlassWorkspaceAllocFromHost(hlb.data(), G*sizeof(LayoutB));
    LayoutX*    dLC    = (LayoutX*)   g_catlassWorkspaceAllocFromHost(hlc.data(), G*sizeof(LayoutX));

    typename Kernel::Arguments args{G,(uint8_t*)dS,(uint8_t*)dAlpha,(uint8_t*)dBeta,
        params->inputAddr[0],(uint8_t*)dLA,params->inputAddr[1],(uint8_t*)dLB,
        params->outputAddr[0],(uint8_t*)dLC,
        params->inputAddr[0],(uint8_t*)dLWA,params->inputAddr[1],(uint8_t*)dLWB,
        params->outputAddr[0],params->outputAddr[0]};

    Catlass::RunKernel<Kernel>(args, stream, blockNum);
    aclrtSynchronizeStream(stream);
}
