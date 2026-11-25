#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemv/block/block_gemv.hpp"
#include "catlass/gemv/kernel/kernel_gemv_aiv.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"
#include "catlass/gemv/tile/tile_vmad.hpp"
#include "catlass/gemv/tile/tile_vmuls.hpp"
#include "catlass/gemv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif

using namespace Catlass;
using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutX = layout::VectorLayout;
using LayoutY = layout::VectorLayout;
using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::GemvAtlasA2;
using UBTileShape = GemvShape<32, 512>;
using AType = Gemm::GemmType<float, LayoutA>;
using XType = Gemm::GemmType<float, LayoutX>;
using YType = Gemm::GemmType<float, LayoutY>;
using BiasType = void;
using TileCopy = Gemv::Tile::TileCopyGemvAiv<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
using TileVmad = Gemv::Tile::TileVmad<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>;
using TileVmuls = Gemv::Tile::TileVmuls<typename DispatchPolicy::ArchTag, XType>;
using GemvBlock = Gemv::Block::BlockGemv<DispatchPolicy, UBTileShape, AType, XType, YType, BiasType, TileCopy, TileVmad, TileVmuls>;
using GemvKernel = Gemv::Kernel::KernelGemvAiv<GemvBlock, void>;

static uint32_t getSplitNum(uint32_t m, uint32_t n, uint32_t M1, uint32_t N1, uint32_t maxSplit)
{
    (void)n;(void)N1;
    uint32_t bn=(m-1)/M1+1, s1=1, s2=1, mo=0;
    for(uint32_t i=1; i<=maxSplit; i+=1){uint32_t o=(i*bn)%40; if(!o) o=40; if(o>mo){mo=o;s1=i;}}
    mo=0;
    for(uint32_t i=1; i<=maxSplit; i<<=1){uint32_t o=(i*bn)%40; if(!o) o=40; if(o>mo){mo=o;s2=i;}}
    return (s1-s2)>4?s1:s2;
}

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::GemmParams* params)
{
    uint32_t m = params->m, k = params->k;
    uint32_t split = getSplitNum(m, k, UBTileShape::M, UBTileShape::N, 20);

    typename GemvKernel::Arguments arguments{
        GemvCoord{m, k}, params->inputAddr[0], params->inputAddr[1],
        params->outputAddr[0], params->outputAddr[0], params->alpha, params->beta, split};

    Catlass::RunKernel<GemvKernel>(arguments, stream, blockNum);
}
