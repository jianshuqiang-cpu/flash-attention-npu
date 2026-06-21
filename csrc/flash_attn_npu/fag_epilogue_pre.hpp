/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fag_block.h"
#include "kernel_operator.h"
#include "fag_common/common_header.h"

namespace Catlass::Epilogue::Block {

// FAGPre 是 FlashAttention Gradient 反向的 Vector 侧预处理阶段。
// 它在真正的 Cube/Vector 反向计算前执行，负责把 fp32 的 dq/dk/dv workspace 清零。
// 后续 Cube/Vector 阶段会在这些 workspace 上累加梯度，最后 FAGPost 再将其 cast 并写回输出 tensor。
template <
    typename ElementVecDtype
>
class BlockEpilogue<
    EpilogueAtlasA2FAGPre,
    ElementVecDtype>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    AscendC::TPipe *pipe;
    // 需要清零的 fp32 workspace。dq 使用 qSize，dk/dv 使用 kvSize。
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

    // 当前 AIV core 编号。
    uint32_t cBlockIdx;
    // dq workspace 的按 core 切分参数。
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    // dk/dv workspace 的按 core 切分参数，二者形状相同，因此共用一套 kvPre* 参数。
    uint32_t kvPreBlockFactor;
    uint32_t kvPreBlockTotal;
    uint32_t kvPreBlockTail;

    // 当前 core 实际清零的元素数和起始 offset。
    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in, __gm__ uint8_t *dq, 
    __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace, __gm__ uint8_t * tiling_in)
    {
        cBlockIdx = AscendC::GetBlockIdx();
        pipe = pipe_in;

        // 从 tiling_data 中读取 workspace 偏移和展平元素数量。
        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)tiling_in);
        int64_t dqWorkSpaceOffset = tilingData.GetValue(TILING_DQ_WORKSPACE_OFFSET);
        int64_t dkWorkSpaceOffset = tilingData.GetValue(TILING_DK_WORKSPACE_OFFSET);
        int64_t dvWorkSpaceOffset = tilingData.GetValue(TILING_DV_WORKSPACE_OFFSET);
        int64_t qSize = tilingData.GetValue(TILING_Q_SIZE);
        int64_t kvSize = tilingData.GetValue(TILING_KV_SIZE);

        AscendC::GlobalTensor<uint32_t> tilingDataU32;
        tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)tiling_in);;
        uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);

        // 计算 dq 清零任务的切分：每个 core 至多负责 qPreBlockFactor 个 fp32 元素。
        qPreBlockFactor = (qSize + coreNum - 1) / coreNum;
        qPreBlockTotal = (qSize + qPreBlockFactor - 1) / qPreBlockFactor;
        int64_t qPreTailNumTmp = qSize % qPreBlockFactor;
        qPreBlockTail = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

        // 计算 dk/dv 清零任务的切分：dk 和 dv 元素数相同，共用 kvPreBlockFactor。
        kvPreBlockFactor = (kvSize + coreNum - 1) / coreNum;
        kvPreBlockTotal = (kvSize + kvPreBlockFactor - 1) / kvPreBlockFactor;
        int64_t kvPreTailNumTmp = kvSize % kvPreBlockFactor;
        kvPreBlockTail = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

        // 绑定 fp32 workspace。这里按 float 指针寻址，所以 byte offset 需要除以 sizeof(float)。
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvWorkSpaceOffset / sizeof(float));

        // 计算当前 core 负责清零的 dq 起点和长度；最后一个有效 core 使用 tail 长度。
        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
        // dk/dv 使用相同的起点和长度。
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
        dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    // 主入口：只在 AIV core 上执行 workspace 清零。
    CATLASS_DEVICE
    void operator()()
    {
        // 清零 dq workspace 中属于当前 core 的分片。
        if (g_coreType == AscendC::AIV && cBlockIdx < qPreBlockTotal) {
            AscendC::InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }

        // 清零 dk/dv workspace 中属于当前 core 的分片。
        if (g_coreType == AscendC::AIV && cBlockIdx < kvPreBlockTotal) {
            AscendC::InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
            AscendC::InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
        }
    }

};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
