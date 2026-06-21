/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fag_block.h"
#include "kernel_operator.h"
#include "fag_common/common_header.h"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TQue;

namespace Catlass::Epilogue::Block {

// FAGPost 是 FlashAttention Gradient 反向的最后一个 Vector 后处理阶段。
// 前面的 Cube/Vector 阶段会在 fp32 workspace 中累加 dq/dk/dv；本阶段负责：
//   1. 从 workspace 读取 fp32 梯度；
//   2. 对 dq/dk 乘以 softmax_scale；
//   3. cast 到输出 dtype，例如 FP16/BF16；
//   4. 写回最终 dq/dk/dv 输出 tensor。
template <
    typename ElementVecDtype
>
class BlockEpilogue<
    EpilogueAtlasA2FAGPost,
    ElementVecDtype>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPost;
    using ArchTag = typename DispatchPolicy::ArchTag;

    constexpr static uint32_t POST_BUFFER_NUM = 1;

    AscendC::TPipe *pipe;
    // 输入 UB 用于承接 fp32 workspace，输出 UB 用于承接 cast 后的目标 dtype 数据。
    TBuf<QuePosition::VECIN> inBuffer;
    TBuf<QuePosition::VECOUT> outBuffer;

    // 输入：前序 Cube 阶段在 fp32 workspace 中累积出的 dq/dk/dv。
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    // 输出：最终写回 Python/PyTorch 可见的 dq/dk/dv tensor。
    AscendC::GlobalTensor<ElementVecDtype> dqGm, dkGm, dvGm;

    // 当前 AIV core 编号。
    int64_t cBlockIdx;
    // ubBaseSize 是单次循环最多处理的 fp32 元素数量对应的 UB 容量。
    int64_t ubBaseSize;
    // dq 使用 qPost* 切分；dk/dv 形状一致，复用 kvPost* 切分。
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    int64_t kvPostBlockFactor;
    uint64_t kvPostBlockTotal;
    int64_t kvPostBaseNum;
    int64_t kvPostTailNum;
    // dq/dk 需要乘以 softmax_scale，dv 不需要缩放。
    float scaleValue;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in, __gm__ uint8_t *dq, 
    __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace, __gm__ uint8_t * tiling_in)
    {
        cBlockIdx = GetBlockIdx();
        pipe = pipe_in;

        // 从 tiling_data 中读取 workspace 偏移、总元素数量和 scale。
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
        int64_t mixCoreNum = (coreNum + 1) / 2;

        AscendC::GlobalTensor<float> tilingDataFp;
        tilingDataFp.SetGlobalBuffer((__gm__ float *)tiling_in);
        scaleValue = tilingDataFp.GetValue(TILING_SCALE_VALUE * CONST_2);

        // 绑定最终输出 tensor。
        dqGm.SetGlobalBuffer((__gm__ ElementVecDtype *)dq);
        dkGm.SetGlobalBuffer((__gm__ ElementVecDtype *)dk);
        dvGm.SetGlobalBuffer((__gm__ ElementVecDtype *)dv);

        // 绑定 fp32 累加 workspace。
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvWorkSpaceOffset / sizeof(float));

        // 计算 post 阶段每次循环能处理多少 fp32 元素。
        // POST_COEX_NODE=3 表示为 dq/dk/dv 三类后处理预留共存空间，ubBaseSize 再按 256 元素对齐。
        constexpr static uint32_t POST_COEX_NODE = 3;
        constexpr static uint32_t WORKSPACE_NUM_ALIGN = 256;
        uint32_t curPostCoexNode =  POST_COEX_NODE;
        uint32_t ubSize = ArchTag::UB_SIZE;
        ubBaseSize = ubSize / curPostCoexNode / POST_BUFFER_NUM;
        ubBaseSize = ubBaseSize / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN; // align

        // dq 总元素数为 qSize，每次循环处理 qPostBaseNum 个 fp32 元素。
        qPostBaseNum = ubBaseSize / sizeof(float);
        qPostBlockTotal = qSize;

        int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
        int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;

        qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
        qPostBlockFactor = (qPostBlockOuterTotal + coreNum - 1) / coreNum;

        // dk/dv 总元素数为 kvSize，二者形状相同，因此共用 kvPost* 切分参数。
        kvPostBaseNum = qPostBaseNum;
        kvPostBlockTotal = kvSize;

        int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
        int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;

        kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
        kvPostBlockFactor = (kvPostBlockOuterTotal + coreNum - 1) / coreNum;


        pipe->InitBuffer(inBuffer, ubBaseSize * 2);
        pipe->InitBuffer(outBuffer, ubBaseSize);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    // 主入口：每个 AIV core 负责 dq 的一段和 dk/dv 的一段，完成 scale、cast 和写回。
    CATLASS_DEVICE
    void operator()()
    {
        // 当前 core 负责的 dq 元素范围，单位是展平后的元素下标。
        uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
        uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

        if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
            qEnd = qPostBlockTotal;
        }
        event_t Mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {

            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<ElementVecDtype> vecOut = outBuffer.Get<ElementVecDtype>();
            uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            // fp32 每 8 个元素为 32B，对齐后从 dq workspace 搬入 UB。
            DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B

            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            // dq 与 softmax score 相关，需要乘以 softmax_scale 后再输出。
            Muls(vecIn, vecIn, scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            // 输出 dtype 通常是 fp16/bf16，每 16 个元素为 32B，对齐后写回 dq。
            DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // 当前 core 负责的 dk/dv 元素范围。dk 和 dv 形状相同，所以共用这段范围。
        uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
        if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
            kvEnd = kvPostBlockTotal;
        }

        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<ElementVecDtype> vecOut = outBuffer.Get<ElementVecDtype>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

            // dk 同样来自 score 方向梯度，需要乘以 softmax_scale。
            Muls(vecIn, vecIn, scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);

            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            // 输出 dtype 通常是 fp16/bf16，每 16 个元素为 32B，对齐后写回 dk。
            DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);

        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // dv 不需要乘 softmax_scale，只需要从 fp32 workspace cast 到输出 dtype 后写回。
        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<ElementVecDtype> vecOut = outBuffer.Get<ElementVecDtype>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            // 输出 dtype 通常是 fp16/bf16，每 16 个元素为 32B，对齐后写回 dv。
            DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            if (i + kvPostBaseNum < kvEnd) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            }
        }

    }

};
    
}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP
