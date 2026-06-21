 /**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fag_block.h"
#include "kernel_operator.h"
#include "fag_common/common_header.h"
#include "fag_common/vector_addr.h"

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

// FAGOp 是 FlashAttention Gradient 反向中的 Vector 侧核心 epilogue。
// 它消费 VectorAddr 生成的 VecAddrInfo，从 Cube 侧 workspace 读取两类矩阵乘中间结果：
//   1. mm2WorkspaceGm：通常对应 QK^T 相关分数，用于重新计算 softmax 概率 P。
//   2. mm1WorkspaceGm：通常对应 dO·V^T 相关结果，用于结合 softmax 导数生成 dS。
// 最终它会把 P 写入 dropWorkSpaceGm，并把 dS 写入 mulWorkSpaceGm，供后续 Cube2/Cube3 继续计算 dq/dk/dv。
template <
    typename ElementVecDtype,
    InputLayout inputLayout
>
class BlockEpilogue<
    EpilogueAtlasA2FAGOp,
    ElementVecDtype,
    std::integral_constant<InputLayout, inputLayout>>
{
public:
    using DispatchPolicy = EpilogueAtlasA2FAGOp;
    using ArchTag = typename DispatchPolicy::ArchTag;

    static constexpr InputLayout getLayout() {
        return std::integral_constant<InputLayout, inputLayout>::value;
    }
    AscendC::TPipe *pipe;
    // 整个 FAGOp 使用一块手动切分的 UB，下面的 *Begin 常量描述各临时区偏移。
    TBuf<> unifiedBuffer;

    // 全局内存视图：atten mask、Cube 中间结果、softmax lse、softmax gradient 辅助结果。
    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    // dropWorkSpaceGm 存放重新计算出的 P，mulWorkSpaceGm 存放反向 softmax 后的 dS。
    GlobalTensor<ElementVecDtype> dropWorkSpaceGm, mulWorkSpaceGm;
    GlobalTensor<float> rowLseGm;
    GlobalTensor<float> sfmgWorkspaceGm;

    constexpr static uint32_t DTYPE_FACTOR = sizeof(float) / sizeof(ElementVecDtype);
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(float);
    constexpr static uint32_t input_block_num = 32 / sizeof(ElementVecDtype);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;

    // UB 内各区域的起始偏移。该文件没有用多个 TBuf，而是用 unifiedBuffer 手动分片。
    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t U8Begin = 58 * 1024;
    constexpr static uint32_t T2BlockBegin = 66 * 1024;

    constexpr static uint32_t DbBegin = 74 * 1024;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static  uint32_t AttenMaskDimS2 = 2048;


    // blockIdx 是实际 AIV core 编号。两个 Vector core 对应一个 Cube 逻辑 core：
    // cubeBlockIdx = blockIdx / 2，subIdx = blockIdx % 2。
    // subIdx=0 处理一个 128x128 子块的上半部分 Q 行，subIdx=1 处理下半部分 Q 行。
    uint32_t blockIdx;
    uint32_t cubeBlockIdx;
    uint32_t subIdx;

        
    // taskId 用于 ping-pong workspace 选择，blockLen 是当前 VecAddrInfo 中有效子块数量。
    int32_t taskId = 0;
    int32_t pingpongIdx = 0;
    int32_t blockLen = 0;

    // org shape info
    int64_t b;
    int64_t nheads_k;
    int64_t g;
    int64_t cuQSeqLen;
    int64_t cuKSeqLen;
    int64_t headdim;
    int64_t seq_q;
    int64_t seq_k;

    float scaleValue;

    int32_t cubeBaseMN;

    // s1 表示 Q 方向，s2 表示 K 方向。Cube 基本块大小固定为 128x128。
    // s1VecSize 是把一个 Cube 块沿 Q 方向拆给两个 Vector sub core 后，每个半块的基准行数。
    int32_t s1VecSize;
    int32_t s2VecSize;
    constexpr static int32_t S1_CUBESIZE = 128;
    constexpr static int32_t S2_CUBESIZE = 128;
    
    // 当前 sub core 实际处理的 Q 行数、K 列数以及 K 方向 16 对齐后的长度。
    int32_t s1Extend;
    int32_t s2Extend;
    int32_t s2ExtendAlign;
    int32_t s1CubeExtend;
    int32_t s2CubeExtend;
    
    int32_t curSeqQIdx;
    int32_t curSeqKIdx;

    // offset 
    int32_t sfmgOffset = 0;
    int32_t lseOffset = 0;

    int64_t copyInOffset = 0;
    int64_t copyOutOffset = 0;
    DataCopyParams copyInParam;
    DataCopyParams copyOutParam;

    __gm__ uint8_t *cu_seq_qlen_addr;
    __gm__ uint8_t *cu_seq_kvlen_addr;

    SoftMaxTiling softmaxTilingData;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in, __gm__ uint8_t *row_lse,
    __gm__ uint8_t *atten_mask, __gm__ uint8_t *cu_seq_qlen,
    __gm__ uint8_t *cu_seq_kvlen, __gm__ uint8_t * workspace, int32_t batchIn, __gm__ uint8_t * tiling_in)
    {
        b = batchIn;
        // ub分配
        pipe = pipe_in;

        blockIdx = GetBlockIdx();
        cubeBlockIdx = blockIdx / 2;
        subIdx = blockIdx % 2;
        // curSeqQIdx 用于选择当前 Vector sub core 处理 Cube 块的上半/下半 Q 行。
        curSeqQIdx = subIdx;
        curSeqKIdx = 0;

        // 一个 task 最多 16 个 128x128 子块，ping-pong 双缓冲时每个 cubeBlockIdx 预留两份空间。
        cubeBaseMN = 16 * 128 * 128;

        cu_seq_qlen_addr = cu_seq_qlen;
        cu_seq_kvlen_addr = cu_seq_kvlen;


        // 从 tiling_data 中读取形状、workspace offset、scale 和 softmax tiling 参数。
        // Host 侧在 fag_tiling.cpp 中填充这些字段，Device 侧在这里解释使用。

        AscendC::GlobalTensor<uint64_t> tilingData;
        tilingData.SetGlobalBuffer((__gm__ uint64_t *)tiling_in);
        b = tilingData.GetValue(TILING_B);
        nheads_k = tilingData.GetValue(TILING_N2);
        g = tilingData.GetValue(TILING_G);
        headdim = tilingData.GetValue(TILING_D);
        seq_q = tilingData.GetValue(TILING_T1) / b;
        seq_k = tilingData.GetValue(TILING_T2) / b;

        int64_t sfmgWorkSpaceOffset = tilingData.GetValue(TILING_SFMG_WORKSPACE_OFFSET);
        int64_t mm1WorkSpaceOffset = tilingData.GetValue(TILING_MM1_WORKSPACE_OFFSET);
        int64_t mm2WorkSpaceOffset = tilingData.GetValue(TILING_MM2_WORKSPACE_OFFSET);
        int64_t pWorkSpaceOffset = tilingData.GetValue(TILING_P_WORKSPACE_OFFSET);
        int64_t dsWorkSpaceOffset = tilingData.GetValue(TILING_DS_WORKSPACE_OFFSET);

        AscendC::GlobalTensor<float> tilingDataFp;
        tilingDataFp.SetGlobalBuffer((__gm__ float *)tiling_in);
        scaleValue = tilingDataFp.GetValue(TILING_SCALE_VALUE * CONST_2);

        AscendC::GlobalTensor<uint32_t> tilingHostU32;
        tilingHostU32.SetGlobalBuffer((__gm__ uint32_t *)tiling_in);
        uint32_t coreNum = tilingHostU32.GetValue(TILING_CORE_NUM * CONST_2);
        softmaxTilingData.srcM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2);
        softmaxTilingData.srcK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 1);
        softmaxTilingData.srcSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 2);
        softmaxTilingData.outMaxM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 3);
        softmaxTilingData.outMaxK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 4);
        softmaxTilingData.outMaxSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 5);
        softmaxTilingData.splitM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 6);
        softmaxTilingData.splitK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 7);
        softmaxTilingData.splitSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 8);
        softmaxTilingData.reduceM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 9);
        softmaxTilingData.reduceK = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 10);
        softmaxTilingData.reduceSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 11);
        softmaxTilingData.rangeM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 12);
        softmaxTilingData.tailM = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 13);
        softmaxTilingData.tailSplitSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 14);
        softmaxTilingData.tailReduceSize = tilingHostU32.GetValue(TILING_SOFTMAX_TILING_DATA * CONST_2 + 15);

        pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);
        // 绑定 GM tensor。rowLseGm 是前向 softmax 的 LSE；attenMaskU8Gm 是 causal mask。
        rowLseGm.SetGlobalBuffer((__gm__ float *)row_lse);
        attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);

        // mm1/mm2 是 Cube1 写入的 float 中间结果；mul/drop 是本 Vector op 写回的低精度中间结果。
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + mm1WorkSpaceOffset));
        mulWorkSpaceGm.SetGlobalBuffer((__gm__ ElementVecDtype *)(workspace + dsWorkSpaceOffset));
        
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + mm2WorkSpaceOffset));
        dropWorkSpaceGm.SetGlobalBuffer((__gm__ ElementVecDtype *)(workspace + pWorkSpaceOffset));

        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + sfmgWorkSpaceOffset));
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    // 根据 batch 索引返回 TND 变长布局下 Q 和 K 的真实序列长度。
    CATLASS_DEVICE
    void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &cuSeqQlen, int64_t &cuSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            cuSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[0];
            cuSeqKvlen = ((__gm__ int32_t *)cu_seq_kvlen_addr)[0];
        } else {
            cuSeqQlen =
                ((__gm__ int32_t *)cu_seq_qlen_addr)[bIdx] - ((__gm__ int32_t *)cu_seq_qlen_addr)[bIdx - 1];
            cuSeqKvlen =
                ((__gm__ int32_t *)cu_seq_kvlen_addr)[bIdx] - ((__gm__ int32_t *)cu_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    // 从 GM 拷贝 uint8 causal mask 到 UB。mask 原始宽度按 2048 存储，这里按当前块尺寸截取。
    CATLASS_DEVICE
    void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset, uint32_t s1Extend, uint32_t s2Extend)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = s1Extend;
        intriParams.blockLen = s2Extend * sizeof(uint8_t);
        intriParams.srcStride = (AttenMaskDimS2 - s2Extend) * sizeof(uint8_t);
        intriParams.dstStride = 0;
        intriParams.rsv = 0;
        DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
    }

    // 将 bool/uint8 mask 应用到分数矩阵上：mask 为 true 的位置写入极小值，softmax 后趋近 0。
    CATLASS_DEVICE
    void CalcAttenMaskBool(
        LocalTensor<float> &dstTensor,
        LocalTensor<uint8_t> srcTensor,
        uint32_t s1Extend,
        uint32_t s2SrcExtend,
        uint32_t s2MaskExtend = 128,
        uint8_t maskType = 0)
    {
        LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);

        float scalar;
        if constexpr (AscendC::IsSameType<float, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;
            scalar = *((float *)&tmp);
        } else {
            uint16_t tmp = 0xFBFF;
            scalar = *((ElementVecDtype *)&tmp);
        }

        AscendC::SelectWithBytesMaskShapeInfo info;
        info.firstAxis = s1Extend;
        info.srcLastAxis = s2SrcExtend;
        // info.maskLastAxis = (s2SrcExtend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
        info.maskLastAxis = s2MaskExtend;
        dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
        srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
        AscendC::SelectWithBytesMask<float, uint8_t, false>(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    }

    // 读取前向保存的 row_lse。当前实现随后 Duplicate 为 1.0f，用作 SimpleSoftMax 的辅助输入。
    CATLASS_DEVICE
    void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
    {
        int64_t nheads = nheads_k * g;
        AscendC::DataCopyExtParams lseCopyParam;
        lseCopyParam.blockCount = s1Extend;
        lseCopyParam.blockLen = sizeof(float);
        lseCopyParam.srcStride = (nheads - 1) * sizeof(float);
        lseCopyParam.dstStride = 0;
        lseCopyParam.rsv = 0;
        AscendC::DataCopyPad(dstTensor, rowLseGm[softMaxOffset], lseCopyParam, {false, 0, 0, 0});
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        for (uint32_t i = 0; i < s1Extend; i++) {
            AscendC::Brcb(dstTensor[64 * 8 + i * 8], dstTensor[i * 8], 1, {1, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(dstTensor, 1.0f, s1Extend * 8);
    }

    // 计算 softmax 概率 P。规则块走 SimpleSoftMax，非规则块退化为 Sub + Exp 的手写路径。
    CATLASS_DEVICE
    void CalcSoftMax(LocalTensor<float>& dstTensor, LocalTensor<float>& src0Tensor, 
                    LocalTensor<float>& src1Tensor, uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling& tiling)
    {
        bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

        if (isBasicBlock) {
            AscendC::LocalTensor<uint8_t> sharedTmp = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
            uint32_t shapeArray1[2];
            shapeArray1[0] = s1Extend;
            shapeArray1[1] = s2Extend;
            dstTensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            src0Tensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            AscendC::SimpleSoftMax<float, false, true>(dstTensor, src1Tensor, src1Tensor[64 * 8], src0Tensor,
                                        sharedTmp, tiling);
        } else {
            uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

            for(uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
                uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
                AscendC::Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[64 * 8],
                        subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(dstTensor[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num],
                    subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1),
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8)});
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    // 子图 A：从 mm2Workspace 读取分数块，乘 softmax_scale，应用 causal mask，重新计算 P，写入 dropWorkSpace。
    CATLASS_DEVICE
    void SubGrapA(int64_t curIdx, const VecBlockInfo &blockInfo, event_t mte2WaitMte3A)
    {
        uint32_t ubBufferOffset = 0;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }

        AscendC::LocalTensor<float> vecInBuffer3 =
            unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);
        
        CopyInSoftMax(vecInBuffer3, s1Extend, lseOffset);

        AscendC::LocalTensor<float> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);

        AscendC::DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        AscendC::PipeBarrier<PIPE_V>();
        // 将 Cube 得到的分数块乘以 softmax_scale，恢复前向 softmax 前的缩放分数。
        AscendC::Muls(vecClc2Buffer, vecClc2Buffer, scaleValue, s1Extend * s2ExtendAlign);
            
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        if (blockInfo.SeqQIdx == blockInfo.SeqKIdx) {
            // 只有位于对角线的块需要 causal mask；严格下三角块全部有效，不需要额外 mask。
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8[curSeqQIdx * s1VecSize * 128], s1Extend, s2ExtendAlign, S2_CUBESIZE, 0);
        }

        ///////////////////////////////////////////////////////////////
        // simpleSoftMax
        ///////////////////////////////////////////////////////////////
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), DbBegin);
        CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1Extend, s2Extend, s2ExtendAlign, softmaxTilingData);
        LocalTensor<float> vecDropBuffer = simpleSoftmaxResBuf;

        ///////////////////////////////////////////////////////////////
        // 将 float 概率 P cast 到 ElementVecDtype 后写入 p workspace。
        ///////////////////////////////////////////////////////////////
        LocalTensor<ElementVecDtype> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementVecDtype>(17 * 1024 / sizeof(ElementVecDtype), ubBufferOffset + T1Begin);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(vecCopyOutBuffer, vecDropBuffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);

        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        DataCopyPad(dropWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < blockLen - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }
    }
    
    // 子图 B：从 mm1Workspace 读取 dP 类中间结果，减去 sfmg 辅助项，再乘以子图 A 得到的 P，写入 mulWorkSpace 作为 dS。
    CATLASS_DEVICE
    void SubGrapB(int64_t curIdx, const VecBlockInfo &blockInfo, event_t mte2WaitMte3B)
    {
        uint32_t ubBufferOffset = DbBegin;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }

        // 读取 sfmg 辅助项。它通常表示每行 softmax 反向中需要广播相减的归约项。
        LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
        DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset], s1Extend * 8);

        LocalTensor<float> vecClc1Buffer = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), ubBufferOffset + T1Begin);
        
        // 读取 Cube1 生成的 dP 类中间结果。
        DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        ///////////////////////////////////////////////////////////////
        // softmax 反向的逐行广播相减：dP - sum(dP * P)。
        ///////////////////////////////////////////////////////////////
        uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
            uint32_t subMaskCout =
                (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3,
                subMaskCout, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                static_cast<uint8_t>(s2ExtendAlign / 8), 1});
        }

        ///////////////////////////////////////////////////////////////
        // 再乘以 P，得到 dS = P * (dP - rowsum(dP * P))。
        ///////////////////////////////////////////////////////////////
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), DbBegin);
        Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1Extend * s2ExtendAlign);

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<ElementVecDtype> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementVecDtype>(17 * 1024 / sizeof(ElementVecDtype), ubBufferOffset + T1Begin);
        Cast(vecCopyOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);

        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        // 写回 dS 到 ds workspace，后续 Cube2/Cube3 会读取它计算 dq/dk。
        DataCopyPad(mulWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < blockLen - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

    // 主入口：逐个消费 VectorAddr 生成的 VecBlockInfo，计算当前 task 内所有 attention 子块的 P 和 dS。
    CATLASS_DEVICE
    void operator()(const VecAddrInfo &addrs)
    {
        taskId = addrs.taskId;
        // Cube/Vector 通过 taskId 在两份 workspace 中 ping-pong，避免读写同一缓冲区。
        pingpongIdx = taskId % 2;
        blockLen = addrs.blockLength;

        event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        if (taskId == 0) {
            // mask 在同一个 Vector core 内可复用，首个 task 预取一次 128x128 causal mask 到 UB。
            LocalTensor<uint8_t> attenMaskUbuint8 =
                    unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), BoolBegin);
            CopyInAttenMaskBool(attenMaskUbuint8, 0, 128, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < blockLen; ++i) {
            
            auto &blockInfo = addrs.VecBlkInfo[i];

            ///////////////////////////////////////////////////////////////
            // 标量侧准备：根据 blockInfo 计算当前 Vector sub core 负责的形状、LSE 偏移、sfmg 偏移和 workspace 拷贝参数。
            ///////////////////////////////////////////////////////////////
            
            if constexpr (getLayout() == InputLayout::TND) {
                GetSeqQlenKvlenByBidx(blockInfo.batchIdx, cuQSeqLen, cuKSeqLen);
            } else {
                cuQSeqLen = seq_q;
                cuKSeqLen = seq_k;
            }

            s1CubeExtend = blockInfo.lengthy;
            s2CubeExtend = 128;

            // 一个 Cube 结果块沿 Q 方向拆给两个 Vector core：subIdx=0 处理前半，subIdx=1 处理后半。
            s1VecSize = (s1CubeExtend + 1) / 2;
            s2VecSize = 128;

            s1Extend = subIdx ? s1CubeExtend - s1VecSize : s1VecSize;
            s2Extend = blockInfo.lengthx;
            s2ExtendAlign = (s2Extend + 15) / 16 * 16;

            // 计算 row_lse 偏移。TND 用 cu_seqlens 找 batch 起点，BSND 用固定 seq_q 找 batch 起点。
            int64_t globalSeqStart = 0;
            if constexpr (getLayout() == InputLayout::TND) {
                globalSeqStart = (blockInfo.batchIdx > 0)
                ? ((__gm__ int32_t *)cu_seq_qlen_addr)[blockInfo.batchIdx - 1]
                : 0;
            } else {
                globalSeqStart = (blockInfo.batchIdx > 0)
                ? seq_q
                : 0;
            }
            int64_t seqOffsetInBlock = blockInfo.SeqQIdx * S1_CUBESIZE + curSeqQIdx * s1VecSize;
            int64_t nheads = nheads_k * g;
            int64_t headIdx = blockInfo.nheadsKIdx * g + blockInfo.gIdx;
            lseOffset = (globalSeqStart + seqOffsetInBlock) * nheads + headIdx;

            // 计算 sfmgWorkspace 中当前 Q 行的偏移。sfmg 每行按 8 个 float 存储广播辅助项。
            sfmgOffset = 0;
            if (blockInfo.batchIdx > 0) {
                if constexpr (getLayout() == InputLayout::TND) {
                    sfmgOffset = ((__gm__ int32_t *)cu_seq_qlen_addr)[blockInfo.batchIdx - 1] * nheads_k * g * 8;
                } else {
                    sfmgOffset = seq_q * nheads_k * g * 8;
                }
            }
            sfmgOffset += ((blockInfo.nheadsKIdx * g + blockInfo.gIdx) * cuQSeqLen + blockInfo.SeqQIdx * S1_CUBESIZE + curSeqQIdx * s1VecSize) * 8;
            
            // copyInOffset 指向 Cube 写入的 float workspace；curSeqQIdx 选择当前 sub core 的半块起点。
            copyInOffset = 
                cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset + curSeqQIdx * s1VecSize * s2CubeExtend;
            copyInParam = {
                static_cast<uint16_t>(s1Extend),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(float)), 
                0
            };

            // copyOutOffset 指向 Vector 写回的低精度 workspace，slot 布局与 copyIn 保持一致。
            copyOutOffset = 
                (cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset) +
                (curSeqQIdx * s1VecSize * s2CubeExtend);
            copyOutParam = {
                static_cast<uint16_t>(s1Extend),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(ElementVecDtype)),
                0,
                static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(ElementVecDtype))
            };

            ///////////////////////////////////////////////////////////////
            // Vector 侧计算：A 子图重算 P，B 子图利用 P 和 sfmg 计算 dS。
            ///////////////////////////////////////////////////////////////
            event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            SubGrapA(i, blockInfo, mte2WaitMte3A);
            SubGrapB(i, blockInfo, mte2WaitMte3B);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

};
    
}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP
