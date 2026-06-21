/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// init_outputs.hpp 实现了 FlashAttention 前向推理中"输出初始化"阶段的 Epilogue。
// 它是 CATLASS 框架 BlockEpilogue 的一个模板特化，策略标签为 EpilogueAtlasA2InitOutWhenZero。
//
// 在 FlashAttention 的在线 softmax 流水线中，O（注意力输出）需要在所有 KV 分块上累加更新：
//   首个分块: O = P × V
//   后续分块: O = O * exp(dm) + P × V（重缩放 + 累加）
//   最后分块: O = O / rowsum（归一化）
// 因此在进入 KV 循环之前，必须将全局内存中的 O 初始化为 0；
// 同时 LSE（log-sum-exp）初始化为接近 +inf 的大值，标记"尚未处理任何 KV 分块"，
// 使得第一个 KV 分块的 online softmax 能正确计算 rowmax/rowsum。
//
// 该 Epilogue 仅在 Vector 核（__DAV_C220_VEC__）上执行，
// 主要用于 kvSLoopNumTotal <= 0（无有效 KV 序列）的边界情况，直接将输出清零/置 inf。
// 对于正常有 KV 数据的路径，首个分块的初始化由 rescale_o.hpp 中的"首块直接赋值"逻辑隐式完成。

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_INIT_OUTPUTS_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_INIT_OUTPUTS_HPP_T

#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fa_block.h"

namespace Catlass::Epilogue::Block {

// BlockEpilogue 模板特化：策略 = EpilogueAtlasA2InitOutWhenZero
// AttnOutType_ : 注意力输出 O 的类型（元素类型 + 布局），如 half/bf16，BSND/TND
// LseOutType_  : LSE 输出的类型（元素类型 + 布局），LSE 通常为 float
// LSE_MODE_    : LSE 输出模式，OUT_ONLY 时需要同时初始化 LSE，NONE 时只初始化 O
template <
    class AttnOutType_,
    class LseOutType_,
    LseModeT LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2InitOutWhenZero<LSE_MODE_>,
    AttnOutType_,
    LseOutType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2InitOutWhenZero<LSE_MODE_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementAttnOut = typename AttnOutType_::Element;  // O 的元素类型（half / bfloat16_t）
    using ElementLseOut = typename LseOutType_::Element;    // LSE 的元素类型（float）

    using LayoutAttnOut = typename AttnOutType_::Layout;    // O 的布局描述
    using LayoutLseOut = typename LseOutType_::Layout;      // LSE 的布局描述

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;
    static constexpr float ATTN_OUT_INI = 0;        // 注意力输出 O 初始值：0
    static constexpr float LSE_OUT_INI = 3e+99;     // LSE 初始值：接近 +inf，表示"尚未处理任何分块"
    static constexpr uint32_t HALF_ELEM_NUM_PER_BLK = 16;   // 每个 block 中 half 元素个数
    static constexpr uint32_t FLOAT_ELEM_NUM_PER_BLK = 8;   // 每个 block 中 float 元素个数
    static constexpr uint32_t HALF_ELEM_NUM_PER_RPT = 128;  // 每次 repeat 处理的 half 元素个数
    static constexpr uint32_t FLOAT_ELEM_NUM_PER_RPT = 64;  // 每次 repeat 处理的 float 元素个数
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;  // UB 中每个逻辑块的字节大小（16KB）

    // 构造函数：从共享 resource 中分配 UB 上的临时张量
    // attnOutUbTensor 从 UB 偏移 0 开始，lseOutUbTensor 从偏移 6*16KB = 98304 字节开始
    // 这些偏移与 online_softmax.hpp / rescale_o.hpp 中的 UB 布局协调，避免互相覆盖
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t ATTN_OUT_INIT_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LSE_OUT_INIT_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;

        attnOutUbTensor = resource.ubBuf.template GetBufferByByte<ElementAttnOut>(ATTN_OUT_INIT_UB_TENSOR_OFFSET);
        lseOutUbTensor = resource.ubBuf.template GetBufferByByte<ElementLseOut>(LSE_OUT_INIT_UB_TENSOR_OFFSET);
    }

    // SubCoreCompute：单个 sub-core 实际执行初始化的核心函数
    // 流程：
    //   1. 在 UB 上用 Duplicate 填充初始值（O→0, LSE→+inf）
    //   2. 通过 DataCopyPad 将 UB 中的初始值写回全局内存（GM）
    //   3. O 和 LSE 使用不同的事件 ID（EVENT_ID6 / EVENT_ID7）做 V→MTE3 同步
    //
    // 参数说明：
    //   gOutput        : GM 上的注意力输出张量（O）
    //   gLse           : GM 上的 LSE 张量
    //   layoutOutput   : O 的布局，shape(1) 为 oHiddenSize = qHeads * embedV
    //   layoutLse      : LSE 的布局，shape(1) 为 qHeads
    //   qSThisSubBlock : 当前 sub-core 负责的 query 序列行数（S 维）
    //   qNThisSubBlock : 当前 sub-core 负责的 query head 数（N 维）
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementAttnOut> gOutput,
        AscendC::GlobalTensor<ElementLseOut> gLse,
        const LayoutAttnOut &layoutOutput,
        const LayoutLseOut &layoutLse,
        uint32_t qSThisSubBlock, uint32_t qNThisSubBlock)
    {
        uint32_t oHiddenSize = layoutOutput.shape(1);  // O 的总隐藏维度 = num_heads * head_dim
        uint32_t qHeads = layoutLse.shape(1);          // query head 数量
        uint32_t embedV = oHiddenSize / qHeads;        // 每个 head 的维度（head_dim）
        uint32_t embedRoundV = RoundUp(embedV, HALF_ELEM_NUM_PER_BLK);  // 按 16 元素对齐后的 head_dim
        AscendC::PipeBarrier<PIPE_ALL>();

        // ---- 第一部分：初始化注意力输出 O 为 0 ----
        // 等待 MTE3（DMA 写）→V（Vector）通道就绪，确保之前的 GM 写操作完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        // 在 UB 上将 attnOutUbTensor 的前 embedRoundV*qSThisSubBlock 个元素填充为 0
        AscendC::Duplicate(attnOutUbTensor, static_cast<ElementAttnOut>(ATTN_OUT_INI), embedRoundV * qSThisSubBlock);
        // 通知 V→MTE3：UB 数据准备好，可以开始 DMA 写回
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        // 对每个负责的 head，将 UB 中的 0 值通过 DataCopyPad 写到 GM
        // DataCopyExtParams 参数：
        //   qSThisSubBlock            : 拷贝行数（query 序列维）
        //   embedV * sizeof(Element)  : 每行连续字节数（一个 head 的 head_dim）
        //   0                         : 源行间隔（UB 中连续）
        //   (oHiddenSize - embedV) * sizeof(Element) : 目的行间隔（跨 head 的步长）
        //   0                         : 目的起始偏移
        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
            AscendC::DataCopyPad(
                gOutput[qNIdx * embedV],
                attnOutUbTensor,
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embedV * sizeof(ElementAttnOut),
                    0, (oHiddenSize - embedV) * sizeof(ElementAttnOut), 0));
        }
        // 通知 MTE3→V：GM 写完成，释放 EVENT_ID6
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);

        // ---- 第二部分：初始化 LSE 为 +inf（仅在 OUT_ONLY 模式下）----
        // LSE 的形状是 [total_q_tokens, qHeads]，每个元素是 float
        // 初始化为 3e+99（接近 float 最大值），使得 online softmax 第一个分块的 rowmax 能正确更新
        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            // 在 UB 上填充 LSE 初始值，需要 qSThisSubBlock * 8 个 float（每 block 8 个 float，向上对齐）
            AscendC::Duplicate(lseOutUbTensor, LSE_OUT_INI, qSThisSubBlock * FLOAT_ELEM_NUM_PER_BLK);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            // 对每个 head，写 qSThisSubBlock 行的 LSE 值
            // 每行 LSE 只有 1 个 float 元素，行间隔为 (qHeads - 1) * sizeof(float)
            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                AscendC::DataCopyPad(
                    gLse[qNIdx],
                    lseOutUbTensor,
                    AscendC::DataCopyExtParams(
                        qSThisSubBlock, sizeof(ElementLseOut),
                        0, (qHeads - 1) * sizeof(ElementLseOut), 0));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // operator()：Epilogue 的入口函数，负责将一个 query block 拆分给多个 sub-core 并行执行
    //
    // 一个 query block 的维度是 qSBlockSize（序列维 tile，如 128）× qNBlockSize（head 维 tile）。
    // Ascend NPU 的 Vector 核支持 sub-core 并行，该函数根据 subBlockIdx 和 subBlockNum 计算
    // 当前 sub-core 应负责的行/列范围和 GM 偏移，然后调用 SubCoreCompute 执行初始化。
    //
    // 拆分策略：
    //   - 当 qNBlockSize == 1（head 维只有 1 个 tile）：按 S 维（序列行）拆分
    //   - 当 qNBlockSize > 1（head 维有多个 tile）：按 N 维（head）拆分
    //
    // 参数说明：
    //   gOutput      : GM 上的输出 O 张量起始地址
    //   gLse         : GM 上的 LSE 张量起始地址
    //   layoutOutput : O 的布局描述
    //   layoutLse    : LSE 的布局描述
    //   qSBlockSize  : query 序列维 tile 大小
    //   qNBlockSize  : query head 维 tile 大小
    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementAttnOut> gOutput,
        AscendC::GlobalTensor<ElementLseOut> gLse,
        const LayoutAttnOut &layoutOutput,
        const LayoutLseOut &layoutLse,
        uint32_t qSBlockSize, uint32_t qNBlockSize)
    {
        uint32_t rowNum = qSBlockSize * qNBlockSize;  // 当前 block 总行数
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t embedV = oHiddenSize / qHeads;

        // 获取当前 sub-core 的编号和 sub-core 总数
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        // 计算每个 sub-core 负责的 head 数（N 维拆分）
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        // 当前 sub-core 实际负责的 head 数
        // qNBlockSize==1 时固定为 1；否则 subBlockIdx==1（最后一个）负责余数，其余均分
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 1
            : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;

        // 计算每个 sub-core 负责的行数
        // qNBlockSize==1 时按 S 维拆分；否则按 N 维拆分，每个 sub-core 负责完整 qSBlockSize 行
        uint32_t rowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetSubBlock = subBlockIdx * rowSplitSubBlock;

        // 计算 O 张量的 GM 偏移：行偏移和列偏移
        // 按 S 维拆分时：行偏移 = rowOffsetSubBlock，列偏移 = 0
        // 按 N 维拆分时：行偏移 = 0，列偏移 = subBlockIdx * qNSplitSubBlock * embedV（跨 head 的字节偏移）
        uint32_t outRowOffsetSubBlock = (qNBlockSize == 1U) ? rowOffsetSubBlock : 0;
        uint32_t outColOffsetSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embedV;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? rowActualSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetSubBlock, outColOffsetSubBlock));
        auto gOutputSubBlock = gOutput[outOffsetSubBlock];
        auto layoutOutputSubBlock = layoutOutput;

        // 计算 LSE 张量的 GM 偏移
        // LSE 形状为 [total_q_tokens, qHeads]，每个元素是 float
        // 按 S 维拆分时：行偏移 = rowOffsetSubBlock，列偏移 = 0
        // 按 N 维拆分时：行偏移 = 0，列偏移 = subBlockIdx * qNSplitSubBlock（head 偏移）
        uint32_t outLseRowOffsetSubBlock = (qNBlockSize == 1U) ?
            rowOffsetSubBlock : 0;
        uint32_t outLseColOffsetSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;
        int64_t lseOffsetSubBlock =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetSubBlock, outLseColOffsetSubBlock));
        auto gLseThisSubBlock = gLse[lseOffsetSubBlock];
        auto layoutLseThisSubBlock = layoutLse;

        // 仅当当前 sub-core 有实际工作（行数 > 0）时才执行初始化
        if (rowActualSubBlock > 0U) {
            SubCoreCompute(
                gOutputSubBlock, gLseThisSubBlock,
                layoutOutputSubBlock, layoutLseThisSubBlock,
                qSThisSubBlock, qNThisSubBlock);
        }
    }
private:
    AscendC::LocalTensor<ElementAttnOut> attnOutUbTensor;  // UB 上用于暂存 O 初始值的局部张量
    AscendC::LocalTensor<ElementLseOut> lseOutUbTensor;    // UB 上用于暂存 LSE 初始值的局部张量
};
}
#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_INIT_OUTPUTS_HPP_T
