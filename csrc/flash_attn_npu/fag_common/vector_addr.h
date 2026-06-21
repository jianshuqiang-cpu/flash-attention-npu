/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef __VECTORAddr_H__
#define __VECTORAddr_H__

#include "common_header.h"

// VectorAddr 用于 FAG 反向计算中 Vector 侧任务的块信息生成。
// CubeAddr 生成的是较粗粒度的矩阵乘地址；VectorAddr 会进一步展开为最多 16 个 128x128 子块，
// 并为 EpilogueFAGOp 提供 batch/head/Q块/K块、块内真实长度和 workspace 偏移等信息。
template<MaskType maskType, InputLayout inputLayout>
class VectorAddr {

public:
    // 基础形状参数。
    int32_t batch;
    int32_t nheads;
    int32_t headdim;
    // GQA/MQA 分组大小，nheads = nheads_k * g。
    int32_t g;
    // 当前 Vector 侧逻辑 core 编号。调用方通常使用 GetBlockIdx() / 2，与 Cube 侧 core 对齐。
    int32_t coreId = 0;

    // coreSegmentBlockNum 是全局 segment 计数，用于按 coreNum 轮转分配任务。
    // SegmentBlockNum 是当前 segment 内的块计数，目前仅维护但未参与后续逻辑。
    // blockNum 是当前 VecAddrInfo 内 workspace 位置计数，最多容纳 16 个 128x128 子块。
    int32_t coreSegmentBlockNum = 0;
    int32_t SegmentBlockNum = 0;
    int32_t blockNum = 0;

    // 当前遍历位置：batch、Q head、Q 序列块、K 序列块。
    int32_t batchIdx;
    int32_t nheadsIdx;
    int32_t qSeqIdx;
    int32_t seqKIdx;
    // 当前 batch 的 Q/K 实际长度。
    int32_t qSeqlen;
    int32_t kSeqlen;
    // Q/K 序列按 128 token 分块后的块数和尾块真实长度。
    int32_t s1BlockNum;
    int32_t s1TailLength;
    int32_t s2BlockNum;
    int32_t s2TailLength;
    // 一次沿 Q 方向处理的块数。通常为 2，尾部或只有一个块时为 1。
    int32_t s1GuardInterval;

    // 单次地址表最多描述 16 个 128x128 子块。
    int32_t limit;
    // 参与 Vector 侧任务划分的逻辑 core 数。
    int32_t coreNum;
    // 当前 addr_mapping 调用轮次，用于让每个 core 每轮只取到属于自己的 segment。
    int32_t roundId;

    // overFlag 表示全局遍历是否尚未结束。
    int32_t overFlag = 1;

    // 输出地址表，addr_mapping 会把本轮属于当前 core 的 VecBlockInfo 写入这里。
    struct VecAddrInfo * globalVecAddr;

    // 变长序列前缀和地址。TND 使用 cu_seq_qlen_addr 计算每个 batch 的真实长度。
    __gm__ uint8_t *cu_seq_qlen_addr;
    __gm__ uint8_t *cu_seq_kvlen_addr;
    // 下面几个 GM 地址成员当前文件未直接使用，保留给地址生成类统一接口或后续扩展。
    __gm__ uint8_t *q_gm_addr;
    __gm__ uint8_t *k_gm_addr;
    __gm__ uint8_t *v_gm_addr;
    __gm__ uint8_t *dout_gm_addr;
    __gm__ uint8_t *user_gm_addr;

    // 将“块数”换算为真实 token 长度。
    // 非尾块长度为 len * 128；如果覆盖到尾块，则最后一个块只贡献 s_tail 个 token。
    __aicore__ uint64_t getSeqRealLength(int32_t sIdx, int32_t len, int32_t s_block_num, int32_t s_tail) {
        if (s_tail == 0) {
            return len * 128;
        } else {
            if (sIdx + len == s_block_num) {
                return (len - 1) * 128 + s_tail;
            } else {
                return len * 128;
            }
        }
    }
    
    // 返回第 i 个 batch 结束位置之前累计的 Q token 数。
    // TND 下直接读取 cu_seqlens；BSND 下每个 batch 定长，因此用 (i + 1) * qSeqlen。
    __aicore__ int64_t getTotalLen(int32_t i) {
        int64_t cuTotalSeqQlen = 0;
        if (inputLayout == InputLayout::TND) {
            cuTotalSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[i];
        } else {
            cuTotalSeqQlen = (i + 1) * qSeqlen;
        }
        return cuTotalSeqQlen;
    }

    // 计算左矩阵起始元素偏移。
    // 布局按 [total_q, nheads, headdim] 展平，qSeqIdx 每次跳过 128 个 token。
    // 与 CubeAddr 不同，这里每次通过 getTotalLen(batchIdx - 1) 动态获取当前 batch 前缀长度。
    __aicore__ uint64_t getLeftAddr(int32_t batchIdx, int32_t nheadsIdx, int32_t qSeqlen, int32_t qSeqIdx, int32_t headdim) {
        if (batchIdx == 0) {
            return (qSeqIdx * 128 * nheads + nheadsIdx) * headdim;
        } else {
            return getTotalLen(batchIdx - 1) * nheads * headdim + (qSeqIdx * 128 * nheads + nheadsIdx) * headdim;
        }
    }

    // 计算右矩阵起始元素偏移。
    // GQA/MQA 场景下多个 query head 共享同一个 key/value head，所以用 nheadsIdx / g 映射到 KV head。
    // 当前文件中该函数未被 addr_mapping 调用，保留给地址生成类统一接口或后续扩展。
    __aicore__ uint64_t getRightAddr(int32_t batchIdx, int32_t nheadsIdx, int32_t kSeqlen, int32_t seqKIdx, int32_t headdim) {
        if (batchIdx == 0) {
            return (seqKIdx * 128 * (nheads / g) + (nheadsIdx / g)) * headdim;
        } else {
            return getTotalLen(batchIdx - 1) * (nheads / g) * headdim + (seqKIdx * 128 * (nheads / g) + (nheadsIdx / g)) * headdim;
        }
    }

    // 计算当前块在临时 workspace 中的偏移。
    // 每个逻辑子块按 128 x 128 矩阵预留空间。
    __aicore__ uint64_t getOutAddr(int32_t workspacePos) {
        return workspacePos * 128 * 128;
    }

    // 把当前逻辑子块转换为 Vector 侧可直接消费的 VecBlockInfo。
    // blockId 对应 Cube 写入 workspace 的 128x128 子块序号；row/col 是当前 Q 组和 K 组内的局部块坐标。
    __aicore__ __inline__ void getOffset(VecBlockInfo &vecPhyAddr, int32_t blockId, int row, int col) 
    {
        vecPhyAddr.batchIdx = batchIdx;
        vecPhyAddr.nheadsIdx = nheadsIdx;
        vecPhyAddr.SeqQIdx = qSeqIdx + row;
        vecPhyAddr.SeqKIdx = seqKIdx + col;
        // 拆分 query head 到 KV head 与组内 head，用于 GQA/MQA。
        vecPhyAddr.nheadsKIdx = nheadsIdx / g;
        vecPhyAddr.gIdx = nheadsIdx % g;
        vecPhyAddr.offset = blockId * 128 * 128;
        
        // Q 方向真实长度。普通块是 128，尾块使用 s1TailLength。
        vecPhyAddr.lengthy = 128;
        if ((row + qSeqIdx == s1BlockNum - 1) && s1TailLength > 0) {
            vecPhyAddr.lengthy = s1TailLength;
        } 

        // K 方向真实长度。普通块是 128，尾块使用 s2TailLength。
        vecPhyAddr.lengthx = 128;
        if ((col + seqKIdx == s2BlockNum - 1) && s2TailLength > 0) {
            vecPhyAddr.lengthx = s2TailLength;
        }
    }

    // 生成一轮 Vector 侧块信息表。
    // 返回值为 overFlag：1 表示后续仍有任务，0 表示所有 batch/head/seq 块已经遍历完。
    __aicore__ int32_t addr_mapping(struct VecAddrInfo * vecAddrInfo) {
        globalVecAddr = vecAddrInfo;
        globalVecAddr->blockLength = 0;

        int32_t loopCnt = 0;
        while (overFlag) {
            // guardLen 表示在 causal 下，当前 Q 分组允许覆盖到的 K 块数。
            // 对于 qSeqIdx 开始、跨度 s1GuardInterval 的 Q 块，K 方向最多处理到对角线附近。
            int32_t guardLen = qSeqIdx + s1GuardInterval - seqKIdx;
            // 当前地址表还剩多少个 128x128 子块容量。
            int32_t reserve = limit - blockNum;
            
            // TODO sparsemode = 0
            // 计算剩余容量在当前 Q 跨度下，最多还能完整覆盖多少个 K 块。
            int32_t realLenAlign = (reserve + s1GuardInterval - 1) / s1GuardInterval;
            if (realLenAlign >= guardLen) {
                // 当前 segment 归属于本 core 时，才把块信息写入输出表；否则只推进全局遍历状态。
                if (coreSegmentBlockNum % coreNum == coreId) {
                    int32_t blockId = blockNum;
                    // x 是 K 方向局部块编号，y 是 Q 方向局部块编号。
                    for (int x = 0; x < guardLen; x++){
                        for (int y = 0; y < s1GuardInterval; y++){
                                // TODO sparsemode = 0
                                // causal 约束：只保留 Q 块编号 >= K 块编号的下三角区域。
                                if (qSeqIdx + y >= seqKIdx + x){
                                    getOffset(globalVecAddr->VecBlkInfo[blockId], blockId, y, x);
                                    blockId ++;
                                }
                        }
                    }
                    // TODO sparsemode = 0
                    // blockLength 使用 workspace slot 上界，与 blockId 递增数量保持同一套 slot 编号。
                    globalVecAddr->blockLength = blockNum + s1GuardInterval * guardLen - (s1GuardInterval + 1) % 2;
                }
                // 当前 Q 分组的 K 范围可以完整放入本地址表，推进到下一个 Q 分组。
                // TODO sparsemode = 0
                blockNum += s1GuardInterval * guardLen - (s1GuardInterval + 1) % 2;
                qSeqIdx += s1GuardInterval;
                seqKIdx = 0;
                // 如果下一个 Q 块已经是尾块，则改成一次只处理 1 个 Q 块，避免越界。
                if (qSeqIdx == s1BlockNum - 1) {
                    s1GuardInterval = 1;
                }
            } else {
                // 当前地址表容量不足以容纳完整 K 范围，只处理一部分 K 块，下一轮继续同一个 Q 分组。
                int32_t realLen = (reserve / s1GuardInterval);
                if (coreSegmentBlockNum % coreNum == coreId) {
                    int32_t blockId = blockNum;
                    for (int x = 0; x < realLen; x++){
                        for (int y = 0; y < s1GuardInterval; y++) {
                                // TODO sparsemode = 0
                                // causal 约束：只保留 Q 块编号 >= K 块编号的下三角区域。
                                if (qSeqIdx + y >= seqKIdx + x) {
                                    getOffset(globalVecAddr->VecBlkInfo[blockId], blockId, y, x);
                                    blockId ++;
                                }
                        }
                    }
                    globalVecAddr->blockLength = blockNum + s1GuardInterval * realLen;
                }
                blockNum += s1GuardInterval * realLen;
                seqKIdx += realLen;
            }
            SegmentBlockNum++;
            // 已经遍历到最后一个 batch 的最后一个 head，并且 Q 块全部完成，标记全局结束。
            if ((qSeqIdx == s1BlockNum) && (batchIdx == batch - 1) && (nheadsIdx == nheads - 1)) {
                overFlag = 0;
                break;
            }

            // 当前 head 的 Q 块遍历完成后，切换到下一个 head 或下一个 batch。
            if (qSeqIdx == s1BlockNum) {
                if (nheadsIdx == nheads - 1) {
                    batchIdx++;
                    nheadsIdx = 0;
                    if (inputLayout == InputLayout::TND) { 
                        qSeqlen = getSeqLen(batchIdx);
                        kSeqlen = getSeqLen(batchIdx);
                    }
                    s1BlockNum = (qSeqlen + 127) / 128;
                    s1TailLength = qSeqlen % 128;
                    s2BlockNum = (kSeqlen + 127) / 128;
                    s2TailLength = kSeqlen % 128;
                } else {
                    nheadsIdx++;
                }
                qSeqIdx = 0;
                seqKIdx = 0;
                s1GuardInterval = (s1BlockNum == 1) ? 1 : 2;
            }

            // 一个地址表最多支持 16 个 out slot。达到阈值后进入下一个 segment。
            // 这里使用 >= 15，是因为部分 causal/边界块会占用非整数的逻辑容量，提前切段可避免写满越界。
            if (blockNum >= 15) {
                coreSegmentBlockNum++;
                SegmentBlockNum = 0;
                blockNum = 0;
                // 每个 addr_mapping 调用只推进到当前 round 覆盖的 core 数量边界。
                // 这样 Cube 侧与 Vector 侧可以按 taskId 分轮流水执行。
                if (coreSegmentBlockNum == roundId * coreNum) {
                    break;
                }
            }
        }
        roundId++;
        return overFlag;
    }

    // TND 变长布局下，根据 cu_seqlens 计算第 i 个 batch 的真实序列长度。
    // 注意调用方在 TND 模式下会把 cu_seq_qlen_addr 传成原始 cu_seqlens + 1，
    // 因此这里 i == 0 时读取的是第一个样本长度。
    __aicore__ int64_t getSeqLen(int32_t i) {
        int64_t cuSeqQlen;
        if (i == 0) {
            cuSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[0];
        } else {
            cuSeqQlen = ((__gm__ int32_t *)cu_seq_qlen_addr)[i] - ((__gm__ int32_t *)cu_seq_qlen_addr)[i - 1];
        }
        return cuSeqQlen;
    }

    // 初始化地址生成器状态。每个 Vector 逻辑 core 持有一个 VectorAddr 实例，并从第一个 batch/head/Q/K 块开始遍历。
    __aicore__ void init(int32_t batchIn, int32_t nheadsIn, int32_t gIn, int32_t headDimIn, uint32_t coreIdx,
        uint32_t seq_q_len, uint32_t seq_k_len,
        __gm__ uint8_t *cu_seq_qlen, __gm__ uint8_t *cu_seq_kvlen, uint32_t totalCoreNum) {
        
        batch = batchIn;
        nheads = nheadsIn;
        g = gIn;
        headdim = headDimIn;

        cu_seq_qlen_addr = cu_seq_qlen;
        cu_seq_kvlen_addr = cu_seq_kvlen;

        coreSegmentBlockNum = 0;
        SegmentBlockNum = 0;
        blockNum = 0;

        batchIdx = 0;
        nheadsIdx = 0;
        qSeqIdx = 0;
        seqKIdx = 0;
        if (inputLayout == InputLayout::TND) {         
            qSeqlen = getSeqLen(batchIdx);
            kSeqlen = getSeqLen(batchIdx);
        } else {
            qSeqlen = seq_q_len;
            kSeqlen = seq_k_len;
        }
        s1BlockNum = (qSeqlen + 127) / 128;
        s1TailLength = qSeqlen % 128;
        s2BlockNum = (kSeqlen + 127) / 128;
        s2TailLength = kSeqlen % 128;
        s1GuardInterval = (s1BlockNum == 1) ? 1 : 2;

        limit = 16;
        coreNum = totalCoreNum;
        coreId = coreIdx;

        roundId = 1;
        overFlag = 1;
    }
};
#endif // __VECTORAddr_H__
