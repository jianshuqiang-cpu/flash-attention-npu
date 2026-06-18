/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

/**
 * FAInferTilingData 结构体
 *
 * 作用：Flash Attention 推理时的核心 tiling 参数容器。
 * 该结构体保存了 Ascend NPU 上 Flash Attention kernel 所需的全部 tiling 策略参数，
 * 包括注意力维度、序列长度、Paged KV Cache 配置、多核任务调度、workspace 大小等。
 * 在 C++ 层（flash_api.cpp）计算 tiling 后，通过该结构体传递给底层 AscendC kernel。
 *
 * 字段分类：
 *   1. 注意力维度参数：numHeads, embeddingSize, embeddingSizeV, kvHeads
 *   2. 序列长度参数：maxQSeqlen, maxKvSeqlen
 *   3. Paged KV Cache 参数：numBlocks, blockSize, maxNumBlocksPerBatch
 *   4. Batch/任务调度参数：batch, firstBatchTaskNum, totalTaskNum
 *   5. Mask 参数：maskType
 *   6. Workspace 大小参数：mm1OutSize, smOnlineOutSize, mm2OutSize, UpdateSize, workSpaceSize
 *   7. 数值参数：scaleValue, softcapValue
 *   8. 对齐填充：padding1, padding2
 */
struct FAInferTilingData {
    // ==================== 注意力维度参数 ====================
    uint32_t numHeads;          // Q 的注意力头数
    uint32_t embeddingSize;     // Q/K 的 head_dim（每个头的维度）
    uint32_t embeddingSizeV;    // V 的 head_dim（允许与 Q/K 不同，用于支持 V 的独立维度）
    // ==================== Paged KV Cache 参数 ====================
    uint32_t numBlocks;         // 总块数（paged KV cache 的块总数）
    uint32_t blockSize;         // 每块的 token 数（通常为 128 或 256）
    // ==================== 序列长度参数 ====================
    uint32_t maxQSeqlen;        // Q 的最大序列长度
    uint32_t maxKvSeqlen;       // KV 的最大序列长度
    // ==================== KV 头数参数 ====================
    uint32_t kvHeads;           // KV 的头数（支持 MQA/GQA，kvHeads <= numHeads）
    // ==================== Batch/任务调度参数 ====================
    uint32_t batch;             // batch size
    uint32_t maxNumBlocksPerBatch;  // 每个 batch 最大块数（用于 paged KV cache 索引）
    uint32_t firstBatchTaskNum; // 首批任务数（FFTS 多核调度用，控制首批核的任务分配）
    uint32_t totalTaskNum;      // 总任务数（多核并行的任务总数，决定核间负载划分）
    // ==================== Mask 参数 ====================
    uint32_t maskType;          // 掩码类型（0=无掩码, 1=causal, 2=prefix 等，对应 MaskType 枚举）
    // ==================== Workspace 大小参数 ====================
    uint64_t mm1OutSize;        // 第一次矩阵乘（QK^T）输出的 workspace 大小（字节）
    uint64_t smOnlineOutSize;   // online softmax 中间结果的 workspace 大小（字节）
    uint64_t mm2OutSize;        // 第二次矩阵乘（PV）输出的 workspace 大小（字节）
    uint64_t UpdateSize;        // 状态更新缓冲大小（保存 logsumexp LSE 和最大值 m，字节）
    uint64_t workSpaceSize;     // 总 workspace 大小（字节，用于一次性分配）
    // ==================== 数值参数 ====================
    float scaleValue;           // softmax 缩放因子（通常为 1/sqrt(head_dim)）
    float softcapValue;         // softcap 值（>0 启用 softcapping attention，限制注意力分数范围）
    // ==================== 对齐填充 ====================
    uint64_t padding1;          // 对齐填充字段1（保证结构体内存对齐，预留扩展）
    uint64_t padding2;          // 对齐填充字段2（保证结构体内存对齐，预留扩展）

    // ==================== Getter 方法 ====================
    // 作用：提供对私有字段的只读访问，const 保证线程安全
    uint32_t get_numHeads() const { return numHeads; }              // 获取 Q 的注意力头数
    uint32_t get_embeddingSize() const { return embeddingSize; }    // 获取 Q/K 的 head_dim
    uint32_t get_embeddingSizeV() const { return embeddingSizeV; } // 获取 V 的 head_dim
    uint32_t get_numBlocks() const { return numBlocks; }            // 获取 paged KV cache 总块数
    uint32_t get_blockSize() const { return blockSize; }           // 获取每块的 token 数
    uint32_t get_maxQSeqlen() const { return maxQSeqlen; }         // 获取 Q 的最大序列长度
    uint32_t get_maxKvSeqlen() const { return maxKvSeqlen; }       // 获取 KV 的最大序列长度
    uint32_t get_kvHeads() const { return kvHeads; }               // 获取 KV 的头数
    uint32_t get_batch() const { return batch; }                   // 获取 batch size
    uint32_t get_maxNumBlocksPerBatch() const { return maxNumBlocksPerBatch; }  // 获取每 batch 最大块数
    uint32_t get_firstBatchTaskNum() const { return firstBatchTaskNum; }        // 获取首批任务数
    uint32_t get_totalTaskNum() const { return totalTaskNum; }     // 获取总任务数
    uint32_t get_maskType() const { return maskType; }             // 获取掩码类型
    uint64_t get_mm1OutSize() const { return mm1OutSize; }         // 获取 QK^T 输出大小
    uint64_t get_smOnlineOutSize() const { return smOnlineOutSize; }  // 获取 online softmax 中间结果大小
    uint64_t get_mm2OutSize() const { return mm2OutSize; }         // 获取 PV 输出大小
    uint64_t get_UpdateSize() const { return UpdateSize; }         // 获取状态更新缓冲大小
    uint64_t get_workSpaceSize() const { return workSpaceSize; }   // 获取总 workspace 大小
    float get_scaleValue() const { return scaleValue; }           // 获取 softmax 缩放因子
    float get_softcapValue() const { return softcapValue; }        // 获取 softcap 值
    uint64_t get_padding1() const { return padding1; }            // 获取对齐填充字段1
    uint64_t get_padding2() const { return padding2; }            // 获取对齐填充字段2

    // ==================== Setter 方法 ====================
    // 作用：提供对字段的写入接口，用于在 C++ 层计算 tiling 后设置参数
    void set_numHeads(uint32_t value) { numHeads = value; }                // 设置 Q 的注意力头数
    void set_embeddingSize(uint32_t value) { embeddingSize = value; }      // 设置 Q/K 的 head_dim
    void set_embeddingSizeV(uint32_t value) { embeddingSizeV = value; }    // 设置 V 的 head_dim
    void set_numBlocks(uint32_t value) { numBlocks = value; }              // 设置 paged KV cache 总块数
    void set_blockSize(uint32_t value) { blockSize = value; }              // 设置每块的 token 数
    void set_maxQSeqlen(uint32_t value) { maxQSeqlen = value; }             // 设置 Q 的最大序列长度
    void set_maxKvSeqlen(uint32_t value) { maxKvSeqlen = value; }           // 设置 KV 的最大序列长度
    void set_kvHeads(uint32_t value) { kvHeads = value; }                  // 设置 KV 的头数
    void set_batch(uint32_t value) { batch = value; }                     // 设置 batch size
    void set_maxNumBlocksPerBatch(uint32_t value) { maxNumBlocksPerBatch = value; }  // 设置每 batch 最大块数
    void set_firstBatchTaskNum(uint32_t value) { firstBatchTaskNum = value; }          // 设置首批任务数
    void set_totalTaskNum(uint32_t value) { totalTaskNum = value; }         // 设置总任务数
    void set_maskType(uint32_t value) { maskType = value; }                // 设置掩码类型
    void set_mm1OutSize(uint64_t value) { mm1OutSize = value; }            // 设置 QK^T 输出大小
    void set_smOnlineOutSize(uint64_t value) { smOnlineOutSize = value; }  // 设置 online softmax 中间结果大小
    void set_mm2OutSize(uint64_t value) { mm2OutSize = value; }             // 设置 PV 输出大小
    void set_UpdateSize(uint64_t value) { UpdateSize = value; }            // 设置状态更新缓冲大小
    void set_workSpaceSize(uint64_t value) { workSpaceSize = value; }      // 设置总 workspace 大小
    void set_scaleValue(float value) { scaleValue = value; }              // 设置 softmax 缩放因子
    void set_softcapValue(float value) { softcapValue = value; }          // 设置 softcap 值
    void set_padding1(uint64_t value) { padding1 = value; }               // 设置对齐填充字段1
    void set_padding2(uint64_t value) { padding2 = value; }               // 设置对齐填充字段2
};

#endif
