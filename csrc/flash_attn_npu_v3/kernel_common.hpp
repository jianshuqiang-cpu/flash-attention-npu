/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// =============================================================================
// kernel_common.hpp — FlashAttention v3 kernel 通用定义
// =============================================================================
// 本头文件定义了 AscendC kernel 端使用的通用常量、枚举、工具函数和参数结构体。
// 这些定义被 mha_fwd_kvcache.cpp（kernel 实现）和 flash_api.cpp（Host 入口）共同引用，
// 是 Host 端与 Device（AI Core）端之间共享的"契约"。
//
// 主要内容：
//   1. KernelCommon 命名空间 — 常量定义（分块大小、同步信号 ID、缓冲区大小等）
//   2. AlignUp / Max — 通用模板工具函数（对齐、取最大值）
//   3. FaiKenel 子命名空间 — 枚举定义（掩码类型、输入布局、流水线类型）
//   4. FAIKernelParams 结构体 — Device 端 kernel 的全局内存（GM）参数集合
//   5. GetQNBlockTile / GetQSBlockTile — 分块（Tiling）计算函数
// =============================================================================

#ifndef KERNEL_COMMON
#define KERNEL_COMMON


namespace KernelCommon {
    // =========================================================================
    // 一、硬件同步信号 ID（HardEvent 信号量）
    // =========================================================================
    // Ascend NPU 的 Cube 引擎（矩阵乘）和 Vector 引擎（softmax/rescale）通过
    // HardEvent 信号量实现双引擎流水线同步。以下 ID 标识三个关键同步点：
    //   QK_READY_ID    — Cube 完成 QK^T 矩阵乘后通知 Vector 可做 softmax
    //   SOFTMAX_READY_ID — Vector 完成 softmax 后通知 Cube 可做 PV 矩阵乘
    //   PV_READY_ID    — Cube 完成 PV 矩阵乘后通知 Vector 可做 rescale 输出
    // =========================================================================
    constexpr uint32_t QK_READY_ID = 1;        // QK^T 计算完成信号
    constexpr uint32_t SOFTMAX_READY_ID = 2;   // softmax 计算完成信号
    constexpr uint32_t PV_READY_ID = 3;        // PV 计算完成信号

    // =========================================================================
    // 二、分块与缓冲区常量
    // =========================================================================
    // PRE_LAUNCH        — 预取缓冲数量，用于 Cube/Vector 流水线预取（2 个缓冲交替使用）
    // N_SPLIT_HELPER    — N 维分块对齐因子，保证 N 维分块数是 2 的倍数（双引擎对齐）
    // MAX_KV_STACK_LEN  — 单次处理的 KV 序列最大长度（512 行），即一个 tile 最多遍历 512 行 KV
    // Q_TILE_CEIL       — Q 序列分块大小（128 行），即一个 tile 处理 128 行 query
    // WORKSPACE_BLOCK_SIZE_DB — workspace 单块大小 = 128 × 512 = 65536 元素
    // =========================================================================
    constexpr uint32_t PRE_LAUNCH = 2;
    constexpr uint32_t N_SPLIT_HELPER = 2;
    constexpr uint32_t MAX_KV_STACK_LEN = 512;
    constexpr uint32_t Q_TILE_CEIL = 128;
    constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;

    // =========================================================================
    // 三、L1 缓存与缓冲区限制
    // =========================================================================
    // L1_MAX_SIZE       — AI Core L1 缓存最大字节数（512KB），tiling 需保证数据不超限
    // L1_MAX_N_NUM      — L1 中 N 维最大元素数（128）
    // DOUBLE_BUFFER     — 双缓冲系数（2），用于 Cube/Vector 流水线双缓冲
    // COMP_TRIU_MASK_DIM_LEN — 因果掩码矩阵维度（2048×2048），上三角掩码
    // =========================================================================
    constexpr uint32_t L1_MAX_SIZE = 524288;
    constexpr uint32_t L1_MAX_N_NUM = 128;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr uint32_t COMP_TRIU_MASK_DIM_LEN = 2048;

    // =========================================================================
    // 四、通用数值常量
    // =========================================================================
    constexpr uint32_t NUM_32 = 32;
    constexpr uint32_t NUM_128 = 128;
    constexpr uint32_t NUM_256 = 256;

    // -------------------------------------------------------------------------
    // AlignUp — 向上对齐函数（模板）
    // -------------------------------------------------------------------------
    // 将值 a 向上对齐到 b 的整数倍。例如 AlignUp(10, 4) = 12。
    // 这是 AscendC kernel 中常用的地址/大小对齐工具，确保数据访问满足
    // 硬件对齐要求（如 32 字节对齐、16 元素对齐等）。
    //
    // 参数：
    //   a — 待对齐的值
    //   b — 对齐基准（对齐粒度）
    // 返回：不小于 a 的最小 b 的整数倍；若 b 为 0 则返回 0
    // -------------------------------------------------------------------------
    template <typename T>
    __aicore__ inline
    T AlignUp(T a, T b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b * b;
    }

    // -------------------------------------------------------------------------
    // Max — 取最大值函数（模板）
    // -------------------------------------------------------------------------
    // 返回两个值中的较大者。kernel 端不使用 std::max（可能不支持），
    // 因此提供此内联实现。用于 tiling 计算中保证分块大小不小于某下限。
    //
    // 参数：
    //   a, b — 待比较的两个值
    // 返回：a 和 b 中的较大者
    // -------------------------------------------------------------------------
    template <typename T>
    __aicore__ inline
    T Max(T a, T b)
    {
        return (a > b) ? a : b;
    }

    // =========================================================================
    // FaiKenel 子命名空间 — FlashAttention kernel 专用枚举与常量
    // =========================================================================
    namespace FaiKenel {
        // BLOCK_SIZE — 矩阵分块的基本元素数（16），对应 Ascend NPU Cube 引擎
        //              单次矩阵乘的最小处理粒度（16×16 块）
        constexpr uint32_t BLOCK_SIZE = 16;

        // ---------------------------------------------------------------------
        // cvPipeLineType — Cube/Vector 流水线类型枚举
        // ---------------------------------------------------------------------
        // 用于区分普通注意力与分块掩码注意力的流水线模式：
        //   FAI_COMMON_NORMAL     — 普通模式（标准前向，无分块掩码）
        //   FAI_COMMON_CHUNK_MASK — 分块掩码模式（如滑动窗口注意力 SWA）
        // ---------------------------------------------------------------------
        enum class cvPipeLineType : uint32_t {
            FAI_COMMON_NORMAL = 0,
            FAI_COMMON_CHUNK_MASK = 1,
        };

        // ---------------------------------------------------------------------
        // MaskType — 注意力掩码类型枚举
        // ---------------------------------------------------------------------
        // 决定 softmax 阶段如何对 attention score 施加掩码：
        //   NO_MASK      — 无掩码（全注意力，所有位置可见）
        //   MASK_CAUSAL  — 因果掩码（下三角，仅可见当前位置及之前的 KV）
        //   MASK_SPEC     — 推测解码掩码（用于 speculative decoding 场景）
        // ---------------------------------------------------------------------
        enum class MaskType : uint32_t {
            NO_MASK = 0,
            MASK_CAUSAL = 1,
            MASK_SPEC = 2
        };

        // ---------------------------------------------------------------------
        // inputLayout — 输入张量布局枚举
        // ---------------------------------------------------------------------
        // 区分两种内存布局，影响 kernel 端地址计算方式：
        //   BSND — 标准布局 (batch, seqlen, num_heads, head_dim)，定长序列
        //   TND  — 变长布局 (total_tokens, num_heads, head_dim)，用于 varlen 场景
        //          total_tokens 是所有 batch 序列长度之和，通过 cu_seqlens 索引
        // ---------------------------------------------------------------------
        enum class inputLayout : uint32_t {
            BSND = 0,
            TND = 1
        };
    };

    // =========================================================================
    // FAIKernelParams — Device 端 kernel 全局内存（GM）参数结构体
    // =========================================================================
    // 封装了 FlashAttention kernel 所需的所有 Device 端张量地址（GM_ADDR）。
    // Host 端（flash_api.cpp）获取各张量的 data_ptr 后构造此结构体，
    // 传递给 kernel 的 operator() 作为输入。
    //
    // 字段说明：
    //   q / k / v       — query / key / value 张量的 GM 地址
    //   mask            — 因果掩码矩阵的 GM 地址（无掩码时为 nullptr）
    //   blockTables     — 分页 KV-Cache 的页表地址（非分页模式时为 nullptr）
    //   actualQseqlen   — query 序列长度信息（变长模式为 cu_seqlens，否则为空）
    //   actualKvseqlen  — KV 序列长度信息（变长模式为 cu_seqlens，否则为 seqused_k）
    //   o               — 输出张量（attention output）的 GM 地址
    //   lse             — logsumexp 输出的 GM 地址（用于反向传播）
    //   workSpace       — workspace 显存的 GM 地址（中间计算缓冲）
    //   tiling          — FAInferTilingData 分块参数的 GM 地址
    // =========================================================================
    struct FAIKernelParams {
        GM_ADDR q;               // query 输入
        GM_ADDR k;               // key 输入
        GM_ADDR v;               // value 输入
        GM_ADDR mask;            // 因果掩码矩阵
        GM_ADDR blockTables;     // 分页 KV-Cache 页表
        GM_ADDR actualQseqlen;   // query 序列长度（变长模式用）
        GM_ADDR actualKvseqlen;  // KV 序列长度
        GM_ADDR o;               // attention 输出
        GM_ADDR lse;             // logsumexp 输出
        GM_ADDR workSpace;       // workspace 中间缓冲
        GM_ADDR tiling;          // 分块参数（FAInferTilingData）

        // 默认构造函数（空实现，kernel 端使用）
        __aicore__ inline FAIKernelParams() {}

        // 带参构造函数 — 由 Host 端调用，传入所有 GM 地址
        __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
                GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_, GM_ADDR tiling_)
            : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
                actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_) {}
    };

    // -------------------------------------------------------------------------
    // GetQNBlockTile — 计算每个 tile 在 N（head 头数）维度上的分块大小
    // -------------------------------------------------------------------------
    // 在 GQA/MQA 场景下，query 头数 num_heads 是 KV 头数 num_heads_k 的
    // groupSize 倍。本函数决定一个 tile 内同时处理多少个 query 头。
    //
    // 计算逻辑：
    //   1. 若 qSeqlen 不为 0，则 (Q_TILE_CEIL / qSeqlen) 得到一个 tile 能容纳的头数，
    //      再按 N_SPLIT_HELPER(2) 向下对齐，保证 N 维分块是 2 的倍数
    //      （与 Cube/Vector 双引擎流水线对齐）
    //   2. 结果不超过 groupSize（GQA 分组数）
    //   3. 结果不小于 1
    //
    // 注意：此函数为 kernel 端版本（__aicore__ inline），
    //       flash_api.cpp 中有对应的 Host 端版本（使用 std::min/std::max）。
    //
    // 参数：
    //   qSeqlen   — 当前 batch 的 query 序列长度
    //   groupSize — GQA 分组数 = num_heads / num_heads_k
    // 返回：N 维度每个 tile 处理的 query 头数
    // -------------------------------------------------------------------------
    __aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        // 计算一个 tile 能容纳多少个头，并按 N_SPLIT_HELPER 对齐
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        // 不超过 GQA 分组数 groupSize
        qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
        // 至少为 1
        qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
        return qNBlockTile;
    }

    // -------------------------------------------------------------------------
    // GetQSBlockTile — 计算每个 tile 在 S（query 序列）维度上的分块大小
    // -------------------------------------------------------------------------
    // 决定一个 tile 内处理多少行 query（S 维度的分块粒度）。
    // 当前实现固定返回 Q_TILE_CEIL(128)，即每个 tile 处理 128 行 query。
    // kvSeqlen 参数为预留接口，便于未来根据 KV 序列长度动态调整。
    //
    // 参数：
    //   kvSeqlen — 当前 batch 的 KV 序列长度（当前未使用）
    // 返回：S 维度每个 tile 处理的 query 行数（固定 128）
    // -------------------------------------------------------------------------
    __aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }
}
#endif