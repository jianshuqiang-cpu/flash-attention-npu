/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// kernel_common.hpp 是 FlashAttention NPU 前向推理 kernel 的"公共定义"头文件，
// 被 mha_fwd_kvcache.cpp、flash_api.cpp 等前向核心文件 include。
//
// 本文件集中提供四类基础构件：
//   1) Cube/Vector 跨核流水线同步事件 ID 与深度流水参数
//      （QK_READY_ID / SOFTMAX_READY_ID / PV_READY_ID / PRE_LAUNCH）；
//   2) Tile / L1 / Workspace / Mask 等硬件与分块相关常量
//      （Q_TILE_CEIL、MAX_KV_STACK_LEN、L1_MAX_SIZE、COMP_TRIU_MASK_DIM_LEN 等）；
//   3) 枚举类型：流水线类型 cvPipeLineType、注意力 Mask 类型 MaskType、输入布局 inputLayout；
//   4) Host/Device 通用的参数结构体 FAIKernelParams 与 tile 计算函数
//      GetQNBlockTile / GetQSBlockTile，以及通用工具模板 AlignUp / Max。
//
// 其中 Q_TILE_CEIL=128 是前向 kernel 的核心 tile 参数：
//   - Query 序列维度 (M 维) 固定按 128 行分块（GetQSBlockTile 返回 128）；
//   - Query head 维度 (N 维) 由 GetQNBlockTile 根据 qSeqlen 与 GQA groupSize 动态计算；
//   - Cube GEMM 的 L1 tile shape 固定为 GemmShape<128, 128, 128>。
//
// 跨核同步链（CrossCoreFlag）：
//   Cube 核: Q*K^T ─SetFlag(QK_READY_ID=1)─> Vector 核: softmax/mask
//   Vector 核: softmax 完成 ─SetFlag(SOFTMAX_READY_ID=2)─> Cube 核: P*V
//   Cube 核: P*V 完成 ─SetFlag(PV_READY_ID=3)─> Vector 核: online softmax rescale
//
// 注意：本文件中的枚举 MaskType/inputLayout 仅用于前向路径；
// 反向路径（FAG，fag_common/common_header.h）另有同名独立定义。

#ifndef KERNEL_COMMON
#define KERNEL_COMMON


namespace KernelCommon {
    // ---- Cube/Vector 跨核事件 ID，用于 CrossCoreFlag 三段式流水同步 ----
    constexpr uint32_t QK_READY_ID = 1;          // Q*K^T 矩阵乘完成事件，Vector 核据此开始 softmax
    constexpr uint32_t SOFTMAX_READY_ID = 2;     // softmax（含 mask）完成事件，Cube 核据此开始 P*V
    constexpr uint32_t PV_READY_ID = 3;          // P*V 矩阵乘完成事件，Vector 核据此做 O 重缩放/累加

    // ---- 深度流水参数 ----
    constexpr uint32_t PRE_LAUNCH = 2;           // KV 块预发射深度：流水线上提前 2 个 KV tile，总缓冲槽 = PRE_LAUNCH+1 = 3
    constexpr uint32_t N_SPLIT_HELPER = 2;       // head 维分块对齐辅助值，保证 qNBlockTile 为 2 的倍数，便于 sub-core 对半拆分
    constexpr uint32_t MAX_KV_STACK_LEN = 512;   // 每次外层 KV 循环处理的最大 K/V token 数（stack tile 长度上限）

    // ---- Query tile 与 Workspace ----
    constexpr uint32_t Q_TILE_CEIL = 128;        // Query 序列维度 (M 维) 的基础 tile 大小：128 行
                                                 // 它是 L1 GEMM tile M 维大小、QSBlockTile、workspace 尺寸的基础
    constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
                                                 // 单核单缓冲 workspace 元素数：128*512 = 65536
                                                 // 实际 workspace = blockDim * 此值 * (PRE_LAUNCH+1) * 4（S/P/OTmp/Update 四区域）

    // ---- L1 缓存相关 ----
    constexpr uint32_t L1_MAX_SIZE = 524288;     // Cube 核 L1 总大小：512KB
    constexpr uint32_t L1_MAX_N_NUM = 128;       // K 列方向 (N 维) L1 tile 最大大小：128，受硬件约束
    constexpr uint32_t DOUBLE_BUFFER = 2;        // V 矩阵在 L1 中使用双缓冲（ping-pong）预留 2 份空间

    // ---- Causal Mask 相关 ----
    constexpr uint32_t COMP_TRIU_MASK_DIM_LEN = 2048;  // 预生成 causal mask 矩阵尺寸：2048×2048（严格上三角为1）
                                                       // host 侧 at::triu(ones, 1) 生成；支持最长 2048 token 的因果注意力

    // ---- 硬件对齐常量 ----
    constexpr uint32_t NUM_32 = 32;              // 32 元素对齐单位（适配 Ascend C220 32 元素处理粒度）
    constexpr uint32_t NUM_128 = 128;            // 128 元素对齐单位（embed 维向上对齐到 128）
    constexpr uint32_t NUM_256 = 256;            // 256 元素对齐单位（K 维 L1 tile 大小下限）

    // AlignUp：将 a 向上对齐到 b 的整数倍；b=0 时返回 0 以避免除零。
    // 通用工具模板，device 侧代码多数场景直接使用 AscendC 内置的 RoundUp/CeilDiv。
    template <typename T>
    __aicore__ inline
    T AlignUp(T a, T b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b * b;
    }

    // Max：返回两个值中的较大者。通用工具模板，device 侧 softmax 行最大值
    // 计算使用的是 AscendC::Max<half/float, false>，而非此模板。
    template <typename T>
    __aicore__ inline
    T Max(T a, T b)
    {
        return (a > b) ? a : b;
    }

    namespace FaiKenel {
        constexpr uint32_t BLOCK_SIZE = 16;      // 数据搬运/矩阵运算基础对齐单元：16 个元素
                                                 // 用于 embed 维度与 row 总数向上 RoundUp（half 下 16 元素 = 32 字节）

        // Cube-Vector 流水线类型枚举（预留扩展，当前未实际使用）
        enum class cvPipeLineType : uint32_t {
            FAI_COMMON_NORMAL = 0,       // 普通流水线
            FAI_COMMON_CHUNK_MASK = 1,   // chunk mask 流水线（未来扩展）
        };

        // 注意力 Mask 类型枚举：作为 FAInferKernel 的模板参数 MASK_TYPE，
        // 通过 if constexpr 在编译期裁剪 mask 相关代码分支
        enum class MaskType : uint32_t {
            NO_MASK = 0,        // 无 mask（非因果，如双向 encoder、MLA 的部分场景）
            MASK_CAUSAL = 1,    // 因果（causal / autoregressive）mask：上三角置 -inf
            MASK_SPEC = 2       // 特殊 mask（预留）
        };

        // 输入数据布局枚举：作为 FAInferKernel 的模板参数 INPUT_LAYOUT
        enum class inputLayout : uint32_t {
            BSND = 0,   // Batch-Seqlen-Head-Dim：训练/带 padding 的标准 4D 布局
            TND = 1     // Token-Head-Dim：variable-length / packed 3D 布局，
                        // 通过 cumulative seqlen 数组前缀差分还原每个 batch 的真实长度，避免 padding 浪费
        };
    };

    // FAIKernelParams：Device 侧 kernel 参数打包结构体。
    // 全局 kernel 函数 FAInfer 的参数是 12 个独立 GM_ADDR（uint64_t），
    // 在 device 入口处将其中 11 个张量地址（除 fftsAddr 外）打包成本结构体，
    // 再传入 FAInferKernel::operator() 使用。
    //
    // 字段说明：
    //   q/k/v          : Q/K/V 全局内存指针（K、V 共用 ElementK 类型）
    //   mask           : 预生成的 causal mask 指针（2048×2048 uint8，仅 causal 路径使用）
    //   blockTables    : Paged KV cache 页表（int32，每个 batch 一个 page 索引序列）
    //   actualQseqlen  : 实际 Q 序列长度数组（int32，cumulative seqlen，TND 布局用于前缀差分）
    //   actualKvseqlen : 实际 KV 序列长度数组（int32）
    //   o              : 注意力输出 O 指针
    //   lse            : log-sum-exp 输出指针（float，用于反向/重计算/前缀缓存）
    //   workSpace      : 临时 workspace 指针，内部再划分为 S/P/OTmp/Update 四个区域
    //   tiling         : 宿主机打包好的 FAInferTilingData 标量参数指针
    struct FAIKernelParams {
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR mask;
        GM_ADDR blockTables;
        GM_ADDR actualQseqlen;
        GM_ADDR actualKvseqlen;
        GM_ADDR o;
        GM_ADDR lse;
        GM_ADDR workSpace;
        GM_ADDR tiling;

        __aicore__ inline FAIKernelParams() {}

        __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
                GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_, GM_ADDR tiling_)
            : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
                actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_) {}
    };

    // GetQNBlockTile：计算 Query head 维度（N 维）的分块大小。
    //
    // 参数：
    //   qSeqlen   : 当前 batch 实际 Q 序列长度
    //   groupSize : GQA group size = qHeads / kvHeads，即每个 KV head 对应的 Q head 数
    //               - MHA: groupSize = 1；MQA: groupSize = qHeads；GQA: 1 < groupSize < qHeads
    //
    // 逻辑：
    //   1) 若 qSeqlen > 0，先算 (128/qSeqlen) 并向下对齐到 N_SPLIT_HELPER(=2) 的倍数；
    //      qSeqlen 越短（如 decode qSeqlen=1），返回值越大（128），一次处理更多 head；
    //      qSeqlen 越长（如 prefill qSeqlen≈128），返回值越小。
    //   2) 再 clip 到 [1, groupSize]：不超过 groupSize，保证一个 block 内的 Q heads 属于同一 KV head；
    //      不小于 1，保证至少处理 1 个 head。
    //
    // 返回值 qNBlockTile 决定了每个 task 在 head 维处理多少个 Q head；
    // 总 head 块数 = CeilDiv(groupSize, qNBlockTile) * kvHeads。
    __aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
        qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
        return qNBlockTile;
    }

    // GetQSBlockTile：返回 Query 序列维度（S 维，即 token 行方向）的 tile 大小。
    // 当前实现固定返回 Q_TILE_CEIL=128，即每个 Q 序列块最多处理 128 行 token；
    // 参数 kvSeqlen 暂未参与动态调节（为未来按 kv 序列长度自适应 tile 预留接口）。
    // Host 侧 flash_api.cpp 有等价副本，两处需保持一致。
    __aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }
}
#endif
