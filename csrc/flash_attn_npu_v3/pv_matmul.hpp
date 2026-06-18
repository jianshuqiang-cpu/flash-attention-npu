/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_PV_HPP_T
#define CATLASS_GEMM_BLOCK_MMAD_PV_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "fa_block.h"

/**
 * @file pv_matmul.hpp
 * @brief FlashAttention v3 PV (P×V) 矩阵乘法 BlockMmad 特化 —— Cube 引擎专用
 *
 * 本文件实现了 FlashAttention v3 中第二个矩阵乘法 (P × V) 的 Cube 引擎计算块。
 * 在 FlashAttention 的两阶段 GEMM 中：
 *   1. QK 阶段: S = Q × K^T          (由 qk_matmul.hpp 负责)
 *   2. PV 阶段: O = P × V            (由本文件负责)
 *
 * 其中 P 是经过 Online Softmax 处理后的注意力权重矩阵 (已归一化)，
 * V 是 Value 矩阵，O 是最终输出。
 *
 * 核心特性：
 * - 支持 Paged KV Cache: 通过 block_table 间接寻址访问 V 矩阵
 * - L1/L0 多级 Ping-Pong 缓冲: 实现 Cube 引擎内部软件流水线
 * - 跨核同步: 等待 Vector 引擎完成 Online Softmax (softmaxFlag 信号量)
 * - 三重循环结构: nL1 (N维) × mL1 (M维) × kL1 (K维) + kL0 (L0分块)
 *
 * 内存层级 (Ascend NPU Cube 引擎):
 *   GM (Global Memory) → L1 (L1 Buffer) → L0A/L0B (L0 Buffer) → L0C (累加器)
 *   其中 L0A/L0B/L0C 使用 Ping-Pong 双缓冲以重叠数据搬运与计算
 *
 * 与 qk_matmul.hpp 的区别：
 * - QK 中 A=Q (预加载), B=K (KV循环); PV 中 A=P (来自Softmax), B=V (预加载)
 * - PV 需要等待 softmaxFlag (Vector→Cube 同步), QK 需要通知 qkReady (Cube→Vector)
 * - PV 的 B (V) 在循环外预加载到 L1, QK 的 B (K) 在循环内加载
 */

namespace Catlass::Gemm::Block {

/**
 * @brief PV 矩阵乘法 BlockMmad 特化类模板
 *
 * @tparam PAGED_CACHE_FLAG_ 是否启用 Paged KV Cache (分页缓存)
 * @tparam ENABLE_UNIT_FLAG_ 是否启用单元优化标志
 * @tparam L1TileShape_     L1 层 Tile 形状 (M, K 维度)
 * @tparam L0TileShape_     L0 层 Tile 形状 (N, K 维度)
 * @tparam AType_          左矩阵类型 (P 矩阵, 来自 Softmax 输出)
 * @tparam BType_          右矩阵类型 (V 矩阵)
 * @tparam CType_          输出矩阵类型 (O 矩阵)
 * @tparam BiasType_       偏置类型 (FlashAttention 中通常不使用)
 * @tparam TileCopy_       Tile 搬运策略 (GM→L1, L1→L0, L0C→GM)
 * @tparam TileMmad_       L0 层矩阵乘法指令封装
 *
 * 该特化针对 MmadAtlasA2FAIPVT 调度策略, 专为 FlashAttention 的 PV 阶段设计。
 * A2 代表 Atlas A2 架构, FAI 代表 FlashAttention Inference, PVT 代表 P×V 计算。
 */
template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmad<
    MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_> {
public:
    // ======================== 类型别名 (Type Aliases) ========================
    using DispatchPolicy = MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;  // 调度策略
    using ArchTag = typename DispatchPolicy::ArchTag;                                // 架构标签 (Atlas A2)
    using L1TileShape = L1TileShape_;                                                // L1 Tile 形状
    using L0TileShape = L0TileShape_;                                                // L0 Tile 形状
    using ElementA = typename AType_::Element;                                        // 左矩阵元素类型 (P 的元素类型, 通常 half/bfloat16)
    using LayoutA = typename AType_::Layout;                                          // 左矩阵布局 (P 的布局)
    using ElementB = typename BType_::Element;                                       // 右矩阵元素类型 (V 的元素类型)
    using LayoutB = typename BType_::Layout;                                          // 右矩阵布局 (V 的布局)
    using ElementC = typename CType_::Element;                                        // 输出矩阵元素类型 (O 的元素类型)
    using LayoutC = typename CType_::Layout;                                          // 输出矩阵布局 (O 的布局, 必须为 RowMajor)
    using TileMmad = TileMmad_;                                                        // L0 层 Mmad 指令封装
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;                              // GM→L1 A 矩阵搬运 (P: GM→L1)
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;                              // GM→L1 B 矩阵搬运 (V: GM→L1)
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;                              // L1→L0A 搬运 (P: L1→L0A)
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;                              // L1→L0B 搬运 (V: L1→L0B)
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;                              // L0C→GM 搬运 (O: L0C→GM)
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;  // 累加器类型 (通常 float)
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;                              // A 在 L1 中的布局
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;                              // B 在 L1 中的布局
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;                              // A 在 L0 中的布局
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;                              // B 在 L0 中的布局
    using LayoutCInL0 = layout::zN;                                                    // C 在 L0 中的布局 (zN: Z型N主序, Cube 引擎输出格式)

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;           // L1 A 对齐辅助器
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;            // L1 B 对齐辅助器

    // ======================== 硬件常量 (Constants) ========================
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;                        // Ping-Pong 阶数 (通常为 2, 双缓冲)
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);  // L1 A 单缓冲大小 (字节)
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);  // L1 B 单缓冲大小 (字节)
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;                           // L0A 硬件总大小
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;                           // L0B 硬件总大小
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;                           // L0C 硬件总大小 (累加器)
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;              // L0A 单个 Ping-Pong 缓冲大小
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;              // L0B 单个 Ping-Pong 缓冲大小
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;              // L0C 单个 Ping-Pong 缓冲大小
    static constexpr uint32_t BLOCK_SIZE = 16;                                        // 矩阵分块基本单位 (16×16, Cube 引擎最小计算单元)
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;                                 // Embedding 维度分块大小 (D=128 per head)
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;                               // 单元块堆叠数
    static constexpr uint32_t KV_BASE_BLOCK = 512;                                    // KV Cache 基础块大小
    static constexpr uint32_t KV_SPLIT_SIZE = 128;                                   // KV 分块大小
    static constexpr uint32_t LOAB_BLOCK = 1;                                         // L0A/L0B 块数
    static constexpr uint32_t COORD_DIM0 = 0;                                         // 坐标维度 0 (M/行)
    static constexpr uint32_t COORD_DIM1 = 1;                                         // 坐标维度 1 (N/列)
    static constexpr uint32_t COORD_DIM2 = 2;                                         // 坐标维度 2 (K/归约维)

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /**
     * @brief 构造函数 —— 分配 L1/L0 内存空间并初始化 Ping-Pong 缓冲
     *
     * 内存布局 (L1 Buffer):
     *   [l1ATensor[0] | l1ATensor[1] | ... | l1ATensor[STAGES-1] | l1BTensor]
     *    ↑ P 矩阵 Ping-Pong 缓冲 (STAGES 个)              ↑ V 矩阵缓冲 (单个, 循环外预加载)
     *
     * 内存布局 (L0 Buffer):
     *   L0A: [l0ATensor[0] | l0ATensor[1]]  Ping-Pong 双缓冲
     *   L0B: [l0BTensor[0] | l0BTensor[1]]  Ping-Pong 双缓冲
     *   L0C: [l0CTensor[0] | l0CTensor[1]]  Ping-Pong 双缓冲 (累加器)
     *
     * @param resource       硬件资源管理器 (提供 L1/L0 Buffer 访问)
     * @param nDyn           N 维动态大小 (embedding 维度的实际分块)
     * @param kDyn           K 维动态大小 (KV 序列长度的实际分块)
     * @param KVStackLen     KV Stack 长度 (单个 KV 块的序列长度, 默认 512)
     * @param l1BufAddrStart L1 Buffer 起始地址偏移 (用于多 Block 共享 L1)
     */
    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource,uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {
        maxKVStackLen = KVStackLen;
        // 分配 L1 内存空间: l1BTensor 放在 l1ATensor 数组之后
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
            L1TileShape::M * kDyn * sizeof(ElementA) * STAGES);
        // 为每个 Ping-Pong 阶段分配 L1A / L0A / L0B / L0C 缓冲
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart +
                L1TileShape::M * kDyn * sizeof(ElementA) * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        l1NDynamic = nDyn;
        l1KDynamic = kDyn;
    }

    __aicore__ inline
    ~BlockMmad() {}

    /**
     * @brief 重置 Paged Cache 块起始偏移为 0
     * 在新的 KV 循环迭代开始时调用, 确保 blockStartOffset 从 0 开始
     */
    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    /**
     * @brief 获取当前 KV 块的实际形状
     * @param actualShape  [in/out] 矩阵形状坐标, 更新 K 维度 (COORD_DIM2)
     * @param nowLen       当前 KV 块的实际序列长度
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& nowLen)
    {        
        actualShape[COORD_DIM2] = nowLen;
    }

    /**
     * @brief 计算 KV 矩阵在 GM 中的偏移 (非分页模式)
     * @param kOffset  [out] 计算得到的 GM 偏移
     * @param nIdx    当前 KV 块索引
     * @param strideKV KV 矩阵的 stride (每行字节数 / 元素数)
     *
     * 非分页模式下, KV 矩阵在 GM 中连续存储, 偏移 = nIdx × maxKVStackLen × strideKV
     */
    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t &strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV;
    }

    /**
     * @brief 计算 KV 矩阵在 GM 中的偏移 (分页模式 / Paged KV Cache)
     * @param gBlockTable     [in] block_table 全局张量 (存储物理块索引)
     * @param kOffset        [out] 计算得到的 GM 偏移
     * @param blockStartOffset 当前块内的起始偏移 (块内位置)
     * @param nowNIdx        当前逻辑块索引
     * @param strideKV       KV 矩阵的 stride
     * @param blockSize      物理块大小 (每个物理块包含的 token 数)
     *
     * 分页模式下, 通过 block_table 间接寻址:
     *   1. 从 block_table 查表得到物理块 ID: blockTableId = gBlockTable.GetValue(nowNIdx)
     *   2. 计算物理偏移: kOffset = blockTableId × blockSize × strideKV + blockStartOffset × strideKV
     * 这样 V 矩阵可以分散存储在不连续的物理块中, 实现高效的内存管理
     */
    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t blockStartOffset, 
        uint32_t nowNIdx, uint32_t &strideKV, uint32_t &blockSize)
    {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        kOffset = blockTableId * blockSize * strideKV + blockStartOffset * strideKV;
    }

    /**
     * @brief 设置分页缓存块的遍历参数
     * @param stackSeqTile   当前需要处理的序列长度 (token 数)
     * @param blockStart     [in/out] 块起始位置
     * @param blockEnd       [out] 最后一个块的实际长度
     * @param curBlockTotalNum [out] 需要遍历的总块数
     * @param blockSize      物理块大小
     *
     * 分页遍历逻辑:
     * - 如果 stackSeqTile 跨越了当前块边界, 需要分多块处理
     * - blockEnd: 最后一个不完整块的实际长度
     * - curBlockTotalNum: 总块数 = ceil((stackSeqTile - blockStart) / blockSize) + 1
     * - 否则 (未跨越边界), 直接处理 stackSeqTile 长度, 块数为 1
     */
    __aicore__ inline
    void setBlockParam(uint32_t stackSeqTile, uint32_t &blockStart, uint32_t &blockEnd, uint32_t &curBlockTotalNum, uint32_t blockSize){
        if(stackSeqTile >= blockStart && blockSize != 0) {
            blockEnd = ((stackSeqTile - blockStart) % blockSize == 0) ? blockSize : (stackSeqTile - blockStart) % blockSize;
            curBlockTotalNum = (((stackSeqTile - blockStart) + blockSize - 1) / blockSize) + 1;
        } else {
            blockStart = stackSeqTile;
            blockEnd = stackSeqTile + blockStartOffset;
            curBlockTotalNum = 1;
        }
    }

    /**
     * @brief 更新分页块的偏移和索引 (处理完一个物理块后调用)
     * @param nowLen       刚处理的块的实际长度
     * @param curBlockIdx  [in/out] 当前块索引, 处理完后递增
     * @param blockSize    物理块大小
     *
     * 偏移更新逻辑:
     * - 如果当前块已填满 (blockStartOffset + nowLen == blockSize), 重置偏移为 0
     * - 否则, 累加偏移: blockStartOffset += nowLen
     * - 块索引递增: curBlockIdx++
     */
    __aicore__ inline
    void updateBlockOffset(uint32_t nowLen, uint32_t &curBlockIdx, uint32_t blockSize){
        if (blockStartOffset + nowLen == blockSize) {
            blockStartOffset = 0;
        } else {
            blockStartOffset += nowLen;
        }
        curBlockIdx++;
    }

    /**
     * @brief PV 矩阵乘法主计算函数 —— 执行 O = P × V
     *
     * 执行流程:
     *   1. 预加载 V 矩阵到 L1 (支持分页/非分页两种模式)
     *   2. 等待 Vector 引擎完成 Online Softmax (softmaxFlag 跨核同步)
     *   3. 三重循环执行 P × V 矩阵乘法:
     *      - 外层 nL1 循环: 遍历 N 维 (embedding/head_dim 分块)
     *      - 中层 mL1 循环: 遍历 M 维 (query 行分块)
     *      - 内层 kL1 循环: 遍历 K 维 (KV 序列分块)
     *      - 最内 kL0 循环: L0 层 K 维分块
     *   4. 将 L0C 累加结果写回 GM
     *
     * Ping-Pong 同步策略:
     * - l1PPingPongFlag: L1A 缓冲 Ping-Pong (P 矩阵 GM→L1)
     * - l0ABPingPongFlag: L0A/L0B 缓冲 Ping-Pong (L1→L0 搬运)
     * - l0CPingPongFlag: L0C 缓冲 Ping-Pong (累加器输出)
     *
     * HardEvent 同步链:
     * - MTE1_MTE2: L1 数据搬运完成 → 可被 MTE2 读取
     * - MTE2_MTE1: GM→L1 搬运完成 → 可被 MTE1 读取
     * - M_MTE1:    Mmad 计算完成 → L0A/L0B 可被覆盖
     * - MTE1_M:    L1→L0 搬运完成 → 可执行 Mmad
     * - M_FIX:      Mmad 完成 → 可执行 FIX (L0C→GM)
     * - FIX_M:      FIX 完成 → L0C 可被覆盖
     *
     * @param gA            P 矩阵全局张量 (左矩阵, 来自 Softmax 输出)
     * @param gB            V 矩阵全局张量 (右矩阵)
     * @param gC            O 矩阵全局张量 (输出)
     * @param gBlockTable   分页块表 (Paged KV Cache 用)
     * @param layoutA/B/C   矩阵布局描述
     * @param actualOriShape 原始形状 (M, N, K)
     * @param nIdx          当前 KV 块索引
     * @param nLoop         KV 块总数
     * @param blockSize     分页物理块大小
     * @param kvSeqlen      KV 序列总长度
     * @param strideKV      KV 矩阵 stride
     * @param blockStackNum 块堆叠数
     * @param softmaxFlag   跨核同步信号量 (等待 Vector 引擎完成 Softmax)
     */
    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementA> gA,
        AscendC::GlobalTensor<ElementB> gB,
        AscendC::GlobalTensor<ElementC> gC,
        AscendC::GlobalTensor<int32_t> gBlockTable,
        LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
        uint32_t &nIdx, uint32_t &nLoop, uint32_t &blockSize, uint32_t kvSeqlen, uint32_t strideKV,
        uint32_t blockStackNum, Arch::CrossCoreFlag softmaxFlag)
    {
        // 解析矩阵形状: rowNum=M(query行数), embed=N(embedding维度), stackSeqTile=K(KV序列长度)
        uint32_t rowNum = actualOriShape[COORD_DIM0];
        uint32_t embed = actualOriShape[COORD_DIM1];
        uint32_t stackSeqTile = actualOriShape[COORD_DIM2];
        GemmCoord actualShape{rowNum, embed, 0};
        uint32_t gBOffset = 0;

        // ==================== 阶段 1: 预加载 V 矩阵到 L1 ====================
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);  // 等待 L1 中 V 数据可用
        if constexpr (PAGED_CACHE_FLAG_) {
            // ---- 分页模式: 通过 block_table 间接寻址, 可能跨越多个物理块 ----
            uint32_t curBlockIdx =  0;
            uint32_t blockStart = blockSize - blockStartOffset;
            uint32_t blockEnd = 0;
            uint32_t curBlockTotalNum = 0;
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);  // 计算需要遍历的物理块数
            while(curBlockIdx < curBlockTotalNum) {
                // 当前块的实际长度: 中间块为完整块, 最后一块可能不完整
                uint32_t nowLen = (curBlockIdx < (curBlockTotalNum-1)) ? (blockSize - blockStartOffset) : (blockEnd - blockStartOffset);
                uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;  // 逻辑块索引
                getBlockShape(actualShape, nowLen);                                   // 更新形状
                getKVOffset(gBlockTable, gBOffset, blockStartOffset, nowNIdx, strideKV, blockSize);  // 查表计算物理偏移
                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
                uint32_t curBlockSize = (curBlockIdx > 0) ? ((curBlockIdx - 1) * blockSize + blockStart) : 0;
                MatrixCoord l1BTileCoord{curBlockSize, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
                copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);  // V: GM→L1 (分页块)
                updateBlockOffset(nowLen, curBlockIdx, blockSize);            // 更新偏移, 准备下一块
            }
        } else {
            // ---- 非分页模式: V 矩阵连续存储, 直接搬运 ----
            getBlockShape(actualShape, stackSeqTile);
            getKVOffset(gBOffset, nIdx, strideKV);
            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
            copyGmToL1B(l1BTensor, gB[gBOffset], layoutBInL1, layoutBTile);  // V: GM→L1 (连续)
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);   // 通知 V 数据已搬运到 L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);  // 等待确认

        // ==================== 阶段 2: 跨核同步 —— 等待 Vector 引擎完成 Softmax ====================
        Arch::CrossCoreWaitFlag(softmaxFlag);  // 阻塞等待 P 矩阵 (Softmax 输出) 就绪

        // ==================== 阶段 3: 三重循环执行 P × V 矩阵乘法 ====================
        uint32_t mL1Loop = CeilDiv(rowNum, L1TileShape::M);        // M 维 L1 分块循环数
        uint32_t kL1Loop = CeilDiv(stackSeqTile, l1KDynamic);      // K 维 L1 分块循环数
        uint32_t nL1Loop = CeilDiv(embed, L0TileShape::N);        // N 维 L0 分块循环数

        // 外层循环: N 维 (embedding/head_dim 分块)
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; nL1Idx++) {
            uint32_t nL1Actual = (nL1Idx < nL1Loop - 1U) ? L0TileShape::N : (embed - nL1Idx * L0TileShape::N);
            // 中层循环: M 维 (query 行分块)
            for (uint32_t mL1Idx = 0; mL1Idx < mL1Loop; mL1Idx++) {
                uint32_t mL1Actual = (mL1Idx < mL1Loop - 1U) ? L1TileShape::M : (rowNum - mL1Idx * L1TileShape::M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);  // 等待 L0C 可写
                // 内层循环: K 维 (KV 序列分块)
                for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
                    uint32_t kL1Actual = (kL1Idx < kL1Loop - 1U) ? l1KDynamic : (stackSeqTile - kL1Idx * l1KDynamic);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);  // 等待 L1A 可读

                    // ---- 3a. P 矩阵: GM → L1 (Ping-Pong) ----
                    MatrixCoord gmATileCoord{mL1Idx * L1TileShape::M, kL1Idx * l1KDynamic};
                    auto gmTileA = gA[layoutA.GetOffset(gmATileCoord)];
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mL1Actual, kL1Actual));
                    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mL1Actual, kL1Actual);
                    copyGmToL1A(l1ATensor[l1PPingPongFlag], gmTileA, layoutAInL1, layoutTileA);  // P: GM→L1
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);            // 通知 L1A 可读

                    // ---- 3b. L0 层 K 维分块循环 ----
                    uint32_t kL0Loop = CeilDiv(kL1Actual, L0TileShape::K);
                    for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                        uint32_t kL0Actual =
                            (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kL1Actual - kL0Idx * L0TileShape::K);
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL1Actual, kL0Actual);
                        MatrixCoord l1ATileCoord{0, kL0Idx * L0TileShape::K};
                        auto l1ATile = l1ATensor[l1PPingPongFlag][layoutAInL1.GetOffset(l1ATileCoord)];

                        // ---- 3b-i. P: L1 → L0A (Ping-Pong) ----
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);  // 等待 L0A 可写
                        if (kL0Idx == 0U) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);  // 首次需等待 GM→L1 完成
                        }
                        copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);  // P: L1→L0A
                        if (kL0Idx == kL0Loop - 1U) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);  // 最后一块, 释放 L1A
                        }

                        // ---- 3b-ii. V: L1 → L0B (Ping-Pong) ----
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nL1Actual);
                        MatrixCoord l1BTileCoord{kL1Idx * l1KDynamic + kL0Idx * L0TileShape::K, L0TileShape::N * nL1Idx};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);  // 等待 L0B 可写
                        copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);  // V: L1→L0B

                        // ---- 3b-iii. 执行 Mmad: L0C = L0A × L0B (累加到 L0C) ----
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);   // 通知 L0A/L0B 已就绪
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);  // 等待确认
                        bool initMmad = (kL1Idx == 0U) && (kL0Idx == 0U);           // 是否为首次计算 (需初始化 L0C)
                        uint32_t mL0Align = (mL1Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;  // M 维 16 对齐
                        tileMmad(l0CTensor[l0CPingPongFlag],
                            l0ATensor[l0ABPingPongFlag],
                            l0BTensor[l0ABPingPongFlag],
                            mL0Align,
                            nL1Actual,
                            kL0Actual,
                            initMmad);  // Cube 引擎矩阵乘: L0C += L0A × L0B
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);      // 释放 L0A
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);  // 释放 L0B
                        l0ABPingPongFlag = 1U - l0ABPingPongFlag;  // 切换 Ping-Pong 缓冲
                    }
                    l1PPingPongFlag = 1U - l1PPingPongFlag;  // 切换 L1A Ping-Pong 缓冲
                }

                // ---- 3c. O: L0C → GM (输出结果) ----
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);   // 通知 Mmad 完成
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);  // 等待确认
                MatrixCoord gmCTileCoord{mL1Idx * L0TileShape::M, L0TileShape::N * nL1Idx};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL1Actual, nL1Actual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL1Actual, nL1Actual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);  // O: L0C→GM
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);  // 释放 L0C
                l0CPingPongFlag = 1U - l0CPingPongFlag;  // 切换 L0C Ping-Pong 缓冲
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);  // 通知 L1 数据已消费完
    }
 
protected:
    // ======================== L1/L0 缓冲张量 ========================
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];                // L1 中 P 矩阵缓冲 (Ping-Pong, STAGES 个)
    AscendC::LocalTensor<ElementB> l1BTensor;                        // L1 中 V 矩阵缓冲 (单个, 循环外预加载)
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];                // L0A 中 P 矩阵缓冲 (Ping-Pong)
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];                // L0B 中 V 矩阵缓冲 (Ping-Pong)
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];      // L0C 累加器缓冲 (Ping-Pong, 存储 O 中间结果)

    // ======================== 计算与搬运组件 ========================
    TileMmad tileMmad;            // L0 层 Mmad 指令封装 (执行 L0C += L0A × L0B)
    CopyGmToL1A copyGmToL1A;     // P 矩阵搬运: GM → L1
    CopyGmToL1B copyGmToL1B;     // V 矩阵搬运: GM → L1
    CopyL1ToL0A copyL1ToL0A;     // P 矩阵搬运: L1 → L0A
    CopyL1ToL0B copyL1ToL0B;     // V 矩阵搬运: L1 → L0B
    CopyL0CToGm copyL0CToGm;     // O 矩阵搬运: L0C → GM

    // ======================== Ping-Pong 标志 ========================
    uint32_t l1PPingPongFlag = 0;   // L1A 缓冲 Ping-Pong 标志 (0/1 切换)
    uint32_t l0CPingPongFlag = 0;   // L0C 缓冲 Ping-Pong 标志 (0/1 切换)
    uint32_t l0ABPingPongFlag = 0;  // L0A/L0B 缓冲 Ping-Pong 标志 (0/1 切换, +2U 用于 L0B)

    // ======================== 动态维度参数 ========================
    uint32_t l1MDynamic = 0;        // M 维动态大小 (query 行数)
    uint32_t l1NDynamic = 0;        // N 维动态大小 (embedding 维度)
    uint32_t l1KDynamic = 0;        // K 维动态大小 (KV 序列长度)

    // ======================== 分页缓存参数 ========================
    uint32_t blockStartOffset = 0;  // 当前物理块内的起始偏移 (Paged KV Cache 用)
    uint32_t maxKVStackLen = 0;     // 最大 KV Stack 长度 (单个 KV 块的序列长度)
};

}

#endif