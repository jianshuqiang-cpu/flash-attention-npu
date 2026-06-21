/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_QK_HPP_T
#define CATLASS_GEMM_BLOCK_MMAD_QK_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "fa_block.h"

/*
 * ============================================================================================
 * QK 矩阵乘法 BlockMmad —— FlashAttention 第一阶段 GEMM (S = Q × K^T)
 * ============================================================================================
 *
 * 【定位】
 *   本文件实现 FlashAttention 两阶段 GEMM 中的第一阶段：QK 阶段。
 *   计算 S = Q × K^T（注意力分数矩阵），其结果 S 会写入 GM，
 *   随后由 Vector 引擎的 Online Softmax 进行缩放、指数化、归一化等处理。
 *
 *   两阶段 GEMM 流水线：
 *     ┌─────────────┐    GM(S)    ┌──────────────────┐   GM(P)   ┌─────────────┐
 *     │ QK 矩阵乘法 │ ──────────► │ Online Softmax    │ ────────► │ PV 矩阵乘法 │
 *     │ (本文件)     │             │ (Vector 引擎)     │           │ (pv_matmul)  │
 *     └─────────────┘             └──────────────────┘           └─────────────┘
 *        Cube 引擎                   Cube+Vector 协同               Cube 引擎
 *
 * 【核心特性】
 *   1. Q 矩阵预加载：Q 在整个 KV 循环中保持不变，通过 loadQGM() 一次性加载到 L1，
 *      后续所有 KV 块复用同一份 Q 数据，避免重复搬运。
 *   2. K 矩阵按块加载：K（即 B 矩阵）按 KV 块从 GM 搬运到 L1，支持 Paged/非 Paged 两种模式。
 *   3. L0 Ping-Pong 三重循环：nL1(外层 KV 块) × mL0(Q 行分块) × kL0(嵌入维分块)，
 *      通过 l0ABPingPongFlag / l0CPingPongFlag / l1KvPingPongFlag 三组标志位实现流水线。
 *   4. 结果直写 GM：与 PV 不同，QK 的结果 S 直接通过 CopyL0CToGm 写回 GM，
 *      供 Vector 引擎读取（PV 的结果保留在 L0C 中等待 Vector 消费）。
 *
 * 【与 pv_matmul.hpp 的 3 大区别】
 *   ┌──────────────┬────────────────────────────┬────────────────────────────┐
 *   │ 对比维度      │ qk_matmul.hpp (本文件)      │ pv_matmul.hpp              │
 *   ├──────────────┼────────────────────────────┼────────────────────────────┤
 *   │ 矩阵角色      │ A=Q, B=K^T, C=S            │ A=P, B=V, C=O              │
 *   │ 跨核同步方向  │ QK 先行，结果写 GM 供 Vector│ PV 后行，等待 Vector softmax│
 *   │ B 矩阵加载    │ K 按块加载（循环内搬运）    │ V 预加载到 L1（循环前搬运） │
 *   │ 结果去向      │ CopyL0CToGm 写回 GM         │ 保留在 L0C（Vector 直接读）│
 *   └──────────────┴────────────────────────────┴────────────────────────────┘
 *
 * 【内存层级】
 *   GM(Q) ──loadQGM──► L1A(Q, 单缓冲, 全程复用)
 *   GM(K) ──copyGmToL1B──► L1B(K, STAGES 缓冲 Ping-Pong) ──copyL1ToL0B──► L0B
 *   L1A(Q) ──────────────copyL1ToL0A──► L0A
 *   L0A(Q) × L0B(K) ──tileMmad──► L0C(S) ──copyL0CToGm──► GM(S)
 *
 * ============================================================================================
 */

namespace Catlass::Gemm::Block {

/*
 * BlockMmad 模板特化 —— QK 矩阵乘法
 *
 * 模板参数说明：
 *   PAGED_CACHE_FLAG_ : 是否启用 Paged KV Cache（分页缓存）模式
 *   ENABLE_UNIT_FLAG_ : 是否启用单元级标志位（用于细粒度同步控制）
 *   L1TileShape_      : L1 层 Tile 形状（M/K/N 维度）
 *   L0TileShape_      : L0 层 Tile 形状（M/K/N 维度）
 *   AType_/BType_/CType_ : A/B/C 矩阵的类型包装（Element + Layout）
 *   BiasType_         : 偏置数据类型（QK 阶段一般不使用偏置）
 *   TileCopy_         : 数据搬运算子集合（GM↔L1, L1↔L0）
 *   TileMmad_         : L0 层矩阵乘法算子（Cube 引擎 mmad 指令封装）
 *
 * 调度策略命名：MmadAtlasA2FAIQKT
 *   A2  = Atlas A2（昇腾 A2 系列芯片）
 *   FAI = FlashAttention Inference（推理场景）
 *   QKT = Q × K^T（QK 转置乘法）
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
    MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_> {
public:
    // ============================== 类型别名（Type Aliases） ==============================
    using DispatchPolicy = MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;  // 调度策略
    using ArchTag = typename DispatchPolicy::ArchTag;                                // 架构标签（Atlas A2）
    using L1TileShape = L1TileShape_;                                                // L1 层 Tile 形状
    using L0TileShape = L0TileShape_;                                                // L0 层 Tile 形状
    using ElementA = typename AType_::Element;                                       // A 矩阵元素类型（Q）
    using LayoutA = typename AType_::Layout;                                         // A 矩阵布局
    using ElementB = typename BType_::Element;                                       // B 矩阵元素类型（K）
    using LayoutB = typename BType_::Layout;                                         // B 矩阵布局
    using ElementC = typename CType_::Element;                                       // C 矩阵元素类型（S）
    using LayoutC = typename CType_::Layout;                                         // C 矩阵布局
    using TileMmad = TileMmad_;                                                      // L0 矩阵乘法算子
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;                             // GM → L1A 搬运（Q）
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;                             // GM → L1B 搬运（K）
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;                             // L1A → L0A 搬运（Q）
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;                             // L1B → L0B 搬运（K）
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;                             // L0C → GM 搬运（S 结果回写）
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;  // 累加器类型（通常 float）
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;                             // A 在 L1 中的布局
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;                             // B 在 L1 中的布局
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;                             // A 在 L0 中的布局
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;                             // B 在 L0 中的布局
    using LayoutCInL0 = layout::zN;                                                  // C 在 L0 中固定为 zN 布局（Cube 引擎要求）

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;           // L1A 对齐辅助器
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;           // L1B 对齐辅助器

    // ============================== 硬件常量（Hardware Constants） ==============================
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;                      // 流水线级数（Ping-Pong 缓冲数，通常为 2）
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);  // L1A 单缓冲大小（字节）
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);  // L1B 单缓冲大小（字节）
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;                         // L0A 总大小（硬件分配）
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;                         // L0B 总大小（硬件分配）
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;                         // L0C 总大小（硬件分配）
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;            // L0A 每个 Ping-Pong 缓冲大小
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;            // L0B 每个 Ping-Pong 缓冲大小
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;            // L0C 每个 Ping-Pong 缓冲大小
    static constexpr uint32_t BLOCK_SIZE = 16;                                       // 矩阵分块基础大小（Cube 引擎 16×16 基本块）
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;                                // 嵌入维分块大小
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;                              // 单块堆叠数
    static constexpr uint32_t KV_BASE_BLOCK = 512;                                   // KV 基础块大小
    static constexpr uint32_t KV_SPLIT_SIZE = 128;                                   // KV 分块大小
    static constexpr uint32_t COORD_DIM0 = 0;                                        // 坐标维度 0（M / 行数）
    static constexpr uint32_t COORD_DIM1 = 1;                                        // 坐标维度 1（N / KV 序列长度）
    static constexpr uint32_t COORD_DIM2 = 2;                                        // 坐标维度 2（K / 嵌入维度）

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");  // C 矩阵仅支持行主序

    /*
     * 构造函数 —— 分配 L1/L0 缓冲区
     *
     * 内存布局示意（L1 空间）：
     *   ┌─────────────────────────────────────────────────────────────┐
     *   │ L1A (Q 矩阵, 单缓冲)   │ L1B[0] (K) │ L1B[1] (K) │ ...        │
     *   │ l1BufAddrStart       │  M*kDyn    │  nDyn*kDyn │            │
     *   └─────────────────────────────────────────────────────────────┘
     * L0 空间（Ping-Pong 双缓冲）：
     *   L0A[0]/L0A[1] (Q 分块) | L0B[0]/L0B[1] (K 分块) | L0C[0]/L0C[1] (S 累加器)
     *
     * 参数：
     *   resource       : 硬件资源句柄（提供 L1/L0A/L0B/L0C 缓冲区）
     *   nDyn           : N 维动态大小（KV 序列分块大小）
     *   kDyn           : K 维动态大小（嵌入维分块大小）
     *   KVStackLen     : KV 堆栈长度（默认 512，用于偏移计算）
     *   l1BufAddrStart : L1 缓冲区起始地址偏移
     */
    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {   
        maxKVStackLen = KVStackLen;
        // 分配 L1 内存空间
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);  // L1A: Q 矩阵（单缓冲，全程复用）
        for (uint32_t i = 0; i < STAGES; i++) {
            // L1B: K 矩阵 Ping-Pong 缓冲，地址紧跟 L1A 之后
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                L1TileShape::M * kDyn * sizeof(ElementA) + nDyn * kDyn * sizeof(ElementB) * i);
            // L0A/L0B/L0C: 各自 Ping-Pong 双缓冲，按 PINGPONG_BUF_SIZE 等分
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        l1NDynamic = nDyn;  // 记录 N 维动态大小
        l1KDynamic = kDyn;  // 记录 K 维动态大小
    }

    __aicore__ inline
    ~BlockMmad() {}

    /*
     * loadQGM —— 将 Q 矩阵从 GM 预加载到 L1
     *
     * 【作用】Q 矩阵在整个 KV 循环中保持不变，因此只需在循环开始前加载一次到 L1A。
     *        后续所有 KV 块的计算都复用这份 Q 数据，大幅减少 GM→L1 的搬运开销。
     *
     * 【GQA 支持】通过 singleGroupHeads（每组头数）和 qHeads（总头数）实现分组查询注意力：
     *   - tokenNumPerGroup = rowNum / singleGroupHeads：每组 token 数
     *   - 多个 head 共享同一组 Q，通过 stride = qHeads * embed 跨组寻址
     *
     * 参数：
     *   gA               : Q 矩阵的 GM 张量
     *   layoutA          : Q 矩阵的布局描述
     *   rowNum           : Q 的行数（token 数）
     *   singleGroupHeads : 每组的头数（GQA 分组大小）
     *   qHeads           : 总查询头数
     */
    __aicore__ inline
    void loadQGM(
        AscendC::GlobalTensor<ElementA> gA,
        LayoutA layoutA,
        uint32_t rowNum, uint32_t &singleGroupHeads, uint32_t &qHeads)
    {
        uint32_t embed = layoutA.shape(1);                                          // 嵌入维度
        uint32_t rowNumRound = RoundUp(rowNum, L1AAlignHelper::M_ALIGNED);          // 行数对齐向上取整
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;                      // 每组 token 数
        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));  // 单组 tile 布局
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);  // L1 中 A 的布局
        // 执行 GM → L1A 搬运，支持 GQA 多组跨步寻址
        copyGmToL1A(
            l1ATensor, gA,
            layoutAInL1, layoutSingleANd,
            tokenNumPerGroup, qHeads * embed, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);
        // 设置并等待 MTE2_MTE1 事件，确保 GM→L1 搬运完成后 L1 数据可被 MTE1（L1→L0）读取
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }

    /*
     * setBlockParam —— 设置 Paged 缓存的分块遍历参数
     *
     * 【作用】在 Paged KV Cache 模式下，KV 数据按物理块（blockSize）分散存储，
     *        需要计算当前遍历涉及的块数量和最后一块的长度。
     *
     * 参数：
     *   stackSeqTile    : 当前需要遍历的 KV 序列总长度
     *   blockStart      : [输入/输出] 块起始位置
     *   blockEnd        : [输出] 最后一个块的实际长度（可能不满 blockSize）
     *   curBlockTotalNum: [输出] 需要遍历的总块数
     *   blockSize       : 物理块大小
     */
    __aicore__ inline
    void setBlockParam(uint32_t stackSeqTile, uint32_t &blockStart, uint32_t &blockEnd, uint32_t &curBlockTotalNum, uint32_t blockSize){
        if(stackSeqTile >= blockStart && blockSize != 0) {
            // 计算最后一块的长度：若整除则为完整 blockSize，否则为余数
            blockEnd = ((stackSeqTile - blockStart) % blockSize == 0) ? blockSize : (stackSeqTile - blockStart) % blockSize;
            // 总块数 = ceil((stackSeqTile - blockStart) / blockSize) + 1
            curBlockTotalNum = (((stackSeqTile - blockStart) + blockSize - 1) / blockSize) + 1;
        } else {
            // 首次调用或 blockSize 为 0 时的初始化
            curBlockTotalNum = 1;
            blockStart = stackSeqTile;
            blockEnd = stackSeqTile + blockStartOffset;
        }
    }
    
    /*
     * getBlockShape (重载1) —— 获取非 Paged 模式下的实际 N 维形状
     *
     * 【作用】计算当前 nL1 分块的实际 N 维大小。最后一个分块可能不满 l1NDynamic。
     *
     * 参数：
     *   actualShape  : [输出] 更新后的实际形状（仅修改 N 维）
     *   nL1Idx       : 当前 nL1 循环索引
     *   nL1Loop      : nL1 总循环次数
     *   stackSeqTile : KV 序列总长度
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t nL1Idx, uint32_t nL1Loop, uint32_t stackSeqTile)
    {
        uint32_t nSplitSize = l1NDynamic;  // 默认为完整分块大小
        if (nL1Idx == nL1Loop - 1U) {
            // 最后一个分块：取余数
            nSplitSize = stackSeqTile - nL1Idx * l1NDynamic;
        }
        actualShape[COORD_DIM1] = nSplitSize;  // 更新 N 维
    }

    /*
     * getBlockShape (重载2) —— 获取 Paged 模式下当前物理块的实际长度
     *
     * 【作用】在 Paged 模式下，计算本次从物理块中读取的长度 nowLen。
     *        nowLen = min(当前块剩余长度, L1 分块剩余长度)
     *
     * 参数：
     *   actualShape     : [输出] 更新后的实际形状（仅修改 N 维）
     *   blockStartOffset: [输入] 当前块内已读取的偏移
     *   l1NResDynamic   : [输入] L1 分块剩余需读取的长度
     *   kvL1Len         : [输入] 当前 L1 分块已读取的长度
     *   nowLen          : [输出] 本次读取的长度
     *   blockSize       : [输入] 物理块大小
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& blockStartOffset, uint32_t& l1NResDynamic, uint32_t& kvL1Len, uint32_t& nowLen, uint32_t& blockSize)

    {
        // 取当前块剩余长度与 L1 分块剩余长度的较小值
        nowLen = (blockSize - blockStartOffset < l1NResDynamic - kvL1Len) ?
                blockSize - blockStartOffset :
                l1NResDynamic - kvL1Len;
        actualShape[COORD_DIM1] = nowLen;  // 更新 N 维为本次读取长度
    }

    /*
     * getKVOffset (重载1) —— 非 Paged 模式下计算 K 矩阵在 GM 中的偏移
     *
     * 【作用】连续存储模式下，K 矩阵偏移计算：
     *        kOffset = nIdx * maxKVStackLen * strideKV + nowNIdx * l1NDynamic * strideKV
     *   - nIdx    : 外层 KV stack 索引
     *   - nowNIdx : 内层 nL1 分块索引
     *   - strideKV: K 矩阵的行步长（嵌入维）
     */
    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t nowNIdx, uint32_t strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV + nowNIdx * l1NDynamic * strideKV;
    }

    /*
     * getKVOffset (重载2) —— Paged 模式下通过 block_table 间接寻址计算 K 矩阵偏移
     *
     * 【作用】Paged KV Cache 模式下，KV 数据按物理块分散存储，通过 block_table 间接寻址：
     *   1. blockTableId = gBlockTable.GetValue(nowNIdx)：查表获取物理块 ID
     *   2. kOffset = blockTableId * blockSize * strideKV + startOffset * strideKV
     *      - blockTableId * blockSize * strideKV：物理块基地址
     *      - startOffset * strideKV：块内偏移
     *
     * 参数：
     *   gBlockTable : block_table 全局张量（存储物理块 ID 映射）
     *   kOffset     : [输出] 计算得到的 GM 偏移
     *   nowNIdx     : 逻辑块索引
     *   startOffset : 块内起始偏移
     *   strideKV    : K 矩阵行步长
     *   blockSize    : 物理块大小
     */
    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t nowNIdx, 
        uint32_t startOffset, uint32_t strideKV, uint32_t blockSize)
    {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);  // 查表获取物理块 ID
        kOffset = blockTableId * blockSize * strideKV + startOffset * strideKV;  // 物理块基地址 + 块内偏移
    }

    /*
     * resetBlockStart —— 重置 Paged 缓存的块起始偏移
     *
     * 【作用】在新的 KV stack 开始遍历时，将块内偏移重置为 0。
     */
    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    /*
     * updateBlockOffset —— 更新 Paged 缓存的块偏移并推进块索引
     *
     * 【作用】每次读取 nowLen 长度后，更新块内偏移：
     *   - 若当前块已读满（blockStartOffset + nowLen == blockSize）：
     *     重置偏移为 0，并推进到下一个物理块（curBlockIdx++）
     *   - 否则：累加偏移量
     *
     * 参数：
     *   nowLen      : 本次读取的长度
     *   curBlockIdx : [输入/输出] 当前物理块索引
     *   blockSize   : 物理块大小
     */
    __aicore__ inline
    void updateBlockOffset(uint32_t nowLen, uint32_t &curBlockIdx, uint32_t blockSize){
        if(blockStartOffset + nowLen == blockSize){
            // 当前块已读满，推进到下一块
            blockStartOffset = 0;
            curBlockIdx++;
        } else{
            // 当前块未读满，累加偏移
            blockStartOffset += nowLen;
        }
    }

    /*
     * operator() —— QK 矩阵乘法主计算函数
     *
     * 【作用】执行 S = Q × K^T 的完整计算流程，结果写入 GM 供 Vector 引擎读取。
     *
     * 【执行流程】
     *   1. 初始化：解析形状参数，计算 nL1 循环次数，Paged 模式下设置分块参数
     *   2. 外层循环 nL1（KV 块遍历）：
     *      a. Paged 模式：内层 while 循环跨多个物理块搬运 K 到 L1B
     *      b. 非 Paged 模式：直接搬运一个 nL1 分块的 K 到 L1B
     *   3. 中层循环 mL0（Q 行分块）：
     *      - 等待 L0C 可写（FIX_M）
     *   4. 内层循环 kL0（嵌入维分块）：
     *      - L1A → L0A 搬运 Q 分块（copyL1ToL0A）
     *      - L1B → L0B 搬运 K 分块（copyL1ToL0B）
     *      - 执行 Cube 矩阵乘法 tileMmad （L0A × L0B → L0C）
     *      - Ping-Pong 切换 l0ABPingPongFlag
     *   5. mL0 循环结束：L0C → GM 回写结果 S（copyL0CToGm），切换 l0CPingPongFlag
     *   6. nL1 循环结束：切换 l1KvPingPongFlag
     *
     * 【HardEvent 同步链】
     *   MTE1_MTE2 : 等待 L1B 可被新数据覆盖（上一轮 K 已被 L0 读取完毕）
     *   MTE2_MTE1 : 等待 GM→L1B 搬运完成，L1B 数据可被 L1→L0 读取
     *   M_MTE1    : 等待 L0A/L0B 可被新数据覆盖（上一轮已被 Cube 计算完毕）
     *   MTE1_M    : 等待 L1→L0 搬运完成，L0A/L0B 数据可被 Cube 读取
     *   M_FIX     : 等待 Cube 计算完成，L0C 可被回写
     *   FIX_M     : 等待 L0C→GM 回写完成，L0C 可被新结果覆盖
     *
     * 参数：
     *   gA, gB, gC       : Q/K/S 的 GM 张量
     *   gBlockTable      : Paged 模式的 block_table（非 Paged 模式不使用）
     *   layoutA/B/C      : 各矩阵的布局
     *   actualOriShape   : 原始形状 (M=rowNum, N=stackSeqTile, K=embed)
     *   nIdx             : 外层 KV stack 索引
     *   nLoop            : 外层 KV stack 总循环数
     *   blockSize        : Paged 物理块大小
     *   strideKV         : K 矩阵行步长
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
                    uint32_t nIdx, uint32_t nLoop, uint32_t blockSize, uint32_t strideKV)
    {
        // ---- 阶段1：初始化与参数解析 ----
        uint32_t rowNum = actualOriShape[COORD_DIM0];        // M 维：Q 的行数（token 数）
        uint32_t stackSeqTile = actualOriShape[COORD_DIM1];   // N 维：KV 序列总长度
        uint32_t embed = actualOriShape[COORD_DIM2];         // K 维：嵌入维度

        GemmCoord actualShape{rowNum, 0, embed};             // 实际计算形状（N 维动态更新）
        uint32_t gBOffset = 0;                                // K 矩阵在 GM 中的偏移

        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);  // L1A 布局

        uint32_t tileNNumPerBaseBlock = blockSize / l1NDynamic;  // 每个基础块包含的 nL1 分块数
        uint32_t nL1Loop = CeilDiv(stackSeqTile, l1NDynamic);    // nL1 外层循环次数
        uint32_t curBlockIdx =  0;                               // 当前物理块索引（Paged 模式）
        uint32_t blockStart = 0;                                 // 块起始位置
        uint32_t blockEnd = 0;                                   // 最后一块长度
        uint32_t curBlockTotalNum = 0;                           // 总块数
        if constexpr (PAGED_CACHE_FLAG_){
            // Paged 模式：初始化分块遍历参数
            blockStart = blockSize - blockStartOffset;
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);
        }
        // ---- 阶段2：外层循环 nL1（KV 块遍历）----
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t mActual = actualShape.m();
            uint32_t kActual = actualShape.k();
            uint32_t nActual = actualShape.n();
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
            if constexpr (PAGED_CACHE_FLAG_){
                // ---- Paged 模式：跨多个物理块搬运 K 到 L1B ----
                uint32_t l1NResDynamic = (nL1Idx < (nL1Loop-1)) ? l1NDynamic : (stackSeqTile - nL1Idx * l1NDynamic);  // 本 nL1 分块需读取的总长度
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(embed, l1NResDynamic);
                uint32_t kvL1Len = 0;  // 已读取长度
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);  // 等待 L1B 可被新数据覆盖
                while(kvL1Len < l1NResDynamic){
                    // 内层循环：跨物理块填充一个 nL1 分块
                    uint32_t nowLen = 0;
                    uint32_t curBlockSize = (curBlockIdx < (curBlockTotalNum-1)) ? blockSize : blockEnd;  // 当前块大小（最后一块可能不满）
                    uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;  // 逻辑块索引
                    getBlockShape(actualShape, blockStartOffset, l1NResDynamic, kvL1Len, nowLen, curBlockSize);  // 计算本次读取长度
                    getKVOffset(gBlockTable, gBOffset, nowNIdx, blockStartOffset, strideKV, blockSize);  // block_table 间接寻址
                    auto layoutBTile = layoutB.GetTileLayout(MakeCoord(embed, nowLen));
                    MatrixCoord l1BTileCoord{0, kvL1Len};  // L1B 内偏移（按已读长度定位）
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);  // GM → L1B 搬运
                    kvL1Len += nowLen;  // 累加已读长度
                    updateBlockOffset(nowLen, curBlockIdx, blockSize);  // 更新块偏移/推进块索引
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);  // 通知 L1B 数据就绪
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = l1NResDynamic;
            } else {
                // ---- 非 Paged 模式：直接搬运一个 nL1 分块的 K ----
                getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);  // 计算实际 N 维
                getKVOffset(gBOffset, nIdx, nL1Idx, strideKV);  // 连续存储偏移
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = actualShape.n();
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);  // 等待 L1B 可写
                copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB[gBOffset], layoutBInL1, layoutBTile);  // GM → L1B
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);  // 通知 L1B 就绪
            }
            // ---- 阶段3：中层循环 mL0（Q 行分块）+ 内层循环 kL0（嵌入维分块）----
            uint32_t mL0Loop = CeilDiv(mActual, L0TileShape::M);  // M 维 L0 分块数
            uint32_t kL0Loop = CeilDiv(kActual, L0TileShape::K);  // K 维 L0 分块数
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1U) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);  // 本块 M 实际大小
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);  // 等待 L0C 可写（上一轮已回写 GM）
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);  // 本块 K 实际大小

                    // --- L1A → L0A 搬运 Q 分块 ---
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);  // 等待 L0A 可写
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);  // L1A → L0A

                    // --- L1B → L0B 搬运 K 分块 ---
                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    if ((mL0Idx == 0U) && (kL0Idx == 0U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);  // 首次需等待 L1B 数据就绪
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);  // 等待 L0B 可写（使用偏移+2 的独立事件）
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);  // L1B → L0B
                    if ((mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);  // 末次通知 L1B 可被新数据覆盖
                    }

                    // --- Cube 矩阵乘法 L0A × L0B → L0C ---
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);  // 通知 L0A/L0B 数据就绪
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);  // 等待 Cube 可读
                    bool initMmad = (kL0Idx == 0U);  // 首个 kL0 块需初始化累加器（清零），后续块累加
                    uint32_t mL0Align = (mL0Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;  // M 维对齐到 16
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL0Align,
                        nActual,
                        kL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);      // 通知 L0A 可被新数据覆盖
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);  // 通知 L0B 可被新数据覆盖
                    l0ABPingPongFlag = 1U - l0ABPingPongFlag;  // 切换 L0A/L0B Ping-Pong
                }
                // ---- 阶段4：L0C → GM 回写结果 S ----
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);  // 通知 Cube 计算完成
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);  // 等待 L0C 可回写
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};  // GM 中 S 的 tile 坐标
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);  // L0C → GM
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);  // 通知 L0C 可被新结果覆盖
                l0CPingPongFlag = 1U - l0CPingPongFlag;  // 切换 L0C Ping-Pong
            }
            l1KvPingPongFlag = 1U - l1KvPingPongFlag;  // 切换 L1B Ping-Pong
        }
    }
protected:
    // ============================== 数据成员（Data Members） ==============================

    // --- L1/L0 缓冲张量 ---
    AscendC::LocalTensor<ElementA> l1ATensor;                       // L1A: Q 矩阵（单缓冲，全程复用）
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];              // L1B: K 矩阵（Ping-Pong 缓冲）
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];              // L0A: Q 分块（Ping-Pong 缓冲）
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];              // L0B: K 分块（Ping-Pong 缓冲）
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];   // L0C: S 累加器（Ping-Pong 缓冲）

    // --- 计算与搬运组件 ---
    TileMmad tileMmad;          // L0 矩阵乘法算子（Cube 引擎 mmad 指令封装）
    CopyGmToL1A copyGmToL1A;    // GM → L1A 搬运（Q 预加载）
    CopyGmToL1B copyGmToL1B;    // GM → L1B 搬运（K 按块加载）
    CopyL1ToL0A copyL1ToL0A;    // L1A → L0A 搬运（Q 分块）
    CopyL1ToL0B copyL1ToL0B;    // L1B → L0B 搬运（K 分块）
    CopyL0CToGm copyL0CToGm;    // L0C → GM 搬运（S 结果回写）

    // --- Ping-Pong 标志位 ---
    uint32_t l1KvPingPongFlag = 0;   // L1B (K) Ping-Pong 标志：0/1 切换
    uint32_t l0CPingPongFlag = 0;   // L0C (S) Ping-Pong 标志：0/1 切换
    uint32_t l0ABPingPongFlag = 0;  // L0A/L0B (Q/K 分块) Ping-Pong 标志：0/1 切换

    // --- 动态维度 ---
    uint32_t l1MDynamic = 0;         // M 维动态大小（Q 行数）
    uint32_t l1NDynamic = 0;         // N 维动态大小（KV 分块大小）
    uint32_t l1KDynamic = 0;         // K 维动态大小（嵌入维分块）

    // --- Paged 缓存参数 ---
    uint32_t blockStartOffset = 0;   // 当前物理块内已读取的偏移
    uint32_t maxKVStackLen = 0;       // KV 堆栈最大长度（用于偏移计算）
};

}

#endif