/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

/**
 * ============================================================================
 * qk_matmul.hpp —— FlashAttention NPU 前向推理中 Q×K^T 矩阵乘（Cube 核）
 * ============================================================================
 *
 * 【文件定位】
 *   本文件实现 CATLASS 框架中 BlockMmad 的模板特化，调度策略为 MmadAtlasA2FAIQKT，
 *   对应 FlashAttention 前向推理的第一个矩阵乘：
 *
 *       S = Q × K^T
 *
 *   其中：
 *     - Q (ElementA) : Query 矩阵，形状 [rowNum, embed]，
 *                      rowNum = qSBlockSize × qNBlockSize（序列块行数 × 当前 group 内 head 数）；
 *     - K (ElementB) : Key 矩阵，以 RowMajor 存储为 [strideK/embed, embed] 物理形状，
 *                      逻辑上对每个 KV head 形状为 [kvSeqlen, embed]，矩阵乘中取转置 K^T；
 *     - S (ElementC) : 注意力分数矩阵（未加 mask、未做 softmax），形状 [rowNum, stackSeqTile]，
 *                      写入 workspace，供 Vector 核 online_softmax 消费。
 *
 * 【与 PV 矩阵乘的对偶性】
 *   QK 阶段（本文件）         ：A=Q（单缓冲常驻 L1），B=K（Ping-Pong 加载）
 *   PV 阶段（pv_matmul.hpp）  ：A=P（Ping-Pong 加载），B=V（单缓冲预加载到 L1）
 *   两者在循环结构、ping-pong 分配上互为镜像：
 *   ┌──────────┬──────────────────────┬──────────────────────┐
 *   │          │ QK (S = Q × K^T)     │ PV (O = P × V)        │
 *   ├──────────┼──────────────────────┼──────────────────────┤
 *   │ A 矩阵   │ Q 单缓冲常驻 L1      │ P Ping-Pong 动态加载  │
 *   │ B 矩阵   │ K Ping-Pong 动态加载 │ V 单缓冲预加载到 L1   │
 *   │ 跨核信号 │ CrossCoreSetFlag    │ CrossCoreWaitFlag    │
 *   │          │   (QK 完成→Vector)   │   (等待 softmaxReady) │
 *   │ COORD    │ DIM1=N(stackSeqTile) │ DIM1=N(embed)        │
 *   │          │ DIM2=K(embed)        │ DIM2=K(stackSeqTile) │
 *   │ initMmad │ (kL0Idx == 0)        │ (kL1Idx==0)&&(kL0==0)│
 *   │ 外层循环 │ nL1(stack)→mL0→kL0   │ nL1(embed)→mL1→kL1→kL0│
 *   │ K/V 加载 │ nL1 循环内逐块加载   │ operator() 开头一次性 │
 *   └──────────┴──────────────────────┴──────────────────────┘
 *
 * 【内存层级】
 *   GM(全局内存) ──MTE2──> L1(片上SRAM) ──MTE1──> L0A/L0B(Cube输入缓冲)
 *                                                          │
 *                                                    Cube(MMAD)
 *                                                          │
 *                                                    L0C(Cube输出) ──FIX──> GM(S)
 *
 * 【L1 内存布局（从 l1BufAddrStart=0 开始，PV 的 L1 紧接其后）】
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ L1A (Q 单缓冲)   : rowNum × embed 个 ElementA           │
 *   │ L1B[0] (K slot0) : nDyn × kDyn 个 ElementB              │
 *   │ L1B[1] (K slot1) : nDyn × kDyn 个 ElementB              │
 *   └──────────────────────────────────────────────────────────┘
 *   Q 在 operator() 之前通过 loadQGM() 一次性加载到 L1，全程复用；
 *   K 在 nL1 循环内 Ping-Pong 加载（每个 KV 子块搬运一次）。
 *
 * 【核心循环嵌套】
 *   nL1(KV序列/N维, 步长l1NDynamic) → mL0(Q行/M维, 步长128) → kL0(embed/K维, 步长128)
 *
 * 【跨核/事件同步】
 *   - EVENT_ID3 (MTE2_MTE1): Q 加载完成信号（loadQGM 内 Set/Wait）
 *   - EVENT_ID0 (MTE1_M/M_FIX/MTE2_MTE1): 复用为 L0A/L0B 就绪、Cube 完成、K 加载完成信号
 *   - l1KvPingPongFlag : L1B (K) Ping-Pong（K 块 GM→L1 同步）
 *   - l0ABPingPongFlag : L0A/L0B Ping-Pong（L1→L0 搬运同步，L0A 用 flag+0，L0B 用 flag+2）
 *   - l0CPingPongFlag  : L0C Ping-Pong（Cube→GM 输出同步）
 *   - CrossCoreSetFlag(qkReady) 在 operator() 返回后由 mha_fwd_kvcache.cpp 调用方设置，
 *     通知 Vector 核开始 online_softmax
 *
 * 【分页 KV Cache 支持】
 *   通过模板参数 PAGED_CACHE_FLAG_ 编译期分支：
 *   - false: K 连续存储，单次 DataCopy 即可搬运到 L1B；
 *   - true : K 按 blockSize 分页存储，在 nL1 循环内逐页搬运并拼接
 *           到 L1B 的 Ping-Pong slot 连续区域。
 *
 * 【GQA (Grouped Query Attention) 支持】
 *   loadQGM() 使用特殊的 DataCopy 模式，支持多头分组：
 *   将同一 group 内共享同一 KV head 的多个 Q head 批量加载到 L1，
 *   通过 tokenNumPerGroup（每 group 的 token 数）和 qHeads*embed（GM 步长）
 *   参数描述 GQA 的 interleaved 内存布局。
 * ============================================================================
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

namespace Catlass::Gemm::Block {

/**
 * @brief BlockMmad 对 MmadAtlasA2FAIQKT 调度策略的特化：FlashAttention QK 矩阵乘
 *
 * @tparam PAGED_CACHE_FLAG_  是否启用分页 KV Cache 寻址
 * @tparam ENABLE_UNIT_FLAG_  单元标志（预留，当前版本未使用）
 * @tparam L1TileShape_       L1 层分块形状 GemmShape<M, N, K>，实例化时为 <128, 128, 128>
 * @tparam L0TileShape_       L0 层分块形状 GemmShape<M, N, K>，实例化时为 <128, 128, 128>
 * @tparam AType_             A 矩阵（Q）的元素类型与布局
 * @tparam BType_             B 矩阵（K）的元素类型与布局
 * @tparam CType_             C 矩阵（S，QK分数）的元素类型与布局
 * @tparam BiasType_          偏置类型（QK 阶段无偏置）
 * @tparam TileCopy_          各层 DataCopy 策略集合（含 GQA 多 pattern 拷贝）
 * @tparam TileMmad_          Cube MMAD 算子策略
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
    // ======================== 类型别名（CATLASS 框架约定） ========================
    using DispatchPolicy = MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;                       // 架构标签：Atlas A2 (C220)
    using L1TileShape = L1TileShape_;                                       // L1 分块 <M=128, N=128(dyn), K=128>
    using L0TileShape = L0TileShape_;                                       // L0 分块 <M=128, N=128, K=128>
    using ElementA = typename AType_::Element;                              // Q 矩阵元素类型（half/bf16）
    using LayoutA = typename AType_::Layout;                                // Q 矩阵布局（RowMajor）
    using ElementB = typename BType_::Element;                              // K 矩阵元素类型（half/bf16）
    using LayoutB = typename BType_::Layout;                                // K 矩阵布局（RowMajor）
    using ElementC = typename CType_::Element;                              // S 矩阵元素类型（float/half）
    using LayoutC = typename CType_::Layout;                                // S 矩阵布局（RowMajor）
    using TileMmad = TileMmad_;                                             // Cube MMAD 算子实例

    // 各层 DataCopy 算子类型
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;                    // GM→L1: Q 矩阵搬运（GQA 多 pattern）
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;                    // GM→L1: K 矩阵搬运
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;                    // L1→L0: Q 子块搬运
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;                    // L1→L0: K 子块搬运
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;                    // L0→GM: S 结果写回

    // 累加器类型（由 A、B 元素类型自动推导，half×half→float）
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // 各层内存中的布局
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;                    // Q 在 L1 中的布局
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;                    // K 在 L1 中的布局
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;                    // Q 在 L0A 中的布局（Cube 专用格式）
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;                    // K 在 L0B 中的布局（Cube 专用格式）
    using LayoutCInL0 = layout::zN;                                         // S 在 L0C 中的布局（zN 分形格式）

    // L1 对齐辅助
    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    // ======================== 编译期常量 ========================
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;              // Ping-Pong 阶段数 = 2
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);  // 静态 L1A 大小（Q 用动态大小）
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);  // 静态 L1B 大小（K 用动态大小）
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;                 // L0A 总大小（架构定义）
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;                 // L0B 总大小
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;                 // L0C 总大小
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;    // L0A 单个 Ping-Pong slot
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;    // L0B 单个 Ping-Pong slot
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;    // L0C 单个 Ping-Pong slot
    static constexpr uint32_t BLOCK_SIZE = 16;                              // Cube MMAD 行对齐单位（元素）
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;                       // Embedding 维分块大小（预留）
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;                     // 单元块堆叠数（预留）
    static constexpr uint32_t KV_BASE_BLOCK = 512;                          // KV 基础块大小（预留）
    static constexpr uint32_t KV_SPLIT_SIZE = 128;                          // KV 分割大小（预留）
    static constexpr uint32_t COORD_DIM0 = 0;                               // GemmCoord 第0维：M = rowNum（Q 行数 × head数）
    static constexpr uint32_t COORD_DIM1 = 1;                               // GemmCoord 第1维：N = stackSeqTile（KV 序列长度）
    static constexpr uint32_t COORD_DIM2 = 2;                               // GemmCoord 第2维：K = embed（头维度）

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /**
     * @brief 构造函数：分配 L1/L0 各层缓冲
     *
     * @param resource        架构资源对象（含 l1Buf, l0ABuf, l0BBuf, l0CBuf）
     * @param nDyn            N 维（KV序列）动态 tile 大小，受 L1 容量约束（典型值 128）
     * @param kDyn            K 维（embed）动态大小，对齐后值（典型值 128/256）
     * @param KVStackLen      单个 KV stack 的最大序列长度（分页时用于换算 blockTable 索引）
     * @param l1BufAddrStart  L1 缓冲区起始字节偏移（QK 通常从 0 开始）
     *
     * L1 缓冲布局（从 l1BufAddrStart 开始）：
     *   ┌──────────────────────────────┐ ← l1BufAddrStart
     *   │ L1A (Q 单缓冲)               │
     *   │ M × kDyn × sizeof(ElementA)  │  大小为 L1TileShape::M × kDyn (静态M=128)
     *   ├──────────────────────────────┤ ← L1TileShape::M × kDyn × sizeof(ElementA)
     *   │ L1B[0] (K Ping-Pong slot 0)  │
     *   │ nDyn × kDyn × sizeof(ElementB)│
     *   ├──────────────────────────────┤ ← 上一位置 + nDyn×kDyn×sizeof(B)
     *   │ L1B[1] (K Ping-Pong slot 1)  │
     *   │ nDyn × kDyn × sizeof(ElementB)│
     *   └──────────────────────────────┘
     */
    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {
        maxKVStackLen = KVStackLen;
        // 分配 L1 空间
        // L1A (Q) 从起始地址开始，单缓冲（全程复用）
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; i++) {
            // L1B[i] (K) 的 Ping-Pong slot：紧跟在 Q 缓冲之后
            // 每个 slot 大小为 nDyn × kDyn 个 ElementB
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
                L1TileShape::M * kDyn * sizeof(ElementA) + nDyn * kDyn * sizeof(ElementB) * i);
            // L0A/L0B/L0C Ping-Pong slot：从对应 L0 buffer 起始地址均分
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
     * @brief 将 Q 矩阵从 GM 加载到 L1（在 operator() 之前调用一次）
     *
     * 此函数执行 GQA-aware 的批量 DataCopy：
     * 在 GQA 模式下，同一 group 内的多个 Q head 共享同一个 KV head，
     * Q 矩阵在 GM 中按 [qSBlockSize, qHeads, embed]  interleaved 存储。
     *
     * @param gA                Q 矩阵 GM 张量
     * @param layoutA           Q 的布局描述符（RowMajor [rowNum, embed]）
     * @param rowNum            总行数 = qSBlockSize × singleGroupHeads
     * @param singleGroupHeads  [in/out] 当前 group 内的 Q head 数（= qNBlockSize）
     * @param qHeads            [in/out] 全局 Q head 总数（用于计算 GM 行步长）
     *
     * GQA DataCopy 参数说明：
     *   - tokenNumPerGroup = rowNum / singleGroupHeads = qSBlockSize
     *     每个 group 内的 token 数（序列块大小）
     *   - qHeads * embed : GM 侧单个 token 的行跨度（跨所有 Q head）
     *   - 单次搬运 tile 形状: [singleGroupHeads, embed]（一个 group 内的所有 head × embed）
     *   - rowNumRound : 行方向上对齐到 L1AAlignHelper::M_ALIGNED 的大小
     *
     * 加载完成后通过 EVENT_ID3 同步（SetFlag+WaitFlag 确保 DMA 完成），
     * 不使用 EVENT_ID0 以避免与 operator() 内其他事件冲突。
     */
    __aicore__ inline
    void loadQGM(
        AscendC::GlobalTensor<ElementA> gA,
        LayoutA layoutA,
        uint32_t rowNum, uint32_t &singleGroupHeads, uint32_t &qHeads)
    {
        uint32_t embed = layoutA.shape(1);
        uint32_t rowNumRound = RoundUp(rowNum, L1AAlignHelper::M_ALIGNED);
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;
        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
        copyGmToL1A(
            l1ATensor, gA,
            layoutAInL1, layoutSingleANd,
            tokenNumPerGroup, qHeads * embed, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }

    /**
     * @brief 分页模式：计算当前 stackSeqTile 范围内涉及的页块参数
     *
     * @param[in]  stackSeqTile      当前 KV stack tile 的序列长度
     * @param[in,out] blockStart     [in] 首页起始偏移（= blockSize - blockStartOffset）
     *                               [out] 首页实际可用长度
     * @param[out] blockEnd          末页实际可用长度
     * @param[out] curBlockTotalNum  需要加载的物理块总数
     * @param[in]  blockSize         每个物理块的 token 数
     *
     * 与 PV 版本逻辑相同：
     *   - stackSeqTile > blockStart 且 blockSize≠0：跨多页加载
     *   - 否则：单页（所有数据在当前物理块内）
     */
    __aicore__ inline
    void setBlockParam(uint32_t stackSeqTile, uint32_t &blockStart, uint32_t &blockEnd, uint32_t &curBlockTotalNum, uint32_t blockSize){
        if(stackSeqTile >= blockStart && blockSize != 0) {
            blockEnd = ((stackSeqTile - blockStart) % blockSize == 0) ? blockSize : (stackSeqTile - blockStart) % blockSize;
            curBlockTotalNum = (((stackSeqTile - blockStart) + blockSize - 1) / blockSize) + 1;
        } else {
            curBlockTotalNum = 1;
            blockStart = stackSeqTile;
            blockEnd = stackSeqTile + blockStartOffset;
        }
    }

    /**
     * @brief 非分页模式：设置当前 nL1 块的实际 N 维长度
     *
     * @param[out] actualShape   实际 GEMM 形状，N 维会被设置为当前 nL1 块长度
     * @param[in]  nL1Idx        当前 N 维（KV序列）块索引
     * @param[in]  nL1Loop       N 维总循环数
     * @param[in]  stackSeqTile  总序列长度
     *
     * 非尾块取 l1NDynamic，尾块取 stackSeqTile - nL1Idx*l1NDynamic（处理非对齐尾块）
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t nL1Idx, uint32_t nL1Loop, uint32_t stackSeqTile)
    {
        uint32_t nSplitSize = l1NDynamic;
        if (nL1Idx == nL1Loop - 1U) {
            nSplitSize = stackSeqTile - nL1Idx * l1NDynamic;
        }
        actualShape[COORD_DIM1] = nSplitSize;
    }

    /**
     * @brief 分页模式：计算当前页要加载的 K 子块长度（处理页内残留）
     *
     * @param[out] actualShape      实际 GEMM 形状，N 维设为 nowLen
     * @param[in]  blockStartOffset 当前物理块内偏移
     * @param[in]  l1NResDynamic    当前 nL1 块剩余需要加载的 token 数
     * @param[in]  kvL1Len          已累积加载到 L1B 的 token 数
     * @param[out] nowLen           当前页应加载的 token 数
     * @param[in]  blockSize        物理块大小（或末页可用长度 curBlockSize）
     *
     * nowLen = min(当前物理块内剩余, L1B slot 剩余容量)
     *
     * 取两者较小值是因为一次 K Ping-Pong 加载可能需要多个物理页拼接，
     * 也可能一个物理页跨越两个 nL1 块。
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& blockStartOffset, uint32_t& l1NResDynamic, uint32_t& kvL1Len, uint32_t& nowLen, uint32_t& blockSize)

    {
        nowLen = (blockSize - blockStartOffset < l1NResDynamic - kvL1Len) ?
                blockSize - blockStartOffset :
                l1NResDynamic - kvL1Len;
        actualShape[COORD_DIM1] = nowLen;
    }

    /**
     * @brief 非分页模式：计算 K 矩阵在 GM 中的起始偏移
     *
     * @param[out] kOffset   K 矩阵 GM 元素偏移
     * @param[in]  nIdx      当前 KV stack 索引
     * @param[in]  nowNIdx   当前 nL1 块索引（在 stack 内）
     * @param[in]  strideKV  K 矩阵行 stride（每行元素数 = kvHeads*embed）
     *
     * 非分页布局中：
     *   kOffset = nIdx * maxKVStackLen * strideKV + nowNIdx * l1NDynamic * strideKV
     *   即：跳到指定 stack 起点 + 跳过已处理的 nL1 块
     */
    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t nowNIdx, uint32_t strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV + nowNIdx * l1NDynamic * strideKV;
    }

    /**
     * @brief 分页模式：通过 blockTable 计算 K 矩阵 GM 偏移
     *
     * @param      gBlockTable  块表
     * @param[out] kOffset      计算得到的 GM 元素偏移
     * @param[in]  nowNIdx      当前逻辑块号
     * @param[in]  startOffset  块内起始偏移
     * @param[in]  strideKV     K 行 stride
     * @param[in]  blockSize    物理块大小
     *
     * 分页布局中：物理块号 = blockTable[nowNIdx]，kOffset = 物理块号*blockSize*strideKV + startOffset*strideKV
     */
    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t nowNIdx,
        uint32_t startOffset, uint32_t strideKV, uint32_t blockSize)
    {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        kOffset = blockTableId * blockSize * strideKV + startOffset * strideKV;
    }

    /**
     * @brief 重置分页块内偏移
     */
    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    /**
     * @brief 分页模式：更新块内偏移和块索引
     *
     * 与 PV 版本略有差异：
     *   - 只有当 blockStartOffset+nowLen==blockSize（正好到页边界）时才 curBlockIdx++
     *   - 若跨页则 blockStartOffset 不归零（保持页内偏移），curBlockIdx 也不自增
     *   这是因为 QK 的分页加载在 nL1 循环内，一个 nL1 块可能连续累积多页。
     */
    __aicore__ inline
    void updateBlockOffset(uint32_t nowLen, uint32_t &curBlockIdx, uint32_t blockSize){
        if(blockStartOffset + nowLen == blockSize){
            blockStartOffset = 0;
            curBlockIdx++;
        } else{
            blockStartOffset += nowLen;
        }
    }

    /**
     * @brief 执行 Q×K^T 矩阵乘
     *
     * 前置条件：loadQGM() 已在外部调用，Q 矩阵已加载到 L1A 并完成 EVENT_ID3 同步。
     *
     * 执行流程：
     *
     * 【阶段0：分页参数准备】
     *   若 PAGED_CACHE_FLAG_：调用 setBlockParam 计算当前 stack 涉及的物理块信息
     *
     * 【阶段1：nL1 循环（KV 序列分块，步长 l1NDynamic）】
     *   每个 nL1 块对应 K 矩阵的 [nL1Idx*l1NDynamic : (nL1Idx+1)*l1NDynamic] 行区间：
     *
     *   1a. 分页模式：
     *       Wait MTE1_MTE2(l1KvPingPongFlag) → 等待 L1B slot 空闲
     *       while (kvL1Len < l1NResDynamic)：逐页加载 K 并拼接到 L1B 当前 slot
     *         - 计算 nowLen（当前页贡献 token 数）
     *         - 查 blockTable 获取物理地址
     *         - copyGmToL1B 搬入 L1B[kvL1Len:]
     *         - kvL1Len += nowLen
     *         - updateBlockOffset 更新分页状态
     *       Set MTE2_MTE1(l1KvPingPongFlag) → 标记 K 加载完成
     *
     *   1b. 非分页模式：
     *       Wait MTE1_MTE2 → 单次 copyGmToL1B 加载连续 K → Set MTE2_MTE1
     *
     * 【阶段2：mL0 循环（Q 行分块，步长 L0TileShape::M=128）】
     *   Wait FIX_M(l0CPingPongFlag) → 等待上一 mL0 块 S 写回完成
     *
     *   【阶段3：kL0 循环（embed/K 维，步长 L0TileShape::K=128）】
     *     - L1A→L0A：搬运 Q 的 [mL0Actual × kL0Actual] 子块
     *     - L1B→L0B：搬运 K 的 [kL0Actual × nActual] 子块（首块需 Wait MTE2_MTE1 等 K 完整到 L1）
     *     - 末子块 Set MTE1_MTE2(l1KvPingPongFlag)：释放 L1B slot
     *     - Set/Wait MTE1_M(EVENT_ID0)：等 L0A/B 就绪
     *     - tileMmad(initMmad = (kL0Idx==0))：
     *         kL0Idx==0 时初始化累加器（新的 mL0 块开始），否则累加
     *     - 释放 L0A/B Ping-Pong slot，切换 flag
     *
     *   【阶段4：S 写回】
     *   Wait M_FIX(EVENT_ID0) → 等 Cube 完成
     *   copyL0CToGm：将 S 的 [mL0Actual × nActual] 子块写回 GM workspace
     *   Set FIX_M(l0CPingPongFlag) → 释放 L0C slot
     *
     * 【阶段5：切换 K Ping-Pong flag】
     *   l1KvPingPongFlag ^= 1，进入下一个 nL1 块
     *
     * @param gA           Q 矩阵 GM 张量（仅用于接口一致性，Q 已通过 loadQGM 预加载）
     * @param gB           K 矩阵 GM 张量
     * @param gC           S 输出 GM 张量（workspace 中 QK 分数）
     * @param gBlockTable  分页块表
     * @param layoutA      Q 的布局描述符
     * @param layoutB      K 的布局描述符
     * @param layoutC      S 的布局描述符
     * @param actualOriShape  原始 GEMM 形状 {M=rowNum, N=stackSeqTile, K=embed}
     * @param nIdx         当前 KV stack 索引
     * @param nLoop        KV stack 总数
     * @param blockSize    分页块大小
     * @param strideKV     K 矩阵行 stride
     *
     * 注意：operator() 返回后，调用方（mha_fwd_kvcache.cpp）会调用
     * Arch::CrossCoreSetFlag(qkReady) 通知 Vector 核 QK 完成。
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
                    uint32_t nIdx, uint32_t nLoop, uint32_t blockSize, uint32_t strideKV)
    {
        // 从 actualOriShape 解析三维（注意 QK 的维度映射与 PV 不同）：
        // M = rowNum（Q行×head数），N = stackSeqTile（KV序列），K = embed（头维度）
        uint32_t rowNum = actualOriShape[COORD_DIM0];
        uint32_t stackSeqTile = actualOriShape[COORD_DIM1];
        uint32_t embed = actualOriShape[COORD_DIM2];

        GemmCoord actualShape{rowNum, 0, embed};
        uint32_t gBOffset = 0;

        // Q 在 L1 中的布局描述符（行主序，[rowNum, embed]）
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);

        // 计算每个基础块包含多少个 nL1 tile（分页模式下用于逻辑块号换算）
        uint32_t tileNNumPerBaseBlock = blockSize / l1NDynamic;
        uint32_t nL1Loop = CeilDiv(stackSeqTile, l1NDynamic);   // N 维（KV序列）外层循环数
        uint32_t curBlockIdx =  0;
        uint32_t blockStart = 0;
        uint32_t blockEnd = 0;
        uint32_t curBlockTotalNum = 0;

        // ==================== 阶段 0: 分页参数预计算 ====================
        if constexpr (PAGED_CACHE_FLAG_){
            blockStart = blockSize - blockStartOffset;         // 首页起始偏移
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);
        }

        // ==================== 阶段 1: nL1 外层循环（KV 序列分块） ====================
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t mActual = actualShape.m();
            uint32_t kActual = actualShape.k();
            uint32_t nActual = actualShape.n();
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

            if constexpr (PAGED_CACHE_FLAG_){
                // ---------- 分页模式：逐页加载 K 并拼接到 L1B 当前 Ping-Pong slot ----------
                uint32_t l1NResDynamic = (nL1Idx < (nL1Loop-1)) ? l1NDynamic : (stackSeqTile - nL1Idx * l1NDynamic);
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(embed, l1NResDynamic);
                uint32_t kvL1Len = 0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                // 循环直到当前 nL1 块所需的 K 全部加载到 L1B
                while(kvL1Len < l1NResDynamic){
                    uint32_t nowLen = 0;
                    uint32_t curBlockSize = (curBlockIdx < (curBlockTotalNum-1)) ? blockSize : blockEnd;
                    uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;
                    getBlockShape(actualShape, blockStartOffset, l1NResDynamic, kvL1Len, nowLen, curBlockSize);
                    getKVOffset(gBlockTable, gBOffset, nowNIdx, blockStartOffset, strideKV, blockSize);
                    auto layoutBTile = layoutB.GetTileLayout(MakeCoord(embed, nowLen));
                    MatrixCoord l1BTileCoord{0, kvL1Len};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);
                    kvL1Len += nowLen;
                    updateBlockOffset(nowLen, curBlockIdx, blockSize);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = l1NResDynamic;
            } else {
                // ---------- 非分页模式：单次 DataCopy 加载 K 的 nL1 块 ----------
                getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
                getKVOffset(gBOffset, nIdx, nL1Idx, strideKV);
                mActual = actualShape.m();
                kActual = actualShape.k();
                nActual = actualShape.n();
                layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB[gBOffset], layoutBInL1, layoutBTile);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
            }

            // ==================== 阶段 2/3/4: mL0 + kL0 嵌套计算 ====================
            uint32_t mL0Loop = CeilDiv(mActual, L0TileShape::M);     // M 维（Q 行）循环数
            uint32_t kL0Loop = CeilDiv(kActual, L0TileShape::K);     // K 维（embed）循环数
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1U) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);
                // 等待 L0C→GM 写回完成，L0C Ping-Pong slot 可被覆写
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);

                    // ---- L1A→L0A：搬运 Q 的 [mL0Actual × kL0Actual] 子块 ----
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);

                    // ---- L1B→L0B：搬运 K 的 [kL0Actual × nActual] 子块 ----
                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    // 每个 mL0 块的首个 kL0 子块，需额外等待 K 的 GM→L1 完整完成
                    if ((mL0Idx == 0U) && (kL0Idx == 0U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);
                    // 最后一个 mL0 的最后一个 kL0 子块完成后，释放 L1B Ping-Pong slot
                    if ((mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                    }

                    // ---- 等待 L0A/L0B 都就绪后执行 Cube MMAD ----
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    // 每个 mL0 块的首个 kL0 子块初始化累加器，后续 kL0 子块累加
                    bool initMmad = (kL0Idx == 0U);
                    // MMAD 要求 M 维对齐到 BLOCK_SIZE(16)
                    uint32_t mL0Align = (mL0Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL0Align,
                        nActual,
                        kL0Actual,
                        initMmad);
                    // 释放 L0A 和 L0B Ping-Pong slot
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    // 切换 L0 Ping-Pong flag（0↔1）
                    l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                }
                // ---- L0C→GM：写回当前 (mL0, nL1) 块的 S 结果 ----
                // 等待 Cube 计算完成
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                // 释放 L0C Ping-Pong slot
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
            // 切换 L1B (K) Ping-Pong flag（0↔1）
            l1KvPingPongFlag = 1U - l1KvPingPongFlag;
        }
    }
protected:
    // ======================== L1/L0 缓冲成员 ========================
    AscendC::LocalTensor<ElementA> l1ATensor;                           // L1 中 Q 矩阵的单缓冲（loadQGM 加载后全程复用）
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];                   // L1 中 K 矩阵的 Ping-Pong 缓冲（STAGES=2 个 slot）
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];                   // L0A Ping-Pong 缓冲（Cube A/Q 输入）
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];                   // L0B Ping-Pong 缓冲（Cube B/K 输入）
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];         // L0C Ping-Pong 缓冲（Cube C/S 累加器/输出）

    // ======================== DataCopy/MMAD 算子实例 ========================
    TileMmad tileMmad;                                                  // Cube MMAD 算子
    CopyGmToL1A copyGmToL1A;                                            // GM→L1: Q 搬运算子（GQA-aware 多 pattern）
    CopyGmToL1B copyGmToL1B;                                            // GM→L1: K 搬运算子
    CopyL1ToL0A copyL1ToL0A;                                            // L1→L0: Q 搬运算子
    CopyL1ToL0B copyL1ToL0B;                                            // L1→L0: K 搬运算子
    CopyL0CToGm copyL0CToGm;                                            // L0→GM: S 写回算子

    // ======================== Ping-Pong 标志（运行时状态） ========================
    uint32_t l1KvPingPongFlag = 0;                                     // L1B (K) Ping-Pong 标志（0/1 切换）
    uint32_t l0CPingPongFlag = 0;                                      // L0C (S输出) Ping-Pong 标志（0/1 切换）
    uint32_t l0ABPingPongFlag = 0;                                     // L0A/L0B Ping-Pong 标志（0/1 切换）
    // 注：l0ABPingPongFlag 同时控制 L0A（flag+0）和 L0B（flag+2），
    // 使用 +2 偏移确保事件号不冲突（0/1 给 L0A，2/3 给 L0B）。

    // ======================== 动态分块大小 ========================
    uint32_t l1MDynamic = 0;                                           // L1 M 维动态大小（预留）
    uint32_t l1NDynamic = 0;                                           // L1 N 维动态大小（KV 序列分块步长）
    uint32_t l1KDynamic = 0;                                           // L1 K 维动态大小（embed 对齐后大小）

    // ======================== 分页 KV Cache 状态 ========================
    uint32_t blockStartOffset = 0;                                     // 当前物理块内偏移（跨页残留位置）
    uint32_t maxKVStackLen = 0;                                        // 单个 KV stack 最大序列长度
};

}

#endif
