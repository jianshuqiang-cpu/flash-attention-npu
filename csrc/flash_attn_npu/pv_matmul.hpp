/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

/**
 * ============================================================================
 * pv_matmul.hpp —— FlashAttention NPU 前向推理中 P×V 矩阵乘（Cube 核）
 * ============================================================================
 *
 * 【文件定位】
 *   本文件实现 CATLASS 框架中 BlockMmad 的模板特化，调度策略为 MmadAtlasA2FAIPVT，
 *   对应 FlashAttention 前向推理的第二个矩阵乘：
 *
 *       O = P × V
 *
 *   其中：
 *     - P (ElementA) : Softmax 输出的注意力权重矩阵，形状 [rowNum, stackSeqTile]，
 *                      由 Vector 核上的 online_softmax 计算完成后写入 GM；
 *     - V (ElementB) : Value 矩阵，形状 [stackSeqTile, embed]；
 *     - O/C(ElementC): 输出矩阵（中间结果 OTmp），形状 [rowNum, embed]，后续由
 *                      rescale_o.hpp 对各 KV stack 累加并做最终归一化。
 *
 * 【与 QK 矩阵乘的对偶性】
 *   QK 阶段（qk_matmul.hpp）：A=Q（单缓冲常驻 L1），B=K（Ping-Pong 加载）
 *   PV 阶段（本文件）       ：A=P（Ping-Pong 加载），B=V（单缓冲预加载到 L1）
 *   两者在循环结构、ping-pong 分配上互为镜像：
 *   ┌─────────┬──────────────────────┬──────────────────────┐
 *   │         │ QK (S = Q × K^T)     │ PV (O = P × V)        │
 *   ├─────────┼──────────────────────┼──────────────────────┤
 *   │ A 矩阵  │ Q 单缓冲常驻 L1      │ P Ping-Pong 动态加载  │
 *   │ B 矩阵  │ K Ping-Pong 动态加载 │ V 单缓冲预加载到 L1   │
 *   │ 跨核等待│ 无（QK 先于 softmax）│ CrossCoreWaitFlag    │
 *   │         │                      │   (等待 softmaxReady) │
 *   │ COORD   │ DIM1=N(stackSeqTile) │ DIM1=N(embed)        │
 *   │         │ DIM2=K(embed)        │ DIM2=K(stackSeqTile) │
 *   └─────────┴──────────────────────┴──────────────────────┘
 *
 * 【内存层级】
 *   GM(全局内存) ──MTE2──> L1(片上SRAM) ──MTE1──> L0A/L0B(Cube输入缓冲)
 *                                                          │
 *                                                    Cube(MMAD)
 *                                                          │
 *                                                    L0C(Cube输出) ──FIX──> GM
 *
 * 【L1 内存布局（与 QK 共享 L1，PV 的 L1 起始地址偏移为 L1_QK_SIZE）】
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │ L1A[0]  (P Ping-Pong slot 0)   : M × kDyn 个 ElementA              │
 *   │ L1A[1]  (P Ping-Pong slot 1)   : M × kDyn 个 ElementA              │
 *   │ L1B     (V 单缓冲)             : stackSeqTile × embed 个 ElementB  │
 *   └─────────────────────────────────────────────────────────────────────┘
 *   P 是动态加载的（每个 kL1 块切换 Ping-Pong slot），V 在 operator() 开头一次性加载。
 *
 * 【核心循环嵌套】
 *   nL1(embed/N维) → mL1(rowNum/M维) → kL1(seq/K维, P Ping-Pong) → kL0(L0子块)
 *
 * 【跨核/事件同步】
 *   - EVENT_ID4 (MTE1_MTE2): 管道排空信号，operator() 开始时等待上次排空
 *   - EVENT_ID0 (MTE2_MTE1): V 加载完成信号
 *   - CrossCoreWaitFlag(softmaxFlag): 等待 Vector 核完成 Softmax，P 矩阵就绪
 *   - l1PPingPongFlag  : L1A Ping-Pong（P 块 GM→L1 同步）
 *   - l0ABPingPongFlag : L0A/L0B Ping-Pong（L1→L0 搬运同步，L0A 用 flag+0，L0B 用 flag+2）
 *   - l0CPingPongFlag  : L0C Ping-Pong（Cube→GM 输出同步）
 *
 * 【分页 KV Cache 支持】
 *   通过模板参数 PAGED_CACHE_FLAG_ 编译期分支：
 *   - false: V 连续存储，单次 DataCopy 即可搬运到 L1；
 *   - true : V 按 blockSize 分页存储，通过 blockTable 间接寻址，逐页搬运并拼接
 *           到 L1B 连续区域。
 * ============================================================================
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

namespace Catlass::Gemm::Block {

/**
 * @brief BlockMmad 对 MmadAtlasA2FAIPVT 调度策略的特化：FlashAttention PV 矩阵乘
 *
 * @tparam PAGED_CACHE_FLAG_  是否启用分页 KV Cache 寻址
 * @tparam ENABLE_UNIT_FLAG_  单元标志（预留，当前版本未使用）
 * @tparam L1TileShape_       L1 层分块形状 GemmShape<M, N, K>，实例化时为 <128, 128, 256>
 * @tparam L0TileShape_       L0 层分块形状 GemmShape<M, N, K>，实例化时为 <128, 128, 128>
 * @tparam AType_             A 矩阵（P）的元素类型与布局
 * @tparam BType_             B 矩阵（V）的元素类型与布局
 * @tparam CType_             C 矩阵（OTmp）的元素类型与布局
 * @tparam BiasType_          偏置类型（PV 阶段无偏置）
 * @tparam TileCopy_          各层 DataCopy 策略集合
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
    // ======================== 类型别名（CATLASS 框架约定） ========================
    using DispatchPolicy = MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;                       // 架构标签：Atlas A2 (C220)
    using L1TileShape = L1TileShape_;                                       // L1 分块 <M=128, N=128, K=256>
    using L0TileShape = L0TileShape_;                                       // L0 分块 <M=128, N=128, K=128>
    using ElementA = typename AType_::Element;                              // P 矩阵元素类型（half/bf16）
    using LayoutA = typename AType_::Layout;                                // P 矩阵布局（RowMajor）
    using ElementB = typename BType_::Element;                              // V 矩阵元素类型（half/bf16）
    using LayoutB = typename BType_::Layout;                                // V 矩阵布局（RowMajor）
    using ElementC = typename CType_::Element;                              // O/OTmp 元素类型（float/half）
    using LayoutC = typename CType_::Layout;                                // O 矩阵布局（RowMajor）
    using TileMmad = TileMmad_;                                             // Cube MMAD 算子实例

    // 各层 DataCopy 算子类型
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;                    // GM→L1: P 矩阵搬运
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;                    // GM→L1: V 矩阵搬运
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;                    // L1→L0: P 子块搬运
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;                    // L1→L0: V 子块搬运
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;                    // L0→GM: O 结果写回

    // 累加器类型（由 A、B 元素类型自动推导，half×half→float）
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // 各层内存中的布局
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;                    // P 在 L1 中的布局
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;                    // V 在 L1 中的布局
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;                    // P 在 L0A 中的布局（Cube 专用格式）
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;                    // V 在 L0B 中的布局（Cube 专用格式）
    using LayoutCInL0 = layout::zN;                                         // O 在 L0C 中的布局（zN 分形格式）

    // L1 对齐辅助
    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    // ======================== 编译期常量 ========================
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;              // Ping-Pong 阶段数 = 2
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);  // 单个 L1A slot 大小
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);  // L1B 大小（未直接使用，V 大小动态）
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
    static constexpr uint32_t LOAB_BLOCK = 1;                               // LOAB 块大小（预留）
    static constexpr uint32_t COORD_DIM0 = 0;                               // GemmCoord 第0维：M = rowNum（Q 行数）
    static constexpr uint32_t COORD_DIM1 = 1;                               // GemmCoord 第1维：N = embed（Embedding 维度）
    static constexpr uint32_t COORD_DIM2 = 2;                               // GemmCoord 第2维：K = stackSeqTile（KV 序列长度）

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /**
     * @brief 构造函数：分配 L1/L0 各层 Ping-Pong 缓冲
     *
     * @param resource        架构资源对象（含 l1Buf, l0ABuf, l0BBuf, l0CBuf）
     * @param nDyn            N 维（embed）动态大小
     * @param kDyn            K 维（seq）动态分块大小 = nDyn*kDynQK / M
     * @param KVStackLen      单个 KV stack 的最大序列长度（分页时用于换算 blockTable 索引）
     * @param l1BufAddrStart  L1 缓冲区起始字节偏移（QK 阶段已占用前 L1_QK_SIZE 字节）
     *
     * L1 缓冲布局（从 l1BufAddrStart 开始）：
     *   ┌─────────────────────┐ ← l1BufAddrStart
     *   │ L1A[0]: M*kDyn 元素 │  (P Ping-Pong slot 0)
     *   ├─────────────────────┤
     *   │ L1A[1]: M*kDyn 元素 │  (P Ping-Pong slot 1)
     *   ├─────────────────────┤ ← l1BufAddrStart + M*kDyn*sizeof(ElementA)*STAGES
     *   │ L1B: V 矩阵缓冲      │  (V 单缓冲，连续存放整个 stackSeqTile×embed)
     *   └─────────────────────┘
     */
    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource,uint32_t nDyn, uint32_t kDyn, uint32_t KVStackLen = 512, uint32_t l1BufAddrStart = 0)
    {
        maxKVStackLen = KVStackLen;
        // 分配 L1 空间
        // L1B (V) 位于 P Ping-Pong 缓冲之后
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart +
            L1TileShape::M * kDyn * sizeof(ElementA) * STAGES);
        for (uint32_t i = 0; i < STAGES; i++) {
            // L1A[i] (P) 的 Ping-Pong slot：每个 slot 大小为 M*kDyn 个 ElementA
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart +
                L1TileShape::M * kDyn * sizeof(ElementA) * i);
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
     * @brief 重置分页块内偏移（用于多个 KV stack 连续调用时的状态复位）
     */
    __aicore__ inline
    void resetBlockStart(){
        blockStartOffset = 0;
    }

    /**
     * @brief 设置当前块的实际 K 维长度
     * @param[out] actualShape  实际 GEMM 形状，K 维会被设置为 nowLen
     * @param[in]  nowLen       当前要处理的 K 维长度
     */
    __aicore__ inline
    void getBlockShape(GemmCoord &actualShape, uint32_t& nowLen)
    {
        actualShape[COORD_DIM2] = nowLen;
    }

    /**
     * @brief 非分页模式：计算 V 矩阵在 GM 中的起始偏移
     *
     * @param[out] kOffset   V 矩阵起始字节偏移对应的元素索引
     * @param[in]  nIdx      当前 KV stack 索引
     * @param[in]  strideKV  V 矩阵行 stride（每行元素个数）
     *
     * 连续存储布局中，第 nIdx 个 stack 的 V 起始位置 = nIdx * maxKVStackLen * strideKV
     */
    __aicore__ inline
    void getKVOffset(uint32_t &kOffset, uint32_t nIdx, uint32_t &strideKV)
    {
        kOffset = nIdx * maxKVStackLen * strideKV;
    }

    /**
     * @brief 分页模式：通过 blockTable 查找物理块号并计算 V 矩阵在 GM 中的偏移
     *
     * @param      gBlockTable       块表（block_table），存储每个逻辑块对应的物理块号
     * @param[out] kOffset           计算得到的 V 数据 GM 元素偏移
     * @param[in]  blockStartOffset  当前在物理块内的偏移（处理跨块残留）
     * @param[in]  nowNIdx           当前逻辑块号（= nIdx*maxKVStackLen/blockSize + curBlockIdx）
     * @param[in]  strideKV          V 矩阵行 stride
     * @param[in]  blockSize         每个物理块包含的 token 数（如 128）
     *
     * 分页存储布局中：
     *   物理块号 = blockTable[nowNIdx]
     *   起始偏移 = 物理块号 * blockSize * strideKV + blockStartOffset * strideKV
     */
    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t blockStartOffset,
        uint32_t nowNIdx, uint32_t &strideKV, uint32_t &blockSize)
    {
        uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
        kOffset = blockTableId * blockSize * strideKV + blockStartOffset * strideKV;
    }

    /**
     * @brief 分页模式：计算当前 stackSeqTile 范围内涉及的页块参数
     *
     * @param[in]  stackSeqTile      当前 KV stack tile 的序列长度
     * @param[in,out] blockStart     [in] 首页起始偏移（= blockSize - blockStartOffset 处理跨页残留）
     *                               [out] 首页实际可用长度
     * @param[out] blockEnd          末页实际可用长度
     * @param[out] curBlockTotalNum  需要加载的物理块总数
     * @param[in]  blockSize         每个物理块的 token 数
     *
     * 分两种情况：
     *   1. stackSeqTile > blockStart（需要跨多页）：
     *      - 首页可能是残块（blockStart 个 token），末页也可能是残块（blockEnd 个 token）
     *      - 总块数 = ⌈(stackSeqTile - blockStart) / blockSize⌉ + 1
     *   2. stackSeqTile ≤ blockStart（所有数据在当前页内）：
     *      - 仅需 1 个块，长度 = stackSeqTile + blockStartOffset（含之前残留）
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
     * @brief 分页模式：更新块内偏移和块索引（在加载完一个块后调用）
     *
     * @param[in]  nowLen        刚加载的块长度
     * @param[out] curBlockIdx   当前块索引（自增）
     * @param[in]  blockSize     物理块大小
     *
     * 若加上 nowLen 后正好达到 blockSize，说明已到页边界，blockStartOffset 清零；
     * 否则累加 nowLen 表示仍在同一页内。
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
     * @brief 执行 P×V 矩阵乘
     *
     * 执行流程分两大阶段：
     *
     * 【阶段1：V 矩阵预加载到 L1】（与 Cube 计算无关，可与其他工作重叠）
     *   1. 等待 EVENT_ID4 确认上次 PV 调用的 MTE1/MTE2 管道已排空
     *   2. 若 PAGED_CACHE_FLAG：逐页查找 blockTable 并搬运到 L1B 连续区域
     *      否则：单次 DataCopy 将连续 V 数据搬运到 L1B
     *   3. SetFlag/WaitFlag EVENT_ID0 等待 V 搬运完成
     *
     * 【阶段2：跨核同步——等待 Softmax 完成】
     *   4. CrossCoreWaitFlag(softmaxFlag)：阻塞等待 Vector 核 online_softmax 完成，
     *      P 矩阵已写入 GM 的 workspace 区域
     *
     * 【阶段3：三层嵌套 GEMM 计算】
     *   nL1(embed分块, 步长128)
     *     mL1(Q行分块, 步长128)
     *       FIX_M 同步：等待上一 mL1 块的 L0C→GM 写回完成
     *       kL1(序列分块, 步长l1KDynamic)
     *         MTE1_MTE2 同步：等待 L1A Ping-Pong slot 空闲
     *         GM→L1A：搬运 P 的 [mL1Actual × kL1Actual] 子块
     *         kL0(L0子块, 步长L0TileShape::K=128)
     *           M_MTE1 同步：等待 L0A/L0B Ping-Pong slot 空闲
     *           L1A→L0A、L1B→L0B：搬运子块到 Cube 输入缓冲
     *           SetFlag/WaitFlag(EVENT_ID0, MTE1_M)：等待 L0A/L0B 都就绪
     *           tileMmad()：Cube 矩阵乘累加
     *           释放 L0A/L0B Ping-Pong slot
     *       M_FIX 同步：等待 Cube 计算完成
     *       L0C→GM：写回 O/OTmp 子块
     *
     * 【阶段4：尾部信号】
     *   5. SetFlag(EVENT_ID4)：标记 PV 管道已排空，供下次 PV 调用使用
     *
     * @param gA           P 矩阵（Softmax 输出）GM 张量
     * @param gB           V 矩阵 GM 张量
     * @param gC           O/OTmp 输出 GM 张量
     * @param gBlockTable  分页块表（非分页模式传空 tensor）
     * @param layoutA      P 的布局描述符
     * @param layoutB      V 的布局描述符
     * @param layoutC      O 的布局描述符
     * @param actualOriShape  原始 GEMM 形状 {M=rowNum, N=embed, K=stackSeqTile}
     * @param nIdx         当前 KV stack 索引（输入/输出，分页模式下可能被修改）
     * @param nLoop        KV stack 总数
     * @param blockSize    分页块大小
     * @param kvSeqlen     有效 KV 序列长度
     * @param strideKV     V 矩阵行 stride
     * @param blockStackNum 块堆叠数
     * @param softmaxFlag  跨核同步信号（SOFTMAX_READY_ID=2），PV 开始计算前等待此信号
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
        // 从 actualOriShape 解析三维：M=行(Q行数), N=列(embedding), K=归约(KV序列长度)
        uint32_t rowNum = actualOriShape[COORD_DIM0];
        uint32_t embed = actualOriShape[COORD_DIM1];
        uint32_t stackSeqTile = actualOriShape[COORD_DIM2];
        GemmCoord actualShape{rowNum, embed, 0};
        uint32_t gBOffset = 0;

        // V 在 L1 中的布局描述符（行主序，[stackSeqTile, embed]）
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);

        // ==================== 阶段 1: V 矩阵 GM→L1 预加载 ====================
        // 等待上一次 PV 调用的 MTE1/MTE2 管道排空（EVENT_ID4 为 PV 专用排空信号）
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);

        if constexpr (PAGED_CACHE_FLAG_) {
            // ---------- 分页模式：逐页加载 V 到 L1B 的连续区域 ----------
            uint32_t curBlockIdx =  0;
            uint32_t blockStart = blockSize - blockStartOffset;  // 首页起始偏移（处理上次跨页残留）
            uint32_t blockEnd = 0;
            uint32_t curBlockTotalNum = 0;
            setBlockParam(stackSeqTile, blockStart, blockEnd, curBlockTotalNum, blockSize);
            while(curBlockIdx < curBlockTotalNum) {
                // 计算当前页要加载的 token 数：非末页取 blockSize-blockStartOffset，末页取 blockEnd-blockStartOffset
                uint32_t nowLen = (curBlockIdx < (curBlockTotalNum-1)) ? (blockSize - blockStartOffset) : (blockEnd - blockStartOffset);
                // 将 stack 索引和块索引组合为全局逻辑块号
                uint32_t nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx;
                getBlockShape(actualShape, nowLen);
                getKVOffset(gBlockTable, gBOffset, blockStartOffset, nowNIdx, strideKV, blockSize);
                // 当前页的 tile 布局和 L1B 写入起始位置
                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
                uint32_t curBlockSize = (curBlockIdx > 0) ? ((curBlockIdx - 1) * blockSize + blockStart) : 0;
                MatrixCoord l1BTileCoord{curBlockSize, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
                copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);
                updateBlockOffset(nowLen, curBlockIdx, blockSize);
            }
        } else {
            // ---------- 非分页模式：单次 DataCopy 加载 V ----------
            getBlockShape(actualShape, stackSeqTile);
            getKVOffset(gBOffset, nIdx, strideKV);
            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
            copyGmToL1B(l1BTensor, gB[gBOffset], layoutBInL1, layoutBTile);
        }

        // 等待 V 从 GM→L1 搬运完成（MTE2 管道完成信号）
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        // ==================== 阶段 2: 跨核同步（等待 Softmax 完成） ====================
        // Cube 核阻塞，直到 Vector 核完成 online_softmax 并通过 CrossCoreSetFlag 通知
        // 此时 P 矩阵已写入 GM workspace，可以安全加载
        Arch::CrossCoreWaitFlag(softmaxFlag);

        // ==================== 阶段 3: P×V 三层嵌套 GEMM 计算 ====================
        // 计算各维循环次数
        uint32_t mL1Loop = CeilDiv(rowNum, L1TileShape::M);     // M 维（Q 行）外层循环数
        uint32_t kL1Loop = CeilDiv(stackSeqTile, l1KDynamic);   // K 维（序列）分块循环数
        uint32_t nL1Loop = CeilDiv(embed, L0TileShape::N);      // N 维（embedding）分块循环数

        // 最外层：N 维（embedding 列）分块
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; nL1Idx++) {
            uint32_t nL1Actual = (nL1Idx < nL1Loop - 1U) ? L0TileShape::N : (embed - nL1Idx * L0TileShape::N);
            // 中间层：M 维（Q 行）分块
            for (uint32_t mL1Idx = 0; mL1Idx < mL1Loop; mL1Idx++) {
                uint32_t mL1Actual = (mL1Idx < mL1Loop - 1U) ? L1TileShape::M : (rowNum - mL1Idx * L1TileShape::M);
                // 等待 L0C→GM 写回完成，L0C Ping-Pong slot 可被覆写
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                // 内层：K 维（序列长度）分块（P 矩阵 GM→L1 的 Ping-Pong 粒度）
                for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
                    uint32_t kL1Actual = (kL1Idx < kL1Loop - 1U) ? l1KDynamic : (stackSeqTile - kL1Idx * l1KDynamic);
                    // 等待 L1A Ping-Pong slot 空闲（前一个 kL1 块的 L1→L0 搬运已完成）
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                    // GM→L1A：搬运 P 矩阵的 [mL1Actual × kL1Actual] 子块
                    MatrixCoord gmATileCoord{mL1Idx * L1TileShape::M, kL1Idx * l1KDynamic};
                    auto gmTileA = gA[layoutA.GetOffset(gmATileCoord)];
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mL1Actual, kL1Actual));
                    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mL1Actual, kL1Actual);
                    copyGmToL1A(l1ATensor[l1PPingPongFlag], gmTileA, layoutAInL1, layoutTileA);
                    // 通知：L1A 当前 slot 的 P 数据已可用
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);

                    // L0 内层：将 L1 中的 P/V 子块搬运到 L0 并执行 Cube MMAD
                    uint32_t kL0Loop = CeilDiv(kL1Actual, L0TileShape::K);
                    for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                        uint32_t kL0Actual =
                            (kL0Idx < kL0Loop - 1U) ? L0TileShape::K : (kL1Actual - kL0Idx * L0TileShape::K);
                        // ---- L1A→L0A：搬运 P 的 [mL1Actual × kL0Actual] 子块 ----
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL1Actual, kL0Actual);
                        MatrixCoord l1ATileCoord{0, kL0Idx * L0TileShape::K};
                        auto l1ATile = l1ATensor[l1PPingPongFlag][layoutAInL1.GetOffset(l1ATileCoord)];

                        // 等待 L0A Ping-Pong slot 空闲
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                        // kL0Idx==0 时需等待 L1A 的 GM→L1 搬运完整完成
                        if (kL0Idx == 0U) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);
                        }
                        copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);
                        // kL0Idx 为最后一个子块时，释放 L1A Ping-Pong slot（L1→L0 全部完成）
                        if (kL0Idx == kL0Loop - 1U) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                        }

                        // ---- L1B→L0B：搬运 V 的 [kL0Actual × nL1Actual] 子块 ----
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nL1Actual);
                        MatrixCoord l1BTileCoord{kL1Idx * l1KDynamic + kL0Idx * L0TileShape::K, L0TileShape::N * nL1Idx};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];

                        // 等待 L0B Ping-Pong slot 空闲（L0B 使用 flag+2 以与 L0A 独立）
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                        copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);

                        // ---- 等待 L0A/L0B 都就绪后执行 Cube MMAD ----
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        // 仅在每个 (mL1, nL1) 的第一个 k 块初始化累加器（后续累加）
                        bool initMmad = (kL1Idx == 0U) && (kL0Idx == 0U);
                        // MMAD 要求 M 维对齐到 BLOCK_SIZE(16)
                        uint32_t mL0Align = (mL1Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                        tileMmad(l0CTensor[l0CPingPongFlag],
                            l0ATensor[l0ABPingPongFlag],
                            l0BTensor[l0ABPingPongFlag],
                            mL0Align,
                            nL1Actual,
                            kL0Actual,
                            initMmad);
                        // 释放 L0A 和 L0B Ping-Pong slot
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                        // 切换 L0 Ping-Pong flag（0↔1）
                        l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                    }
                    // 切换 L1A Ping-Pong flag（0↔1）
                    l1PPingPongFlag = 1U - l1PPingPongFlag;
                }
                // ---- L0C→GM：写回当前 (mL1, nL1) 块的 O/OTmp 结果 ----
                // 等待 Cube 计算完成
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                MatrixCoord gmCTileCoord{mL1Idx * L0TileShape::M, L0TileShape::N * nL1Idx};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL1Actual, nL1Actual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL1Actual, nL1Actual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                // 释放 L0C Ping-Pong slot
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
        }
        // ==================== 阶段 4: 管道排空信号 ====================
        // 标记 PV MTE1/MTE2 管道已排空，供下一次 PV 调用（或下一个 KV stack）使用
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    }

protected:
    // ======================== L1/L0 缓冲成员 ========================
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];          // L1 中 P 矩阵的 Ping-Pong 缓冲（STAGES=2 个 slot）
    AscendC::LocalTensor<ElementB> l1BTensor;                  // L1 中 V 矩阵的单缓冲（预加载后全程复用）
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];          // L0A Ping-Pong 缓冲（Cube A 矩阵输入）
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];          // L0B Ping-Pong 缓冲（Cube B 矩阵输入）
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES]; // L0C Ping-Pong 缓冲（Cube C 累加器/输出）

    // ======================== DataCopy/MMAD 算子实例 ========================
    TileMmad tileMmad;                                         // Cube MMAD 算子
    CopyGmToL1A copyGmToL1A;                                   // GM→L1: P 搬运算子
    CopyGmToL1B copyGmToL1B;                                   // GM→L1: V 搬运算子
    CopyL1ToL0A copyL1ToL0A;                                   // L1→L0: P 搬运算子
    CopyL1ToL0B copyL1ToL0B;                                   // L1→L0: V 搬运算子
    CopyL0CToGm copyL0CToGm;                                   // L0→GM: O 写回算子

    // ======================== Ping-Pong 标志（运行时状态） ========================
    uint32_t l1PPingPongFlag = 0;                             // L1A (P) Ping-Pong 标志（0/1 切换）
    uint32_t l0CPingPongFlag = 0;                             // L0C (输出) Ping-Pong 标志（0/1 切换）
    uint32_t l0ABPingPongFlag = 0;                            // L0A/L0B Ping-Pong 标志（0/1 切换）
    // 注：l0ABPingPongFlag 同时控制 L0A（flag+0）和 L0B（flag+2），
    // 使用 +2 偏移确保 L0A 和 L0B 的事件号不冲突（0/1 给 L0A，2/3 给 L0B）。

    // ======================== 动态分块大小 ========================
    uint32_t l1MDynamic = 0;                                  // L1 M 维动态大小（预留，当前固定为 L1TileShape::M）
    uint32_t l1NDynamic = 0;                                  // L1 N 维动态大小（embed 分块）
    uint32_t l1KDynamic = 0;                                  // L1 K 维动态大小（序列分块步长）

    // ======================== 分页 KV Cache 状态 ========================
    uint32_t blockStartOffset = 0;                            // 当前物理块内偏移（跨页残留位置）
    uint32_t maxKVStackLen = 0;                               // 单个 KV stack 最大序列长度（换算 blockTable 索引用）
};

}

#endif
