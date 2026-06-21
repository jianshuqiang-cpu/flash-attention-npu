/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

/**
 * ============================================================================
 * online_softmax_low_prec.hpp —— 半精度(FP16)版 Online Softmax Vector Epilogue
 * ============================================================================
 *
 * 【文件定位】
 *   本文件是 CATLASS BlockEpilogue 的偏特化版本，中间计算精度 SM_DTYPE_=half（FP16）。
 *   对应高精度版本 online_softmax.hpp（SM_DTYPE_=float，FP32）。
 *
 *   在 FlashAttention 前向推理 pipeline 中，本 epilogue 运行在 Vector 核上，位于
 *   Cube 核完成 Q*K^T 之后、Cube 核开始 P*V 之前，由 softmaxReady 跨核事件同步。
 *
 * 【核心算法 —— Online Softmax（分块安全 softmax）】
 *   FlashAttention 将 K/V 序列切成若干 KV stack tile（沿 N 维），逐 tile 流式计算。
 *   设当前为第 t 个 tile，在线 softmax 维护两个跨 tile 累积量：
 *     m_t = max( m_{t-1}, rowmax(S_t) )           // 全局行最大值
 *     l_t = exp(m_{t-1} - m_t) * l_{t-1} + rowsum(exp(S_t - m_t))  // 全局归一化分母
 *   并输出当前 tile 的概率矩阵：
 *     P_t = exp(S_t - m_t)
 *   P_t 被写回 GM 供后续 P*V Cube matmul 使用；m_t/l_t 保留在 UB，
 *   到最后一个 tile 后由 rescale_o epilogue 统一做 O 的归一化加权求和。
 *
 *   S 已由 Cube 侧乘好 scaleValue(=1/sqrt(d))，但本 epilogue 仍要再 ScaleS 一次？
 *   —— 不：Cube 输出的 S=Q*K^T 仍需在 Vector 侧乘 scaleValue（Cube 侧做的是 MMAD，
 *   scaleValue 在 Vector 侧通过 Muls 融合执行，保持与高精度版本一致的入口点）。
 *
 * 【与高精度版本(online_softmax.hpp)的关键差异】
 *   1. 所有中间张量(lm/hm/gm/dm/ll/gl/tv)均使用 half（FP16），节省 UB；
 *   2. UB 中 S 区容量 16384 half = 32KB（高版本 8192 float = 32KB）；
 *   3. 增加独立的 computeUbTensor（scale/mask 后 S 从此计算），lsUbTensor 仅作
 *      GM->UB 搬运着陆区；高版本直接在 lsUbTensor[sUbOffset] 原地计算；
 *   4. maskUbTensor16（UpCast 后的 half mask）复用 LS 区的 0 偏移（S 已搬到 compute）；
 *   5. 行规约用 Add + WholeReduceSum（两级）；高版本用 BlockReduceSum 三级级联；
 *   6. 无 256 列特化路径（只有 512 列特化 + 通用尾块路径）；
 *   7. 不支持 softcap（softcapValue_ 参数被注释）；
 *   8. 无 preLoad 预取流水线（row 循环内 DMA 与计算串行，无 DMA/Compute overlap）；
 *   9. P 输出直接 DataCopy<half> 即可（无需 float->half/bf16 Cast）；
 *  10. isLastNoMaskStackTile 参数接收但未使用（流水排空同步简化）。
 *
 * 【UB 内存布局（字节偏移，单 Vector 核 ~192KB UB）】
 *
 *     0KB ┌──────────────────────────────────────────┐
 *         │ lsUbTensor (S原始输入, half)              │ 32KB (2 × UB_UINT8_BLOCK_SIZE)
 *         │ → maskUbTensor16 在 mask 阶段复用此区域   │    (S已搬到compute,可安全复用)
 *    32KB ├──────────────────────────────────────────┤
 *         │ computeUbTensor (scale+mask后的S, half)  │ 32KB (pingpong 双缓冲, 各 16KB?)
 *    64KB ├──────────────────────────────────────────┤
 *         │ lpUbTensor (P=exp(S-m) 输出, ElementOut) │ ~16KB
 *         │                                          │
 *   160KB ├──────────────────────────────────────────┤
 *         │ tvUbTensor (Brcb 广播/临时, half)        │ 8KB (8 × UB_UINT8_VECTOR_SIZE)
 *  168KB  │ lm(1KB) │ hm(1KB) │ gm(1KB) │            │
 *         │ ll(1KB) │ gl(1KB) │ dm(1KB) │            │
 *   176KB ├──────────────────────────────────────────┤
 *         │ maskUbTensor (int8 mask, GM→UB)          │ 16KB (1 × UB_UINT8_BLOCK_SIZE)
 *   192KB └──────────────────────────────────────────┘
 *
 * 【两个 operator() 重载】
 *   重载1（无 mask）：完全在对角线下方的 KV block 或 NO_MASK 编译期路径使用；
 *                    调用者在外部已 CrossCoreWaitFlag(qkReady)。
 *   重载2（带 causal mask）：跨越因果对角线的 KV block 使用；内部等待 qkReady，
 *                    加载 mask→UpCast→ApplyMask(-6e4)→softmax。
 *
 * 【当前状态】
 *   本文件作为低精度预留路径存在，flash_api.cpp 中所有 kernel 实例化均使用
 *   IntermCalcPrec=float（即高版本 online_softmax.hpp），本 half 版本未被实例化。
 * ============================================================================
 */
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fa_block.h"

namespace Catlass::Epilogue::Block {

/**
 * BlockEpilogue 偏特化：EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>
 *
 * 模板参数：
 *   OutputType_ : P 输出类型（GemmType<ElementP, RowMajor>，通常为 half/bf16）
 *   InputType_  : S 输入类型（GemmType<half, RowMajor>，Q*K^T 的 FP16 结果）
 *   MaskType_   : mask 类型（GemmType<int8, RowMajor>，预生成的上三角 mask）
 *   LSE_MODE_   : LSE 输出模式（NONE / OUT_ONLY）
 *
 * 注意：第二个模板参数 SM_DTYPE_=half 决定了走本 low_prec 偏特化
 */
template <
    class OutputType_,
    class InputType_,
    class MaskType_,
    LseModeT LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>,
    OutputType_,
    InputType_,
    MaskType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;

    // ----------------------------- 硬件常量 -----------------------------
    // Ascend C Vector 指令的基本处理宽度：
    //   一个 Vector 指令一次处理 256 字节 = 128 个 half = 64 个 float
    //   一个 Block（DMA 对齐单位）= 32 字节 = 16 个 half = 8 个 float
    //   一个 Repeat（V 指令重复步长）= 256 字节（连续8个block）
    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;      // 一个 block 包含 8 个 float
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;    // 一个 vector 包含 64 个 float(256B)
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;    // 一个 vector 包含 128 个 half(256B)
    static constexpr uint32_t BLOCK_SIZE = 16;           // 一个 block 包含 16 个 half
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;   // 一个 vector 的 uint8 元素数
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;   // mask 区大小(16KB,以uint8计)
    static constexpr uint32_t VECTOR_SIZE = 128;         // 默认 vector 大小(half 元素数)
    // S 在 UB 中单 pingpong 槽位的最大元素数：16384 half = 32KB
    // 对应高版本 float 的 8192（同样32KB），half 版本元素数翻倍
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;

    // 行规约临时缓冲区大小（high prec 版本用于多级 BlockReduceSum）
    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    // SetBlockReduceMask 专用的掩码位宽常量（当前 low prec 版本未调用该函数，保留）
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    // 每个 Vector 子核一次行循环最多处理的行数
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    // RowsumSPECTILE512 专用：一次性处理 4 个 half-vector（共 512 列）时的分块索引
    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;

    /**
     * 构造函数：分配 UB 张量，初始化 scaleValue
     *
     * UB 分区偏移（按字节计，GetBufferByByte 接口）：
     *   LS   = 0 * 16KB            : S 原始 GM->UB 着陆区
     *   COMP = 2 * 16KB = 32KB     : S 计算区（scale/mask/exp 均在此）
     *   LP   = 4 * 16KB = 64KB     : P 输出区（写给 GM）
     *   TV   = 10 * 16KB = 160KB   : Brcb 广播临时区
     *   LM   = TV + 8 vector       : 局部行 max（当前 tile）
     *   HM   = TV + 9 vector       : 新的全局行 max（m_new）
     *   GM   = TV + 10 vector      : 旧的全局行 max（m_old）
     *   LL   = TV + 11 vector      : 局部行 sum（当前 tile 的 l_local）
     *   GL   = TV + 12 vector      : 全局行 sum（l_new）
     *   DM   = TV + 13 vector      : delta-m = exp(m_old - m_new) 重缩放因子
     *   MASK = 11 * 16KB = 176KB   : int8 mask 区
     *
     * 注：maskUbTensor16（half mask）复用 LS_UB_TENSOR_OFFSET=0 区域，
     *     因为执行 mask 相关流程时 S 已从 ls 搬到 compute 区，ls 区可重用。
     */
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_, float /*softcapValue_*/ = 0.0f)
    {
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t COMPUTE_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 0;

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;

        scaleValue = static_cast<half>(scaleValue_);
        lsUbTensor = resource.ubBuf.template GetBufferByByte<half>(LS_UB_TENSOR_OFFSET);
        computeUbTensor = resource.ubBuf.template GetBufferByByte<half>(COMPUTE_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<half>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<half>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    /**
     * SetVecMask：按有效元素数 len 设置 Vector 指令的谓词掩码
     *   - len >= 128：全1掩码（两个64位掩码字均置0xffff...ffff）
     *   - len < 64：低位掩码字的前 len 位置1，高位置0
     *   - 64 <= len < 128：高位掩码字的前 (len-64) 位置1，低位全1
     *
     * 用于尾块处理（columnNum 不是 128 的整数倍时，仅对有效元素计算）。
     */
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        const int32_t MAX_MASK_LEN = 128;
        const int32_t HALF_MASK_LEN = 64;
        if (len >= MAX_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - HALF_MASK_LEN > 0 ? len - HALF_MASK_LEN : 0;
        int32_t lowMask = len - HALF_MASK_LEN >= 0 ? HALF_MASK_LEN : len;
        if (len < HALF_MASK_LEN) {
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    /**
     * SetBlockReduceMask：块归约专用掩码（16 元素粒度）
     *   - len > 16：全1掩码
     *   - 否则：将 subMask 重复到 4 个 16-bit lane 中（高低位掩码字均填4份）
     *
     * 注意：当前 low prec 版本行规约走 WholeReduceSum，本函数保留但未使用。
     */
    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        const int32_t MAX_LEN = 16;
        if (len > MAX_LEN) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = (static_cast<uint64_t>(1) << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    /**
     * RowsumSPECTILE512：512列特化的行求和快速路径
     *   - 输入 srcUb 形状为 (numRowsRound, 512) half，已对齐到 BLOCK_SIZE
     *   - 使用 3 次 Add 将 4 个 128-half vector 合并成 1 个 128-half vector：
     *       step1: v[0] += v[1]   → v[0] 存前半累加
     *       step2: v[2] += v[3]   → v[2] 存后半累加
     *       step3: v[0] += v[2]   → v[0] 存全部 512 列的成对和
     *   - 最后 WholeReduceSum 在 128-half 内做完整行加和，得到每行一个 half 的 rowsum
     *
     * 相比通用路径 RowsumTAILTILE，省掉了 for 循环内多次 PipeBarrier 的开销。
     */
    __aicore__ inline
    void RowsumSPECTILE512(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowsumUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::Add<half, false>(
            srcUb,
            srcUb,
            srcUb[HALF_VECTOR_SIZE],
            (uint64_t)0,
            numRowsRound,
            AscendC::BinaryRepeatParams(
                1, 1, 1,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE));
        AscendC::Add<half, false>(
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_2],
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_2],
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_3],
            (uint64_t)0,
            numRowsRound,
            AscendC::BinaryRepeatParams(
                1, 1, 1,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<half, false>(
            srcUb,
            srcUb,
            srcUb[HALF_VECTOR_SIZE * SPLIT_COL_IDX_2],
            (uint64_t)0,
            numRowsRound,
            AscendC::BinaryRepeatParams(
                1, 1, 1,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE,
                numElemsAligned / BLOCK_SIZE));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<half, false>(
            rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
            numElemsAligned / BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * RowsumTAILTILE：通用列宽的行求和
     *   - 若 numElems <= 128：单次 WholeReduceSum 即可（用掩码处理尾块）
     *   - 若 numElems > 128：先 for 循环把所有 128-half 完整块 Add 合并到 srcUb[0]，
     *     再对不足 128 的尾块做掩码 Add，最后 WholeReduceSum 得到每行和
     *
     * 用于列数非 512（即 KV tile 的尾块）时的 rowsum(exp(S - m)) 计算。
     */
    __aicore__ inline
    void RowsumTAILTILE(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowsumUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems <= HALF_VECTOR_SIZE) {
            SetVecMask(numElems);
            AscendC::WholeReduceSum<half, false>(
                rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            for (uint32_t vmaxIdx = 1; vmaxIdx < numElems / HALF_VECTOR_SIZE; vmaxIdx++) {
                AscendC::Add<half, false>(
                    srcUb,
                    srcUb,
                    srcUb[vmaxIdx * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (numElems % HALF_VECTOR_SIZE > 0) {
                SetVecMask(numElems % HALF_VECTOR_SIZE);
                AscendC::Add<half, false>(
                    srcUb,
                    srcUb,
                    srcUb[numElems / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::WholeReduceSum<half, false>(
                rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * RowmaxTAILTILE：通用列宽的行求最大值
     *   - 若 numElems <= 128：单次 WholeReduceMax（带尾块掩码）
     *   - 否则：
     *       1. DataCopy 把第一块 128 half 搬到 lsUbTensor（作为初始max）
     *       2. for 循环对每个完整 128-half 块与当前累积做 Max 归并
     *       3. 对不足 128 的尾块做掩码 Max 归并
     *       4. WholeReduceMax 在 128 half 内取每行最大值
     *
     * 注意：low prec 版本没有 RowmaxSPECTILE512 特化（高版本有），一律走通用路径。
     */
    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowmaxUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems <= HALF_VECTOR_SIZE) {
            SetVecMask(numElems);
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            AscendC::DataCopy(
                lsUbTensor,
                srcUb,
                AscendC::DataCopyParams(
                    numRowsRound,
                    HALF_VECTOR_SIZE / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t vmaxIdx = 1; vmaxIdx < numElems / HALF_VECTOR_SIZE; vmaxIdx++) {
                AscendC::Max<half, false>(
                    lsUbTensor,
                    lsUbTensor,
                    srcUb[vmaxIdx * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (numElems % HALF_VECTOR_SIZE > 0) {
                SetVecMask(numElems % HALF_VECTOR_SIZE);
                AscendC::Max<half, false>(
                    lsUbTensor,
                    lsUbTensor,
                    srcUb[numElems / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    numRowsRound,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE,
                        numElemsAligned / BLOCK_SIZE));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, lsUbTensor, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * CopySGmToUb：将 S=Q*K^T 从 GM 搬到 UB(lsUbTensor)
     *   - rowNumCurLoop 行 × columnNumRound 列（columnNumPad 含行跨度padding）
     *   - 源地址由外层根据 rowOffsetIoGm 和 MatrixCoord 偏移传入 gInput
     */
    __aicore__ inline
    void CopySGmToUb(AscendC::GlobalTensor<half> gInput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            lsUbTensor,
            gInput,
            AscendC::DataCopyParams(rowNumCurLoop,
                columnNumRound / BLOCK_SIZE,
                (columnNumPad - columnNumRound) / BLOCK_SIZE,
                0));
    }

    /**
     * CopyMaskGmToUb：将 causal 上三角 mask 从 GM 搬到 UB(maskUbTensor)
     *
     * 因为一个 row batch 可能覆盖多个 Q head（当 qNBlockSize>1 时 GQA 多头并行），
     * 而 mask 是按 (qS, kvS) 形状存放（每个 head 共享），所以要分三段加载：
     *   prologue    : 头部残差行（不足一个完整 tokenNumPerHead）
     *   integral    : 中间若干个完整 head，每个 head 逐 token 拷贝
     *   epilogue    : 尾部残差行
     *
     * DataCopyPad 支持带 padding 的 DMA，自动把 stride 间的无效字节跳过。
     */
    __aicore__ inline
    void CopyMaskGmToUb(AscendC::GlobalTensor<ElementMask> gMask, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum)
    {
        uint32_t innerUbRowOffset = 0;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset],
                gMask[proTokenIdx * maskStride],
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset],
                gMask,
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset],
                gMask,
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
        }
    }

    /**
     * ScaleS：对 S 执行 S = scaleValue * S
     *   - 从 lsUbTensor 读，写到 computeUbTensor
     *   - scaleValue = 1/√d（softmax 的温度缩放）
     *   - 同时完成 ls → compute 的搬运（Muls 本身就是二元运算，但这里 src/dst 不同）
     */
    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<half, false>(
            computeUbTensor,
            lsUbTensor,
            scaleValue,
            (uint64_t)0,
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * UpCastMask：将 int8 mask 向上转型为 half mask
     *   - int8 的 1（需要mask）→ half 的 1.0，后续 Muls(-6e4) 变 -6e4
     *   - int8 的 0（可见）→ half 的 0.0
     *   - 高版本需要两步 Cast(int8→half→float)，本low_prec版本只需一步(int8→half)
     */
    template<typename ElementMaskDst, typename ElementMaskSrc>
    __aicore__ inline 
    void UpCastMask(
        const AscendC::LocalTensor<ElementMaskDst> &maskUbTensorDst,
        const AscendC::LocalTensor<ElementMaskSrc> &maskUbTensorSrc,
        uint32_t rowNumCurLoop,
        uint32_t columnNumRound)
    {
        AscendC::Cast<ElementMaskDst, ElementMaskSrc, false>(
            maskUbTensorDst, maskUbTensorSrc, AscendC::RoundMode::CAST_NONE, (uint64_t)0,
            CeilDiv(rowNumCurLoop * columnNumRound, (uint32_t)(REPEAT_SIZE_IN_BYTE / sizeof(ElementMaskDst))),
            AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * ApplyMask：将 causal mask 应用到 S 上
     *   - 先把 half mask 乘以 -6e4（half 能表示的接近最小的负数，softmax 后趋近 0）
     *   - 然后把 -6e4 * mask 加到 computeUbTensor 上
     *     mask=1(需mask)位置 → S += -6e4 → exp(S-m)≈0
     *     mask=0(可见)  位置 → S += 0    → 正常 softmax
     *
     *   - 若 mask 覆盖全部列（maskColumnRound == columnNumRound）：整块 Add 一次完成
     *   - 否则：按列偏移 addMaskUbOffset 仅对需要 mask 的列做 Add（三角边界行）
     */
    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        AscendC::Muls<half, false>(
            maskUbTensor16,
            maskUbTensor16,
            (half)-6e4,
            (uint64_t)0,
            (rowNumCurLoop * maskColumnRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (maskColumnRound == columnNumRound) {
            AscendC::Add<half, false>(
                computeUbTensor,
                computeUbTensor,
                maskUbTensor16,
                (uint64_t)0,
                (rowNumCurLoop * maskColumnRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            uint32_t loop = maskColumnRound / HALF_VECTOR_SIZE;
            for (uint32_t i = 0; i < loop; i++) {
                AscendC::Add<half, false>(
                    computeUbTensor[addMaskUbOffset + i * HALF_VECTOR_SIZE],
                    computeUbTensor[addMaskUbOffset + i * HALF_VECTOR_SIZE],
                    maskUbTensor16[i * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1,
                        1,
                        1,
                        columnNumRound / BLOCK_SIZE,
                        columnNumRound / BLOCK_SIZE,
                        maskColumnRound / BLOCK_SIZE));
            }
            if (maskColumnRound % HALF_VECTOR_SIZE > 0) {
                SetVecMask(maskColumnRound % HALF_VECTOR_SIZE);
                AscendC::Add<half, false>(
                    computeUbTensor[addMaskUbOffset + loop * HALF_VECTOR_SIZE],
                    computeUbTensor[addMaskUbOffset + loop * HALF_VECTOR_SIZE],
                    maskUbTensor16[loop * HALF_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(1,
                        1,
                        1,
                        columnNumRound / BLOCK_SIZE,
                        columnNumRound / BLOCK_SIZE,
                        maskColumnRound / BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * CalcLocalRowMax：计算当前 tile S 的行最大值 m_local
     *   - 输入 computeUbTensor（已 scale/mask），输出 lmUbTensor[rowOffset]
     *   - 每行一个 half 值（rowOffset 是行内循环偏移，用于多 pingpong 槽）
     */
    __aicore__ inline
    void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        RowmaxTAILTILE(
            computeUbTensor,
            lmUbTensor[rowOffset],
            tvUbTensor,
            rowNumCurLoopRound,
            columnNum,
            columnNumRound);
    }

    /**
     * UpdateGlobalRowMax：将当前 tile 的 m_local 与历史 m_old 合并，得到 m_new
     *
     *   若 isFirstStackTile（第 0 个 KV tile）：
     *       m_new = m_local；直接拷贝即可
     *   否则：
     *       m_new = max(m_local, m_old)                     // hm = max(lm, gm)
     *       delta_m = exp(m_old - m_new)                    // dm = exp(gm - hm)
     *       （delta_m 是重缩放因子：旧 P 需要乘以它才能在新的 m_new 下保持一致）
     *
     *   最后无条件：m_old = m_new（gm ← hm 为下一个 tile 准备）
     */
    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            AscendC::Max<half, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub<half, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<half, false>(dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * CalcExp：计算 P_tile = exp(S - m_new)
     *   1. Brcb 将 hmUbTensor[rowOffset]（每行一个 half）按列广播为整块
     *      → tvUbTensor（形状与 S 同）
     *   2. computeUbTensor = computeUbTensor - tvUbTensor（逐元素减行最大值）
     *      尾块列使用 SetVecMask 保护
     *   3. computeUbTensor = exp(computeUbTensor)（逐元素取指数，得到 P_tile）
     *
     *   BrcbRepeatParams(1, 8)：每个标量重复 8 次（每次覆盖 16 字节 = 8 个 half），
     *   连续 Brcb 后 tv 中每行都填满同一个 hm 值。
     */
    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t rowOffset)
    {
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint16_t>(),
            hmUbTensor[rowOffset].template ReinterpretCast<uint16_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < columnNum / HALF_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<half, false>(
                computeUbTensor[subIdx * HALF_VECTOR_SIZE],
                computeUbTensor[subIdx * HALF_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / BLOCK_SIZE, columnNumRound / BLOCK_SIZE, 1));
        }
        if (columnNum % HALF_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % HALF_VECTOR_SIZE);
            AscendC::Sub<half, false>(
                computeUbTensor[columnNum / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                computeUbTensor[columnNum / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / BLOCK_SIZE, columnNumRound / BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<half, false>(
            computeUbTensor,
            computeUbTensor,
            (uint64_t)0,
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * CalcLocalRowSum：计算当前 tile P 的行和 l_local = rowsum(exp(S - m_new))
     *   - 列数 == 512：走 RowsumSPECTILE512 快速路径
     *   - 否则：走通用 RowsumTAILTILE 路径
     *   - 结果写入 llUbTensor[rowOffset]
     */
    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        if (columnNum == 512U) {
            RowsumSPECTILE512(computeUbTensor,
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowsumTAILTILE(computeUbTensor,
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    /**
     * UpdateGlobalRowSum：将当前 tile 的 l_local 与历史 l_old 合并，得到 l_new
     *
     *   若 isFirstStackTile：
     *       l_new = l_local
     *   否则：
     *       l_new = delta_m * l_old + l_local
     *             = exp(m_old - m_new) * l_old + rowsum(exp(S_tile - m_new))
     *
     *   注意：此处不写回 gl → 别的位置（gl 保留为 l_new 供后续 tile 使用），
     *   最后由 rescale_o epilogue 在全部 tile 结束后用 l_new 归一化 O。
     */
    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            AscendC::Mul<half, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add<half, false>(
                glUbTensor[rowOffset],
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    /**
     * MoveP：将计算好的 P_tile 从 computeUbTensor 搬到 lpUbTensor（P输出缓冲）
     *   - low prec 版本 P 本身就是 half，直接 DataCopy 即可（无需 cast）
     *   - 高版本此处需要 DownCastP（float→half/bf16）
     */
    __aicore__ inline
    void MoveP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = 1;
        repeatParams.srcStride = 0;
        repeatParams.blockLen = CeilDiv(rowNumCurLoop * columnNumRound, BLOCK_SIZE);
        AscendC::DataCopy<half>(lpUbTensor, computeUbTensor, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /**
     * CopyPUbToGm：将 P_tile 从 lpUbTensor 写回 GM
     *   - 写到 gOutput（实际是 workspace 中的 gP 区域）
     *   - columnNumPad - columnNumRound 为列方向的 padding 间隔
     *   - P_tile 写回后 Cube 侧即可通过 PV matmul 读取 P*V
     */
    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(gOutput,
            lpUbTensor,
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    /**
     * SubCoreCompute：单个子核（sub-core）上的核心 softmax 计算
     *
     * 执行顺序（online softmax 三步法）：
     *   ① CalcLocalRowMax       → m_local
     *   ② UpdateGlobalRowMax    → m_new = max(m_local, m_old), dm = exp(m_old - m_new)
     *   ③ CalcExp               → P_tile = exp(S - m_new)
     *   ④ MoveP + CopyPUbToGm   → 把 P_tile 写给 GM（供 Cube PV matmul 使用）
     *   ⑤ CalcLocalRowSum       → l_local = rowsum(P_tile)
     *   ⑥ UpdateGlobalRowSum    → l_new = dm * l_old + l_local
     *
     * 其中 ④ 和 ⑤ 通过 EVENT_ID0 做 MTE3(写GM) 与 V(计算) 之间的软流水：
     *   - CalcExp 后 SetFlag(V_MTE2→EVENT_ID0)，MoveP 后 WaitFlag(MTE3_V→EVENT_ID0)
     *     （等前一轮的写GM完成后才开始下一轮的MoveP？实际是为了pingpong双缓冲不冲突）
     *
     * LSE_MODE=OUT_ONLY 时，isFirstStackTile && isFirstRowLoop 需要等待 EVENT_ID4，
     * 该事件由外部在 LSE 数据预加载到 UB 后 SetFlag。
     */
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isFirstRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        UpdateGlobalRowMax(rowNumCurLoop,
            rowNumCurLoopRound,
            columnNum,
            columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        MoveP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    /**
     * operator() 重载1：无 causal mask 版本
     *
     * 使用场景：
     *   - MASK_TYPE=NO_MASK 编译期路径（双向注意力）
     *   - MASK_TYPE=MASK_CAUSAL 且 doTriUMask=false（当前 KV tile 完全在对角线下方）
     *
     * 调用约定：调用者已在外部 CrossCoreWaitFlag(qkReady)，本函数不等待跨核事件。
     *
     * 执行流程：
     *   1. 解析行/列形状，计算子核划分（rowSplitSubBlock 等）
     *   2. 按 rowNumTile 分批循环（UB S 区容量 32KB / 列宽 决定每批行数）
     *   3. 每批：CopySGmToUb → ScaleS → SubCoreCompute
     *      其中 CopySGmToUb 与 ScaleS 通过 EVENT_ID0 做 MTE2/V 软流水
     *
     * @param gOutput            输出 P 的 GM tensor（workspace 中 gP）
     * @param gInput             输入 S 的 GM tensor（workspace 中 gS，Cube 写的 QK^T）
     * @param isFirstStackTile   是否是当前 Q row block 的第一个 KV tile
     * @param isLastNoMaskStackTile 最后一个无mask的KV tile（本版本接收但未使用）
     * @param qSBlockSize        Q 序列维 tile 行数（128 或尾块）
     * @param qNBlockSize        Q head 维 tile 并行head数
     * @param curStackTileMod    当前 KV tile 的 pingpong 槽位（curStackTileMod % PRE_LAUNCH）
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<half> gInput,
        const LayoutOutput &layoutOutput, const LayoutInput &layoutInput, GemmCoord actualBlockShape,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        // 两个 Vector 子核（sub-core 0 和 sub-core 1）的任务划分
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        // 子核行拆分策略：
        //   qNBlockSize == 1（decode 单头）：按序列维对半拆（qSBlockSize/2）
        //   qNBlockSize >  1（prefill 多头）：按 head 维拆，每子核 qNSplitSubBlock 个head
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        // 每批最大行数 = UB S 单槽元素数 / 列宽，向下对齐到 BLOCK_SIZE，且不超过 64 行
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];

            // 等待上一批 ScaleS 完成后再搬下一批（避免ls区域读写冲突）
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);

            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            SubCoreCompute(
                gOutputCurLoop,
                layoutOutputCurLoop,
                rowOffsetCurLoop,
                isFirstStackTile,
                (rowLoopIdx == 0U),
                columnNumRound,
                pingpongFlag,
                curStackTileMod);
        }
    }

    /**
     * operator() 重载2：带 causal mask 版本
     *
     * 使用场景：MASK_TYPE=MASK_CAUSAL 且 doTriUMask=true（当前 KV tile 跨越因果对角线）
     *
     * 与重载1相比，额外工作：
     *   - 内部 CrossCoreWaitFlag(qkReady)（等 Cube 完成 Q*K^T）
     *   - 根据 triUp/triDown/kvSStartIdx/kvSEndIdx 计算 mask 的 GM 偏移和 mask 列范围
     *   - CopyMaskGmToUb → UpCastMask → ApplyMask（把 -6e4 加到需要 mask 的位置）
     *   - mask 加载使用独立的 EVENT_ID1/EVENT_ID3 流水，避免和 S 流水冲突
     *
     * causal mask 区域计算：
     *   对每个 Q 位置 i，只能注意到 KV 位置 j < i（严格下三角）
     *   设 triUp = noSkipKvS - qSBlockSize（本 block Q 行的最小可见 KV 位置 - 1）
     *      triDown = noSkipKvS（本 block Q 行的最大可见 KV 位置）
     *   若 triUp 在当前 KV stack 内（triUp >= kvSStartIdx）：
     *     → 部分列需要 mask（三角边界），需要计算 maskColumn/addMaskUbOffset
     *   若 triUp 在当前 KV stack 之前：
     *     → 整块都需要 mask（上三角不可见区域）
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<half> gInput,
        AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown,
        uint32_t kvSStartIdx, uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        // 子核行划分（同重载1）
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = AscendC::Std::min(qSBlockSize, rowActualThisSubBlock);

        // 序列维对半拆时，子核1的 mask 行偏移需要加上 rowOffsetThisSubBlock
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1U) ? rowOffsetThisSubBlock : 0;

        // mask 应用区域计算
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            // triUp 在当前 KV stack 内：从 triUpRoundDown 开始的列需要 mask
            uint32_t triUpRoundDown = RoundDown(triUp, BLOCK_SIZE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0U;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            // triUp 在当前 KV stack 之前：整块列都需要 mask（上三角不可见）
            gmOffsetMaskRow = 0U;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0U;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, BLOCK_SIZE);

        // 计算 mask 在 GM 中的起始位置
        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        // 行循环分批（同重载1）
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);

        // 若子核分配的行数为0（极端情况），仅等待跨核信号后直接返回
        if (rowActualThisSubBlock == 0U) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }
        // 等待 Cube 完成 Q*K^T
        Arch::CrossCoreWaitFlag(qkReady);
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            // mask 分段参数（prologue/integral/epilogue），用于多头并行时正确拼接 mask
            uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
            uint32_t proTokenNum = AscendC::Std::min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) %
                tokenNumPerHeadThisSubBlock;
            uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
            uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;

            // 1) 搬运 S：GM(ls着陆) → compute → ScaleS（EVENT_ID0 流水）
            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
            
            // 2) 搬运 mask：GM→UB(mask区) → UpCast(int8→half)（EVENT_ID3 流水）
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            CopyMaskGmToUb(
                gMaskThisSubBlock,
                maskColumn,
                maskColumnRound,
                maskStride,
                tokenNumPerHeadThisSubBlock,
                proTokenIdx,
                proTokenNum,
                integralHeadNum,
                epiTokenNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            // 3) 应用 mask：mask*-6e4 加到 computeUbTensor 上
            ApplyMask(
                (pingpongFlag * MAX_UB_S_ELEM_NUM),
                rowNumCurLoop,
                columnNumRound,
                maskColumnRound,
                addMaskUbOffset);

            // 4) Online softmax 主体计算（同重载1）
            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            SubCoreCompute(
                gOutputCurLoop,
                layoutOutputCurLoop,
                rowOffsetCurLoop,
                isFirstStackTile,
                (rowLoopIdx == 0),
                columnNumRound,
                pingpongFlag,
                curStackTileMod);
        }
    }

private:
    half scaleValue;                              // softmax 缩放因子(=1/√d)，已转为half
    AscendC::LocalTensor<half> lsUbTensor;        // S GM→UB 着陆区(half, 复用mask16空间)
    AscendC::LocalTensor<half> computeUbTensor;   // S 计算区(half, scale+mask+exp在此)
    AscendC::LocalTensor<ElementOutput> lpUbTensor; // P 输出缓冲(half/bf16, 待写GM)
    AscendC::LocalTensor<ElementMask> maskUbTensor; // int8 mask 原始缓冲
    AscendC::LocalTensor<half> maskUbTensor16;    // half mask (复用LS区0偏移)
    AscendC::LocalTensor<half> lmUbTensor;        // 当前 tile 局部行 max (m_local)
    AscendC::LocalTensor<half> hmUbTensor;        // 合并后新的全局行 max (m_new)
    AscendC::LocalTensor<half> gmUbTensor;        // 历史全局行 max (m_old)
    AscendC::LocalTensor<half> dmUbTensor;        // delta-m = exp(m_old - m_new) 重缩放因子
    AscendC::LocalTensor<half> llUbTensor;        // 当前 tile 局部行 sum (l_local)
    AscendC::LocalTensor<half> tvUbTensor;        // Brcb 广播/规约临时缓冲
    AscendC::LocalTensor<half> glUbTensor;        // 全局行 sum (l_new)
};

}

#endif