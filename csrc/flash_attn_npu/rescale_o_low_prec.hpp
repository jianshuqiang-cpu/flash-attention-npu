/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

/**
 * ============================================================================
 * rescale_o_low_prec.hpp —— FlashAttention NPU 前向推理 O 重缩放/累加/归一化 epilogue（低精度 half 版，预留）
 * ============================================================================
 *
 * 【文件定位】
 *   本文件实现 CATLASS 框架 BlockEpilogue 的模板特化，调度策略为
 *   EpilogueAtlasA2RescaleOT<LSE_MODE_, half>（half = 低精度中间计算），
 *   对应 FlashAttention 前向推理最后一个 Vector 核阶段：对 PV 输出 OTmp 做跨 KV stack 的
 *   在线累加和最终归一化，产生最终输出 O 和（可选的）LSE。
 *
 *   与 rescale_o.hpp（高精度 float 版）的核心区别：
 *   - 所有中间张量 (lo/dm/gl/tv/hm/gm) 均使用 half (FP16) 而非 float；
 *   - 向量指令使用 half 宽度（HALF_VECTOR_SIZE=128 vs FLOAT_VECTOR_SIZE=64）；
 *   - Brcb 以 uint16_t 重解释（half=16位），行对齐到 HALF_BLOCK_SIZE=16；
 *   - 无需 float→half 的 Cast（go 本身已是 half/ElementOutput）；
 *   - LSE 输出需要额外 half→float 的 Cast 转换（新增 tvUbTensor32 和 lse16/32 缓冲）。
 *
 *   注意：当前 flash_api.cpp 中所有 kernel 实例化均使用 IntermCalcPrec=float，
 *   即 rescale_o.hpp（高精度版）是实际运行的版本；本 low_prec 版本仅作为预留路径，未被实例化。
 *
 * 【核心算法（online softmax 的 O 累加）】
 *   记 m_t、l_t 为 online softmax 维护的全局 running max/sum，
 *   dm_t = exp(m_{t-1} - m_t) 为当前 tile 的缩放因子。
 *
 *   - 首块 (isFirstStackTile)：
 *       O = OTmp              （直接使用 PV 输出）
 *   - 中间块：
 *       O = O * dm + OTmp     （旧 O 乘缩放因子后加上新 PV 输出）
 *   - 末块 (isLastStackTile)：
 *       O = O / l_T           （除以全局 rowsum 完成 softmax 归一化）
 *       LSE = ln(l_T) + m_T   （可选：输出 log-sum-exp）
 *
 * 【内存层级（仅 Vector 核，使用 UB）】
 *   GM (gInput=OTmp, gUpdate=O中间量, gOutput=O, gLse=LSE)
 *     │ MTE2 (GM→UB) / MTE3 (GM→UB for update)
 *     ▼
 *   UB (Unified Buffer, 静态分区):
 *     - loUbTensor: OTmp（当前 tile 的 PV 输出）
 *     - goUbTensor: O 累加结果
 *     - dmUbTensor: dm 缩放因子（softmax 产出，按 curStackTileMod 三槽）
 *     - glUbTensor: l 全局累加和（最终归一化用）
 *     - gmUbTensor: m 全局最大值（LSE 计算用）
 *     - tvUbTensor/tvUbTensor32: 向量广播临时缓冲
 *     - hmUbTensor: 预留（与 softmax 共享偏移）
 *     - lse16/lse32_ubuf_tensor: LSE 中间/输出缓冲
 *     ── 所有 UB 张量通过固定字节偏移从 ubBuf 分配，偏移与 online_softmax_low_prec.hpp 完全一致，
 *        实现 Vector 核 softmax→rescale 的零拷贝数据传递
 *
 * 【Sub-core 拆分】
 *   Atlas A2 每个 AI Core 含 2 个 Vector 子核，通过 AscendC::GetSubBlockIdx/Num 获取：
 *   - Decode (qNBlockSize==1): 沿序列(行)维拆分
 *     sub-core0 处理前半行，sub-core1 处理后半行；
 *   - Prefill (qNBlockSize>1): 沿 head(列)维拆分
 *     sub-core0 处理前 qNSplitSubBlock 个 head，sub-core1 处理剩余 head。
 *
 * 【行循环（UB O 容量溢出处理）】
 *   MAX_UB_O_ELEM_NUM=8192 个 half 元素，当 rowNum*embed > 8192 时，
 *   需要多轮 rowLoop 处理，每轮将 go 通过 gUpdate(GM) 做溢出写回/读回：
 *   - needRowLoop=1 时，非首块在 *dm 之前先从 gUpdate 加载上一轮的 go，
 *     非末块在累加完成后将 go 写回 gUpdate。
 *
 * 【Multi-head 输出三段式（prologue/integral/epilogue）】
 *   行循环的每行包含多个 head，一次 DataCopyPad 时需要分成：
 *   - prologue: 起始不完整 head 的部分 token
 *   - integral: 若干个完整 head（步长 = qSThisSubBlock 行）
 *   - epilogue: 末尾不完整 head 的部分 token
 *
 * 【事件同步】
 *   - EVENT_ID0: MTE2_V（lo/go DMA 完成）、V_MTE3（O 写回 GM 排空）、M_FIX（Cube完成... 此处复用）
 *   - EVENT_ID1: go(gUpdate) DMA 完成
 *   - EVENT_ID3: V_MTE2（lo 可被覆写，sub-core 间传递）
 *   - EVENT_ID4: V_MTE3/MTE3_V（LSE 排空/释放）
 *   - EVENT_ID5: V_MTE3（非末块 go 写回 gUpdate 排空）
 *   - EVENT_ID6: MTE3_MTE2（SubCoreCompute 入口排空）
 * ============================================================================
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_LOW_PREC_O_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_LOW_PREC_O_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fa_block.h"

namespace Catlass::Epilogue::Block {

/**
 * @brief BlockEpilogue 对 EpilogueAtlasA2RescaleOT<half> 的特化：低精度 O 重缩放
 *
 * @tparam OutputType_  最终输出 O 的类型 + 布局（half/bf16, RowMajor）
 * @tparam InputType_   PV 输出 OTmp 的类型 + 布局（half, RowMajor）
 * @tparam UpdateType_  O 中间溢出缓冲 gUpdate 的类型 + 布局（half）
 * @tparam LseType_     LSE 输出类型 + 布局（float）
 * @tparam LSE_MODE_    LSE 输出模式（NONE=0 不输出, OUT_ONLY=1 输出 LSE）
 */
template <
    class OutputType_,
    class InputType_,
    class UpdateType_,
    class LseType_,
    LseModeT LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2RescaleOT<LSE_MODE_, half>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, half>;
    using ArchTag = typename DispatchPolicy::ArchTag;                        // Atlas A2 架构标签

    using ElementOutput = typename OutputType_::Element;                     // O 元素类型（half/bf16）
    using ElementInput = typename InputType_::Element;                       // OTmp 元素类型（half）
    using ElementUpdate = typename UpdateType_::Element;                     // gUpdate 元素类型（half）
    using ElementLse = typename LseType_::Element;                           // LSE 元素类型（float）

    using LayoutOutput = typename OutputType_::Layout;                       // O 布局（RowMajor）
    using LayoutInput = typename InputType_::Layout;                         // OTmp 布局
    using LayoutUpdate = typename UpdateType_::Layout;                       // gUpdate 布局
    using LayoutLse = typename LseType_::Layout;                             // LSE 布局

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;           // LSE 输出模式

    // ======================== 编译期常量（half 低精度版本） ========================
    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;                      // 每个 half block 元素数（预留）
    static constexpr uint32_t BLOCK_SIZE = 16;                               // 通用块大小（对齐）
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;                 // half 向量计算元素数
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;                 // float 向量计算元素数（LSE 用）
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;                     // half 每行元素数（预留）
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;                    // float 每行元素数（预留）
    static constexpr uint32_t MULTIPLIER = 2;                                // 倍增因子（预留）
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;                          // float 块大小（8个float=32字节）
    static constexpr uint32_t HALF_BLOCK_SIZE = 16;                          // half 块大小（16个half=32字节）
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;                        // float 向量宽度（64个float=256字节）
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;                        // half 向量宽度（128个half=256字节）
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;                   // UB 1KB 向量步长
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;                   // UB 16KB 块步长
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;                          // dm 缓冲大小（预留）
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;                         // ll 缓冲大小（预留）
    static constexpr uint32_t VECTOR_SIZE = 128;                             // 默认向量宽度（=HALF_VECTOR_SIZE）
    static constexpr uint32_t NUM4 = 4;                                      // 常量4（预留）
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;                      // UB 中 O 最大 half 元素数（8192×2B=16KB）
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;                    // 每个 sub-core 支持最大行数（dm 三槽×256行）
    static constexpr uint32_t SIZE_OF_16BIT = 2;                             // half/bf16 字节大小

    /**
     * @brief 构造函数：从 UB 按固定字节偏移分配各张量
     *
     * UB 内存布局（与 online_softmax_low_prec.hpp 完全一致，实现零拷贝共享）：
     *
     *   偏移(KB)   张量           用途
     *   ─────────────────────────────────────────────────────
     *   0-16       (softmax 使用)  P workspace / mask / S 等
     *   ...
     *   96-112     loUbTensor     OTmp 加载缓冲（当前 tile 的 PV 输出）
     *   112-128    (softmax 使用)  mask32 等
     *   128-160    goUbTensor     O 累加缓冲
     *   160-169    tvUbTensor     广播临时缓冲（Brcb 目的地）
     *              /tvUbTensor32  （LSE 时复用为 float 广播缓冲）
     *   169-170    hmUbTensor     预留（softmax 局部max，rescale 中未使用）
     *   170-172    gmUbTensor     m 全局最大值（LSE 用）
     *              /lse32_ubuf_tensor （LSE float 输出时复用 GM 偏移）
     *   172-173    glUbTensor     l 全局累加和（最终归一化用）
     *              /lse16_ubuf_tensor （LSE half 中间时复用 GL 偏移）
     *   173-...    dmUbTensor     dm 缩放因子三槽（每槽 MAX_ROW_NUM_SUB_CORE 个 half）
     */
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        // UB 各张量的字节偏移（单位：字节）
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;       // 96KB
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;       // 128KB
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;      // 160KB

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;   // 169KB
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;  // 170KB
        constexpr uint32_t LSE32_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE; // 与GM复用
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;  // 172KB
        constexpr uint32_t LSE16_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE; // 与GL复用
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;  // 173KB

        loUbTensor = resource.ubBuf.template GetBufferByByte<half>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);
        tvUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);
        lse16_ubuf_tensor = resource.ubBuf.template GetBufferByByte<half>(LSE16_UB_TENSOR_OFFSET);
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE32_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    /**
     * @brief 设置 half 向量 mask（处理非对齐尾向量）
     *
     * half 向量宽度为 128 个元素（两个 64 位 mask 寄存器各管 64 个 half）。
     *
     * @param len 实际有效元素数（0~128）
     *
     * 逻辑：
     * - len >= 128: 全 mask 开启（两个 64 位寄存器全 1）
     * - len >= 64 : 高64位全1，低64位取 (1<<(len-64))-1
     * - len < 64  : 高64位全0，低64位取 (1<<len)-1
     */
    __aicore__ inline
    void SetMask(int32_t len)
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
     * @brief 将 O 从 UB 写回 GM，支持 multi-head 三段式（prologue/integral/epilogue）
     *
     * 由于行循环中每轮的行可能跨越多个 head 边界，输出时需要分三段处理：
     *   - prologue: 起始不完整 head 的 proTokenNum 个 token
     *   - integral: integralHeadNum 个完整 head（每个 qSThisSubBlock 行连续）
     *   - epilogue: 末尾不完整 head 的 epiTokenNum 个 token
     *
     * 每段使用 DataCopyPad，设置：
     *   - 每段行数（proTokenNum/qSThisSubBlock/epiTokenNum）
     *   - 源行长度 = embed × 2B（half 连续元素）
     *   - GM 行 stride 间隔 = (oHiddenSize - embed) × 2B（跳过同一 token 的其他 head 列）
     *
     * @param gOutput         GM 输出 O 张量
     * @param proTokenIdx     prologue 在当前行循环内的起始 token 索引
     * @param proTokenNum     prologue token 数（可能为0）
     * @param epiTokenNum     epilogue token 数（可能为0）
     * @param integralHeadNum 完整 head 数
     * @param qSThisSubBlock  每个 head 在当前 sub-core 内行数（序列维）
     * @param embed           活动 head 维度（= d_head）
     * @param oHiddenSize     O 张量最后一维步长（包含所有 head 列，= numHeads×d_head）
     */
    __aicore__ inline
    void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t epiTokenNum, uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
        }
    }

    /**
     * @brief 单个 sub-core 单轮行循环的核心计算
     *
     * 执行流程：
     *
     * 【A. 非首块 O 重缩放准备 (isFirstStackTile==false)】
     *   A1. Wait EVENT_ID3，等待上一轮 SubCoreCompute 的 lo DMA 完成（避免 lo 缓冲覆写）
     *   A2. DataCopy gInput(OTmp) → loUbTensor
     *   A3. Set EVENT_ID0，标记 lo DMA 已启动
     *   A4. Wait EVENT_ID6(MTE3_MTE2)，MTE3 管道排空（避免与 gUpdate/gLse 写冲突）
     *   A5. Brcb: 将 dm[dmUbOffsetCurStackTile] 按行广播到 tvUbTensor
     *       （dm 是 per-row 的 scalar，Brcb 复制到整行向量）
     *   A6. 若 needRowLoop：从 gUpdate(GM) 加载上一轮溢出的 go → goUbTensor
     *   A7. 全 mask 下逐 HALF_VECTOR_SIZE 执行 go = go * dm（half Mul），尾向量用 SetMask 处理
     *   A8. Wait EVENT_ID0（lo DMA 完成）
     *   A9. 全量 Add: go = lo + go（half Add）
     *   A10. Set EVENT_ID3，通知下一轮 lo 缓冲已可复用
     *
     * 【A'. 首块 (isFirstStackTile==true)】
     *   A'1. DataCopy gInput(OTmp) → goUbTensor（直接 go = OTmp）
     *   A'2. Set/Wait EVENT_ID0
     *
     * 【B. 末块归一化+输出 (isLastStackTile==true)】
     *   B1. Brcb: 将 gl[rowOffsetLoop] 按行广播到 tvUbTensor（l 是 per-row scalar）
     *   B2. 全 mask 下逐 HALF_VECTOR_SIZE 执行 go = go / gl（half Div），尾向量 SetMask
     *   B3. PipeBarrier + Set/Wait V_MTE3(EVENT_ID0)：等 Vector 排空
     *   B4. CopyOToGm：将归一化后的 O 写回 GM（三段式）
     *   B5. 若 LSE_MODE==OUT_ONLY 且 isLastRowLoop：
     *       - Ln<half>(lse16, gl): lse16 = ln(l)（half）
     *       - Add<half>(lse16, lse16, gm): lse16 += m（LSE = ln(l)+m，half）
     *       - Cast<float,half>(lse32, lse16): half→float 转换
     *       - Brcb<float>(tv32, lse32): 广播到向量（后续 DataCopyPad 用）
     *       - DataCopyPad tv32 → gLse（写回 GM，float 格式）
     *       - 支持 qNThisSubBlock==0（单head连续写）和多head（逐head带stride写）
     *
     * 【B'. 非末块且 needRowLoop】
     *   B'1. Set/Wait V_MTE3(EVENT_ID5)：等 Vector 排空
     *   B'2. DataCopy go → gUpdate(GM)：溢出写回，供下一轮行循环读回
     *
     * 【C. 尾部】
     *   Set EVENT_ID6(MTE3_MTE2)：标记 MTE3→MTE2 可继续（SubCoreCompute 结束）
     *
     * @param gOutput          最终输出 O（仅末块写）
     * @param gInput           OTmp（PV 输出，当前 tile）
     * @param gUpdate          O 中间溢出缓冲（GM）
     * @param gLse             LSE 输出（仅末块且 LSE_MODE=OUT_ONLY 写）
     * @param layoutOutput     O 布局
     * @param layoutInput      OTmp 布局
     * @param layoutUpdate     gUpdate 布局
     * @param layoutLse        LSE 布局
     * @param qNThisSubBlock   当前 sub-core 处理的 Q head 数（prefill 拆分用）
     * @param qSThisSubBlock   当前 sub-core 每个 head 的序列行数
     * @param totalRowNum      当前 sub-core 总行数
     * @param isFirstStackTile 首块标志
     * @param isLastStackTile  末块标志
     * @param curStackTileMod  dm 三槽索引（= (stackSeqCount - PRE_LAUNCH) % 3）
     * @param needRowLoop      是否需要行循环（UB 溢出）
     * @param isLastRowLoop    最后一轮行循环标志（LSE 仅在此轮输出）
     * @param rowOffsetLoop    当前行循环起始偏移
     * @param proTokenIdx      prologue token 索引
     * @param proTokenNum      prologue token 数
     * @param epiTokenNum      epilogue token 数
     * @param integralHeadNum  完整 head 数
     */
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        uint32_t qNThisSubBlock, uint32_t qSThisSubBlock, uint32_t totalRowNum,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        uint32_t needRowLoop, uint32_t isLastRowLoop, uint32_t rowOffsetLoop,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp(curRowNum, HALF_BLOCK_SIZE);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        // dm 三槽索引：每槽 MAX_ROW_NUM_SUB_CORE 行，加上行内偏移
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        if (!isFirstStackTile) {
            // ---- 非首块：加载 lo(OTmp)、加载旧 go(gUpdate)、go=go*dm+lo ----
            // 等待上一轮 lo 缓冲消费完成
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            // GM(OTmp) → UB(lo)：搬运当前 tile 的 PV 输出
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        // 等待 MTE3 管道排空（gUpdate/gLse 等 MTE3 写操作完成）
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            // 全 mask 开启
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            // Brcb 广播 dm[dmOffset] 到 tvUbTensor：将每行的 dm scalar 复制到该行所有向量位置
            // BrcbRepeatParams(1,8): 每个 scalar 重复 8 次（一个 half block 含 16 元素，8 次 16 位重复 = 128 bit = 8 half？）
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint16_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                // 行循环溢出：从 GM(gUpdate) 加载上一轮保存的 go
                AscendC::DataCopy(
                    goUbTensor, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block：逐向量 half Mul（旧 O 乘缩放因子 exp(m_prev - m_new)）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / HALF_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<half, false>(
                    goUbTensor[vmul_idx * HALF_VECTOR_SIZE],
                    goUbTensor[vmul_idx * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
            }
            // 尾向量：处理 embed 非 HALF_VECTOR_SIZE 对齐的情况
            if (embed % HALF_VECTOR_SIZE > 0) {
                SetMask(embed % HALF_VECTOR_SIZE);
                AscendC::Mul<half, false>(
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // 等待 lo(OTmp) DMA 完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go：OTmp 加到重缩放后的旧 O 上（half Add）
            AscendC::Add<half, false>(
                goUbTensor,
                goUbTensor,
                loUbTensor,
                (uint64_t)0,
                (curRowNum * embedRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // 通知：lo 缓冲已被消费（下一轮 SubCoreCompute 可覆写）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // ---- 首块：go = lo（直接复制 OTmp 到 O 累加缓冲）----
            AscendC::DataCopy(
                goUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // ---- 末块：除以 l 做最终归一化，写回 O，可选计算 LSE ----
            // *** gl_block = expand_to_block(gl)：广播 l（全局rowsum）到整行
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                glUbTensor.ReinterpretCast<uint16_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block：逐向量 half Div（归一化）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / HALF_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<half, false>(
                    goUbTensor[vdiv_idx * HALF_VECTOR_SIZE],
                    goUbTensor[vdiv_idx * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
            }
            // 尾向量
            if (embed % HALF_VECTOR_SIZE > 0) {
                SetMask(embed % HALF_VECTOR_SIZE);
                AscendC::Div<half, false>(
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    goUbTensor[embed / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / HALF_BLOCK_SIZE, embedRound / HALF_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // 等待 Vector 管道排空，确保 Div 完成
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // ***move O to GM：三段式写回最终输出
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    // ---- LSE 输出：LSE = ln(l) + m（half），再 Cast 到 float 写 GM ----
                    AscendC::PipeBarrier<PIPE_V>();
                    // lse16 = ln(gl) （gl 已被 Brcb 消费完，复用 GL 偏移）
                    AscendC::Ln<half, false>(
                        lse16_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, HALF_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    // lse16 = ln(gl) + gm = LSE (half 精度)
                    AscendC::Add<half, false>(
                        lse16_ubuf_tensor,
                        lse16_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, HALF_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    // lse32 = (float)lse16：half→float 转换（复用 GM 偏移）
                    AscendC::Cast<float, half, false>(
                        lse32_ubuf_tensor,
                        lse16_ubuf_tensor,
                        AscendC::RoundMode::CAST_NONE,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 4));
                    AscendC::PipeBarrier<PIPE_V>();

                    // *** lse_block = expand_to_block(lse32)：广播 float LSE 到向量
                    AscendC::Brcb(
                        tvUbTensor32.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

                    // LSE 写回 GM（float），支持 head 维 strided 写入
                    if (qNThisSubBlock == 0U) {
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor32,
                            AscendC::DataCopyExtParams(
                                totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            AscendC::DataCopyPad(
                                gLse[qNIdx],
                                tvUbTensor32[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                AscendC::DataCopyExtParams(
                                    qSBlockSize, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            }
        } else if (needRowLoop) {
            // ---- 非末块且 UB 溢出：go → gUpdate(GM) 保存，供下一轮行循环读回 ----
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
        }
        // 标记 SubCoreCompute 完成，MTE3→MTE2 管道可继续（下一次 SubCoreCompute 入口 Wait EVENT_ID6 可通过）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    /**
     * @brief 执行 O 重缩放/累加/归一化 epilogue（主入口）
     *
     * 执行流程：
     * 1. 计算行分块参数：UB 最多容纳 MAX_UB_O_ELEM_NUM/embed 行 HALF 元素，
     *    按 HALF_BLOCK_SIZE(16) 向下对齐得到 rowNumTile
     * 2. 获取 sub-core 索引/数量（GetSubBlockIdx/Num）
     * 3. 根据 qNBlockSize 决定 sub-core 拆分策略：
     *    - Decode（qNBlockSize==1）：沿行（序列）维拆分
     *    - Prefill（qNBlockSize>1）：沿列（head）维拆分
     * 4. 计算各 sub-core 的行/列偏移，以及 gOutput/gLse 的 tensor slice
     * 5. 计算 rowLoop 数，若 inRowActualThisSubBlock > rowNumTile 则 needRowLoop=1
     * 6. 逐行循环：
     *    - 计算当前轮 rowActualCurLoop、gOutput/gInput/gUpdate 的 tensor slice
     *    - 计算 multi-head 三段式参数（proTokenNum/integralHeadNum/epiTokenNum）
     *    - 调用 SubCoreCompute 执行核心计算
     *
     * @param gOutput          最终输出 O（GM）
     * @param gInput           PV 输出 OTmp（GM workspace）
     * @param gUpdate          O 中间溢出缓冲（GM workspace）
     * @param gLse             LSE 输出（GM，可选）
     * @param layoutOutput     O 布局描述符
     * @param layoutInput      OTmp 布局描述符
     * @param layoutUpdate     gUpdate 布局描述符
     * @param layoutLse        LSE 布局描述符
     * @param actualBlockShape GEMM 形状 {M=rowNum, N=embed}
     * @param qSBlockSize      Q 序列维块大小（每个 head 的行数）
     * @param qNBlockSize      当前 group 内 Q head 数（sub-core 拆分依据）
     * @param isFirstStackTile 首块标志
     * @param isLastStackTile  末块标志
     * @param curStackTileMod  dm 三槽索引（0/1/2）
     */
    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        GemmCoord actualBlockShape,
        uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        // UB 中 go 最大能容纳的行数
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        // 向下对齐到 HALF_BLOCK_SIZE=16，保证 DMA/向量指令对齐
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, HALF_BLOCK_SIZE);

        // 获取当前 sub-core 索引和总 sub-core 数（每 AI Core 2 个 Vector 子核）
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        // ==================== Sub-core 拆分计算 ====================
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0
                                  : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                       : qNSplitSubBlock;
        // 行拆分：Decode 沿行拆分，Prefill 每个 sub-core 处理完整 qSBlockSize 行（head 内全部 token）
        uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
        // 输出偏移：Decode 沿行偏移，Prefill 沿列(head)偏移
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embed;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        // LSE 输出偏移（与 O 同理）
        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ?
            inRowOffsetThisSubBlock : 0;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;
        int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        if (inRowActualThisSubBlock > 0U) {
            // ==================== 行循环（UB 容量溢出处理） ====================
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;

            // 每轮行循环可能跨越多个 head 边界，三段式参数
            uint32_t proTokenIdx = 0;      // prologue 起始 token 索引（在当前轮内）
            uint32_t proTokenIdxPre = 0;   // 预留
            uint32_t proTokenNum = 0;      // prologue token 数（不完整起始 head）
            uint32_t epiTokenNum = 0;      // epilogue token 数（不完整末尾 head）
            uint32_t integralHeadNum = 0;  // 完整 head 数
            uint32_t qSRemian = qSThisSubBlock;
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                // 计算各 GM tensor 的本轮起始偏移
                int64_t offsetOutput =
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput;
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                // 计算三段式参数（prologue/integral/epilogue）
                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(
                    gOutputCurLoop,
                    gInputCurLoop,
                    gUpdateCurLoop,
                    gLseThisSubBlock,
                    layoutOutputCurLoop,
                    layoutInputCurLoop,
                    layoutUpdateCurLoop,
                    layoutOutLseThisSubBlock,
                    qNThisSubBlock,
                    qSThisSubBlock,
                    inRowActualThisSubBlock,
                    isFirstStackTile,
                    isLastStackTile,
                    curStackTileMod,
                    needRowLoop,
                    (rowLoopIdx == rowLoop - 1U),
                    rowOffsetLoop,
                    proTokenIdx,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum);
            }
        }
    }

private:
    // ======================== UB 张量成员 ========================
    AscendC::LocalTensor<half> loUbTensor;                    // OTmp 加载缓冲（当前 tile PV 输出）
    AscendC::LocalTensor<half> dmUbTensor;                    // dm 缩放因子三槽缓冲（每槽 MAX_ROW_NUM_SUB_CORE 行）
    AscendC::LocalTensor<half> hmUbTensor;                    // 预留（softmax 局部 max，rescale 未使用）
    AscendC::LocalTensor<half> glUbTensor;                    // l 全局累加和（最终归一化用，末块）
    AscendC::LocalTensor<half> tvUbTensor;                    // Brcb 广播临时缓冲（half）
    AscendC::LocalTensor<float> tvUbTensor32;                 // Brcb 广播临时缓冲（float，LSE 输出用，与 tvUbTensor 同地址）
    AscendC::LocalTensor<ElementOutput> goUbTensor;           // O 累加结果缓冲
    AscendC::LocalTensor<half> gmUbTensor;                    // m 全局最大值（LSE 计算用，末块）
    AscendC::LocalTensor<half> lse16_ubuf_tensor;             // LSE 中间缓冲（half，与 glUbTensor 同偏移）
    AscendC::LocalTensor<float> lse32_ubuf_tensor;            // LSE 输出缓冲（float，与 gmUbTensor 同偏移）
};

}

#endif
