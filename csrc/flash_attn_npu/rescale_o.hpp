/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

/**
 * ============================================================================
 * rescale_o.hpp —— FlashAttention NPU 前向推理 O 重缩放/累加/归一化 epilogue（高精度 float 版，**实际使用版本**）
 * ============================================================================
 *
 * 【文件定位】
 *   本文件实现 CATLASS 框架 BlockEpilogue 的模板特化，调度策略为
 *   EpilogueAtlasA2RescaleOT<LSE_MODE_, float>（float = FP32 高精度中间计算），
 *   对应 FlashAttention 前向推理最后一个 Vector 核阶段：对 PV 输出 OTmp 做跨 KV stack 的
 *   在线累加和最终归一化，产生最终输出 O 和（可选的）LSE。
 *
 *   ⭐ **重要**：本文件是 flash_api.cpp 中所有 kernel 实例化实际使用的版本
 *   （IntermCalcPrec=float），rescale_o_low_prec.hpp（half版）仅作为预留路径未被实例化。
 *
 * 【与 rescale_o_low_prec.hpp（低精度 half 版）的核心区别】
 *   - 所有中间张量 (lo/dm/gl/tv/hm/gm) 均使用 **float (FP32)** 而非 half；
 *   - 向量指令使用 float 宽度（FLOAT_VECTOR_SIZE=64 vs HALF_VECTOR_SIZE=128）；
 *   - Brcb 以 uint32_t 重解释（float=32位），行对齐到 FLOAT_BLOCK_SIZE=8；
 *   - 末块 Div 后需要 float→half/bf16 的 **Cast**（bf16 用 CAST_RINT 四舍五入，half 用 CAST_NONE）；
 *   - go 使用**双视图共享同一片 UB**：goUbTensor32(float) 和 goUbTensor16(ElementOutput)
 *     指向同一 GO_UB_TENSOR_OFFSET 地址，Cast 结果原地写回；
 *   - LSE 计算全 float 精度，无需 half→float Cast；lse32 直接复用 gl 偏移（gl 已用完）；
 *   - SetMask 简单：单个 64 位 mask 寄存器控制 64 个 float 元素；
 *   - 不需要独立的 tvUbTensor32/lse16_ubuf_tensor/lse32_ubuf_tensor 等额外缓冲。
 *
 * 【核心算法（online softmax 的 O 累加）】
 *   记 m_t、l_t 为 online softmax 维护的全局 running max/sum，
 *   dm_t = exp(m_{t-1} - m_t) 为当前 tile 的缩放因子。
 *
 *   - 首块 (isFirstStackTile)：
 *       O = OTmp              （直接使用 PV 输出，float 精度）
 *   - 中间块：
 *       O = O * dm + OTmp     （旧 O 乘缩放因子后加上新 PV 输出，float精度）
 *   - 末块 (isLastStackTile)：
 *       O = O / l_T           （除以全局 rowsum 完成 softmax 归一化）
 *       O_out = Cast<half/bf16>(O) （float→输出精度转换）
 *       LSE = ln(l_T) + m_T   （可选：输出 log-sum-exp，float精度）
 *
 * 【内存层级（仅 Vector 核，使用 UB）】
 *   GM (gInput=OTmp, gUpdate=O中间量, gOutput=O, gLse=LSE)
 *     │ MTE2 (GM→UB) / MTE3 (GM→UB for update, UB→GM for output)
 *     ▼
 *   UB (Unified Buffer, 静态分区):
 *     - loUbTensor: OTmp（当前 tile 的 PV 输出, float）
 *     - goUbTensor32/goUbTensor16: O 累加结果（float视图/输出视图共享地址）
 *     - dmUbTensor: dm 缩放因子（softmax 产出，三槽 ×256 float）
 *     - glUbTensor: l 全局累加和（最终归一化用, float, 末块复用为lse32）
 *     - gmUbTensor: m 全局最大值（LSE 计算用, float）
 *     - tvUbTensor: 向量广播临时缓冲（float, 同时用于Brcb dm/gl/lse）
 *     - hmUbTensor: 预留（softmax 局部max，rescale 不读）
 *     ── 所有 UB 张量通过固定字节偏移从 ubBuf 分配，偏移与 online_softmax.hpp 完全一致，
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
 *   MAX_UB_O_ELEM_NUM=8192 个元素，float 版本每元素 4B = 32KB。
 *   embed=128(float)时每行128×4B=512B，maxRowNumPerLoop=8192/128=64行。
 *   rowNumTile=RoundDown(64,FLOAT_BLOCK_SIZE=8)=64行。
 *   当 sub-core 行数>64 时多轮 rowLoop，通过 gUpdate(GM) 做溢出写回/读回。
 *
 * 【Multi-head 输出三段式（prologue/integral/epilogue）】
 *   CopyOToGm 将UB连续O转为GM multi-head strided布局：
 *   - prologue: 起始不完整 head 的部分 token
 *   - integral: 若干个完整 head（步长=qSThisSubBlock行）
 *   - epilogue: 末尾不完整 head 的部分 token
 *
 * 【事件同步】
 *   - EVENT_ID0: MTE2_V（lo/go DMA完成）/ V_MTE3（Cast后O可写GM）
 *   - EVENT_ID1: go(gUpdate) DMA完成（行循环溢出加载）
 *   - EVENT_ID3: V_MTE2（lo 可被覆写）
 *   - EVENT_ID4: V_MTE3/MTE3_V（LSE 排空/释放）
 *   - EVENT_ID5: V_MTE3（非末块 go→gUpdate 溢出写回排空）
 *   - EVENT_ID6: MTE3_MTE2（SubCoreCompute 入口排空）
 * ============================================================================
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fa_block.h"

namespace Catlass::Epilogue::Block {

/**
 * @brief BlockEpilogue 对 EpilogueAtlasA2RescaleOT<float> 的特化：高精度 O 重缩放
 *
 * @tparam OutputType_  最终输出 O 的类型 + 布局（half/bf16, RowMajor）
 * @tparam InputType_   PV 输出 OTmp 的类型 + 布局（float/half, RowMajor）
 * @tparam UpdateType_  O 中间溢出缓冲 gUpdate 的类型 + 布局（float）
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
    EpilogueAtlasA2RescaleOT<LSE_MODE_, float>,
    OutputType_,
    InputType_,
    UpdateType_,
    LseType_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;                        // Atlas A2 架构标签

    using ElementOutput = typename OutputType_::Element;                     // O 输出元素类型（half/bf16）
    using ElementInput = typename InputType_::Element;                       // OTmp 输入元素类型（PV 输出）
    using ElementUpdate = typename UpdateType_::Element;                     // gUpdate 元素类型（float）
    using ElementLse = typename LseType_::Element;                           // LSE 元素类型（float）

    using LayoutOutput = typename OutputType_::Layout;                       // O 布局（RowMajor）
    using LayoutInput = typename InputType_::Layout;                         // OTmp 布局
    using LayoutUpdate = typename UpdateType_::Layout;                       // gUpdate 布局
    using LayoutLse = typename LseType_::Layout;                             // LSE 布局

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;           // LSE 输出模式

    // ======================== 编译期常量（float 高精度版本） ========================
    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;                      // half block 元素数（用于oHiddenSize等）
    static constexpr uint32_t BLOCK_SIZE = 16;                               // 通用块大小（对齐）
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;                 // half 向量元素数（DataCopyPad用）
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;                 // float 向量计算元素数
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;                    // half 每行元素数（预留）
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;                   // float 每行元素数（预留）
    static constexpr uint32_t MULTIPLIER = 2;                                // 倍增因子（预留）
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;                          // float 块大小（8个float=32字节）
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;                        // float 向量宽度（64个float=256字节）
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;                   // UB 1KB 向量步长
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;                   // UB 16KB 块步长
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;                          // dm 缓冲大小（预留）
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;                         // ll 缓冲大小（预留）
    static constexpr uint32_t VECTOR_SIZE = 128;                             // 默认向量宽度（用于SetMask判断）
    static constexpr uint32_t NUM4 = 4;                                      // 常量4（预留）
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;                      // UB 中 O 最大元素数（8192个float=32KB）
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;                    // 每个 sub-core 最大行数（dm 三槽×256行）
    static constexpr uint32_t SIZE_OF_16BIT = 2;                             // half/bf16 字节大小（输出DataCopy用）

    /**
     * @brief 构造函数：从 UB 按固定字节偏移分配各张量
     *
     * UB 内存布局（与 online_softmax.hpp 完全一致，实现零拷贝共享）：
     *
     *   偏移(KB)   张量                    大小      用途
     *   ─────────────────────────────────────────────────────────
     *   0-16       (softmax 使用)          16KB     P workspace/mask/S等
     *   ...
     *   96-128     loUbTensor              32KB     OTmp 加载缓冲（当前 tile PV 输出, float）
     *   128-160    goUbTensor32/16         32KB     O 累加（float视图）/ 输出（half视图，共享地址）
     *   160-169    tvUbTensor              ~9KB     Brcb 广播临时缓冲（float）
     *   169-170    hmUbTensor              1KB      预留（softmax 局部max，rescale 未使用）
     *   170-172    gmUbTensor              2KB      m 全局最大值（LSE 用, float）
     *   172-173    glUbTensor/             1KB      l 全局累加和（归一化用, float）
     *              lse32_ubuf_tensor       (1KB)    LSE 输出（float, 末块复用 gl 偏移）
     *   173-...    dmUbTensor              ~3KB     dm 缩放因子三槽(3×256=768个float=3KB)
     *
     * 关键双视图共享：
     *   - goUbTensor32(float) 和 goUbTensor16(ElementOutput) 起始地址相同（128KB），
     *     前者用于 float Mul/Add/Div 计算，后者在末块 Cast 后用于 DataCopyPad 输出。
     *     Cast 指令将 float 结果原地转换为 half/bf16 写入 goUbTensor16（同地址区域，
     *     因 half 元素仅 2B，float 4B，Cast 后数据只占前半部分）。
     */
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        // UB 各张量字节偏移
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;       // 96KB
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;       // 128KB
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;      // 160KB

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;   // 169KB
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;  // 170KB
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;  // 172KB
        constexpr uint32_t LSE_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE; // 与GL复用
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;  // 173KB

        // Allocate UB space
        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);  // 与go16同地址，float视图
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    /**
     * @brief 设置 float 向量 mask（处理非对齐尾向量）
     *
     * float 向量宽度为 64 个元素，使用两个 64 位 mask 寄存器：
     *   - 高64位（maskHigh）: 控制 [0:63] 的前 64 个 float（对应第一个256B）
     *   - 低64位（maskLow） : 控制 [64:127] 的后 64 个 float（对应第二个256B）
     *
     * 但 float 一次向量操作只处理 64 个元素（一个 64 位 mask），
     * 这里的双mask逻辑是为了兼容VECTOR_SIZE=128的通用接口。
     *
     * @param len 实际有效元素数（0~64，因为float向量64元素）
     *
     * 逻辑：
     * - len == VECTOR_SIZE(128): 两个mask全1
     * - len >= FLOAT_VECTOR_SIZE(64): 低64位mask全1，高64位mask按位设置前(len-64)个bit
     * - len < 64: 低64位mask按位设置前len个bit，高64位mask全0
     */
    __aicore__ inline
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = static_cast<uint64_t>(len) % static_cast<uint64_t>(FLOAT_VECTOR_SIZE);
        for (uint64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    /**
     * @brief 将 O 从 UB 写回 GM，支持 multi-head 三段式（prologue/integral/epilogue）
     *
     * 由于行循环中每轮的行可能跨越多个 head 边界，输出时分三段处理：
     *   - prologue: 起始不完整 head 的 proTokenNum 个 token
     *   - integral: integralHeadNum 个完整 head（每个 qSThisSubBlock 行连续）
     *   - epilogue: 末尾不完整 head 的 epiTokenNum 个 token
     *
     * DataCopyPad 参数说明（输出走 half/bf16 视图 goUbTensor16）：
     *   - 每段行数：proTokenNum/qSThisSubBlock/epiTokenNum
     *   - 源行长度 = embed × SIZE_OF_16BIT（连续 half/bf16 元素字节数）
     *   - GM 行间隔（尾部padding）= (oHiddenSize - embed) × SIZE_OF_16BIT
     *     （跳过同一 token 的其他 head 列，实现 strided 写入）
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
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
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
     *   A1. Wait EVENT_ID3(V_MTE2)：等待上轮 lo 缓冲消费完成
     *   A2. DataCopy gInput(OTmp, float) → loUbTensor
     *   A3. Set EVENT_ID0(MTE2_V)：标记 lo DMA 已启动
     *   A4. Wait EVENT_ID6(MTE3_MTE2)：MTE3 管道排空
     *   A5. Brcb<uint32_t>: 将 dm[dmOffset] 按行广播到 tvUbTensor（float scalar→整行）
     *       BrcbRepeatParams(1,8): 每个 float scalar 重复8次 = 256bit = 8 float
     *   A6. 若 needRowLoop：DataCopy gUpdate(GM float) → goUbTensor32（加载上轮溢出go）
     *   A7. 全 mask 下逐 FLOAT_VECTOR_SIZE(64) 执行 go=go*dm（float Mul），尾向量SetMask
     *   A8. Wait EVENT_ID0：lo DMA 完成
     *   A9. 全量 Add: go = lo + go（float Add）
     *   A10. Set EVENT_ID3(V_MTE2)：lo 缓冲可覆写
     *
     * 【A'. 首块 (isFirstStackTile==true)】
     *   A'1. DataCopy gInput(OTmp) → goUbTensor32（直接 go=OTmp，float）
     *   A'2. Set/Wait EVENT_ID0
     *
     * 【B. 末块归一化+输出 (isLastStackTile==true)】
     *   B1. Brcb<uint32_t>: 将 gl[rowOffsetLoop] 按行广播到 tvUbTensor
     *   B2. 逐64元素 Div: go = go / gl（float Div），尾向量SetMask
     *   B3. **float→half/bf16 Cast**：
     *       - bf16 输出: Cast<bf16,float> 用 CAST_RINT（就近舍入到偶数）
     *       - half 输出: Cast<half,float> 用 CAST_NONE（直接截断）
     *       Cast 结果写入 goUbTensor16（与go32同地址，原地转换）
     *       UnaryRepeatParams(1,1,4,8): 4个float→8个bf16/half？ 实际是blk重复
     *   B4. Set/Wait V_MTE3(EVENT_ID0)：等Vector管道排空
     *   B5. CopyOToGm：三段式写回最终 O（用goUbTensor16）
     *   B6. 若 LSE_MODE==OUT_ONLY 且 isLastRowLoop：
     *       - Ln<float>(lse32, gl): lse32=ln(l) float精度
     *       - Add<float>(lse32, lse32, gm): lse32 += m → LSE=ln(l)+m
     *       - Brcb<uint32_t>(tv, lse32): 广播 float LSE 到向量
     *       - DataCopyPad tv → gLse（写回GM float，支持strided）
     *         qNThisSubBlock==0时单块连续写，否则逐head带stride写
     *
     * 【B'. 非末块且 needRowLoop】
     *   B'1. Set/Wait V_MTE3(EVENT_ID5)：等Vector排空
     *   B'2. DataCopy goUbTensor32 → gUpdate(GM float)：溢出写回
     *
     * 【C. 尾部】
     *   Set EVENT_ID6(MTE3_MTE2)：SubCoreCompute 完成，下一轮可进入
     *
     * @param gOutput          最终输出 O（仅末块写，ElementOutput类型）
     * @param gInput           OTmp（PV 输出，当前 tile, float GM）
     * @param gUpdate          O 中间溢出缓冲（float GM）
     * @param gLse             LSE 输出（float GM，可选）
     * @param layoutOutput     O 布局
     * @param layoutInput      OTmp 布局
     * @param layoutUpdate     gUpdate 布局
     * @param layoutLse        LSE 布局
     * @param qNThisSubBlock   当前 sub-core 处理的 Q head 数
     * @param qSThisSubBlock   当前 sub-core 每个 head 的序列行数
     * @param totalRowNum      当前 sub-core 总行数
     * @param isFirstStackTile 首块标志
     * @param isLastStackTile  末块标志
     * @param curStackTileMod  dm 三槽索引（= (stackSeqCount-PRE_LAUNCH)%3）
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
        uint32_t curRowNumRound = RoundUp(curRowNum, FLOAT_BLOCK_SIZE);   // float对齐: 向上对齐到8行
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        // dm 三槽索引：每槽 MAX_ROW_NUM_SUB_CORE=256 float 行，加行偏移
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        if (!isFirstStackTile) {
            // ---- 非首块：加载 lo(OTmp)、加载旧 go(gUpdate)、go=go*dm+lo ----
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            // GM(float OTmp) → UB(float lo)
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        // 等待 MTE3 管道排空（gUpdate/gLse写操作完成）
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            // Brcb: 广播 dm[dmOffset] 到 tvUbTensor（float scalar→每行向量）
            // BrcbRepeatParams(1,8): 1个block重复，每个scalar在block内重复8次=256bit=8个float
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                // 行循环溢出：从 GM(gUpdate, float) 加载上一轮保存的 go
                AscendC::DataCopy(
                    goUbTensor32, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block：逐64元素 float Mul（旧O乘缩放因子exp(m_prev-m_new)）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            // 尾向量：embed非64对齐
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // 等待 lo(OTmp float) DMA 完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go：OTmp 加到重缩放后的旧 O 上（float Add）
            AscendC::Add<float, false>(
                goUbTensor32,
                goUbTensor32,
                loUbTensor,
                (uint64_t)0,
                (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // 通知 lo 缓冲已被消费
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // ---- 首块：go = lo（直接复制 float OTmp 到 O 累加缓冲 go32）----
            AscendC::DataCopy(
                goUbTensor32, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // ---- 末块：除以 l 做最终归一化，Cast到输出精度，写回O，可选LSE ----
            // *** gl_block = expand_to_block(gl)：广播 l（全局rowsum）到整行
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint32_t>(),
                glUbTensor.ReinterpretCast<uint32_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block：逐64元素 float Div（归一化）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<float, false>(
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            // 尾向量
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Div<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // *** go = Cast<float→ElementOutput>(go)：float转half/bf16，原地写入go16
            if (std::is_same<ElementOutput, bfloat16_t>::value) {
                // bf16 输出：CAST_RINT（round to nearest even，就近舍入到偶数）
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            } else {
                // half 输出：CAST_NONE（直接截断，half精度无需额外舍入）
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            }
            // 等待 Cast 完成，Vector 管道排空（go16 数据就绪可 MTE3 写 GM）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // ***move O to GM：三段式 DataCopyPad（用goUbTensor16，half/bf16视图）
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    // ---- LSE 输出：LSE = ln(l) + m（float 精度）----
                    AscendC::PipeBarrier<PIPE_V>();
                    // lse32 = ln(gl) （复用 gl 偏移为 lse32，此时 gl 已被Brcb消费但Div后不再使用）
                    AscendC::Ln<float, false>(
                        lse32_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));

                    AscendC::PipeBarrier<PIPE_V>();
                    // lse32 = ln(gl) + gm = LSE（全 float 精度，无需额外 Cast）
                    AscendC::Add<float, false>(
                        lse32_ubuf_tensor,
                        lse32_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();

                    // *** lse_block = expand_to_block(lse32)：广播 float LSE 到向量
                    AscendC::Brcb(
                        tvUbTensor.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

                    // LSE 写回 GM（float）
                    if (qNThisSubBlock == 0U) {
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor,
                            AscendC::DataCopyExtParams(
                                totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            AscendC::DataCopyPad(
                                gLse[qNIdx],
                                tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                AscendC::DataCopyExtParams(
                                    qSBlockSize, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            }
        } else if (needRowLoop) {
            // ---- 非末块且UB溢出：go(float) → gUpdate(GM float) 保存，供下轮读回 ----
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor32, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        }
        // 标记 SubCoreCompute 完成：MTE3→MTE2 可继续
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    /**
     * @brief 执行 O 重缩放/累加/归一化 epilogue（主入口）
     *
     * 执行流程：
     * 1. 计算行分块：maxRowNumPerLoop = 8192/embed，向下对齐FLOAT_BLOCK_SIZE(8)→rowNumTile
     * 2. 获取 sub-core 索引/总数
     * 3. 根据 qNBlockSize 拆分 sub-core 工作：
     *    - Decode(qNBlockSize==1): inRowSplitSubBlock=qSBlockSize/subBlockNum（沿行拆）
     *    - Prefill(qNBlockSize>1): inRowSplitSubBlock=qSBlockSize*qNSplitSubBlock（沿head列拆）
     * 4. 计算各sub-core的行列偏移、gOutput/gLse tensor slice
     * 5. 计算 rowLoop 轮数、needRowLoop
     * 6. 逐行循环：
     *    - 计算当前轮 rowActualCurLoop、GM tensor slice
     *    - 三段式参数 proTokenNum/integralHeadNum/epiTokenNum
     *    - 调用 SubCoreCompute
     *
     * @param gOutput          最终输出 O（GM half/bf16）
     * @param gInput           PV 输出 OTmp（GM float/half）
     * @param gUpdate          O 中间溢出缓冲（GM float）
     * @param gLse             LSE 输出（GM float）
     * @param layoutOutput     O 布局
     * @param layoutInput      OTmp 布局
     * @param layoutUpdate     gUpdate 布局
     * @param layoutLse        LSE 布局
     * @param actualBlockShape GEMM 形状 {M=rowNum, N=embed}
     * @param qSBlockSize      Q 序列维块大小
     * @param qNBlockSize      当前 group 内 Q head 数
     * @param isFirstStackTile 首块标志
     * @param isLastStackTile  末块标志
     * @param curStackTileMod  dm 三槽索引
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
        // UB go32 最大容纳行数（float: 8192个float=32KB）
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        // 向下对齐到 FLOAT_BLOCK_SIZE=8
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);

        // Sub-core 索引/总数
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        // ==================== Sub-core 拆分 ====================
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0
                                  : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                       : qNSplitSubBlock;
        // 行拆分：Decode沿行/ Prefill每个sub-core处理完整qSBlockSize行
        uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;
        // 输出偏移：Decode沿行/Prefill沿列(head)
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embed;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        // LSE 输出偏移
        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ?
            inRowOffsetThisSubBlock : 0;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;
        int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        if (inRowActualThisSubBlock > 0U) {
            // ==================== 行循环（UB 溢出处理） ====================
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;

            // 三段式参数
            uint32_t proTokenIdx = 0;
            uint32_t proTokenIdxPre = 0;   // 预留，未使用
            uint32_t proTokenNum = 0;
            uint32_t epiTokenNum = 0;
            uint32_t integralHeadNum = 0;
            uint32_t qSRemian = qSThisSubBlock;  // 预留，未使用
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                // GM tensor 本轮偏移
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

                // 三段式参数计算
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
    // ======================== UB 张量成员（float 高精度版本） ========================
    AscendC::LocalTensor<float> loUbTensor;                     // OTmp 加载缓冲（当前tile PV输出, float）
    AscendC::LocalTensor<float> dmUbTensor;                     // dm 缩放因子三槽（float, 3×256行）
    AscendC::LocalTensor<float> hmUbTensor;                     // 预留（softmax 局部max, rescale未使用）
    AscendC::LocalTensor<float> glUbTensor;                     // l 全局累加和（末块归一化, float, 复用为lse32）
    AscendC::LocalTensor<float> tvUbTensor;                     // Brcb 广播临时缓冲（float, dm/gl/lse共用）
    AscendC::LocalTensor<ElementOutput> goUbTensor16;           // O 累加缓冲（输出类型视图, 与go32同地址）
    AscendC::LocalTensor<float> goUbTensor32;                   // O 累加缓冲（float视图, 用于Mul/Add/Div）
    AscendC::LocalTensor<float> gmUbTensor;                     // m 全局最大值（LSE 计算用, float）
    AscendC::LocalTensor<float> lse32_ubuf_tensor;              // LSE 输出缓冲（float, 与gl复用地址）
};

}

#endif
