/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
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

/*
 * ============================================================================================
 * Rescale O Epilogue (低精度 half 版本) —— FlashAttention 输出重缩放收尾阶段
 * ============================================================================================
 *
 * 【定位】
 *   本文件实现 FlashAttention 的最后一个 Epilogue 阶段：Rescale O (输出重缩放) 。
 *   在 Online Softmax 迭代过程中，O (输出) 是逐 KV 块累加的，每次累加都需要用
 *   旧全局行最大值 (gm_old) 与新全局行最大值 (gm_new) 的比值 dm = exp(gm_old - gm_new)
 *   对已累积的 O 进行缩放，最终再用全局行和 (gl) 归一化得到最终输出。
 *
 *   FlashAttention 三阶段流水线 (Vector 引擎侧) ：
 *     ┌──────────────────┐   dm, gl   ┌──────────────────┐   O_new   ┌──────────────────┐
 *     │ Online Softmax    │ ─────────► │ Rescale O (本文件)│ ────────► │   GM (最终输出)    │
 *     │ (计算 dm/gm/gl)   │            │ (O = O*dm + ΔO)   │           │                  │
 *     └──────────────────┘            └──────────────────┘           └──────────────────┘
 *
 * 【核心公式】
 *   Rescale O 的计算分为三种情况 (由 isFirstStackTile / isLastStackTile 控制) ：
 *
 *   1. 首个 KV 块 (isFirstStackTile == true) ：
 *        go = lo           (直接拷贝输入作为初始 O，无需缩放) 
 *
 *   2. 中间 KV 块 (非首非尾) ：
 *        go = lo + go * dm  (旧 O 乘以缩放因子 dm，加上新增量 lo) 
 *        其中 dm = exp(gm_old - gm_new)，由 Online Softmax 阶段计算并传入
 *
 *   3. 最后一个 KV 块 (isLastStackTile == true) ：
 *        go = go / gl      (用全局行和 gl 归一化，得到最终输出) 
 *        若 LSE_MODE == OUT_ONLY：额外计算 lse = ln(gl) + gm 并写入 GM
 *
 * 【低精度特性 (half 版本) 】
 *   - 数据类型：half (fp16) ，所有 UB 张量以 half 为主
 *   - HALF_VECTOR_SIZE = 128：Vector 引擎单次处理 128 个 half 元素
 *   - HALF_BLOCK_SIZE = 16：数据搬运对齐块大小
 *   - UB 容量：MAX_UB_O_ELEM_NUM = 8192 (half 元素) 
 *   - 与 float 版本 (rescale_o.hpp) 的区别：
 *     * float 版本 VECTOR_SIZE=64，half 版本=128
 *     * float 版本有 DownCastP 步骤，half 版本无需降精度
 *     * float 版本 UB 容量更小 (half 元素数翻倍) 
 *
 * 【子核并行】
 *   每个 AI Core 包含 2 个 sub-block，分别处理一半的行 (qN 维或 qS 维切分) 。
 *
 * 【行循环】
 *   当行数超过 UB 容量时 (needRowLoop) ，将行分多轮处理：
 *   每轮处理 rowNumTile 行，中间结果通过 gUpdate (GM) 暂存。
 *
 * ============================================================================================
 */

namespace Catlass::Epilogue::Block {

/*
 * BlockEpilogue 模板特化 —— Rescale O (half 精度) 
 *
 * 模板参数说明：
 *   OutputType_ : 输出 O 的类型包装 (Element + Layout) 
 *   InputType_  : 输入 (旧 O 或增量 O) 的类型包装
 *   UpdateType_ : 中间更新结果 (needRowLoop 时暂存到 GM) 的类型包装
 *   LseType_    : LSE (LogSumExp) 输出的类型包装
 *   LSE_MODE_   : LSE 模式 (NONE / OUT_ONLY / OUT_AND_UPDATE) 
 *
 * 调度策略：EpilogueAtlasA2RescaleOT<LSE_MODE_, half>
 *   A2       = Atlas A2 (昇腾 A2 系列芯片) 
 *   RescaleOT = Rescale O (输出重缩放) + T (Template 模板) 
 *   half      = 半精度浮点 (fp16) 
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
    using DispatchPolicy = EpilogueAtlasA2RescaleOT<LSE_MODE_, half>;  // 调度策略
    using ArchTag = typename DispatchPolicy::ArchTag;                  // 架构标签 (Atlas A2) 

    using ElementOutput = typename OutputType_::Element;  // 输出 O 元素类型
    using ElementInput = typename InputType_::Element;    // 输入元素类型
    using ElementUpdate = typename UpdateType_::Element;  // 更新元素类型
    using ElementLse = typename LseType_::Element;        // LSE 元素类型

    using LayoutOutput = typename OutputType_::Layout;    // 输出布局
    using LayoutInput = typename InputType_::Layout;      // 输入布局
    using LayoutUpdate = typename UpdateType_::Layout;     // 更新布局
    using LayoutLse = typename LseType_::Layout;           // LSE 布局

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;  // LSE 模式常量

    // ============================== 硬件常量 (Hardware Constants)  ==============================
    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;          // 每个数据块含 16 个 half 元素
    static constexpr uint32_t BLOCK_SIZE = 16;                  // 基础块大小 (16 元素对齐) 
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;     // Vector 单次计算 half 元素数
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;     // Vector 单次计算 float 元素数
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;       // 每行 half 元素数
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;      // 每行 float 元素数
    static constexpr uint32_t MULTIPLIER = 2;                   // 倍数 (half:float = 2:1) 
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;              // float 数据块大小 (8 元素) 
    static constexpr uint32_t HALF_BLOCK_SIZE = 16;             // half 数据块大小 (16 元素) 
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;           // float Vector 宽度
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;           // half Vector 宽度 (核心：单次处理 128 个 half) 
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;      // UB uint8 Vector 大小 (1024 字节) 
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;      // UB uint8 块大小 (16384 字节) 
    static constexpr uint32_t HALF_DM_UB_SIZE = 64;             // dm UB 缓冲 half 元素数
    static constexpr uint32_t HALF_LL_UB_SIZE = 256;           // ll (局部行和) UB 缓冲 half 元素数
    static constexpr uint32_t VECTOR_SIZE = 128;                // Vector 宽度 (half 版本) 
    static constexpr uint32_t NUM4 = 4;                         // 常数 4
    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;         // O 矩阵 UB 最大元素数 (half) 
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;        // 每个子核最大行数
    static constexpr uint32_t SIZE_OF_16BIT = 2;                // 16 位数据字节数 (half = 2 字节) 

    /*
     * 构造函数 —— 分配 UB (Unified Buffer) 缓冲区
     *
     * 【作用】在 UB 中分配各张量的存储空间。UB 是 Vector 引擎的片上高速缓存，
     *        所有 Rescale O 计算都在 UB 中完成。
     *
     * 【UB 内存布局】 (按字节偏移分配) 
     *   偏移                              张量              用途
     *   ─────────────────────────────────────────────────────────────
     *   6 * 16384                          loUbTensor        旧 O (Local O，从 GM 加载的已累积 O) 
     *   8 * 16384                          goUbTensor        新 O (Global O，计算结果输出缓冲) 
     *   10 * 16384                         tvUbTensor        临时向量 (temp vector，存放 dm/gl 的广播值) 
     *   10*16384 + 9*1024                  hmUbTensor        行最大值 (half，来自 Online Softmax) 
     *   10*16384 + 10*1024                 gmUbTensor        全局行最大值 (half) 
     *   10*16384 + 10*1024                 lse32_ubuf_tensor LSE (float，中间计算用) 
     *   10*16384 + 12*1024                 glUbTensor        全局行和 (half) 
     *   10*16384 + 12*1024                 lse16_ubuf_tensor LSE (half，最终输出用) 
     *   10*16384 + 13*1024                 dmUbTensor        缩放因子 dm (half，exp(gm_old-gm_new)) 
     *
     * 参数：
     *   resource : 硬件资源句柄 (提供 UB 缓冲区) 
     */
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        // UB 内存偏移常量 (按 16384 字节块 + 1024 字节向量粒度分配) 
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;                       // lo (旧 O) 
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;                         // go (新 O) 
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;                       // tv (临时向量) 

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;   // hm (行最大值 half) 
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;   // gm (全局行最大值 half) 
        constexpr uint32_t LSE32_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE; // lse32 (LSE float) 
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;   // gl (全局行和 half) 
        constexpr uint32_t LSE16_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE; // lse16 (LSE half) 
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;   // dm (缩放因子 half) 

        // 分配各 UB 张量 (部分地址复用，通过类型重解释区分用途) 
        loUbTensor = resource.ubBuf.template GetBufferByByte<half>(LO_UB_TENSOR_OFFSET);                    // 旧 O
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);                   // 缩放因子 dm
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);                    // 全局行和 gl
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);                  // 临时向量 (half 视图) 
        tvUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);                // 临时向量 (float 视图，与 tvUbTensor 共享内存) 
        goUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);          // 新 O (输出) 
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);                   // 行最大值 hm
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);                   // 全局行最大值 gm
        lse16_ubuf_tensor = resource.ubBuf.template GetBufferByByte<half>(LSE16_UB_TENSOR_OFFSET);         // LSE (half) 
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE32_UB_TENSOR_OFFSET);        // LSE (float) 
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    /*
     * SetMask —— 设置 Vector 引擎的掩码 (Mask) 
     *
     * 【作用】Vector 指令按 128 元素 (half) 处理，当实际元素数不足 128 时，
     *        需要通过 Mask 屏蔽多余的元素，避免越界访问或错误计算。
     *
     * 【掩码机制】AscendC Vector 使用 128 位掩码 (高 64 位 + 低 64 位) ：
     *   - len >= 128：全 1 掩码 (-1, -1) ，处理全部 128 元素
     *   - len < 64：仅低 64 位有效，高 64 位为 0
     *   - 64 <= len < 128：低 64 位全 1，高 64 位部分有效
     *
     * 参数：
     *   len : 实际需要处理的元素数
     */
    __aicore__ inline
    void SetMask(int32_t len)
    {
        const int32_t MAX_MASK_LEN = 128;    // 最大掩码长度 (half Vector 宽度) 
        const int32_t HALF_MASK_LEN = 64;    // 半掩码长度 (64 位) 
        if (len >= MAX_MASK_LEN) {
            // 全 1 掩码：处理全部 128 元素
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        // 计算高/低 64 位掩码的有效位数
        int32_t highMask = len - HALF_MASK_LEN > 0 ? len - HALF_MASK_LEN : 0;  // 高 64 位有效位数
        int32_t lowMask = len - HALF_MASK_LEN >= 0 ? HALF_MASK_LEN : len;       // 低 64 位有效位数
        if (len < HALF_MASK_LEN) {
            // 仅低 64 位有效：lowMask 位为 1，高位为 0
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            // 低 64 位全 1，高 64 位 highMask 位为 1
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    /*
     * CopyOToGm —— 将计算完成的 O 从 UB 搬运回 GM
     *
     * 【作用】在 isLastStackTile 阶段，将归一化后的最终输出 O 从 goUbTensor 写回 GM。
     *        支持三段式搬运 (prologue + integral heads + epilogue) ，处理行未对齐情况。
     *
     * 【三段式搬运】每轮处理的行可能跨越多个 head 的边界：
     *   1. prologue 部分 (s > 0) ：起始不完整的 head
     *   2. integral heads (integralHeadNum 个) ：完整的 head，每个 qSThisSubBlock 行
     *   3. epilogue 部分 (epiTokenNum > 0) ：末尾不完整的 head
     *
     * 参数：
     *   gOutput        : 输出 O 的 GM 张量
     *   proTokenIdx    : prologue 部分起始 token 索引
     *   proTokenNum    : prologue 部分 token 数
     *   epiTokenNum   : epilogue 部分 token 数
     *   integralHeadNum: 完整 head 数
     *   qSThisSubBlock : 每个子核的 qS (序列长度) 大小
     *   embed          : 嵌入维度
     *   oHiddenSize    : 输出隐藏层大小 (可能大于 embed，需补齐) 
     */
    __aicore__ inline
    void CopyOToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t epiTokenNum, uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;    // GM 内偏移
        uint32_t innerGOUbOffset = 0;  // UB 内偏移
        // 1. 搬运 prologue 部分 (起始不完整的 head) 
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }
        // 2. 搬运 integral heads (完整 head，每个 qSThisSubBlock 行) 
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }
        // 3. 搬运 epilogue 部分 (末尾不完整的 head) 
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, embed * SIZE_OF_16BIT, 0, (oHiddenSize - embed) * SIZE_OF_16BIT, 0));
        }
    }

    /*
     * SubCoreCompute —— 子核级 Rescale O 核心计算
     *
     * 【作用】执行单个子核 (sub-block) 的 Rescale O 计算，是本文件的核心函数。
     *        根据当前 KV 块的位置 (首/中/尾) 执行不同的计算逻辑。
     *
     * 【计算流程】
     *   ┌─────────────────────────────────────────────────────────────────┐
     *   │ 情况1: isFirstStackTile (首个 KV 块)                             │
     *   │   go = lo   (直接拷贝输入作为初始 O)                             │
     *   ├─────────────────────────────────────────────────────────────────┤
     *   │ 情况2: 中间 KV 块 (非首非尾)                                     │
     *   │   1. 从 GM 加载旧 O 到 loUbTensor                              │
     *   │   2. 广播 dm (缩放因子) 到 tvUbTensor                            │
     *   │   3. go = go * dm   (旧 O 乘以缩放因子)                         │
     *   │   4. go = lo + go   (加上新增量)                                 │
     *   │   若 needRowLoop：将 go 写回 gUpdate (GM) 暂存                   │
     *   ├─────────────────────────────────────────────────────────────────┤
     *   │ 情况3: isLastStackTile (最后一个 KV 块)                          │
     *   │   1. 广播 gl (全局行和) 到 tvUbTensor                            │
     *   │   2. go = go / gl   (归一化得到最终输出)                         │
     *   │   3. CopyOToGm：将 go 写回 GM                                    │
     *   │   若 LSE_MODE == OUT_ONLY：                                      │
     *   │     lse = ln(gl) + gm，cast float，广播，写回 gLse               │
     *   └─────────────────────────────────────────────────────────────────┘
     *
     * 【HardEvent 同步】
     *   V_MTE2   : Vector → MTE2 (Vector 计算完成，可加载新数据) 
     *   MTE2_V   : MTE2 → Vector (GM→UB 搬运完成，Vector 可计算) 
     *   MTE3_MTE2: MTE3 → MTE2 (GM 写回完成，可加载新数据) 
     *   V_MTE3   : Vector → MTE3 (Vector 计算完成，可写回 GM) 
     *   MTE3_V   : MTE3 → Vector (GM 写回完成，Vector 可继续) 
     *
     * 参数：
     *   gOutput/gInput/gUpdate/gLse : 各 GM 张量
     *   layoutOutput/Input/Update/Lse : 各布局
     *   qNThisSubBlock/qSThisSubBlock : 子核的 qN/qS 大小
     *   totalRowNum                   : 总行数
     *   isFirstStackTile/isLastStackTile : 是否首/尾 KV 块
     *   curStackTileMod               : 当前 stack tile 取模 (用于 dm 偏移) 
     *   needRowLoop/isLastRowLoop     : 是否需要行循环/是否最后一轮
     *   rowOffsetLoop                 : 行循环偏移
     *   proTokenIdx/proTokenNum/epiTokenNum/integralHeadNum : 三段式搬运参数
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
        // ---- 参数解析 ----
        uint32_t curRowNum = layoutInput.shape(0);                    // 当前行数
        uint32_t embed = layoutInput.shape(1);                        // 嵌入维度
        uint32_t embedRound = layoutInput.stride(0);                 // 嵌入维对齐后步长
        uint32_t curRowNumRound = RoundUp(curRowNum, HALF_BLOCK_SIZE);  // 行数对齐到 16
        uint32_t qSBlockSize = layoutOutput.shape(0);                // qS 块大小
        uint32_t oHiddenSize = layoutOutput.shape(1);                // 输出隐藏层大小
        uint32_t qHeads = layoutLse.shape(1);                        // 查询头数
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;  // dm UB 偏移

        if (!isFirstStackTile) {
            // ================================================================
            // 情况2: 中间 KV 块 —— go = lo + go * dm
            // ================================================================
            // 1. 从 GM 加载旧 O 到 loUbTensor
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);  // 等待 loUbTensor 可写
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);  // 通知 loUbTensor 可读
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);  // 等待上一轮 MTE3 写回完成
        if (!isFirstStackTile) {
            // 2. 广播 dm (缩放因子) 到 tvUbTensor：每行复用同一个 dm 值
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint16_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,  // 广播次数 = 行数 / 8 (每 8 行一组) 
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                // 行循环时：从 gUpdate 加载上一轮的中间结果 go
                AscendC::DataCopy(
                    goUbTensor, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // 3. go = go * dm (旧 O 乘以缩放因子，逐行广播乘) 
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
            // 处理 embed 不满 128 的尾部
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
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);  // 等待 loUbTensor 可读
            // 4. go = lo + go (加上新增量) 
            AscendC::Add<half, false>(
                goUbTensor,
                goUbTensor,
                loUbTensor,
                (uint64_t)0,
                (curRowNum * embedRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);  // 通知 loUbTensor 可写
        } else {
            // ================================================================
            // 情况1: 首个 KV 块 —— go = lo (直接拷贝输入作为初始 O) 
            // ================================================================
            AscendC::DataCopy(
                goUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // ================================================================
            // 情况3: 最后一个 KV 块 —— go = go / gl，并写回 GM
            // ================================================================
            // 1. 广播 gl (全局行和) 到 tvUbTensor：每行复用同一个 gl 值
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint16_t>(),
                glUbTensor.ReinterpretCast<uint16_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // 2. go = go / gl (归一化，逐行广播除) 
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
            // 处理 embed 不满 128 的尾部
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
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);  // 通知 go 可写回 GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // 3. 将归一化后的 O 写回 GM
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);

            // 4. 若 LSE_MODE == OUT_ONLY：计算并写回 LSE
            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    // lse = ln(gl)：对全局行和取自然对数
                    AscendC::Ln<half, false>(
                        lse16_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, HALF_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    // lse = lse + gm：加上全局行最大值
                    AscendC::Add<half, false>(
                        lse16_ubuf_tensor,
                        lse16_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, CeilDiv(totalRowNum, HALF_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    // half → float 类型转换 (LSE 输出为 float) 
                    AscendC::Cast<float, half, false>(
                        lse32_ubuf_tensor,
                        lse16_ubuf_tensor,
                        AscendC::RoundMode::CAST_NONE,
                        (uint64_t)0, CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 4));
                    AscendC::PipeBarrier<PIPE_V>();

                    // 广播 lse 到 tvUbTensor32 (每 8 行一组) 
                    AscendC::Brcb(
                        tvUbTensor32.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    
                    // 写回 LSE 到 GM (根据 qNThisSubBlock 选择搬运方式) 
                    if (qNThisSubBlock == 0U) {
                        // 单 head：整体搬运
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor32,
                            AscendC::DataCopyExtParams(
                                totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        // 多 head：逐 head 搬运
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
            // ================================================================
            // 中间 KV 块 + 需要行循环：将 go 写回 gUpdate (GM) 暂存
            // ================================================================
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor, AscendC::DataCopyParams(1, curRowNum * embedRound / HALF_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);  // 通知 MTE3 写回完成
    }

    /*
     * operator() —— Rescale O 主入口函数
     *
     * 【作用】协调整个 Rescale O 计算流程：
     *   1. 计算子核切分参数 (qN 维或 qS 维切分到 2 个 sub-block) 
     *   2. 计算行循环参数 (当行数超过 UB 容量时分多轮处理) 
     *   3. 对每轮行循环计算三段式搬运参数 (prologue/integral/epilogue) 
     *   4. 调用 SubCoreCompute 执行实际计算
     *
     * 【子核切分策略】
     *   - qNBlockSize == 1 (单 head) ：按 qS 维切分，每个子核处理一半的行
     *   - qNBlockSize > 1 (多 head) ：按 qN 维切分，每个子核处理一半的 head
     *
     * 【行循环策略】
     *   - maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed：UB 单次最大行数
     *   - rowNumTile = RoundDown(maxRowNumPerLoop, 16)：对齐到 16 的行数
     *   - needRowLoop = (rowLoop > 1)：是否需要多轮
     *
     * 参数：
     *   gOutput/gInput/gUpdate/gLse : 各 GM 张量
     *   layoutOutput/Input/Update/Lse : 各布局
     *   actualBlockShape : 实际块形状 (M=rowNum, N=embed)
     *   qSBlockSize/qNBlockSize : qS/qN 块大小
     *   isFirstStackTile/isLastStackTile/curStackTileMod : KV 块位置标记
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
        // ---- 阶段1：计算行循环参数 ----
        uint32_t rowNum = actualBlockShape.m();                              // 总行数
        uint32_t embed = actualBlockShape.n();                                // 嵌入维度
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;                // UB 单次最大行数
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, HALF_BLOCK_SIZE);  // 对齐到 16 的行数

        // ---- 阶段2：计算子核切分参数 ----
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();  // 当前子核索引 (0 或 1) 
        uint32_t subBlockNum = AscendC::GetSubBlockNum();  // 子核总数 (2) 

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;  // 每个子核的 qN 分块数
        // 当前子核的 qN 数：单 head 时为 0，否则子核 0 取前半、子核 1 取后半
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ? 0
                                  : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock)
                                                       : qNSplitSubBlock;
        // 当前子核的输入行数切分
        uint32_t inRowSplitSubBlock =
            (qNBlockSize == 1U) ? (qSBlockSize / subBlockNum) : (qSBlockSize * qNSplitSubBlock);
        // 当前子核的实际行数 (子核 1 取后半) 
        uint32_t inRowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subBlockIdx * inRowSplitSubBlock;  // 输入行偏移
        // 输出行/列偏移 (单 head 按行切分，多 head 按列切分) 
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1U) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1U) ? 0 : subBlockIdx * qNSplitSubBlock * embed;
        uint32_t qSThisSubBlock = (qNBlockSize == 1U) ? inRowActualThisSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        // LSE 输出偏移 (与 O 的切分策略一致) 
        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1U) ?
            inRowOffsetThisSubBlock : 0;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1U) ?
            0 : subBlockIdx * qNSplitSubBlock;
        int64_t offsetLse =
            layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        // ---- 阶段3：行循环 + 调用 SubCoreCompute ----
        if (inRowActualThisSubBlock > 0U) {
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);  // 行循环次数
            uint32_t needRowLoop = (rowLoop > 1U) ? 1 : 0;                    // 是否需要行循环

            // 每轮处理的行由多个 head 的若干 token 组成，分为三段：
            //   - prologue head (起始不完整) 
            //   - integral heads (完整 head，若干个) 
            //   - epilogue head (末尾不完整) 
            uint32_t proTokenIdx = 0;      // prologue 部分起始 token 索引
            uint32_t proTokenIdxPre = 0;   // 前一轮的 prologue 起始 token 索引
            uint32_t proTokenNum = 0;      // prologue 部分 token 数
            uint32_t epiTokenNum = 0;      // epilogue 部分 token 数
            uint32_t integralHeadNum = 0;  // 完整 head 数
            uint32_t qSRemian = qSThisSubBlock;
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                // 当前轮的行偏移和实际行数
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1U)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                // 计算 GM 偏移
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

                // 三段式分解：prologue + integral + epilogue
                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = AscendC::Std::min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                // 调用子核级计算
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
    // ============================== 私有数据成员 (Private Data Members)  ==============================
    AscendC::LocalTensor<half> loUbTensor;               // 旧 O (Local O，从 GM 加载的已累积 O) 
    AscendC::LocalTensor<half> dmUbTensor;              // 缩放因子 dm = exp(gm_old - gm_new)
    AscendC::LocalTensor<half> hmUbTensor;              // 行最大值 (half，来自 Online Softmax) 
    AscendC::LocalTensor<half> glUbTensor;              // 全局行和 gl (half) 
    AscendC::LocalTensor<half> tvUbTensor;              // 临时向量 (half 视图，存放广播的 dm/gl) 
    AscendC::LocalTensor<float> tvUbTensor32;           // 临时向量 (float 视图，存放广播的 lse) 
    AscendC::LocalTensor<ElementOutput> goUbTensor;     // 新 O (Global O，计算结果输出缓冲) 
    AscendC::LocalTensor<half> gmUbTensor;             // 全局行最大值 gm (half) 
    AscendC::LocalTensor<half> lse16_ubuf_tensor;      // LSE (half，中间计算) 
    AscendC::LocalTensor<float> lse32_ubuf_tensor;      // LSE (float，最终输出) 
};

}

#endif