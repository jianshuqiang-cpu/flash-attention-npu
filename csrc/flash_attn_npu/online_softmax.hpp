/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

/**
 * ============================================================================
 * online_softmax.hpp —— FP32高精度版 Online Softmax Vector Epilogue（实际使用版本）
 * ============================================================================
 *
 * 【文件定位】
 *   本文件是 CATLASS BlockEpilogue 的偏特化版本，模板参数 float 指定中间计算精度
 *   为 FP32（SM_DTYPE_=float）。对应低精度预留版本 online_softmax_low_prec.hpp
 *   （SM_DTYPE_=half，FP16）。
 *
 *   ⚠️ 本文件是 FlashAttention 前向推理中【实际被使用】的版本！flash_api.cpp 中
 *   所有 kernel 实例化均使用 IntermCalcPrec=float，即本高精度版本。low_prec 版本
 *   仅作为预留路径，未被实例化。
 *
 *   在 FlashAttention 前向推理 pipeline 中，本 epilogue 运行在 Vector 核上，位于
 *   Cube 核完成 Q*K^T 之后、Cube 核开始 P*V 之前，由 qkReady/softmaxReady 跨核
 *   事件同步。
 *
 * 【核心算法 —— Online Softmax（分块安全 softmax）】
 *   FlashAttention 将 K/V 序列切成若干 KV stack tile（沿 N 维），逐 tile 流式计算。
 *   设当前为第 t 个 tile，在线 softmax 维护两个跨 tile 累积量：
 *     m_t = max( m_{t-1}, rowmax(S_t) )                                // 全局行最大值
 *     l_t = exp(m_{t-1} - m_t) * l_{t-1} + rowsum(exp(S_t - m_t))     // 全局归一化分母
 *   并输出当前 tile 的概率矩阵（在减去 m_t 后取 exp，未除以 l_t——除留给 rescale_o）：
 *     P_t = exp(S_t - m_t)
 *   P_t 经 DownCast(float→half/bf16) 后写回 GM 供后续 P*V Cube matmul 使用；
 *   m_t/l_t 保留在 UB 中累积传递，到最后一个 tile 后由 rescale_o epilogue 统一做
 *   O 的归一化加权求和。
 *
 * 【与低精度版本(online_softmax_low_prec.hpp)的关键差异】
 *   1. 所有中间计算张量(lm/hm/gm/dm/ll/gl/tv)均使用 float（FP32），计算精度更高；
 *   2. 无需独立的 computeUbTensor：scale/softcap/mask/exp 等计算全部在 lsUbTensor
 *      上原地执行（in-place）；低版本需要单独 compute buffer；
 *   3. lpUbTensor（P 输出）、maskUbTensor（int8 mask）、maskUbTensor32（float mask）
 *      三者共享同一起始偏移（64KB），通过时间复用来节省 UB；低版本有独立布局；
 *   4. maskUbTensor16（UpCast 中间态 half mask）独立位于 176KB，不与其他张量复用；
 *   5. 行规约采用 BlockReduceSum/Max 三级级联结构：SPECTILE512（512列全对齐快速路径）、
 *      SPECTILE256（256列特化路径）、TAILTILE（通用尾块/非对齐路径）；
 *      低版本仅用 WholeReduceSum 两级且无 256 列特化；
 *   6. 支持 softcap（通过 ApplySoftcap 函数实现 c*tanh(x) = 2c/(1+exp(-2x)) - c）；
 *      低版本不支持；
 *   7. 实现 preLoad=1 的 DMA/计算重叠流水线：DMA 搬运当前行块的 S 数据时，
 *      同时计算前一个行块的 softmax，形成乒乓双缓冲流水；低版本无预取；
 *   8. 需要 DownCastP（float→half/bf16 精度转换），bf16 用 CAST_RINT 四舍五入、
 *      half 用 CAST_NONE 直接截断；低版本 S 本身就是 half 无需 Cast；
 *   9. SubCoreCompute 为模板函数 <bool doTriUMask>，有 mask 与无 mask 路径使用
 *      不同的事件同步策略（doTriUMask=true 时不等待 MTE3_V pingpong 事件，
 *      因为 mask 路径的 S 数据由 mask 处理流程保证就绪）；
 *  10. 带 mask 的 operator() 中，row0（第一行块）的 mask 数据在
 *      CrossCoreWaitFlag(qkReady) 之前就预先加载（CopyMaskGmToUb），
 *      利用跨核等待的时间隐藏 mask DMA 延迟——这是关键优化；
 *  11. SetVecMask 针对 float 64 元素向量逐位构建 mask（因为 float 向量宽度为 64，
 *      而非 half 的 128）；CopyS 使用 FLOAT_BLOCK_SIZE=8（float 块大小为 8 个元素，
 *      对应 32 字节；half 块大小为 16 个元素）。
 *
 * 【UB 内存布局（字节偏移，单 Vector 核 ~192KB UB）】
 *
 *     0KB ┌──────────────────────────────────────────┐
 *         │ lsUbTensor (S原始输入+中间计算, float)    │ 64KB
 *         │ pingpong 双缓冲: buffer0[0:32KB],        │   (2 × MAX_UB_S_ELEM_NUM × 4B)
 *         │                 buffer1[32KB:64KB]       │   (2 × 8192 × 4 = 65536B)
 *         │ scale/softcap/sub/exp 均原地操作          │
 *    64KB ├──────────────────────────────────────────┤
 *         │ lpUbTensor (P输出, ElementOutput=half/bf)│ 时间复用区（不同阶段使用不同张量）
 *         │ maskUbTensor (原始mask, ElementMask=int8)│   lp:    2×8192×2B = 32KB
 *         │ maskUbTensor32 (float mask, UpCast后)    │   mask:  ≤8192B
 *         │ 三者时间复用，不同时活跃                   │   mask32:≤8192×4B=32KB
 *         │                                          │   总计 ≤96KB 可用(64KB~160KB)
 *   160KB ├──────────────────────────────────────────┤
 *         │ tvUbTensor (Brcb广播/规约临时, float)    │ 8KB (8×1024B = 2048 floats)
 *  168KB  ├──────────────────────────────────────────┤
 *         │ lm(1KB) │ hm(1KB) │ gm(1KB) │            │ 6KB
 *         │ ll(1KB) │ gl(1KB) │ dm(1KB) │            │ (6个标量/短向量, 各1KB=256 floats)
 *         │ lm=local max, hm=hist max, gm=global max │
 *         │ ll=local sum, gl=global sum, dm=exp差   │
 *   174KB ├──────────────────────────────────────────┤
 *         │ (空闲间隙 ~2KB)                          │
 *   176KB ├──────────────────────────────────────────┤
 *         │ maskUbTensor16 (half mask, UpCast中间态) │ 16KB (8192 half = 16384B)
 *   192KB └──────────────────────────────────────────┘
 *
 * 【preLoad=1 流水线解释】
 *   行循环 for(rowLoopIdx=0; rowLoopIdx<rowLoopNum+preLoad; rowLoopIdx++)：
 *     - 当 rowLoopIdx < rowLoopNum 时：发起当前行块的 DMA 拷贝(CopyS GM→UB)
 *     - 当 rowLoopIdx >= preLoad 时：计算 delayedRowLoopIdx = rowLoopIdx-preLoad
 *       行块的 softmax（ScaleS/Softcap/Mask/SubCoreCompute）
 *   这样 DMA 搬运 rowLoopIdx 行块的 S 数据时，Vector 核正在计算
 *   delayedRowLoopIdx 行块的 softmax，实现 DMA 与计算的并行重叠。
 *   preLoad=1 表示计算比 DMA 延迟 1 个迭代，使用 pingpong 双缓冲（flag 0/1 交替）。
 *
 * 【两个 operator() 重载】
 *   重载1（无 mask）：完全在对角线下方的 KV block 或 NO_MASK 编译期路径使用；
 *                    调用者在外部已 CrossCoreWaitFlag(qkReady)。
 *   重载2（带 causal mask）：跨越因果对角线的 KV block 使用；内部处理跨核同步，
 *                    且 row0 mask 在 qkReady 之前预加载以隐藏延迟。
 * ============================================================================
 */

// 头文件保护宏（注意：宏名沿用原始 low_prec 版本的命名风格，保持一致）
#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T

// ─── 基础框架头文件 ───
#include "catlass/catlass.hpp"                    // CATLASS 主框架头文件
#include "catlass/arch/cross_core_sync.hpp"       // 跨核事件同步（CrossCoreWaitFlag/SetFlag等）
#include "catlass/arch/resource.hpp"              // 硬件资源管理（UB 缓冲区等）
// ─── Epilogue 相关头文件 ───
#include "catlass/epilogue/dispatch_policy.hpp"   // Epilogue 分发策略（EpilogueAtlasA2OnlineSoftmaxT）
#include "catlass/epilogue/tile/tile_copy.hpp"    // Tile 拷贝操作
// ─── 坐标/布局头文件 ───
#include "catlass/gemm_coord.hpp"                 // GEMM 坐标（GemmCoord）
#include "catlass/matrix_coord.hpp"               // 矩阵坐标（MatrixCoord）
// ─── FlashAttention 特定头文件 ───
#include "fa_block.h"                            // FA block 定义（LseModeT 等枚举）

namespace Catlass::Epilogue::Block {

/// @brief FP32 高精度 Online Softmax Vector Epilogue 偏特化
/// @tparam OutputType_  输出类型（half 或 bfloat16_t），决定 DownCast 取整模式
/// @tparam InputType_   输入类型（Cube 输出 S=Q*K^T 的类型，通常为 float）
/// @tparam MaskType_    Mask 类型（通常为 bool/int8，GM 上的 causal mask）
/// @tparam LSE_MODE_    LSE 输出模式（OUT_ONLY 等，控制是否传输 lse）
template <
    class OutputType_,
    class InputType_,
    class MaskType_,
    LseModeT LSE_MODE_>
class BlockEpilogue<
    EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>,
    OutputType_,
    InputType_,
    MaskType_>
{
public:
    // ─── 类型别名 ───
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;   // 输出元素类型 (half/bf16)
    using ElementInput = typename InputType_::Element;     // 输入元素类型 (float)
    using ElementMask = typename MaskType_::Element;       // Mask 元素类型 (bool/int8)

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;

    // ─── 硬件常量 ───
    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;       // 一个 block 的字节数（32B = 256bit）
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;     // 一个 repeat 的字节数
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;          // float 类型一个 block 的元素数（32B/4B=8）
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;        // float 类型一个向量的元素数（一个向量 = 8个block = 64个float = 256B）
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;        // half 类型一个向量的元素数
    static constexpr uint32_t BLOCK_SIZE = 16;               // half 输出类型的 block 元素数（32B/2B=16）
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;   // UB 上一个 uint8 向量的字节数（1KB）
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;   // UB 上一个 block 的字节数（16KB）
    static constexpr uint32_t VECTOR_SIZE = 128;             // 默认向量大小（half 向量宽度）
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;      // lsUbTensor 单个 pingpong buffer 的最大 float 元素数（32KB）

    // ─── 行规约常量 ───
    static constexpr uint32_t REDUCE_UB_SIZE = 1024;             // 规约临时缓冲区 tv 内的偏移量（1024 floats = 4KB）
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;         // 256列特化路径中首次 BlockReduce 后剩余元素数
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;           // 256列特化路径中二次 BlockReduce 后的 mask 长度
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;        // 每个 Vector 核处理的最大行数（决定 dmUbTensor 大小）
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;            // UB 上 float 行对齐大小

    // ─── 构造函数：分配 UB 空间 ───
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_, float softcapValue_ = 0.0f)
    {
        // 分配 UB 空间
        // UB 偏移量计算（单位：字节），基于 UB_UINT8_BLOCK_SIZE=16KB 和 UB_UINT8_VECTOR_SIZE=1KB
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;                                                          // 0KB：S（float）起始
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;                                    // 64KB：P输出（half/bf16）起始
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;                                  // 64KB：原始mask（int8）起始（与lp复用）
        constexpr uint32_t MASK32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;                                // 64KB：float mask起始（与lp/mask复用）

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;                                   // 160KB：临时/广播缓冲区
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;        // 168KB：local max（当前tile行最大值）

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;        // 169KB：hist max（合并后的行最大值）
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;       // 170KB：global max（跨tile累积行最大值）
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;       // 171KB：local sum（当前tile行求和）
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;       // 172KB：global sum（跨tile累积行求和）
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;       // 173KB：delta max（exp(gm-hm)，缩放因子）

        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;                               // 176KB：half mask（UpCast中间态，独立区域）

        // 保存缩放因子和 softcap 值
        scaleValue = scaleValue_;
        softcapValue = softcapValue_;
        // 从资源管理器获取各张量的 UB LocalTensor
        lsUbTensor = resource.ubBuf.template GetBufferByByte<float>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET);
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);
        maskUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(MASK32_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    // ─── 辅助函数：取较小值 ───
    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return (a > b) ? b : a;
    }

    /// @brief 设置向量计算 mask，针对 float 64 元素向量逐位构建
    /// @param len 有效元素长度（0~128）
    /// 由于 float 向量宽度为 64（两个连续的 64 元素块组成 128 位 mask 模式），
    /// 本函数逐位设置 mask 位：len%64 个有效位放在高位或低位 mask 中。
    /// - len == 128 或 0：mask 全 1（不屏蔽）
    /// - len >= 64：高位 mask 全 1，低位 mask 设 (len%64) 位
    /// - len < 64：高位 mask 全 0，低位 mask 设 len 位
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;  // 取余得到不完整向量中的有效元素数
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;                      // 逐位置位
        }

        if (len == VECTOR_SIZE || len == 0) {
            // 完整向量（128个元素）或空：mask 全 1
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            // 超过64但不足128：第一个64全有效，第二个64设temp位
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            // 不足64：仅第二个64有mask位有效
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    /// @brief 设置 BlockReduce 操作的 mask，处理尾块 block 数不足 8 的情况
    /// @param len 有效的 block 数（1~8）
    /// BlockReduce 的每个 repeat 包含 8 个 block，当尾块不足 8 个 block 时需要屏蔽
    /// 多余的 block。mask 按字节分布在 8 个 block 的起始位置。
    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            // 超过8或非法值，全mask
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        // 构造 subMask：低 len 位为 1
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        // 将 subMask 散布到 8 个 byte 位置（每个 byte 控制一个 block）
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    // ─── 行求和（Rowsum）函数：三级级联 BlockReduceSum ───
    // 行规约通过三级 BlockReduceSum 完成：
    //   第一级：block 内（8个元素一组）归约
    //   第二级：block 间归约
    //   第三级：repeat 间归约
    // 根据列数提供三个特化/通用路径：

    /// @brief 512列特化行求和路径（SPECTILE512）
    /// 512列 = 8个repeat × 8个block = 恰好对齐，三级归约参数统一为 8
    __aicore__ inline
    void RowsumSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第一级 BlockReduceSum：block 内（8元素→1元素）归约
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        // 第二级 BlockReduceSum：block 间归约
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        // 第三级 BlockReduceSum：repeat 间归约，得到最终每行一个 sum
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// @brief 256列特化行求和路径（SPECTILE256）
    /// 256列 = 4个repeat × 8个block，第二级归约后剩余 4 个元素，需用 mask 控制第三级
    __aicore__ inline
    void RowsumSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第一级：block 内归约
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        // 第二级：block 间归约，mask=32位有效（对应4个block × 8行布局）
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        // 第三级：repeat 间归约，mask=4个有效block
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    /// @brief 通用尾块行求和路径（TAILTILE），处理任意列数（含非对齐尾向量）
    /// 先处理完整的 64 元素向量循环累加，再处理最后的不完整尾向量
    __aicore__ inline
    void RowsumTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            // 第一个完整 64 元素向量：两级 BlockReduceSum
            AscendC::BlockReduceSum<float, false>(
                tvUbTensor,
                srcUb,
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb,
                tvUbTensor,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            // 后续完整 64 元素向量：规约后与已有的 rowsum 累加
            for (uint64_t rowSumIdx = 1; rowSumIdx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor,
                    srcUb[rowSumIdx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            // 处理不完整尾向量：先 SetVecMask 屏蔽无效元素
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceSum<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            // 尾 block 归约时设置 BlockReduceMask 屏蔽无效 block
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                // 总列数 < 64：直接输出到 rowsumUb（首次写入）
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb,
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // 已有前面向量的累加结果：规约后 Add 到 rowsumUb
                AscendC::BlockReduceSum<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    // ─── 行求最大值（Rowmax）函数：三级级联 BlockReduceMax ───
    // 结构与 Rowsum 完全对称，只是归约操作为 Max 而非 Add。

    /// @brief 512列特化行最大值路径（SPECTILE512），三级全对齐归约
    __aicore__ inline
    void RowmaxSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// @brief 256列特化行最大值路径（SPECTILE256），mask 控制的三级归约
    __aicore__ inline
    void RowmaxSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    /// @brief 通用尾块行最大值路径（TAILTILE），处理任意列数
    /// 与 RowsumTAILTILE 对称，使用 Max 操作合并各向量段的最大值
    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            // 第一个完整向量
            AscendC::BlockReduceMax<float, false>(
                tvUbTensor,
                srcUb,
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb,
                tvUbTensor,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            // 后续完整向量：规约后取 Max 合并
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor,
                    srcUb[rowmax_idx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(
                    tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb,
                    rowmaxUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            // 不完整尾向量
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceMax<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(rowmaxUb,
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceMax<float, false>(tvUbTensor[REDUCE_UB_SIZE],
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(rowmaxUb,
                    rowmaxUb,
                    tvUbTensor[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }

    // ─── DMA 数据搬运函数 ───

    /// @brief 将 S 矩阵从 GM 拷贝到 UB（lsUbTensor 的 pingpong 缓冲区）
    /// 使用 FLOAT_BLOCK_SIZE=8（float 类型）进行 DataCopy
    /// @param gInput GM 上 S 矩阵的起始 GlobalTensor
    /// @param sUbOffset UB 内 pingpong 偏移（0 或 MAX_UB_S_ELEM_NUM）
    /// @param rowNumCurLoop 当前循环处理的行数
    /// @param columnNumRound 列数（向上对齐到 BLOCK_SIZE=16）
    /// @param columnNumPad 行步长（GM 上每行的实际跨度，含 padding）
    __aicore__ inline
    void CopySGmToUb(
        AscendC::GlobalTensor<ElementInput> gInput,
        uint32_t sUbOffset,
        uint32_t rowNumCurLoop,
        uint32_t columnNumRound,
        uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            lsUbTensor[sUbOffset],
            gInput,
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / FLOAT_BLOCK_SIZE,
                (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE, 0));
    }

    /// @brief 将 causal mask 从 GM 拷贝到 UB（maskUbTensor）
    /// 支持 prologue（非对齐头起始部分）、integral heads（完整头）、epilogue（尾部分）三段拷贝
    __aicore__ inline
    void CopyMaskGmToUb(
        AscendC::GlobalTensor<ElementMask> gMask,
        uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead,
        uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum)
    {
        uint32_t innerUbRowOffset = 0;
        // 拷贝 prologue 部分：起始位置不对齐到 token 边界的行
        if (proTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask[proTokenIdx * maskStride],
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        // 拷贝完整的头（integral heads）
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        // 拷贝 epilogue 部分：最后一个不完整头的剩余行
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
        }
    }

    // ─── 向量计算函数 ───

    /// @brief 对 S 乘以缩放因子 scaleValue（=1/sqrt(d_head)）
    /// S = S * scaleValue，原地操作 lsUbTensor
    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::Muls<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            scaleValue,
            (uint64_t)0,
            CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));

        AscendC::PipeBarrier<PIPE_V>();
    }

    /// @brief 应用 softcap：c * tanh(x)，其中 c = softcapValue
    /// 数学公式转换（避免直接计算 tanh，用 exp/reciprocal 等向量化原语实现）：
    ///   c * tanh(x) = c * (exp(2x) - 1) / (exp(2x) + 1)
    ///              = (2c) / (1 + exp(-2x)) - c
    /// 计算步骤（原地操作 lsUbTensor）：
    ///   1) x = -2 * x
    ///   2) x = exp(x)
    ///   3) x = 1 + x
    ///   4) x = 1 / x
    ///   5) x = 2c * x
    ///   6) x = x - c  → 即得 c*tanh(原始x)
    // softcap * tanh(x) = (2 * softcap) / (1 + exp(-2x)) - softcap
    template <bool hasSoftcap>
    __aicore__ inline
    void ApplySoftcap(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        if constexpr (hasSoftcap) {
            uint32_t repeatTimes = CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE);
            AscendC::UnaryRepeatParams unaryParams(1, 1, 8, 8);

            // 步骤1：x = -2 * x
            AscendC::Muls<float, false>(
                lsUbTensor[sUbOffset], lsUbTensor[sUbOffset],
                -2.0f, (uint64_t)0, repeatTimes, unaryParams);
            AscendC::PipeBarrier<PIPE_V>();

            // 步骤2：x = exp(x) = exp(-2*原始x)
            AscendC::Exp<float, false>(
                lsUbTensor[sUbOffset], lsUbTensor[sUbOffset],
                (uint64_t)0, repeatTimes, unaryParams);
            AscendC::PipeBarrier<PIPE_V>();

            // 步骤3：x = 1 + x = 1 + exp(-2x)
            AscendC::Adds<float, false>(
                lsUbTensor[sUbOffset], lsUbTensor[sUbOffset],
                1.0f, (uint64_t)0, repeatTimes, unaryParams);
            AscendC::PipeBarrier<PIPE_V>();

            // 步骤4：x = 1/x = 1/(1+exp(-2x))
            AscendC::Reciprocal<float, false>(
                lsUbTensor[sUbOffset], lsUbTensor[sUbOffset],
                (uint64_t)0, repeatTimes, unaryParams);
            AscendC::PipeBarrier<PIPE_V>();

            // 步骤5：x = 2c * x = 2c/(1+exp(-2x))
            AscendC::Muls<float, false>(
                lsUbTensor[sUbOffset], lsUbTensor[sUbOffset],
                2.0f * softcapValue, (uint64_t)0, repeatTimes, unaryParams);
            AscendC::PipeBarrier<PIPE_V>();

            // 步骤6：x = x - c = c*tanh(原始x)
            AscendC::Adds<float, false>(
                lsUbTensor[sUbOffset], lsUbTensor[sUbOffset],
                -softcapValue, (uint64_t)0, repeatTimes, unaryParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// @brief Mask 类型提升模板函数：将 mask 从 ElementMaskSrc 类型转换为 ElementMaskDst 类型
    /// 两级 UpCast 路径：int8 → half → float，使用 CAST_NONE（不做舍入，直接转换）
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

    /// @brief 应用 causal mask：将 mask 位置的 S 值设为极小值（-3e38），使其在 softmax 后趋近于 0
    /// mask 值为 1 的位置需要被屏蔽：mask32 = mask32 * (-3e38)，然后加到 S 上
    /// 即：S[mask=1] = S[mask=1] + (-3e38)，S[mask=0] = S[mask=0] + 0
    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        // 将 float mask（0/1）乘以极大负数，得到掩码偏移量
        AscendC::Muls<float, false>(
            maskUbTensor32,
            maskUbTensor32,
            (float)-3e38,
            (uint64_t)0,
            CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (maskColumnRound == columnNumRound) {
            // mask 列数与 S 列数相同：整块相加
            AscendC::Add<float, false>(
                lsUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                maskUbTensor32,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            // mask 列数小于 S 列数：仅在需要加 mask 的区域（addMaskUbOffset 起始）相加
            uint32_t loop = maskColumnRound / FLOAT_VECTOR_SIZE;
            for (uint32_t i = 0; i < loop; i++) {
                AscendC::Add<float, false>(lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + i * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[i * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        maskColumnRound / FLOAT_BLOCK_SIZE));
            }
            // 处理尾向量
            if (maskColumnRound % FLOAT_VECTOR_SIZE > 0) {
                SetVecMask(maskColumnRound % FLOAT_VECTOR_SIZE);
                AscendC::Add<float, false>(lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    lsUbTensor[sUbOffset][addMaskUbOffset + loop * FLOAT_VECTOR_SIZE],
                    maskUbTensor32[loop * FLOAT_VECTOR_SIZE],
                    (uint64_t)0,
                    rowNumCurLoop,
                    AscendC::BinaryRepeatParams(
                        1, 1, 1,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        columnNumRound / FLOAT_BLOCK_SIZE,
                        maskColumnRound / FLOAT_BLOCK_SIZE));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ─── Online Softmax 核心计算步骤 ───

    /// @brief 计算当前 tile 的局部行最大值 lm = rowmax(S_t)
    /// 根据列数分派到三个特化路径：512/256/通用
    __aicore__ inline
    void CalcLocalRowMax(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        if (columnNum == 512) {
            RowmaxSPECTILE512(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowmaxSPECTILE256(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowmaxTAILTILE(
                lsUbTensor[sUbOffset],
                lmUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    /// @brief 更新全局行最大值 gm，计算缩放因子 dm
    /// 首个 tile（isFirstStackTile）：直接将 lm 复制为 hm 和 gm
    /// 非首个 tile：
    ///   hm = max(lm, gm)       // 新的全局最大值
    ///   dm = exp(gm - hm)      // 旧 l 的缩放因子（因为 gm 可能比 hm 小）
    ///   gm = hm                // 更新全局最大值
    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // 首个 tile：hm = lm
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)
            AscendC::Max<float, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm
            AscendC::Sub<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm) = exp(gm - hm)，用于缩放旧的 l
            AscendC::Exp<float, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(
            gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// @brief 计算 P_t = exp(S_t - m_t)：减去行最大值后取指数
    /// 步骤：
    ///   1) Brcb 广播 hm 到 tv（每行一个 max 值 → 扩展为整行 block）
    ///   2) ls = ls - hm_block（逐元素减去行最大值）
    ///   3) ls = exp(ls)（逐元素取指数，得到 P_t）
    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t rowOffset)
    {
        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint32_t>(),
            hmUbTensor[rowOffset].template ReinterpretCast<uint32_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = ls - hm_block
        for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_SIZE; ++subIdx) {
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][subIdx * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
        }
        // 处理尾向量
        if (columnNum % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(columnNum % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                lsUbTensor[sUbOffset][columnNum / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUbTensor,
                (uint64_t)0,
                rowNumCurLoop,
                AscendC::BinaryRepeatParams(
                    1, 1, 0, columnNumRound / FLOAT_BLOCK_SIZE, columnNumRound / FLOAT_BLOCK_SIZE, 1));
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = exp(ls)
        AscendC::Exp<float, false>(
            lsUbTensor[sUbOffset],
            lsUbTensor[sUbOffset],
            (uint64_t)0,
            CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// @brief 计算当前 tile 的局部行和 ll = rowsum(P_t) = rowsum(exp(S_t - m_t))
    /// 根据列数分派到三个特化路径
    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        // *** ll = rowsum(ls32)
        if (columnNum == 512) {
            RowsumSPECTILE512(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else if (columnNum == 256) {
            RowsumSPECTILE256(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            RowsumTAILTILE(
                lsUbTensor[sUbOffset],
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    /// @brief 更新全局行和 gl
    /// 首个 tile：gl = ll
    /// 非首个 tile：gl = dm * gl + ll（即 exp(m_{t-1}-m_t)*l_{t-1} + rowsum(P_t)）
    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // *** gl = ll
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // *** gl = dm * gl（旧的 l 乘以缩放因子 exp(gm-hm)）
            AscendC::Mul<float, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl（加上当前 tile 的行和）
            AscendC::Add<float, false>(
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

    /// @brief 将 P（float）降精度转换为输出类型（half 或 bfloat16_t）
    /// bf16 转换使用 CAST_RINT（就近偶数舍入），因为 bf16 尾数较短需要舍入；
    /// half 转换使用 CAST_NONE（直接截断低位），half 尾数足够不需要额外舍入。
    __aicore__ inline
    void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            // bfloat16：使用 CAST_RINT（四舍五入到最近偶数）
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_RINT,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            // float16/half：使用 CAST_NONE（直接截断）
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    /// @brief 将 P（half/bf16）从 UB 拷贝回 GM，供后续 P*V Cube matmul 使用
    /// 使用 BLOCK_SIZE=16（half 块大小），尾部 padding 填 0
    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            gOutput,
            lpUbTensor[sUbOffset],
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    /// @brief 子核心计算模板函数：执行单个行块的完整 online softmax 计算流程
    /// @tparam doTriUMask 是否为带 causal mask 的路径（影响事件同步策略）
    ///
    /// doTriUMask=true（带mask路径）：
    ///   - 不在 CalcExp 后等待 MTE3_V(pingpongFlag)，因为 mask 路径中 S 数据
    ///     由 ApplyMask 前的 MTE2_V 事件保证就绪；
    ///   - DownCastP→SetFlag(V_MTE3) 后不等 V_MTE3，直接 CalcLocalRowSum；
    ///   - CopyPUbToGm 后用 EVENT_ID0 做 MTE3_MTE2 同步（而非 pingpongFlag）。
    ///
    /// doTriUMask=false（无mask路径）：
    ///   - CalcExp 后等待 MTE3_V(pingpongFlag) 确认 S 数据 DMA 完成；
    ///   - DownCastP→SetFlag(V_MTE3) 后等 V_MTE3 确认 lp 就绪再 CopyP；
    ///   - isLastNoMaskStackTile&&isLastRowLoop 时通过 EVENT_ID0 传递同步信号。
    ///
    /// 计算流程：
    ///   CalcLocalRowMax → UpdateGlobalRowMax → CalcExp →
    ///   [Wait MTE3_V(mask路径跳过)] → DownCastP → SetFlag(V_MTE3) →
    ///   CalcLocalRowSum → SetFlag(V_MTE2) →
    ///   Wait V_MTE3 → CopyPUbToGm → [SetFlag MTE3_V / EVENT_ID0] →
    ///   UpdateGlobalRowSum
    template <bool doTriUMask>
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t isFirstRowLoop, uint32_t isLastRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, FLOAT_BLOCK_SIZE);
        uint32_t columnNum = layoutOutput.shape(1);
        uint32_t columnNumPad = layoutOutput.stride(0);
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            // LSE 仅输出模式下，tv 在最后一个 stack tile 用于传输 lse 数据
            // In lse out-only mode, tv is used in the last stack tile to transport lse
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        // 步骤1：计算当前 tile 行最大值
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        // 步骤2：更新全局行最大值，计算 dm 缩放因子
        UpdateGlobalRowMax(
            rowNumCurLoop, rowNumCurLoopRound,
            columnNum, columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);

        // 步骤3：减去行最大值并取指数，得到 P_t = exp(S_t - m_t)
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        if constexpr (!doTriUMask) {
            // 无 mask 路径：等待 S 的 DMA 搬运完成（mask 路径在 ApplyMask 前已保证数据就绪）
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }

        // 步骤4：将 P 从 float 降精度到 half/bf16
        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        // 步骤5：计算当前 tile 行和
        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        // 步骤6：等待 P 在 UB 中就绪后，拷贝回 GM
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        if constexpr (!doTriUMask) {
            // 无 mask 路径：发出 MTE3_V 信号通知下一轮 DMA 可以使用此 pingpong buffer
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                // 最后一个无 mask tile 的最后一行：通过 EVENT_ID0 通知后续
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            // 有 mask 路径：使用 EVENT_ID0 做 mask DMA 与计算的同步
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        // 步骤7：更新全局行和
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    // ─── operator() 重载 ───

    /// @brief operator() 重载1：无 mask 版本
    /// 用于完全在因果对角线下方的 KV block（所有位置均可见，无需 mask）。
    /// 调用者在外部已执行 CrossCoreWaitFlag(qkReady) 等待 Cube 侧 Q*K^T 完成。
    /// 实现 preLoad=1 的 DMA/计算重叠流水线（见文件头部说明）。
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
        const LayoutOutput &layoutOutput, const LayoutInput &layoutInput, GemmCoord actualBlockShape,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);
        uint32_t columnNumPad = layoutInput.stride(0);

        // 子核拆分：根据 SubBlockIdx 将行分配到两个 Vector 核
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        // 计算每次行循环处理的行数（受限于 UB 中 S 缓冲区大小 MAX_UB_S_ELEM_NUM）
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;  // DMA 预取深度为 1，实现乒乓双缓冲流水

        // preLoad=1 流水线主循环（详见文件头部"preLoad=1 流水线解释"）
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                // DMA 阶段：发起当前行块的 S 数据 GM→UB 搬运
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                // 等待 V→MTE2 信号（上一次使用此 pingpong buffer 的计算已完成，可以覆盖）
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                // 计算阶段：处理延迟行块（比 DMA 滞后 preLoad=1 个迭代）
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                // 等待 MTE2→V 信号（S 数据 DMA 完成，可以开始计算）
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                // 乘以缩放因子 scaleValue
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                // 如果启用 softcap，应用 c*tanh(x) 变换
                if (softcapValue > 0.0f) {
                    ApplySoftcap<true>((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                }
                // 执行核心 online softmax 计算（无 mask 路径，doTriUMask=false）
                SubCoreCompute<false>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    isLastNoMaskStackTile,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod);
            }
        }
    }

    /// @brief operator() 重载2：带 causal mask 版本
    /// 用于跨越因果对角线的 KV block（部分位置需要 mask）。
    /// 关键优化：row0（第一行块）的 mask 数据在 CrossCoreWaitFlag(qkReady) 之前
    /// 预先加载，利用跨核等待 Cube 完成的时间隐藏 mask DMA 延迟。
    /// 同样实现 preLoad=1 的 DMA/计算重叠流水线，且 mask 加载与计算形成二级流水。
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown,
        uint32_t kvSStartIdx, uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t columnNum = actualBlockShape.n();
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE_IN_BYTE);
        uint32_t columnNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);
        // 子核拆分
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
            (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        // 每个头的 token 数及 mask 偏移计算
        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
            rowOffsetThisSubBlock : 0;

        // 计算 mask 在 GM 上的偏移和有效列范围（处理因果对角线对齐）
        // calc mask shift in gm
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            // 对角线在当前 block 内部：mask 从 triUp 位置开始
            uint32_t triUpRoundDown = RoundDown(triUp, BLOCK_SIZE_IN_BYTE);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0;
            maskColumn = kvSEndIdx - triUpRoundDown;
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;
        } else {
            // 对角线在当前 block 之前：整行都可能需要 mask
            gmOffsetMaskRow = 0;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = columnNum;
            addMaskUbOffset = 0;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, BLOCK_SIZE_IN_BYTE);

        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        // 行循环参数（同无 mask 版本）
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        // 本子核无行需要处理：仅等待跨核同步后直接返回
        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        // preLoad=1 流水线主循环（含 mask 处理）
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                // DMA 阶段：搬运 S 数据
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // ★ 关键优化：row0（第一行块）的 mask 在 CrossCoreWaitFlag(qkReady) 之前预加载
                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // 计算 prologue/integral/epilogue 三段 mask 拷贝参数
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    // 等待 EVENT_ID0（mask buffer 空闲）→ 拷贝 mask → 发出 EVENT_ID2
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    // 等待 Cube 侧 Q*K^T 完成（mask 已提前发起 DMA，隐藏延迟）
                    Arch::CrossCoreWaitFlag(qkReady);
                }
                // 搬运 S 数据 GM→UB（pingpong 双缓冲）
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            if (rowLoopIdx >= preLoad) {
                // 计算阶段：处理延迟行块
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                // 等待 mask 数据就绪，执行两级 UpCast：int8→half→float
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
                UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);
                
                // 等待 S 数据就绪，执行 scale、softcap、mask
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                if (softcapValue > 0.0f) {
                    ApplySoftcap<true>((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                }
                ApplyMask(
                    (pingpongFlag * MAX_UB_S_ELEM_NUM),
                    rowNumCurLoop, columnNumRound,
                    maskColumnRound, addMaskUbOffset);
                // 预取下一个行块的 mask（与当前 softmax 计算并行）
                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                // online softmax vectorized compute
                // 执行核心 online softmax 计算（带 mask 路径，doTriUMask=true）
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                SubCoreCompute<true>(
                    gOutputCurLoop,
                    layoutOutputCurLoop,
                    rowOffsetCurLoop,
                    isFirstStackTile,
                    0,
                    (delayedRowLoopIdx == 0),
                    (delayedRowLoopIdx == rowLoopNum - 1),
                    columnNumRound,
                    pingpongFlag,
                    curStackTileMod);
            }
        }
    }

private:
    // ─── 成员变量 ───

    float scaleValue;                                        // softmax 缩放因子（=1/sqrt(d_head)），在 ScaleS 中使用
    float softcapValue;                                      // softcap 参数 c，若 >0 则应用 c*tanh(x) 变换；0 表示不启用
    AscendC::LocalTensor<float> lsUbTensor;                  // S 张量 UB 缓冲区（float），pingpong双缓冲，0KB起始，64KB大小；
                                                             // scale/softcap/mask/sub/exp 均原地操作
    AscendC::LocalTensor<ElementOutput> lpUbTensor;          // P 张量 UB 缓冲区（half/bf16），64KB起始，与mask/mask32时间复用；
                                                             // 存放 DownCast 后的概率矩阵，供 GM 写回
    AscendC::LocalTensor<ElementMask> maskUbTensor;          // 原始 mask UB 缓冲区（ElementMask=int8），64KB起始，与lp/mask32时间复用；
                                                             // GM→UB 直接搬运着陆区
    AscendC::LocalTensor<half> maskUbTensor16;               // half 类型 mask 中间缓冲区，176KB起始，16KB独立区域；
                                                             // UpCast 第一级（int8→half）的目标
    AscendC::LocalTensor<float> maskUbTensor32;              // float 类型 mask 缓冲区，64KB起始，与lp/mask时间复用；
                                                             // UpCast 第二级（half→float）的目标，ApplyMask 使用
    AscendC::LocalTensor<float> lmUbTensor;                  // local max：当前 stack tile 内行最大值，168KB起始，1KB；
                                                             // 每次 Rowmax 后存储当前 tile 的局部行最大值
    AscendC::LocalTensor<float> hmUbTensor;                  // hist max：合并后的历史行最大值，169KB起始，1KB；
                                                             // = max(lm, gm)，即更新后的全局行最大值
    AscendC::LocalTensor<float> gmUbTensor;                  // global max：跨 tile 累积的全局行最大值，170KB起始，1KB；
                                                             // online softmax 中的 m_{t-1}
    AscendC::LocalTensor<float> dmUbTensor;                  // delta max：exp(gm - hm)，173KB起始，1KB；
                                                             // 用于缩放旧的全局行和 l_{t-1}，按 curStackTileMod*MAX_ROW+rowOffset 索引
    AscendC::LocalTensor<float> llUbTensor;                  // local sum：当前 stack tile 内行求和，171KB起始，1KB；
                                                             // 每次 Rowsum 后存储当前 tile 的 exp(S-m_t) 行和
    AscendC::LocalTensor<float> tvUbTensor;                  // temporary vector：临时/广播缓冲区，160KB起始，8KB；
                                                             // Brcb 广播目标、BlockReduce 临时存储等多用途
    AscendC::LocalTensor<float> glUbTensor;                  // global sum：跨 tile 累积的全局行求和（归一化分母），172KB起始，1KB；
                                                             // online softmax 中的 l_t = dm*l_{t-1} + ll
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T