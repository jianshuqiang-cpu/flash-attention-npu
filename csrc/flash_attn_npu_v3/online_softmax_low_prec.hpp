/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// ============================================================================
// 文件说明：online_softmax_low_prec.hpp
// ----------------------------------------------------------------------------
// 本文件实现 FlashAttention v3 的【低精度（half）在线 Softmax】Epilogue 模块。
// 运行于 Ascend NPU 的 Vector 引擎，与 Cube 引擎（负责 QK^T / PV 矩阵乘）协同工作。
//
// 在线 Softmax 算法核心思想（避免物化完整注意力矩阵）：
//   对每个 KV 块迭代：
//     1. Cube 引擎计算 S = QK^T，结果存于 GM
//     2. Vector 引擎拷贝 S 到 UB（Unified Buffer）
//     3. 缩放 S = S * scaleValue（通常 scaleValue = 1/sqrt(d)）
//     4. （可选）应用注意力掩码 Mask
//     5. 计算局部行最大值 lm = max(S_row)
//     6. 更新全局行最大值 gm = max(gm, lm)，并计算缩放因子 dm = exp(gm_old - gm_new)
//     7. 计算 P = exp(S - gm_new)（即 Softmax 分子部分）
//     8. 计算局部行和 ll = sum(P_row)
//     9. 更新全局行和 gl = dm * gl + ll（对历史结果做缩放累加）
//    10. 拷贝 P 到 GM，供 Cube 引擎执行 PV 矩阵乘
//
// 跨引擎同步：通过 Arch::CrossCoreFlag 信号量（qkReady）等待 Cube 引擎完成 QK^T
// ============================================================================

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP_T
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_LOW_PREC_HPP_T

// CATLASS 框架头文件（华为 CANN 模板库，类似 NVIDIA CUTLASS）
#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"   // 跨核同步（Cube↔Vector 信号量）
#include "catlass/arch/resource.hpp"          // 硬件资源管理（UB/L0/L1 缓冲区分配）
#include "catlass/epilogue/dispatch_policy.hpp" // Epilogue 调度策略
#include "catlass/epilogue/tile/tile_copy.hpp"  // Tile 拷贝原语
#include "catlass/gemm_coord.hpp"              // 矩阵坐标（M/N/K）
#include "catlass/matrix_coord.hpp"            // 矩阵元素坐标
#include "fa_block.h"                          // FlashAttention 块定义（调度策略标签）

namespace Catlass::Epilogue::Block {

// ============================================================================
// BlockEpilogue 类模板（低精度 half 特化）
// ----------------------------------------------------------------------------
// 模板参数：
//   - OutputType_  : 输出类型包装（含 Element + Layout）
//   - InputType_   : 输入类型包装（S 矩阵，此处为 half）
//   - MaskType_    : 掩码类型包装
//   - LSE_MODE_    : LogSumExp 输出模式（NONE/OUT_ONLY/OUT_AND_UPDATE）
// 调度策略特化：EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>
//   第二个模板参数 half 表示本文件为低精度（half）特化版本
// ============================================================================
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
    // 类型别名派生：从调度策略和类型包装中提取实际元素类型与布局
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>;
    using ArchTag = typename DispatchPolicy::ArchTag;       // 架构标签（如 AtlasA2）
    using ElementOutput = typename OutputType_::Element;     // 输出元素类型（如 half/float）
    using ElementInput = typename InputType_::Element;       // 输入元素类型（S 矩阵，half）
    using ElementMask = typename MaskType_::Element;         // 掩码元素类型（如 int8/uint8）

    // 布局类型别名
    using LayoutOutput = typename OutputType_::Layout;       // 输出布局（P 矩阵）
    using LayoutInput = typename InputType_::Layout;         // 输入布局（S 矩阵）
    using LayoutMask = typename MaskType_::Layout;           // 掩码布局

    // LogSumExp 输出模式：NONE（不输出）/ OUT_ONLY（仅输出）/ OUT_AND_UPDATE（输出并更新）
    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;

    // ---------------------- 硬件相关常量 ----------------------
    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;        // 块大小（字节），Ascend 对齐基本单位
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;      // 单次重复处理的字节数
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;            // float 类型一个 block 的元素数（32B/4B）
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;          // float 向量长度（256B/4B）
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;          // half 向量长度（256B/2B），关键拆分粒度
    static constexpr uint32_t BLOCK_SIZE = 16;                 // half 类型一个 block 的元素数（32B/2B）
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;     // UB 中 uint8 向量大小（字节）
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;      // UB 中 uint8 块大小（字节），用于地址偏移计算
    static constexpr uint32_t VECTOR_SIZE = 128;              // 默认向量处理宽度
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;       // UB 中 S 矩阵单块最大元素数（用于 Ping-Pong）

    // 行归约相关常量
    static constexpr uint32_t REDUCE_UB_SIZE = 1024;           // 归约用 UB 大小
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;       // 行操作掩码 32
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;         // 行操作掩码 8
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;         // 行操作掩码 4
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;         // 行操作掩码 2
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;       // 单子核最大行数（dm 缓冲区维度）
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;          // UB 中 float 单行大小

    // 列拆分索引常量（用于 512 列特化的 4 路拆分）
    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    // ============================================================================
    // 构造函数：分配 UB（Unified Buffer）空间并初始化各 Tensor
    // ----------------------------------------------------------------------------
    // UB 内存布局（按字节偏移划分）：
    //   [0, 2*UB_UINT8_BLOCK_SIZE)          : lsUbTensor      —— S 矩阵输入缓冲（Ping-Pong）
    //   [2*UB_UINT8_BLOCK_SIZE, 4*...)      : computeUbTensor —— 计算中间结果（缩放/exp 后的 S/P）
    //   [4*UB_UINT8_BLOCK_SIZE, ...)        : lpUbTensor       —— P 矩阵输出缓冲（拷贝到 GM）
    //   [0, ...)                            : maskUbTensor16  —— half 掩掩码（与 lsUbTensor 共享低地址区）
    //   [10*UB_UINT8_BLOCK_SIZE, ...)       : tv/lm/hm/gm/ll/gl/dmUbTensor —— 行归约统计量
    //   [11*UB_UINT8_BLOCK_SIZE, ...)       : maskUbTensor     —— 原始掩码缓冲
    // 参数：
    //   resource   : 硬件资源句柄（提供 ubBuf）
    //   scaleValue_: Softmax 缩放因子（通常为 1/sqrt(head_dim)），转为 half 存储
    // ============================================================================
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        // 各 UB Tensor 的字节偏移起点（基于 UB_UINT8_BLOCK_SIZE=16384 和 UB_UINT8_VECTOR_SIZE=1024 计算）
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;                    // S 输入缓冲偏移
        constexpr uint32_t COMPUTE_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;  // 计算缓冲偏移
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;       // P 输出缓冲偏移
        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 0;               // half 掩掩码偏移（复用低地址）

        // 行归约统计量区域（从 10*UB_UINT8_BLOCK_SIZE 开始，按向量大小递增）
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;                       // 临时向量（广播用）
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;  // 局部行最大值 lm

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;  // 当前行最大值 hm
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE; // 全局行最大值 gm
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;  // 局部行和 ll
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;  // 全局行和 gl
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;  // 缩放因子 dm=exp(gm_old-gm_new)

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;  // 原始掩码缓冲偏移

        // 保存缩放因子（float 转 half，后续 Muls 使用）
        scaleValue = static_cast<half>(scaleValue_);
        // 绑定各 UB Tensor 到对应偏移地址
        lsUbTensor = resource.ubBuf.template GetBufferByByte<half>(LS_UB_TENSOR_OFFSET);          // S 输入
        computeUbTensor = resource.ubBuf.template GetBufferByByte<half>(COMPUTE_UB_TENSOR_OFFSET); // 计算中间结果
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);  // P 输出
        maskUbTensor = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_UB_TENSOR_OFFSET); // 原始掩码
        maskUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_UB_TENSOR_OFFSET);   // half 掩掩码
        lmUbTensor = resource.ubBuf.template GetBufferByByte<half>(LM_UB_TENSOR_OFFSET);          // 局部行最大值
        hmUbTensor = resource.ubBuf.template GetBufferByByte<half>(HM_UB_TENSOR_OFFSET);          // 当前行最大值
        gmUbTensor = resource.ubBuf.template GetBufferByByte<half>(GM_UB_TENSOR_OFFSET);           // 全局行最大值
        dmUbTensor = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);          // 缩放因子
        llUbTensor = resource.ubBuf.template GetBufferByByte<half>(LL_UB_TENSOR_OFFSET);          // 局部行和
        tvUbTensor = resource.ubBuf.template GetBufferByByte<half>(TV_UB_TENSOR_OFFSET);          // 临时向量
        glUbTensor = resource.ubBuf.template GetBufferByByte<half>(GL_UB_TENSOR_OFFSET);          // 全局行和
    }

    // 析构函数（无资源需手动释放，UB 由框架统一管理）
    __aicore__ inline
    ~BlockEpilogue() {}

    // ============================================================================
    // SetVecMask：设置 Vector 引擎的掩码寄存器，用于处理不足一个完整向量的尾部元素
    // ----------------------------------------------------------------------------
    // Ascend Vector 引擎以 128 元素（half）为一个向量单位处理数据。当实际元素数 len
    // 不足 128 时，需通过掩码屏蔽多余元素，避免越界读写。
    // 掩码分为高 64 位和低 64 位，共 128 bit 对应 128 个元素。
    // 参数：len - 实际需要处理的元素数（1~128）
    // ============================================================================
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        const int32_t MAX_MASK_LEN = 128;   // 最大掩码长度（一个完整向量）
        const int32_t HALF_MASK_LEN = 64;   // 半个向量长度（掩码分高低两段）
        if (len >= MAX_MASK_LEN) {
            // 元素数 >= 128，使用全 1 掩码（处理完整向量）
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        // 计算高 64 位和低 64 位的有效元素数
        int32_t highMask = len - HALF_MASK_LEN > 0 ? len - HALF_MASK_LEN : 0;  // 高段有效位数
        int32_t lowMask = len - HALF_MASK_LEN >= 0 ? HALF_MASK_LEN : len;       // 低段有效位数
        if (len < HALF_MASK_LEN) {
            // 元素数 < 64：仅低 64 位有有效位，高 64 位全 0
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            // 元素数 >= 64：低 64 位全 1，高 64 位部分有效
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    // ============================================================================
    // SetBlockReduceMask：设置块归约（WholeReduceSum/Max）的掩码
    // ----------------------------------------------------------------------------
    // 块归约指令以 16 元素为一组进行归约。当有效元素数 len <= 16 时，需设置掩码
    // 屏蔽无效元素。掩码模式：将 16 位子掩码复制到 4 个 16 位段（共 64 位）。
    // 参数：len - 有效元素数（1~16）
    // ============================================================================
    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        const int32_t MAX_LEN = 16;  // 块归约最大有效长度
        if (len > MAX_LEN) {
            // 超过 16，使用全 1 掩码
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        // 构造 16 位子掩码，然后复制到 4 个 16 位段（48/32/16/0 位偏移）
        uint64_t subMask = (static_cast<uint64_t>(1) << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    // ============================================================================
    // RowsumSPECTILE512：针对 512 列特化块的行求和（P 矩阵每行元素求和）
    // ----------------------------------------------------------------------------
    // 512 列恰好是 4 个 HALF_VECTOR_SIZE(128)，采用 4 路拆分归约优化：
    //   步骤1: src[0:128] += src[128:256]       （第0段 += 第1段）
    //   步骤2: src[256:384] += src[384:512]     （第2段 += 第3段）
    //   步骤3: src[0:128] += src[256:384]       （合并前两步结果）
    //   步骤4: WholeReduceSum 对剩余 128 元素做最终归约
    // 参数：
    //   srcUb           : 输入 P 矩阵（computeUbTensor）
    //   rowsumUb        : 输出行和结果（llUbTensor）
    //   tvUbTensor      : 临时缓冲（本函数未使用，保留接口一致性）
    //   numRowsRound    : 行数（按 BLOCK_SIZE 对齐）
    //   numElems        : 实际列数（=512）
    //   numElemsAligned : 对齐后列数
    // ============================================================================
    __aicore__ inline
    void RowsumSPECTILE512(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowsumUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 步骤1: 第0段(0~128) += 第1段(128~256)
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
        // 步骤2: 第2段(256~384) += 第3段(384~512)
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
        // 步骤3: 第0段 += 第2段（合并步骤1和2的结果）
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
        // 步骤4: 对剩余 128 元素做块归约求和
        AscendC::WholeReduceSum<half, false>(
            rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
            numElemsAligned / BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // RowsumTAILTILE：通用行求和（处理任意列数的尾部块）
    // ----------------------------------------------------------------------------
    // 当列数不是 512 时使用此函数。算法：
    //   - 若列数 <= 128：直接 WholeReduceSum
    //   - 若列数 > 128：先逐段(每128)累加到第0段，再 WholeReduceSum
    //   - 尾部不足 128 的部分通过 SetVecMask 屏蔽后累加
    // 参数同 RowsumSPECTILE512
    // ============================================================================
    __aicore__ inline
    void RowsumTAILTILE(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowsumUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems <= HALF_VECTOR_SIZE) {
            // 列数 <= 128：设置掩码后直接归约
            SetVecMask(numElems);
            AscendC::WholeReduceSum<half, false>(
                rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            // 列数 > 128：逐段累加到第0段
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
            // 处理尾部不足 128 的部分
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
            // 对累加后的第0段做最终归约
            AscendC::WholeReduceSum<half, false>(
                rowsumUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // RowmaxTAILTILE：通用行求最大值（处理任意列数的尾部块）
    // ----------------------------------------------------------------------------
    // 算法与 RowsumTAILTILE 类似，但使用 Max 和 WholeReduceMax：
    //   - 若列数 <= 128：直接 WholeReduceMax
    //   - 若列数 > 128：先拷贝第0段到 lsUbTensor，再逐段取 Max 累加，最后归约
    // 注意：求 Max 时不能原地累加（会污染原始数据），故先 DataCopy 到 lsUbTensor
    // 参数：
    //   srcUb           : 输入（P 矩阵或 S 矩阵）
    //   rowmaxUb        : 输出行最大值结果（lmUbTensor）
    //   tvUbTensor      : 临时缓冲（未使用）
    //   numRowsRound    : 行数（对齐）
    //   numElems        : 实际列数
    //   numElemsAligned : 对齐列数
    // ============================================================================
    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<half> &srcUb, const AscendC::LocalTensor<half> &rowmaxUb,
        const AscendC::LocalTensor<half> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        if (numElems <= HALF_VECTOR_SIZE) {
            // 列数 <= 128：设置掩码后直接归约求最大值
            SetVecMask(numElems);
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, srcUb, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            // 列数 > 128：先拷贝第0段(0~128)到 lsUbTensor 作为累加器
            AscendC::DataCopy(
                lsUbTensor,
                srcUb,
                AscendC::DataCopyParams(
                    numRowsRound,
                    HALF_VECTOR_SIZE / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE,
                    (numElemsAligned - HALF_VECTOR_SIZE) / BLOCK_SIZE));
            AscendC::PipeBarrier<PIPE_V>();
            // 逐段(每128)与累加器取 Max
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
            // 处理尾部不足 128 的部分
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
            // 对累加后的第0段做最终归约求最大值
            AscendC::WholeReduceMax<half, false>(
                rowmaxUb, lsUbTensor, (int32_t)0, numRowsRound, 1, 1,
                numElemsAligned / BLOCK_SIZE, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // CopySGmToUb：将 S 矩阵（QK^T 结果）从 GM 拷贝到 UB（lsUbTensor）
    // ----------------------------------------------------------------------------
    // 使用 DataCopy 指令，支持行间 stride 跳过 padding 区域。
    // 参数：
    //   gInput         : GM 中的 S 矩阵
    //   sUbOffset      : UB 内偏移（用于 Ping-Pong，0 或 MAX_UB_S_ELEM_NUM）
    //   rowNumCurLoop  : 当前行数
    //   columnNumRound : 对齐列数（BLOCK_SIZE 的倍数）
    //   columnNumPad    : padding 后列数（含 stride 间隙）
    // ============================================================================
    __aicore__ inline
    void CopySGmToUb(AscendC::GlobalTensor<half> gInput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(
            lsUbTensor,
            gInput,
            AscendC::DataCopyParams(rowNumCurLoop,
                columnNumRound / BLOCK_SIZE,                          // 每行有效 block 数
                (columnNumPad - columnNumRound) / BLOCK_SIZE,        // 行间 stride（跳过 padding）
                0));
    }

    // ============================================================================
    // CopyMaskGmToUb：将注意力掩码从 GM 拷贝到 UB（maskUbTensor）
    // ----------------------------------------------------------------------------
    // 掩码布局可能因 GQA（分组查询注意力）而需要按 head 重复拷贝。
    // 本函数将一行行的 mask 数据按 [proTokenNum, integralHeadNum×tokenNumPerHead, epiTokenNum]
    // 三段拷贝到 UB，支持跨 head 的紧凑布局展开。
    // 参数：
    //   gMask              : GM 中的掩码
    //   columnNum          : 实际列数
    //   columnNumRound     : 对齐列数
    //   maskStride         : 掩码行 stride
    //   tokenNumPerHead    : 每个 head 的 token 数
    //   proTokenIdx        : 前导 token 索引（跨 head 边界的前段）
    //   proTokenNum        : 前导 token 数
    //   integralHeadNum    : 完整 head 数
    //   epiTokenNum        : 尾部 token 数
    // ============================================================================
    __aicore__ inline
    void CopyMaskGmToUb(AscendC::GlobalTensor<ElementMask> gMask, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead, uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum)
    {
        uint32_t innerUbRowOffset = 0;
        // 第一段：前导 token（跨 head 边界的前段，从 proTokenIdx 开始）
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
        // 第二段：完整 head 数（每个 head 拷贝 tokenNumPerHead 行）
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
        // 第三段：尾部 token（不足一个完整 head 的剩余部分）
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

    // ============================================================================
    // ScaleS：对 S 矩阵做缩放 S = S * scaleValue
    // ----------------------------------------------------------------------------
    // scaleValue 通常为 1/sqrt(head_dim)，是 Softmax 的标准缩放。
    // 使用 Muls（标量乘法）指令，结果存于 computeUbTensor。
    // 参数：
    //   sUbOffset      : UB 偏移（Ping-Pong）
    //   rowNumCurLoop  : 行数
    //   columnNumRound: 对齐列数
    // ============================================================================
    __aicore__ inline
    void ScaleS(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** ls = scaleValue * ls
        // Muls: computeUbTensor = lsUbTensor * scaleValue（标量乘法）
        AscendC::Muls<half, false>(
            computeUbTensor,
            lsUbTensor,
            scaleValue,
            (uint64_t)0,
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // UpCastMask：将掩码从原始类型（如 int8/uint8）上转换为 half
    // ----------------------------------------------------------------------------
    // 后续 ApplyMask 需要与 half 类型的 S 矩阵做加法，故需先将掩码转为 half。
    // 使用 Cast 指令，CAST_NONE 表示不做舍入。
    // 模板参数：
    //   ElementMaskDst : 目标类型（half）
    //   ElementMaskSrc : 源类型（如 int8）
    // ============================================================================
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

    // ============================================================================
    // ApplyMask：将掩码应用到 S 矩阵（被屏蔽位置置为 -inf）
    // ----------------------------------------------------------------------------
    // 算法：
    //   1. mask16 = mask16 * (-65504)  —— 将掩码 0/1 转为 0/-65504（half 的 -inf 近似）
    //   2. computeUb = computeUb + mask16  —— 屏蔽位置加上 -65504，Softmax 后趋近于 0
    // 当 maskColumnRound == columnNumRound 时，掩码与 S 列对齐，直接整块加；
    // 否则按 HALF_VECTOR_SIZE 分段加（处理因果掩码的部分列覆盖）。
    // 参数：
    //   sUbOffset       : UB 偏移
    //   rowNumCurLoop   : 行数
    //   columnNumRound  : S 矩阵对齐列数
    //   maskColumnRound : 掩码对齐列数（可能小于 columnNumRound）
    //   addMaskUbOffset : 掩码在 S 中的列偏移（用于因果掩码的对角线偏移）
    // ============================================================================
    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        // 步骤1: mask16 = mask16 * (-65504)，将 0/1 掩码转为 0/-inf
        AscendC::Muls<half, false>(
            maskUbTensor16,
            maskUbTensor16,
            (half)-6e4, // -65504（half 最小值近似，作为 -inf）
            (uint64_t)0,
            (rowNumCurLoop * maskColumnRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        if (maskColumnRound == columnNumRound) {
            // 掩码与 S 列完全对齐：整块相加 computeUb = computeUb + mask16
            AscendC::Add<half, false>(
                computeUbTensor,
                computeUbTensor,
                maskUbTensor16,
                (uint64_t)0,
                (rowNumCurLoop * maskColumnRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            // 掩码列数 < S 列数（因果掩码场景）：按 HALF_VECTOR_SIZE 分段加
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
            // 处理尾部不足 HALF_VECTOR_SIZE 的部分
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

    // ============================================================================
    // CalcLocalRowMax：计算当前块的局部行最大值 lm = max(S_row)
    // ----------------------------------------------------------------------------
    // 调用 RowmaxTAILTILE 对 computeUbTensor（缩放后的 S）逐行求最大值，
    // 结果存于 lmUbTensor[rowOffset]。
    // 参数：
    //   sUbOffset         : UB 偏移（未直接使用，保留接口）
    //   rowNumCurLoopRound: 行数（对齐）
    //   columnNum         : 实际列数
    //   columnNumRound    : 对齐列数
    //   rowOffset         : 行偏移（子核内分块）
    // ============================================================================
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

    // ============================================================================
    // UpdateGlobalRowMax：更新全局行最大值并计算缩放因子 dm
    // ----------------------------------------------------------------------------
    // 在线 Softmax 核心步骤：
    //   - 首个块（isFirstStackTile=1）：hm = lm（直接初始化全局最大值）
    //   - 后续块：
    //       hm = max(lm, gm)          —— 新全局最大值
    //       dm = exp(gm - hm)          —— 历史缩放因子（gm 为旧全局最大值）
    //   - 最后：gm = hm（更新全局最大值）
    // dm 用于后续 UpdateGlobalRowSum 中对历史行和做缩放：gl = dm * gl + ll
    // 参数：
    //   rowNumCurLoop      : 实际行数
    //   rowNumCurLoopRound : 对齐行数
    //   columnNum          : 列数（未使用）
    //   columnNumRound     : 对齐列数（未使用）
    //   dmUbOffsetCurCycle : dm 缓冲偏移（按 curStackTileMod 和 rowOffset 计算）
    //   rowOffset          : 行偏移
    //   isFirstStackTile   : 是否为首个 KV 块
    // ============================================================================
    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // 首个块：hm = lm（直接拷贝局部最大值为全局最大值）
            AscendC::DataCopy(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // 后续块：先设置掩码处理不足 128 的行
            SetVecMask(rowNumCurLoop);
            // *** hm = vmax(lm, gm)  —— 取局部与全局最大值的较大者
            AscendC::Max<half, false>(
                hmUbTensor[rowOffset],
                lmUbTensor[rowOffset],
                gmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));

            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm  —— 旧全局最大值减新全局最大值（<=0）
            AscendC::Sub<half, false>(
                dmUbTensor[dmUbOffsetCurCycle],
                gmUbTensor[rowOffset],
                hmUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)  —— 历史缩放因子（<=1，用于缩小历史累加和）
            AscendC::Exp<half, false>(dmUbTensor[dmUbOffsetCurCycle],
                dmUbTensor[dmUbOffsetCurCycle],
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm  —— 更新全局最大值
        AscendC::DataCopy(gmUbTensor[rowOffset],
            hmUbTensor[rowOffset],
            AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // CalcExp：计算 P = exp(S - gm)（Softmax 分子部分）
    // ----------------------------------------------------------------------------
    // 算法：
    //   1. 将 hm（行最大值）广播到整个 block（Brcb），存于 tvUbTensor
    //   2. computeUb = computeUb - tvUbTensor（逐行减去行最大值，数值稳定）
    //   3. computeUb = exp(computeUb)（得到 P = exp(S-gm)）
    // 减去行最大值是为了数值稳定性（避免 exp 溢出）。
    // 参数：
    //   sUbOffset         : UB 偏移
    //   rowNumCurLoop     : 实际行数
    //   rowNumCurLoopRound: 对齐行数
    //   columnNum         : 实际列数
    //   columnNumRound    : 对齐列数
    //   rowOffset         : 行偏移
    // ============================================================================
    __aicore__ inline
    void CalcExp(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t rowOffset)
    {
        // *** hm_block = expand_to_block(hm), 存放于 tv
        // Brcb: 将 hmUbTensor（每 8 个 float 一组）广播到 tvUbTensor 的整个 block
        AscendC::Brcb(
            tvUbTensor.template ReinterpretCast<uint16_t>(),
            hmUbTensor[rowOffset].template ReinterpretCast<uint16_t>(),
            rowNumCurLoopRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = ls - hm_block  —— 逐行减去行最大值（数值稳定）
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
        // 处理尾部不足 HALF_VECTOR_SIZE 的列
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
        // *** ls = exp(ls)  —— 计算 P = exp(S - gm)
        AscendC::Exp<half, false>(
            computeUbTensor,
            computeUbTensor,
            (uint64_t)0,
            (rowNumCurLoop * columnNumRound + HALF_VECTOR_SIZE - 1) / HALF_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // CalcLocalRowSum：计算当前块的局部行和 ll = sum(P_row)
    // ----------------------------------------------------------------------------
    // 根据列数选择优化路径：
    //   - 列数 == 512：调用 RowsumSPECTILE512（4 路拆分优化）
    //   - 其他：调用 RowsumTAILTILE（通用路径）
    // 结果存于 llUbTensor[rowOffset]。
    // 参数：
    //   sUbOffset         : UB 偏移
    //   rowNumCurLoopRound: 对齐行数
    //   columnNum         : 实际列数
    //   columnNumRound    : 对齐列数
    //   rowOffset         : 行偏移
    // ============================================================================
    __aicore__ inline
    void CalcLocalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoopRound, uint32_t columnNum, uint32_t columnNumRound,
        uint32_t rowOffset)
    {
        // *** ll = rowsum(ls32)
        if (columnNum == 512U) {
            // 512 列特化路径（4 路拆分归约）
            RowsumSPECTILE512(computeUbTensor,
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        } else {
            // 通用行求和路径
            RowsumTAILTILE(computeUbTensor,
                llUbTensor[rowOffset],
                tvUbTensor,
                rowNumCurLoopRound,
                columnNum,
                columnNumRound);
        }
    }

    // ============================================================================
    // UpdateGlobalRowSum：更新全局行和 gl = dm * gl + ll
    // ----------------------------------------------------------------------------
    // 在线 Softmax 核心步骤：
    //   - 首个块（isFirstStackTile=1）：gl = ll（直接初始化）
    //   - 后续块：gl = dm * gl + ll（用缩放因子 dm 缩放历史和，再累加当前块和）
    // dm 来自 UpdateGlobalRowMax，保证 gl 始终是对当前全局最大值归一化的行和。
    // 参数：
    //   sUbOffset         : UB 偏移
    //   rowNumCurLoop     : 实际行数
    //   rowNumCurLoopRound: 对齐行数
    //   dmUbOffsetCurCycle: dm 缓冲偏移
    //   rowOffset         : 行偏移
    //   isFirstStackTile   : 是否为首个 KV 块
    // ============================================================================
    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // 首个块：gl = ll（直接拷贝局部行和为全局行和）
            // *** gl = ll
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // 后续块：gl = dm * gl + ll
            SetVecMask(rowNumCurLoop);
            // *** gl = dm * gl  —— 用缩放因子缩小历史行和
            AscendC::Mul<half, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl  —— 累加当前块行和
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

    // ============================================================================
    // MoveP：将 P 矩阵（exp 后的 S）从 computeUbTensor 拷贝到 lpUbTensor
    // ----------------------------------------------------------------------------
    // lpUbTensor 是 P 矩阵的输出缓冲，后续 CopyPUbToGm 会将其拷贝到 GM，
    // 供 Cube 引擎执行 PV 矩阵乘。
    // 参数：
    //   sUbOffset      : UB 偏移（未使用）
    //   rowNumCurLoop  : 行数
    //   columnNumRound: 对齐列数
    // ============================================================================
    __aicore__ inline
    void MoveP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = 1;                                    // 单次拷贝
        repeatParams.srcStride = 0;                                     // 无 stride
        repeatParams.blockLen = CeilDiv(rowNumCurLoop * columnNumRound, BLOCK_SIZE);  // 总 block 数
        AscendC::DataCopy<half>(lpUbTensor, computeUbTensor, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // CopyPUbToGm：将 P 矩阵从 UB（lpUbTensor）拷贝回 GM
    // ----------------------------------------------------------------------------
    // Cube 引擎从 GM 读取 P 执行 PV 矩阵乘，故需将 P 从 UB 拷回 GM。
    // 参数：
    //   gOutput        : GM 目标地址（P 矩阵）
    //   sUbOffset      : UB 偏移（未使用）
    //   rowNumCurLoop  : 行数
    //   columnNumRound : 对齐列数
    //   columnNumPad   : padding 后列数
    // ============================================================================
    __aicore__ inline
    void CopyPUbToGm(AscendC::GlobalTensor<ElementOutput> gOutput, uint32_t sUbOffset, uint32_t rowNumCurLoop,
        uint32_t columnNumRound, uint32_t columnNumPad)
    {
        AscendC::DataCopy(gOutput,
            lpUbTensor,
            AscendC::DataCopyParams(
                rowNumCurLoop, columnNumRound / BLOCK_SIZE, 0, (columnNumPad - columnNumRound) / BLOCK_SIZE));
    }

    // ============================================================================
    // SubCoreCompute：子核计算编排（单个行块的在线 Softmax 核心流程）
    // ----------------------------------------------------------------------------
    // 本函数编排单个行块（rowNumCurLoop 行）的完整在线 Softmax 流程：
    //   1. CalcLocalRowMax   : 计算局部行最大值 lm
    //   2. UpdateGlobalRowMax: 更新全局行最大值 gm，计算缩放因子 dm
    //   3. CalcExp           : 计算 P = exp(S - gm)
    //   4. MoveP             : 将 P 拷贝到 lpUbTensor
    //   5. CopyPUbToGm       : 将 P 拷贝到 GM（供 Cube 引擎 PV 乘）
    //   6. CalcLocalRowSum   : 计算局部行和 ll
    //   7. UpdateGlobalRowSum: 更新全局行和 gl
    // 通过 HardEvent 标志实现 MTE2（搬数）↔ V（计算）↔ MTE3（搬数）的流水线同步。
    // 参数：
    //   gOutput         : GM 输出（P 矩阵）
    //   layoutOutput    : 输出布局
    //   rowOffset       : 行偏移（子核内分块）
    //   isFirstStackTile: 是否首个 KV 块
    //   isFirstRowLoop  : 是否首个行循环
    //   columnNumRound  : 对齐列数
    //   pingpongFlag    : Ping-Pong 标志（0/1）
    //   curStackTileMod : 当前栈 tile 模数（用于 dm 缓冲区索引）
    // ============================================================================
    __aicore__ inline
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput, const LayoutOutput &layoutOutput,
        uint32_t rowOffset, uint32_t isFirstStackTile, uint32_t isFirstRowLoop,
        uint32_t columnNumRound, uint32_t pingpongFlag,
        uint32_t curStackTileMod)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);                    // 当前行数
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, BLOCK_SIZE);  // 对齐行数
        uint32_t columnNum = layoutOutput.shape(1);                         // 实际列数
        uint32_t columnNumPad = layoutOutput.stride(0);                    // padding 后列数
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;             // Ping-Pong 偏移
        // dm 缓冲偏移：按 curStackTileMod（软件流水阶段）和 rowOffset 计算
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            // LSE 仅输出模式：首个块首个行循环时，等待 MTE3→V 事件（tv 被 lse 占用）
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        // 步骤1: 计算局部行最大值 lm
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);  // 通知 MTE2 可搬数
        // 步骤2: 更新全局行最大值 gm，计算缩放因子 dm
        UpdateGlobalRowMax(rowNumCurLoop,
            rowNumCurLoopRound,
            columnNum,
            columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);
        // 步骤3: 计算 P = exp(S - gm)
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        // 等待 MTE3→V（上一次 CopyPUbToGm 完成），然后搬 P 到 lpUbTensor
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        // 步骤4: 将 P 从 computeUb 拷贝到 lpUbTensor
        MoveP(sUbOffset, rowNumCurLoop, columnNumRound);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);  // 通知 MTE3 可搬数到 GM

        // 步骤5: 计算局部行和 ll
        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);

        // 等待 V→MTE3（MoveP 完成），然后搬 P 到 GM
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        // 步骤6: 将 P 从 lpUbTensor 拷贝到 GM（供 Cube 引擎 PV 乘）
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);  // 通知下次可覆盖
        // 步骤7: 更新全局行和 gl = dm * gl + ll
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    // ============================================================================
    // operator()（无掩码版）：无注意力掩码的在线 Softmax 入口
    // ----------------------------------------------------------------------------
    // 处理无 Mask 场景（如无因果掩码的自回归解码）。
    // 流程：
    //   1. 计算子核行划分（subBlock 0/1 各处理一半行）
    //   2. 按行分块循环（rowNumTile 行/次，Ping-Pong）
    //   3. 每行块：拷贝 S→UB → 缩放 → SubCoreCompute
    // 参数：
    //   gOutput          : GM 输出（P 矩阵）
    //   gInput           : GM 输入（S 矩阵，QK^T 结果）
    //   layoutOutput     : 输出布局
    //   layoutInput      : 输入布局
    //   actualBlockShape : 实际块形状（M=行, N=列）
    //   isFirstStackTile : 是否首个 KV 块
    //   isLastNoMaskStackTile: 未使用（保留接口）
    //   qSBlockSize      : Q 的 S 维块大小
    //   qNBlockSize      : Q 的 N 维块大小
    //   curStackTileMod  : 当前栈 tile 模数
    // ============================================================================
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<half> gInput,
        const LayoutOutput &layoutOutput, const LayoutInput &layoutInput, GemmCoord actualBlockShape,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile,
        uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();                            // 总行数
        uint32_t columnNum = actualBlockShape.n();                         // 实际列数
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);         // 对齐列数
        uint32_t columnNumPad = layoutInput.stride(0);                     // padding 后列数

        // 子核划分：每个核有两个子核（subBlock 0/1），各处理一半行
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();                 // 当前子核索引（0 或 1）
        uint32_t subBlockNum = AscendC::GetSubBlockNum();                 // 子核总数（2）

        // 计算当前子核负责的行范围
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;             // 每子核 N 分块数
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;  // 行偏移起点

        // 计算每循环最大行数（受 UB 容量限制）
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, BLOCK_SIZE);    // 对齐到 BLOCK_SIZE
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);     // 不超过 128 行
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile); // 行循环次数

        // 行分块循环（Ping-Pong）
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;                      // Ping-Pong 标志
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;          // 当前循环行偏移
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;  // GM 中行偏移
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            // 计算 GM 输入偏移并获取当前行块的 S 矩阵
            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];

            // 搬数 MTE2：拷贝 S 从 GM 到 UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // 计算 V：缩放 S
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);

            // 计算 GM 输出偏移并获取当前行块的 P 矩阵输出地址
            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            // 调用子核计算编排（在线 Softmax 核心流程）
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

    // ============================================================================
    // operator()（带掩码版）：带注意力掩码的在线 Softmax 入口
    // ----------------------------------------------------------------------------
    // 处理带 Mask 场景（如因果掩码、自定义掩码）。
    // 相比无掩码版，增加了：
    //   - 因果掩码区域划分（triUp/triDown 对角线上下区域）
    //   - 掩码拷贝（CopyMaskGmToUb）
    //   - 掩码类型转换（UpCastMask）
    //   - 掩码应用（ApplyMask）
    //   - 跨核同步等待 Cube 引擎完成 QK^T（CrossCoreWaitFlag(qkReady)）
    // 参数：
    //   gOutput          : GM 输出（P 矩阵）
    //   gInput           : GM 输入（S 矩阵）
    //   gMask            : GM 掩码
    //   layoutOutput     : 输出布局
    //   layoutInput      : 输入布局
    //   layoutMask       : 掩码布局
    //   actualBlockShape : 实际块形状
    //   isFirstStackTile : 是否首个 KV 块
    //   qSBlockSize      : Q 的 S 维块大小
    //   qNBlockSize      : Q 的 N 维块大小
    //   curStackTileMod  : 当前栈 tile 模数
    //   qkReady          : 跨核信号量（等待 Cube 完成 QK^T）
    //   triUp            : 因果掩码上三角边界（对角线上方起始）
    //   triDown          : 因果掩码下三角边界（对角线下方结束）
    //   kvSStartIdx      : KV 序列起始索引
    //   kvSEndIdx        : KV 序列结束索引
    // ============================================================================
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementOutput> gOutput, AscendC::GlobalTensor<half> gInput,
        AscendC::GlobalTensor<ElementMask> gMask, const LayoutOutput &layoutOutput, const LayoutInput &layoutInput,
        const LayoutInput &layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown,
        uint32_t kvSStartIdx, uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();                            // 总行数
        uint32_t columnNum = actualBlockShape.n();                         // 实际列数
        uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);         // 对齐列数
        uint32_t columnNumPad = layoutInput.stride(0);                     // S padding 后列数
        uint32_t maskStride = layoutMask.stride(0);                        // 掩码行 stride
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();                 // 当前子核索引
        uint32_t subBlockNum = AscendC::GetSubBlockNum();                 // 子核总数

        // 子核行划分（同无掩码版）
        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1U) ?
            0 : (subBlockIdx == 1U) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1U) ? (qSBlockSize / 2U) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1U) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = AscendC::Std::min(qSBlockSize, rowActualThisSubBlock);

        // 掩码偏移：qNBlockSize==1 时按子核偏移，否则从 0 开始
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1U) ? rowOffsetThisSubBlock : 0;

        // 因果掩码区域计算：根据 triUp 和 kvSStartIdx 的关系确定掩码覆盖范围
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            // triUp 在当前 KV 块范围内：掩码从 triUp 对齐位置开始
            uint32_t triUpRoundDown = RoundDown(triUp, BLOCK_SIZE);
            gmOffsetMaskRow = triUp - triUpRoundDown;                     // 掩码行偏移
            gmOffsetMaskColumn = 0U;                                     // 列从 0 开始
            maskColumn = kvSEndIdx - triUpRoundDown;                     // 掩码列数
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;              // S 中掩码起始偏移
        } else {
            // triUp 在当前 KV 块之前：整列都需要掩码
            gmOffsetMaskRow = 0U;
            gmOffsetMaskColumn = kvSStartIdx - triUp;                   // 列偏移
            maskColumn = columnNum;                                      // 全列掩码
            addMaskUbOffset = 0U;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, BLOCK_SIZE);       // 掩码对齐列数

        // 计算掩码 GM 偏移
        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        // 计算每循环最大行数（同无掩码版）
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, HALF_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);

        // 若当前子核无行需处理，仅等待 Cube 完成后返回
        if (rowActualThisSubBlock == 0U) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }
        // 等待 Cube 引擎完成 QK^T 计算（跨核同步）
        Arch::CrossCoreWaitFlag(qkReady);
        // 行分块循环（Ping-Pong）
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum; rowLoopIdx++) {
            uint32_t pingpongFlag = rowLoopIdx % 2U;
            uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
            uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
            uint32_t rowNumCurLoop =
                (rowLoopIdx == rowLoopNum - 1U) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

            // GQA 场景下掩码按 head 重复：计算 pro/integral/epi 三段 token 数
            uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
            uint32_t proTokenNum = AscendC::Std::min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) %
                tokenNumPerHeadThisSubBlock;
            uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
            uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;

            // 搬数 MTE2：拷贝 S 从 GM 到 UB
            int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gInputCurLoop = gInput[offsetInput];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopySGmToUb(
                gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // 计算 V：缩放 S
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
            
            // 搬数 MTE2：拷贝掩码从 GM 到 UB（按 GQA head 展开）
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
            // 计算 V：掩码类型转换（int8 → half）
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            // 计算 V：应用掩码（屏蔽位置置 -inf）
            ApplyMask(
                (pingpongFlag * MAX_UB_S_ELEM_NUM),
                rowNumCurLoop,
                columnNumRound,
                maskColumnRound,
                addMaskUbOffset);

            // online softmax vectorized compute（在线 Softmax 向量化计算）
            int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
            auto gOutputCurLoop = gOutput[offsetOutput];
            auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
            // 调用子核计算编排
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
    // ==================== 私有数据成员 ====================
    half scaleValue;                                       // Softmax 缩放因子（1/sqrt(d)）
    AscendC::LocalTensor<half> lsUbTensor;                 // S 矩阵输入缓冲（GM→UB）
    AscendC::LocalTensor<half> computeUbTensor;             // 计算中间结果（缩放/exp 后的 S/P）
    AscendC::LocalTensor<ElementOutput> lpUbTensor;        // P 矩阵输出缓冲（UB→GM）
    AscendC::LocalTensor<ElementMask> maskUbTensor;        // 原始掩码缓冲（int8/uint8）
    AscendC::LocalTensor<half> maskUbTensor16;              // half 掩掩码（转换后）
    AscendC::LocalTensor<half> lmUbTensor;                 // 局部行最大值 lm（当前块）
    AscendC::LocalTensor<half> hmUbTensor;                 // 当前行最大值 hm（max(lm, gm)）
    AscendC::LocalTensor<half> gmUbTensor;                 // 全局行最大值 gm（跨块累积）
    AscendC::LocalTensor<half> dmUbTensor;                 // 缩放因子 dm=exp(gm_old-gm_new)
    AscendC::LocalTensor<half> llUbTensor;                 // 局部行和 ll（当前块）
    AscendC::LocalTensor<half> tvUbTensor;                 // 临时向量（广播 hm 用）
    AscendC::LocalTensor<half> glUbTensor;                 // 全局行和 gl（跨块累积）
};

}

#endif