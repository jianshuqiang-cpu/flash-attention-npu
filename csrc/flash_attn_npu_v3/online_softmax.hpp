/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// ============================================================================
// 文件说明：online_softmax.hpp — 高精度（float）Online Softmax Epilogue
// ============================================================================
//
// 本文件是 FlashAttention v3 在昇腾 NPU 上的 Vector 引擎 Epilogue 实现，
// 采用 float（FP32）精度执行 Online Softmax 全流程计算。
//
// 与 online_softmax_low_prec.hpp（half 版本）的核心区别：
//   1. 计算精度：所有中间量（S、P、rowmax、rowsum）均使用 float，数值精度更高
//   2. 硬件常量：FLOAT_VECTOR_SIZE=64（half 版为 128），MAX_UB_S_ELEM_NUM=8192（half 版为 16384）
//      —— float 占 4 字节，相同 UB 空间能容纳的元素数减半
//   3. 新增 256 列专用归约：RowsumSPECTILE256 / RowmaxSPECTILE256
//      —— float 下 256=4×64，可做 4 路 split 归约
//   4. 新增 DownCastP：将 float 精度的 P 矩阵降精度为 half/bfloat16，供 Cube 引擎 PV 乘法使用
//   5. ApplyMask 掩码值：使用 -3e38（float 最小值）而非 -65504（half 最小值）
//
// Online Softmax 算法流程（10 步）：
//   1. CopyS       : S=QK^T 从 GM 拷贝到 UB（Vector 引擎接收 Cube 引擎计算结果）
//   2. ScaleS     : S *= scaleValue（1/sqrt(d) 缩放）
//   3. (ApplyMask) : 若有掩码，将 mask 区域置为 -3e38（仅 with-mask 版本）
//   4. CalcLocalRowMax  : 计算当前 KV 块的局部行最大值 lm
//   5. UpdateGlobalRowMax: hm=max(lm,gm), dm=exp(gm-hm), gm=hm（在线更新全局行最大值）
//   6. CalcExp    : P=exp(S-gm)（减去行最大值保证数值稳定性）
//   7. DownCastP  : P 从 float 降精度为 half/bfloat16（本版本特有）
//   8. CalcLocalRowSum  : 计算当前 KV 块的局部行和 ll
//   9. UpdateGlobalRowSum: gl=dm*gl+ll（在线更新全局行和）
//  10. CopyPUbToGm: P 矩阵从 UB 拷贝回 GM（供 Cube 引擎 PV 乘法使用）
//
// 双引擎流水线协作：
//   - Cube 引擎：执行 QK^T 和 PV 矩阵乘法
//   - Vector 引擎（本文件）：执行 Online Softmax 全流程
//   - 跨核同步：CrossCoreFlag qkReady 信号量，Vector 等待 Cube 完成 QK^T
//
// CATLASS 框架头文件说明：
//   - catlass.hpp              : CATLASS 框架主头文件（类比 NVIDIA CUTLASS）
//   - cross_core_sync.hpp      : 跨核同步原语（CrossCoreFlag/CrossCoreWaitFlag）
//   - resource.hpp             : 硬件资源管理（UB/L1/L0 缓冲区分配）
//   - dispatch_policy.hpp      : Epilogue 分派策略（EpilogueAtlasA2OnlineSoftmaxT）
//   - tile_copy.hpp            : Tile 级数据拷贝原语
//   - gemm_coord.hpp           : GEMM 坐标（m/n/k 维度描述）
//   - matrix_coord.hpp         : 矩阵坐标（行列偏移描述）
//   - fa_block.h               : FlashAttention 块级参数定义
// ============================================================================

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "fa_block.h"
 
namespace Catlass::Epilogue::Block {

// ============================================================================
// 类模板：BlockEpilogue — 高精度（float）Online Softmax Epilogue
// ============================================================================
// 模板参数说明：
//   OutputType_  : 输出张量类型（含元素类型 ElementOutput 和布局 LayoutOutput）
//                  —— P 矩阵输出精度，通常为 half 或 bfloat16_t
//   InputType_   : 输入张量类型（含元素类型 ElementInput 和布局 LayoutInput）
//                  —— S 矩阵输入精度，本版本为 float
//   MaskType_    : 掩码张量类型（含元素类型 ElementMask 和布局 LayoutMask）
//                  —— 掩码精度，通常为 int8_t 或 bool
//   LSE_MODE_    : LSE（LogSumExp）模式枚举
//                  —— NONE: 不计算 LSE
//                  —— OUT_ONLY: 仅输出 LSE
//                  —— OUT_AND_UPDATE: 输出并更新 LSE
//
// 分派策略特化：EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>
//   第二个模板参数 float 标识本类为高精度版本（low_prec 版本为 half）
// ============================================================================
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
    using DispatchPolicy = EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    static constexpr LseModeT LSE_MODE = DispatchPolicy::LSE_MODE;

    // ---- 硬件常量定义（float 精度专用）----
    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;       // 内存对齐块大小（字节）
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;     // 单次向量指令重复处理大小（字节）
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;          // float 元素的 Block 大小（8 个 float = 32 字节对齐）
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;       // float 向量处理宽度（64 个 float/指令，half 版为 128）
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;       // half 向量处理宽度（用于掩码转换参考）
    static constexpr uint32_t BLOCK_SIZE = 16;               // 通用 Block 大小（用于 DataCopy 对齐）
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;  // UB uint8 向量宽度（1024 字节）
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 16384;  // UB uint8 Block 大小（16384 字节 = 16KB）
    static constexpr uint32_t VECTOR_SIZE = 128;            // 通用向量大小
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;     // S 矩阵在 UB 中的最大元素数（float 版，half 版为 16384）

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;         // 归约中间缓冲区大小（元素数）
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;    // 256 列专用归约的中间 mask 值
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;     // 256 列专用归约的最终 mask 值
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;   // 每个子核最大行数（dm 缓冲区按此分块）
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;       // UB float 行大小

    // ============================================================================
    // 构造函数：分配 UB（Unified Buffer）空间并初始化各 Tensor
    // ============================================================================
    // UB 内存布局（以字节为单位，UB_UINT8_BLOCK_SIZE = 16384 字节 = 16KB）：
    //
    //   偏移 0x00000 (0  *16KB): lsUbTensor    — S 矩阵缓冲区（float），4 个 16KB = 64KB
    //   偏移 0x10000 (4  *16KB): lpUbTensor    — P 矩阵缓冲区（half/bfloat16），与 mask 重叠
    //   偏移 0x10000 (4  *16KB): maskUbTensor  — 掩码缓冲区（int8），与 lp 重叠（分时复用）
    //   偏移 0x10000 (4  *16KB): maskUbTensor32— 掩码缓冲区（float），与 lp/mask 重叠
    //   偏移 0x16000 (10 *16KB): tvUbTensor    — 临时向量缓冲区（float），8*1024=8KB
    //   偏移 0x16200 (10*16KB+8*1024): lmUbTensor — 局部行最大值 lm（float）
    //   偏移 0x16400 (10*16KB+9*1024): hmUbTensor — 合并后行最大值 hm（float）
    //   偏移 0x16600 (10*16KB+10*1024): gmUbTensor — 全局行最大值 gm（float）
    //   偏移 0x16800 (10*16KB+11*1024): llUbTensor — 局部行和 ll（float）
    //   偏移 0x16A00 (10*16KB+12*1024): glUbTensor — 全局行和 gl（float）
    //   偏移 0x16C00 (10*16KB+13*1024): dmUbTensor — 衰减因子 dm=exp(gm-hm)（float）
    //   偏移 0x1C000 (11 *16KB): maskUbTensor16 — 掩码缓冲区（half），用于掩码类型转换中间态
    //
    // 注意：lpUbTensor/maskUbTensor/maskUbTensor32 共享同一偏移（4*16KB），因为：
    //   - 无掩码路径：使用 lpUbTensor 存储 P 矩阵
    //   - 有掩码路径：先使用 maskUbTensor/maskUbTensor32 处理掩码，再使用 lpUbTensor
    //   - 两者在时间上不重叠，故可分时复用 UB 空间
    // ============================================================================
    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t MASK32_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 11 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        constexpr uint32_t MASK16_UB_TENSOR_OFFSET = 11 * UB_UINT8_BLOCK_SIZE;

        scaleValue = scaleValue_;  // 保存缩放因子 scaleValue = 1/sqrt(d)
        // 从 UB 缓冲区按字节偏移获取各 Tensor
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

    // ============================================================================
    // Min: 通用最小值函数（模板化，支持任意可比较类型）
    // ============================================================================
    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return (a > b) ? b : a;
    }

    // ============================================================================
    // SetVecMask: 设置 Vector 引擎的掩码寄存器
    // ============================================================================
    // 用于处理尾部不足 FLOAT_VECTOR_SIZE(64) 个 float 元素的情况。
    // AscendC 向量指令通过 mask 寄存器控制哪些元素参与计算。
    //
    // 参数：
    //   len : 当前需要处理的元素数量
    //
    // 掩码设置逻辑：
    //   - len == 128 或 len == 0 : 全部元素参与（mask 全 1）
    //   - len >= 64              : 高 64 位全 1，低 (len%64) 位按需设置
    //   - len < 64               : 高 64 位全 0，低 len 位按需设置
    //
    // float 版本与 half 版本的区别：FLOAT_VECTOR_SIZE=64（half 版为 128）
    // ============================================================================
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE || len == 0) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    // ============================================================================
    // SetBlockReduceMask: 设置 BlockReduceSum/BlockReduceMax 的掩码寄存器
    // ============================================================================
    // 用于 BlockReduceSum/BlockReduceMax 指令的尾部处理。
    // BlockReduce 指令按 FLOAT_BLOCK_SIZE(8) 个元素为一组进行归约，
    // 当有效元素数不足 8 时需要通过 mask 屏蔽无效元素。
    //
    // 参数：
    //   len : 有效元素数（1~8），超出范围则全 mask
    //
    // maskValue 构造：将 subMask 复制到 8 个 8-bit 位置（每 8 bit 一组），
    //   形成 64-bit 掩码，覆盖 BlockReduce 指令的所有可能位置。
    // ============================================================================
    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    // ============================================================================
    // RowsumSPECTILE512: 512 列专用行求和（4 路 split 归约）
    // ============================================================================
    // 当列数恰好为 512 时调用。512 = 8 × 64 = 8 × FLOAT_VECTOR_SIZE。
    // 采用 3 级 BlockReduceSum 归约：
    //   第 1 级: srcUb → tvUbTensor       （每 FLOAT_VECTOR_SIZE 个元素求和，8 路）
    //   第 2 级: tvUbTensor → tvUbTensor[REDUCE_UB_SIZE]（再按 FLOAT_BLOCK_SIZE 归约，8 路）
    //   第 3 级: tvUbTensor[REDUCE_UB_SIZE] → rowsumUb（最终归约，8 路）
    //
    // float 版本：512 = 8 × 64（FLOAT_VECTOR_SIZE=64）
    // half  版本：512 = 4 × 128（HALF_VECTOR_SIZE=128），归约路径不同
    // ============================================================================
    __aicore__ inline
    void RowsumSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第 1 级归约：按 FLOAT_VECTOR_SIZE(64) 分组求和
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        // 第 2 级归约：按 FLOAT_BLOCK_SIZE(8) 分组求和
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        // 第 3 级归约：最终归约得到每行求和结果
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // RowsumSPECTILE256: 256 列专用行求和（float 版本特有，half 版无此函数）
    // ============================================================================
    // 当列数恰好为 256 时调用。256 = 4 × 64 = 4 × FLOAT_VECTOR_SIZE。
    // 采用 3 级 BlockReduceSum 归约，但中间级使用不同的 mask 和 repeat 参数：
    //   第 1 级: srcUb → tvUbTensor       （每 FLOAT_VECTOR_SIZE 个元素求和，8 路）
    //   第 2 级: tvUbTensor → tvUbTensor[REDUCE_UB_SIZE]（设置 32 元素 mask，4 路）
    //   第 3 级: tvUbTensor[REDUCE_UB_SIZE] → rowsumUb（设置 4 元素 mask，8 路）
    //
    // 注意：第 2、3 级需要设置专用 mask（ROW_OPS_SPEC_MASK_32=32, ROW_OPS_SPEC_MASK_4=4），
    //   因为 256 列归约后的中间结果布局与 512 列不同。
    // ============================================================================
    __aicore__ inline
    void RowsumSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第 1 级归约：按 FLOAT_VECTOR_SIZE(64) 分组求和
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        // 第 2 级归约：设置 32 元素 mask
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceSum<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        // 第 3 级归约：设置 4 元素 mask
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tvUbTensor[REDUCE_UB_SIZE],
            CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全 mask
    }

    // ============================================================================
    // RowsumTAILTILE: 通用行求和（任意列数）
    // ============================================================================
    // 当列数不是 512 或 256 时调用。处理任意列数的行求和。
    // 算法分两部分：
    //   1. 完整 FLOAT_VECTOR_SIZE(64) 段：循环处理每个完整段，累加到 rowsumUb
    //   2. 尾部段（numElems % 64 > 0）：设置 mask 处理不足 64 的尾部元素
    //
    // 每段归约流程：
    //   BlockReduceSum(srcUb → tvUbTensor)  // 第 1 级：按 FLOAT_BLOCK_SIZE(8) 分组
    //   BlockReduceSum(tvUbTensor → tvUbTensor[REDUCE_UB_SIZE] 或 rowsumUb)  // 第 2 级
    //   Add(rowsumUb += tvUbTensor[REDUCE_UB_SIZE])  // 累加到结果（非首段）
    // ============================================================================
    __aicore__ inline
    void RowsumTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // ---- 第 1 部分：处理完整的 FLOAT_VECTOR_SIZE(64) 段 ----
        if (numElems >= FLOAT_VECTOR_SIZE) {
            // 首段：直接归约到 rowsumUb
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
            // 后续完整段：归约后累加到 rowsumUb
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
                SetVecMask(numRowsRound);  // 设置行数 mask
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
        // ---- 第 2 部分：处理尾部不足 FLOAT_VECTOR_SIZE(64) 的段 ----
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);  // 设置尾部元素 mask
            AscendC::BlockReduceSum<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                // 列数不足 64：直接归约到 rowsumUb（首段即尾段）
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb,
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // 列数 >= 64：归约后累加到 rowsumUb
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

    // ============================================================================
    // RowmaxSPECTILE512: 512 列专用行最大值（4 路 split 归约）
    // ============================================================================
    // 当列数恰好为 512 时调用。结构与 RowsumSPECTILE512 完全对称，
    // 仅将 BlockReduceSum 替换为 BlockReduceMax。
    // 3 级归约：srcUb → tvUbTensor → tvUbTensor[REDUCE_UB_SIZE] → rowmaxUb
    // ============================================================================
    __aicore__ inline
    void RowmaxSPECTILE512(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第 1 级归约：按 FLOAT_VECTOR_SIZE(64) 分组求最大值
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        // 第 2 级归约：按 FLOAT_BLOCK_SIZE(8) 分组求最大值
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        // 第 3 级归约：最终归约得到每行最大值
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================================
    // RowmaxSPECTILE256: 256 列专用行最大值（float 版本特有，half 版无此函数）
    // ============================================================================
    // 当列数恰好为 256 时调用。结构与 RowsumSPECTILE256 完全对称，
    // 仅将 BlockReduceSum 替换为 BlockReduceMax。
    // 3 级归约，中间级使用专用 mask（ROW_OPS_SPEC_MASK_32=32, ROW_OPS_SPEC_MASK_4=4）。
    // ============================================================================
    __aicore__ inline
    void RowmaxSPECTILE256(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // 第 1 级归约：按 FLOAT_VECTOR_SIZE(64) 分组求最大值
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        // 第 2 级归约：设置 32 元素 mask
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        AscendC::BlockReduceMax<float, false>(
            tvUbTensor[REDUCE_UB_SIZE],
            tvUbTensor,
            numRowsRound,
            0, 1, 1, 4);
        AscendC::PipeBarrier<PIPE_V>();
        // 第 3 级归约：设置 4 元素 mask
        SetBlockReduceMask(ROW_OPS_SPEC_MASK_4);
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tvUbTensor[REDUCE_UB_SIZE],
            CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
            0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // 恢复全 mask
    }

    // ============================================================================
    // RowmaxTAILTILE: 通用行最大值（任意列数）
    // ============================================================================
    // 当列数不是 512 或 256 时调用。结构与 RowsumTAILTILE 完全对称，
    // 仅将 BlockReduceSum/Add 替换为 BlockReduceMax/Max。
    //
    // 算法分两部分：
    //   1. 完整 FLOAT_VECTOR_SIZE(64) 段：循环处理每个完整段，取最大值更新 rowmaxUb
    //   2. 尾部段（numElems % 64 > 0）：设置 mask 处理不足 64 的尾部元素
    //
    // 每段归约流程：
    //   BlockReduceMax(srcUb → tvUbTensor)  // 第 1 级：按 FLOAT_BLOCK_SIZE(8) 分组
    //   BlockReduceMax(tvUbTensor → tvUbTensor[REDUCE_UB_SIZE] 或 rowmaxUb)  // 第 2 级
    //   Max(rowmaxUb = max(rowmaxUb, tvUbTensor[REDUCE_UB_SIZE]))  // 取最大值（非首段）
    // ============================================================================
    __aicore__ inline
    void RowmaxTAILTILE(const AscendC::LocalTensor<float> &srcUb, const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tvUbTensor, uint32_t numRowsRound, uint32_t numElems,
        uint32_t numElemsAligned)
    {
        // ---- 第 1 部分：处理完整的 FLOAT_VECTOR_SIZE(64) 段 ----
        if (numElems >= FLOAT_VECTOR_SIZE) {
            // 首段：直接归约到 rowmaxUb
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
            // 后续完整段：归约后取最大值更新 rowmaxUb
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
                SetVecMask(numRowsRound);  // 设置行数 mask
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
        // ---- 第 2 部分：处理尾部不足 FLOAT_VECTOR_SIZE(64) 的段 ----
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);  // 设置尾部元素 mask
            AscendC::BlockReduceMax<float, false>(
                tvUbTensor,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                // 列数不足 64：直接归约到 rowmaxUb（首段即尾段）
                AscendC::BlockReduceMax<float, false>(rowmaxUb,
                    tvUbTensor,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // 列数 >= 64：归约后取最大值更新 rowmaxUb
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

    // ============================================================================
    // CopySGmToUb: S 矩阵从 GM（Global Memory）拷贝到 UB（Unified Buffer）
    // ============================================================================
    // 将 Cube 引擎计算的 S=QK^T 结果从 GM 拷贝到 Vector 引擎的 UB。
    //
    // 参数：
    //   gInput        : GM 中的 S 矩阵（float 精度）
    //   sUbOffset     : UB 中的目标偏移（用于 Ping-Pong 双缓冲：0 或 MAX_UB_S_ELEM_NUM）
    //   rowNumCurLoop : 当前行数
    //   columnNumRound: 对齐后的列数（RoundUp 到 FLOAT_BLOCK_SIZE=8）
    //   columnNumPad  : GM 中的列步幅（含 padding，可能 > columnNumRound）
    //
    // DataCopyParams 参数说明：
    //   - blockCount  = rowNumCurLoop（拷贝的行数）
    //   - blockLen    = columnNumRound / FLOAT_BLOCK_SIZE（每行有效元素对应的 block 数）
    //   - srcStride   = (columnNumPad - columnNumRound) / FLOAT_BLOCK_SIZE（源行间 padding）
    //   - dstStride   = 0（目标行间无 padding，UB 中紧凑存储）
    // ============================================================================
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

    // ============================================================================
    // CopyMaskGmToUb: 掩码矩阵从 GM 拷贝到 UB（支持 GQA 头扩展）
    // ============================================================================
    // 将因果掩码（causal mask）从 GM 拷贝到 UB。
    // GQA（Grouped Query Attention）场景下，多个 Q head 共享同一组 KV head，
    // 掩码需要按 head 重复拷贝。
    //
    // 三段拷贝策略（处理行数不整除 tokenNumPerHead 的情况）：
    //   1. proTokenNum    : 前导部分（不完整的 head 尾部）
    //   2. integralHeadNum: 完整的 head 数（每个 head 拷贝 tokenNumPerHead 行）
    //   3. epiTokenNum    : 尾部部分（不完整的 head 头部）
    //
    // 参数：
    //   gMask              : GM 中的掩码矩阵
    //   columnNum          : 实际列数
    //   columnNumRound     : 对齐后的列数
    //   maskStride         : GM 中掩码的行步幅
    //   tokenNumPerHead    : 每个 head 的 token 数
    //   proTokenIdx        : 前导部分的起始 token 索引
    //   proTokenNum        : 前导部分的 token 数
    //   integralHeadNum    : 完整 head 数
    //   epiTokenNum        : 尾部部分的 token 数
    //
    // 使用 DataCopyPad 处理非对齐的列数（掩码列数通常不满足对齐要求）。
    // ============================================================================
    __aicore__ inline
    void CopyMaskGmToUb(
        AscendC::GlobalTensor<ElementMask> gMask,
        uint32_t columnNum, uint32_t columnNumRound,
        uint32_t maskStride, uint32_t tokenNumPerHead,
        uint32_t proTokenIdx, uint32_t proTokenNum,
        uint32_t integralHeadNum, uint32_t epiTokenNum)
    {
        uint32_t innerUbRowOffset = 0;
        // ---- 第 1 段：前导部分（不完整的 head 尾部）----
        if (proTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask[proTokenIdx * maskStride],
                AscendC::DataCopyExtParams(
                    proTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += proTokenNum * columnNumRound;
        }
        // ---- 第 2 段：完整的 head（循环拷贝 integralHeadNum 个 head）----
        for (uint32_t headIdx = 0; headIdx < integralHeadNum; headIdx++) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    tokenNumPerHead, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
            innerUbRowOffset += tokenNumPerHead * columnNumRound;
        }
        // ---- 第 3 段：尾部部分（不完整的 head 头部）----
        if (epiTokenNum != 0) {
            AscendC::DataCopyPad(
                maskUbTensor[innerUbRowOffset], gMask,
                AscendC::DataCopyExtParams(
                    epiTokenNum, columnNum * sizeof(ElementMask),
                    (maskStride - columnNum) * sizeof(ElementMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementMask>(false, 0, 0, 0));
        }
    }

    // ============================================================================
    // ScaleS: 对 S 矩阵进行缩放 S *= scaleValue
    // ============================================================================
    // scaleValue = 1/sqrt(d)，是 attention 缩放因子的标准做法。
    // 使用 Muls 指令（标量乘法）对 UB 中的 S 矩阵逐元素乘以 scaleValue。
    //
    // 参数：
    //   sUbOffset     : UB 中 S 矩阵的偏移（Ping-Pong 双缓冲）
    //   rowNumCurLoop : 当前行数
    //   columnNumRound: 对齐后的列数
    // ============================================================================
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

    // ============================================================================
    // UpCastMask: 掩码类型提升（低精度 → 高精度）
    // ============================================================================
    // 将掩码从低精度类型（如 int8）转换到高精度类型（如 half 或 float）。
    // 典型转换路径：int8 → half → float（分两步调用本函数）。
    //
    // 模板参数：
    //   ElementMaskDst : 目标类型（half 或 float）
    //   ElementMaskSrc : 源类型（int8 或 half）
    //
    // 参数：
    //   maskUbTensorDst : 目标 UB Tensor
    //   maskUbTensorSrc : 源 UB Tensor
    //   rowNumCurLoop   : 当前行数
    //   columnNumRound  : 对齐后的列数
    //
    // 使用 Cast 指令，CAST_NONE 表示不进行舍入（掩码值 0/1 精确转换）。
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
    // ApplyMask: 将掩码应用到 S 矩阵（掩码区域置为 -3e38）
    // ============================================================================
    // 因果掩码（causal mask）实现：掩码值为 0 的位置表示需要屏蔽（未来 token）。
    // 实现方式：
    //   1. maskUbTensor32 *= -3e38  （掩码值 0→0, 1→-3e38）
    //   2. S += maskUbTensor32      （0 位置不变，1 位置加上 -3e38 变为极小值）
    //
    // float 版本使用 -3e38（float 最小值），half 版本使用 -65504（half 最小值）。
    //
    // 参数：
    //   sUbOffset       : UB 中 S 矩阵的偏移
    //   rowNumCurLoop   : 当前行数
    //   columnNumRound  : S 矩阵对齐后的列数
    //   maskColumnRound : 掩码对齐后的列数（可能 < columnNumRound，表示部分列覆盖）
    //   addMaskUbOffset : 掩码在 S 中的列偏移（用于部分列覆盖场景）
    //
    // 两种情况：
    //   - maskColumnRound == columnNumRound: 掩码覆盖全部列，直接逐元素 Add
    //   - maskColumnRound < columnNumRound : 掩码仅覆盖部分列，需按列偏移分段 Add
    // ============================================================================
    __aicore__ inline
    void ApplyMask(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound, uint32_t maskColumnRound,
        uint32_t addMaskUbOffset)
    {
        // 步骤 1：掩码值转换（0→0, 1→-3e38）
        AscendC::Muls<float, false>(
            maskUbTensor32,
            maskUbTensor32,
            (float)-3e38,
            (uint64_t)0,
            CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤 2：将掩码加到 S 矩阵
        if (maskColumnRound == columnNumRound) {
            // ---- 情况 1：掩码覆盖全部列，直接逐元素 Add ----
            AscendC::Add<float, false>(
                lsUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                maskUbTensor32,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * maskColumnRound, FLOAT_VECTOR_SIZE),
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        } else {
            // ---- 情况 2：掩码仅覆盖部分列，按列偏移分段 Add ----
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
            // 处理尾部不足 FLOAT_VECTOR_SIZE 的列
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

    // ============================================================================
    // CalcLocalRowMax: 计算当前 KV 块的局部行最大值 lm
    // ============================================================================
    // 根据列数分派到不同的专用归约函数：
    //   - 512 列: RowmaxSPECTILE512（3 级归约，8 路 split）
    //   - 256 列: RowmaxSPECTILE256（3 级归约，4 路 split，float 版特有）
    //   - 其他  : RowmaxTAILTILE（通用归约，分段累加）
    //
    // 结果存入 lmUbTensor[rowOffset]，用于后续 UpdateGlobalRowMax 更新全局行最大值。
    // ============================================================================
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

    // ============================================================================
    // UpdateGlobalRowMax: 在线更新全局行最大值（Online Softmax 核心）
    // ============================================================================
    // Online Softmax 算法的关键步骤：避免物化完整 QK^T 矩阵，逐 KV 块更新全局最大值。
    //
    // 算法：
    //   首块 (isFirstStackTile=true):
    //     hm = lm  （直接初始化全局最大值为局部最大值）
    //   后续块 (isFirstStackTile=false):
    //     hm = max(lm, gm)     （合并局部和全局最大值）
    //     dm = exp(gm - hm)    （计算衰减因子，gm < hm 时 dm < 1）
    //   最后:
    //     gm = hm              （更新全局最大值）
    //
    // dm（衰减因子）的作用：后续 UpdateGlobalRowSum 中 gl *= dm，
    //   对之前块的 P 值进行缩放，保证 softmax 分母的一致性。
    //
    // 参数：
    //   dmUbOffsetCurCycle : dm 在 UB 中的偏移（按 curStackTileMod * MAX_ROW_NUM_SUB_CORE 分块）
    //   rowOffset          : 行偏移（子核内行索引）
    //   isFirstStackTile   : 是否为首个 KV 块（决定初始化还是合并）
    // ============================================================================
    __aicore__ inline
    void UpdateGlobalRowMax(uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound, uint32_t columnNum,
        uint32_t columnNumRound, uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // ---- 首块：直接初始化 hm = lm ----
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
            // *** dm = exp(dm)
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

    // ============================================================================
    // CalcExp: 计算指数 P = exp(S - gm)（数值稳定的 softmax）
    // ============================================================================
    // 减去行最大值 gm 保证数值稳定性（避免 exp 溢出）。
    //
    // 算法步骤：
    //   1. hm_block = Brcb(hm)     （将 hm 广播到每个 FLOAT_BLOCK_SIZE 块，存入 tv）
    //   2. S = S - hm_block        （逐元素减，每行减去对应行最大值）
    //   3. S = exp(S)              （计算指数，得到 P 矩阵）
    //
    // Brcb（Broadcast Block）指令：将 1 个元素广播到 FLOAT_BLOCK_SIZE(8) 个连续位置，
    //   使得 hm（每行 1 个值）能与 S（每行 columnNum 个值）逐元素运算。
    //
    // 参数：
    //   sUbOffset       : UB 中 S 矩阵的偏移
    //   rowNumCurLoop   : 当前行数
    //   rowNumCurLoopRound: 对齐后的行数
    //   columnNum       : 实际列数
    //   columnNumRound  : 对齐后的列数
    //   rowOffset       : 行偏移（用于定位 hm）
    // ============================================================================
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
        // 处理尾部不足 FLOAT_VECTOR_SIZE 的列
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

    // ============================================================================
    // CalcLocalRowSum: 计算当前 KV 块的局部行和 ll
    // ============================================================================
    // 对 CalcExp 后的 P 矩阵按行求和，得到局部行和 ll。
    // 根据列数分派到不同的专用归约函数：
    //   - 512 列: RowsumSPECTILE512
    //   - 256 列: RowsumSPECTILE256（float 版特有）
    //   - 其他  : RowsumTAILTILE
    //
    // 结果存入 llUbTensor[rowOffset]，用于后续 UpdateGlobalRowSum 更新全局行和。
    // ============================================================================
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

    // ============================================================================
    // UpdateGlobalRowSum: 在线更新全局行和（Online Softmax 核心）
    // ============================================================================
    // Online Softmax 算法的关键步骤：逐 KV 块更新 softmax 分母（全局行和）。
    //
    // 算法：
    //   首块 (isFirstStackTile=true):
    //     gl = ll  （直接初始化全局行和为局部行和）
    //   后续块 (isFirstStackTile=false):
    //     gl = dm * gl  （用衰减因子缩放之前块的行和）
    //     gl = ll + gl  （加上当前块的局部行和）
    //
    // dm 来自 UpdateGlobalRowMax 的计算结果 dm = exp(gm_old - gm_new)，
    //   当 gm 增大时 dm < 1，对之前块的 P 值进行缩放，保证 softmax 分母的一致性。
    //
    // 参数：
    //   sUbOffset          : UB 中 S 矩阵的偏移（未使用，保留接口一致性）
    //   rowNumCurLoop      : 当前行数
    //   rowNumCurLoopRound : 对齐后的行数
    //   dmUbOffsetCurCycle : dm 在 UB 中的偏移
    //   rowOffset          : 行偏移
    //   isFirstStackTile   : 是否为首个 KV 块
    // ============================================================================
    __aicore__ inline
    void UpdateGlobalRowSum(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t rowNumCurLoopRound,
        uint32_t dmUbOffsetCurCycle, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // ---- 首块：直接初始化 gl = ll ----
            // *** gl = ll
            AscendC::DataCopy(
                glUbTensor[rowOffset],
                llUbTensor[rowOffset],
                AscendC::DataCopyParams(1, rowNumCurLoopRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(rowNumCurLoop);
            // ---- 后续块：缩放并累加 ----
            // *** gl = dm * gl
            AscendC::Mul<float, false>(
                glUbTensor[rowOffset],
                dmUbTensor[dmUbOffsetCurCycle],
                glUbTensor[rowOffset],
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl
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

    // ============================================================================
    // DownCastP: 将 float 精度的 P 矩阵降精度为 half/bfloat16（float 版本特有）
    // ============================================================================
    // Online Softmax 计算在 float 精度下完成后，需要将 P 矩阵降精度为
    // ElementOutput（half 或 bfloat16_t），供 Cube 引擎的 PV 乘法使用。
    //
    // half 版本（online_softmax_low_prec.hpp）无需此步骤，因为 P 本身就是 half 精度。
    //
    // 舍入模式：
    //   - bfloat16_t: CAST_RINT（银行家舍入，适合 bfloat16 的有限精度）
    //   - half      : CAST_NONE（直接截断，half 精度足够）
    //
    // 参数：
    //   sUbOffset     : UB 中 P 矩阵（float）的偏移
    //   rowNumCurLoop : 当前行数
    //   columnNumRound: 对齐后的列数
    //
    // 输出：lpUbTensor[sUbOffset]（ElementOutput 类型，half 或 bfloat16_t）
    // ============================================================================
    __aicore__ inline
    void DownCastP(uint32_t sUbOffset, uint32_t rowNumCurLoop, uint32_t columnNumRound)
    {
        // *** lp = castfp32to16(ls)
        if (std::is_same<ElementOutput, bfloat16_t>::value) {
            // bfloat16 输出：使用 CAST_RINT 舍入模式
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_RINT,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            // half 输出：使用 CAST_NONE 不舍入
            AscendC::Cast<ElementOutput, float, false>(
                lpUbTensor[sUbOffset],
                lsUbTensor[sUbOffset],
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                CeilDiv(rowNumCurLoop * columnNumRound, FLOAT_VECTOR_SIZE),
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
    }

    // ============================================================================
    // CopyPUbToGm: P 矩阵从 UB 拷贝回 GM（供 Cube 引擎 PV 乘法使用）
    // ============================================================================
    // 将降精度后的 P 矩阵（half/bfloat16）从 UB 拷贝回 GM。
    // Cube 引擎的 PV 乘法将从 GM 读取此 P 矩阵。
    //
    // 参数：
    //   gOutput       : GM 中的目标地址（P 矩阵输出位置）
    //   sUbOffset     : UB 中 P 矩阵的偏移（Ping-Pong 双缓冲）
    //   rowNumCurLoop : 当前行数
    //   columnNumRound: 对齐后的列数（按 BLOCK_SIZE=16 对齐）
    //   columnNumPad  : GM 中的列步幅（含 padding）
    //
    // DataCopyParams 参数说明：
    //   - blockCount  = rowNumCurLoop（拷贝的行数）
    //   - blockLen    = columnNumRound / BLOCK_SIZE（每行有效元素对应的 block 数）
    //   - srcStride   = 0（源行间无 padding，UB 中紧凑存储）
    //   - dstStride   = (columnNumPad - columnNumRound) / BLOCK_SIZE（目标行间 padding）
    // ============================================================================
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

    // ============================================================================
    // SubCoreCompute: 子核计算编排（Online Softmax 向量化计算核心）
    // ============================================================================
    // 将 Online Softmax 的各步骤按 HardEvent 流水线编排执行。
    // 模板参数 doTriUMask 控制是否为因果掩码路径（影响 HardEvent 同步策略）。
    //
    // 计算步骤（7 步）：
    //   1. CalcLocalRowMax  : 计算局部行最大值 lm
    //   2. UpdateGlobalRowMax: 更新全局行最大值 gm，计算衰减因子 dm
    //   3. CalcExp          : P = exp(S - gm)
    //   4. DownCastP        : P 从 float 降精度为 half/bfloat16（float 版特有）
    //   5. CalcLocalRowSum  : 计算局部行和 ll
    //   6. CopyPUbToGm      : P 矩阵从 UB 拷贝到 GM
    //   7. UpdateGlobalRowSum: 更新全局行和 gl
    //
    // HardEvent 流水线同步（核内流水）：
    //   - MTE2_V : MTE2（GM→UB 搬运）完成后 V（向量计算）才能开始
    //   - V_MTE3 : V 计算完成后 MTE3（UB→GM 搬运）才能开始
    //   - V_MTE2 : V 计算完成后 MTE2 才能开始（下一轮数据搬运）
    //   - MTE3_V : MTE3 搬运完成后 V 才能开始（下一轮计算）
    //   - MTE3_MTE2: MTE3 完成后 MTE2 才能开始（跨阶段同步）
    //
    // Ping-Pong 双缓冲：pingpongFlag（0 或 1）控制 S/P 矩阵在 UB 中的两组缓冲区交替使用。
    //
    // 参数：
    //   gOutput               : GM 中 P 矩阵输出地址
    //   layoutOutput          : 输出布局描述
    //   rowOffset             : 行偏移（子核内行索引）
    //   isFirstStackTile      : 是否为首个 KV 块
    //   isLastNoMaskStackTile : 是否为最后一个无掩码 KV 块
    //   isFirstRowLoop        : 是否为首个行循环
    //   isLastRowLoop         : 是否为最后一个行循环
    //   columnNumRound        : 对齐后的列数
    //   pingpongFlag          : Ping-Pong 标志（0 或 1）
    //   curStackTileMod       : 当前 KV 块的模（用于 dm 缓冲区分块）
    // ============================================================================
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
        uint32_t sUbOffset = pingpongFlag * MAX_UB_S_ELEM_NUM;  // Ping-Pong 偏移
        uint32_t dmUbOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;  // dm 分块偏移

        // LSE OUT_ONLY 模式：首个块首个行循环需要等待 MTE3 完成（tv 用于传输 lse）
        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            if (isFirstStackTile && isFirstRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            }
        }
        // ---- 步骤 1：计算局部行最大值 lm ----
        CalcLocalRowMax(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        // ---- 步骤 2：更新全局行最大值 gm，计算衰减因子 dm ----
        UpdateGlobalRowMax(
            rowNumCurLoop, rowNumCurLoopRound,
            columnNum, columnNumRound,
            dmUbOffsetCurCycle,
            rowOffset,
            isFirstStackTile);

        // ---- 步骤 3：计算指数 P = exp(S - gm) ----
        CalcExp(sUbOffset, rowNumCurLoop, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        // 无掩码路径：等待 MTE3 完成（上一轮 P 矩阵搬运完毕）
        if constexpr (!doTriUMask) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
        }

        // ---- 步骤 4：P 矩阵降精度 float→half/bfloat16 ----
        DownCastP(sUbOffset, rowNumCurLoop, columnNumRound);
        // 设置 V_MTE3 标志：DownCastP 完成后可以开始 P 矩阵搬运
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);

        // ---- 步骤 5：计算局部行和 ll ----
        CalcLocalRowSum(sUbOffset, rowNumCurLoopRound, columnNum, columnNumRound, rowOffset);
        // 设置 V_MTE2 标志：行和计算完成后可以开始下一轮数据搬运
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);

        // ---- 步骤 6：P 矩阵从 UB 拷贝到 GM ----
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongFlag);  // 等待 DownCastP 完成
        CopyPUbToGm(gOutput, sUbOffset, rowNumCurLoop, columnNumRound, columnNumPad);
        if constexpr (!doTriUMask) {
            // 无掩码路径：设置 MTE3_V 标志，最后一个块需要 MTE3_MTE2 同步
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongFlag);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            // 有掩码路径：设置 MTE3_MTE2 标志（掩码搬运同步）
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        // ---- 步骤 7：更新全局行和 gl ----
        UpdateGlobalRowSum(
            sUbOffset, rowNumCurLoop, rowNumCurLoopRound, dmUbOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    // ============================================================================
    // operator() 无掩码版本：Online Softmax 入口（无因果掩码）
    // ============================================================================
    // 处理无掩码的 Online Softmax 计算流程。
    //
    // 子核并行策略：
    //   - 每个 AI Core 有 2 个子核（subBlockNum=2），各处理一半的行
    //   - qNBlockSize=1 时：按 qSBlockSize/2 分割行
    //   - qNBlockSize>1 时：按 qSBlockSize * (qNBlockSize/2) 分割行
    //
    // 行循环 + Ping-Pong 双缓冲：
    //   - 行数超过单次处理上限时分行循环（rowLoopNum 次）
    //   - preLoad=1：预加载下一轮数据，实现搬运与计算重叠
    //   - pingpongFlag = rowLoopIdx % 2：交替使用两组 UB 缓冲区
    //
    // 每轮循环流程：
    //   1. CopySGmToUb : S 矩阵从 GM 拷贝到 UB（MTE2）
    //   2. ScaleS      : S *= scaleValue（V）
    //   3. SubCoreCompute<false>: Online Softmax 全流程（V + MTE3）
    //
    // 参数：
    //   gOutput               : GM 中 P 矩阵输出地址
    //   gInput                : GM 中 S 矩阵输入地址
    //   layoutOutput/layoutInput: 输出/输入布局描述
    //   actualBlockShape      : 实际块形状（m 行 × n 列）
    //   isFirstStackTile      : 是否为首个 KV 块
    //   isLastNoMaskStackTile : 是否为最后一个无掩码 KV 块
    //   qSBlockSize           : Q 的 S 维块大小
    //   qNBlockSize           : Q 的 N 维块大小
    //   curStackTileMod       : 当前 KV 块的模（用于 dm 缓冲区分块）
    // ============================================================================
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

        // ---- 子核行分割 ----
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;
        // qNBlockSize=1 时 qNThisSubBlock=0（单 N 维，不分割）
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ?
            0 : (subBlockIdx == 1) ? (qNBlockSize - qNSplitSubBlock) : qNSplitSubBlock;
        // 行分割：qNBlockSize=1 时按 qSBlockSize/2 分割，否则按 qSBlockSize*(qNBlockSize/2) 分割
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
            (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;
        // ---- 行循环参数计算 ----
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);  // 限制单次行数不超过 FLOAT_VECTOR_SIZE
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;  // 预加载 1 轮（流水线填充）

        // ---- 行循环 + Ping-Pong 双缓冲 ----
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            // 阶段 1：数据搬运（MTE2）——预加载当前轮
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);  // 等待上一轮 V 计算完成
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);  // 通知 V 可以开始计算
            }
            // 阶段 2：计算（V）——处理上一轮预加载的数据（延迟 1 轮）
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, columnNum));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);  // 等待 MTE2 搬运完成
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
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

    // ============================================================================
    // operator() 带掩码版本：Online Softmax 入口（含因果掩码）
    // ============================================================================
    // 处理带因果掩码（causal mask）的 Online Softmax 计算流程。
    // 因果掩码确保注意力只关注当前位置及之前的 token（自回归生成场景）。
    //
    // 与无掩码版本的关键区别：
    //   1. 跨核同步：CrossCoreWaitFlag(qkReady) 等待 Cube 引擎完成 QK^T
    //   2. 掩码处理：CopyMaskGmToUb → UpCastMask(int8→half→float) → ApplyMask
    //   3. 因果掩码区域计算：根据 triUp/kvSStartIdx 判断掩码覆盖范围
    //   4. SubCoreCompute<true>：使用有掩码路径的 HardEvent 同步策略
    //
    // 因果掩码区域计算逻辑：
    //   - triUp >= kvSStartIdx: 掩码从 triUp 行开始，覆盖部分列（maskColumn < columnNum）
    //     → gmOffsetMaskRow = triUp - triUpRoundDown（行偏移）
    //     → addMaskUbOffset = triUpRoundDown - kvSStartIdx（UB 内列偏移）
    //   - triUp < kvSStartIdx: 掩码从 kvSStartIdx 列开始，覆盖全部列（maskColumn = columnNum）
    //     → gmOffsetMaskColumn = kvSStartIdx - triUp（列偏移）
    //     → addMaskUbOffset = 0（无需 UB 内偏移）
    //
    // 参数：
    //   gOutput/gInput/gMask   : GM 中 P/S/Mask 矩阵地址
    //   layoutOutput/Input/Mask: 输出/输入/掩码布局描述
    //   actualBlockShape       : 实际块形状
    //   isFirstStackTile       : 是否为首个 KV 块
    //   qSBlockSize/qNBlockSize: Q 的 S/N 维块大小
    //   curStackTileMod        : 当前 KV 块的模
    //   qkReady                : 跨核同步信号量（等待 Cube 引擎完成 QK^T）
    //   triUp                  : 上三角掩码起始行（对角线位置）
    //   triDown                : 下三角掩码结束行
    //   kvSStartIdx            : 当前 KV 块的起始索引
    //   kvSEndIdx              : 当前 KV 块的结束索引
    // ============================================================================
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
        // ---- 子核行分割（与无掩码版本相同）----
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

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
            rowOffsetThisSubBlock : 0;

        // ---- 因果掩码区域计算 ----
        // calc mask shift in gm
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t addMaskUbOffset;
        if (triUp >= kvSStartIdx) {
            // 情况 1：掩码起始行 triUp 在当前 KV 块内
            // 掩码仅覆盖 triUp 行之后的部分列
            uint32_t triUpRoundDown = RoundDown(triUp, BLOCK_SIZE_IN_BYTE);
            gmOffsetMaskRow = triUp - triUpRoundDown;       // GM 中掩码的行偏移
            gmOffsetMaskColumn = 0;                          // GM 中掩码的列偏移（从第 0 列开始）
            maskColumn = kvSEndIdx - triUpRoundDown;         // 掩码覆盖的列数
            addMaskUbOffset = triUpRoundDown - kvSStartIdx;  // UB 中掩码的列偏移
        } else {
            // 情况 2：掩码起始行 triUp 在当前 KV 块之前
            // 掩码覆盖全部列
            gmOffsetMaskRow = 0;
            gmOffsetMaskColumn = kvSStartIdx - triUp;        // GM 中掩码的列偏移
            maskColumn = columnNum;                           // 掩码覆盖全部列
            addMaskUbOffset = 0;                              // UB 中无需列偏移
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, BLOCK_SIZE_IN_BYTE);

        // ---- 计算掩码在 GM 中的偏移 ----
        int64_t offsetMask =
            layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn));
        auto gMaskThisSubBlock = gMask[offsetMask];
        auto layoutMaskThisSubBlock = layoutMask;

        // ---- 行循环参数计算 ----
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
        rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        // ---- 空行处理：仅等待跨核同步后返回 ----
        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);  // 即使无数据也需等待 Cube 引擎完成
            return;
        }

        // ---- 行循环 + Ping-Pong 双缓冲 ----
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            // 阶段 1：数据搬运（MTE2）——预加载当前轮
            if (rowLoopIdx < rowLoopNum) {
                uint32_t pingpongFlag = rowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop = (rowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                // 首轮：掩码加载在跨核同步之前
                if (rowLoopIdx == 0) {
                    // GQA 三段分解：proTokenNum/integralHeadNum/epiTokenNum
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskGmToUb(
                        gMaskThisSubBlock,
                        maskColumn, maskColumnRound, maskStride,
                        tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);  // 等待 Cube 引擎完成 QK^T
                }
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongFlag);
                CopySGmToUb(
                    gInputCurLoop, (pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
            }
            // 阶段 2：计算（V）——处理上一轮预加载的数据（延迟 1 轮）
            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t pingpongFlag = delayedRowLoopIdx % 2;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop = (delayedRowLoopIdx == rowLoopNum - 1) ?
                    (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                // ---- 掩码处理：int8 → half → float 两步类型提升 ----
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                UpCastMask<half, ElementMask>(maskUbTensor16, maskUbTensor, rowNumCurLoop, columnNumRound);
                UpCastMask<float, half>(maskUbTensor32, maskUbTensor16, rowNumCurLoop, columnNumRound);
                
                // ---- S 矩阵缩放 + 掩码应用 ----
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);
                ScaleS((pingpongFlag * MAX_UB_S_ELEM_NUM), rowNumCurLoop, columnNumRound);
                ApplyMask(
                    (pingpongFlag * MAX_UB_S_ELEM_NUM),
                    rowNumCurLoop, columnNumRound,
                    maskColumnRound, addMaskUbOffset);
                // ---- 预加载下一轮掩码 ----
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // GQA 三段分解
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
                // ---- Online Softmax 向量化计算 ----
                // online softmax vectorized compute
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
    // ============================================================================
    // 私有成员变量：UB Tensor 语义说明
    // ============================================================================
    // 所有 Tensor 均位于 Vector 引擎的 UB（Unified Buffer）中。
    //
    // S/P 矩阵缓冲区（Ping-Pong 双缓冲，偏移 0 或 MAX_UB_S_ELEM_NUM）：
    float scaleValue;                              // 缩放因子 1/sqrt(d)
    AscendC::LocalTensor<float> lsUbTensor;       // S 矩阵缓冲区（float）—— 存放 QK^T 结果，计算后变为 P=exp(S-gm)
    AscendC::LocalTensor<ElementOutput> lpUbTensor; // P 矩阵缓冲区（half/bfloat16）—— 降精度后的 P，供 Cube PV 乘法

    // 掩码缓冲区（与 lpUbTensor 分时复用同一 UB 区域）：
    AscendC::LocalTensor<ElementMask> maskUbTensor;  // 掩码原始缓冲区（int8）—— 从 GM 拷贝的原始掩码
    AscendC::LocalTensor<half> maskUbTensor16;       // 掩码中间缓冲区（half）—— int8→half 转换中间态
    AscendC::LocalTensor<float> maskUbTensor32;      // 掩码最终缓冲区（float）—— half→float 转换后，用于 ApplyMask

    // 行最大值缓冲区（Online Softmax 状态量）：
    AscendC::LocalTensor<float> lmUbTensor;  // 局部行最大值 lm —— 当前 KV 块的行最大值
    AscendC::LocalTensor<float> hmUbTensor;  // 合并后行最大值 hm —— max(lm, gm) 的结果
    AscendC::LocalTensor<float> gmUbTensor;  // 全局行最大值 gm —— 跨 KV 块累积的行最大值
    AscendC::LocalTensor<float> dmUbTensor;  // 衰减因子 dm —— dm=exp(gm_old-gm_new)，用于缩放之前块的 P 值

    // 行和缓冲区（Online Softmax 状态量）：
    AscendC::LocalTensor<float> llUbTensor;  // 局部行和 ll —— 当前 KV 块的行和
    AscendC::LocalTensor<float> tvUbTensor;  // 临时向量缓冲区 —— 用于 Brcb 广播和归约中间结果
    AscendC::LocalTensor<float> glUbTensor;  // 全局行和 gl —— 跨 KV 块累积的行和（softmax 分母）
};

}

#endif