/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_CATLASS_KERNEL_H
#define OPTEST_CATLASS_KERNEL_H

#include <cstdint>
#include <vector>

#include <acl/acl.h>

#include "catlass_kernel_jit.h"
#include "catlass_kernel_prebuilt.h"
namespace CatlassKernel {

// Forward declarations for prebuilt kernel parameter types (defined in upstream headers).
struct KernelInfo;
struct FAKernelInfo;
struct W4A4QuantMatmulKernelInfo;
struct GemmKernelInfo;

/**
 * @brief 通用的 kernel 执行信息结构体。
 *
 * 包含 GEMM/Batched GEMM 类算子所需的全部参数，包括数据类型、
 * 问题尺寸（M/N/K/batch/group）、布局标志、数据地址列表。
 *
 * 由外部调用方填充，直接传入各 kernel 入口函数作为运行期参数。
 * 对不同的 kernel 实现，部分字段可能不被使用。
 *
 * @note 地址字段 inputAddr/outputAddr 使用 vector<uint8_t*>，
 *       需按具体 kernel 的约定确定元素访问方式。
 */

/**
 * @brief 基础 MatMul kernel。
 * @param blockNum 启用的 AI Core 数量。
 * @param stream   ACL 计算流。
 * @param tParams  编译期模板参数（数据类型、布局）。
 * @param params   运行期参数（M/N/K、地址）。
 */
void BasicMatmul(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params);

/**
 * @brief 基础 MatMul 2D Tiling kernel（Tiling 库版本）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void BasicMatmulTLA(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief MatMul + Add（逐元素加法 fused kernel）。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void MatmulAdd(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief Padding MatMul（带对齐 padding）。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void PaddingMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief Grouped MatMul（多组独立 GEMM 合并执行）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数（g 指定组数，groupList 指定各组尺寸）。
 */
void GroupedMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief Grouped MatMul（混合精度变体）。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void GroupedMatmulMix(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief 优化 MatMul kernel（FFMA/MTE2 特殊优化）。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void OptimizedMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief 优化 MatMul 2D Tiling kernel。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void OptimizedMatmulTLA(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief Grouped MatMul（带逐 token 反量化，按 M 切分）。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void GroupedMatmulSliceMPerTokenDequant(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief Grouped MatMul（带逐 token 反量化，按 K 切分）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void GroupedMatmulSliceKPerTokenDequant(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief 量化 MatMul（weight-only / per-channel 量化）。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void QuantMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief W4A4 量化 MatMul kernel。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo W4A4 量化参数。
 */
void W4A4QuantMatmul(const uint32_t blockNum, aclrtStream stream, const W4A4QuantMatmulKernelInfo& kernelInfo);

/**
 * @brief Split-K 并行 MatMul（K 维度切分并行）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void SplitkMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief Stream-K 并行 MatMul（Stream-K 调度策略）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void StreamkMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief 带 padding 的 Split-K MatMul。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void PaddingSplitkMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief 基础 MatMul（带 Preload ZN 分块预取优化）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void BasicMatmulPreloadZN(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief MatMul + ReLU 激活 fused kernel。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void MatmulRelu(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief MatMul + GELU 激活 fused kernel。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void MatmulGelu(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief MatMul + SiLU 激活 fused kernel。
 * @param blockNum  启用的 AI Core 数量。
 * @param stream    ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void MatmulSilu(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief MatMul Full Load（A 矩阵一次性全加载变体）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void MatmulFullLoadA(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief 标准 GEMM 入口（alpha*A*B + beta*C）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo GEMM 参数（含 alpha/beta）。
 */
void Gemm(const uint32_t aicCoreNum, aclrtStream stream, const GemmKernelInfo& kernelInfo);

/**
 * @brief 小尺寸 MatMul（M ≤ 32 等小规模优化路径）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void SmallMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

/**
 * @brief Batch MatMul（多 batch 并行 GEMM）。
 * @param aicCoreNum AI Core 数量。
 * @param stream     ACL 计算流。
 * @param kernelInfo 通用 kernel 参数。
 */
void BatchedMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo& kernelInfo);

} // namespace CatlassKernel

#endif // SHARED_LIB_CATLASS_KERNEL_H
