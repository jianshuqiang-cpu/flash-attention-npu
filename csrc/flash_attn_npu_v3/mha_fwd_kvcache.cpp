/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// ============================================================================
// 文件说明：mha_fwd_kvcache.cpp
// ----------------------------------------------------------------------------
// 本文件是 FlashAttention v3 前向推理的 AscendC 核函数（Device 侧）实现。
// 核心职责：
//   1. 定义 FAInferKernel 类模板 —— 真正执行 FlashAttention 计算的核类，
//      内部实现 Cube（矩阵乘）与 Vector（softmax/rescale）双引擎流水线。
//   2. 定义 FAInfer 全局核函数 —— 核函数入口，负责根据模板参数（数据类型、
//      分页标志、掩码类型、输入布局、LSE 模式）组装 CATLASS 类型并实例化
//      FAInferKernel，最后调用其 operator() 完成计算。
//
// 计算流程（单次 task，即一个 Q 分块 × KV 分块）：
//   Q × K^T  ->  OnlineSoftmax  ->  P × V  ->  RescaleO
//   (Cube)      (Vector)           (Cube)     (Vector)
//   其中 Cube 与 Vector 通过 HardEvent 信号量（qkReady/softmaxReady/pvReady）同步。
// ============================================================================

// ---- CATLASS 框架头文件（华为 CANN 模板库，类似 NVIDIA CUTLASS）----
#include "catlass/arch/arch.hpp"                       // 架构标签定义
#include "catlass/arch/cross_core_sync.hpp"            // 跨核同步（Cube/Vector 间信号量）
#include "catlass/arch/resource.hpp"                   // 硬件资源管理（L1/L0/UB 缓冲区分配）
#include "catlass/catlass.hpp"                         // CATLASS 公共头文件
#include "catlass/debug.hpp"                           // 调试工具
#include "catlass/epilogue/block/block_epilogue.hpp"    // Epilogue 块（softmax/rescale 的基类模板）
#include "online_softmax_low_prec.hpp"                 // 低精度在线 softmax 实现
#include "online_softmax.hpp"                          // 在线 softmax 实现
#include "rescale_o_low_prec.hpp"                      // 低精度输出 rescale 实现
#include "rescale_o.hpp"                               // 输出 rescale 实现
#include "catlass/epilogue/dispatch_policy.hpp"        // Epilogue 调度策略
#include "catlass/gemm/block/block_mmad.hpp"            // 块级矩阵乘（Cube 引擎封装）
#include "pv_matmul.hpp"                               // P×V 矩阵乘特化
#include "qk_matmul.hpp"                               // Q×K^T 矩阵乘特化
#include "catlass/gemm/dispatch_policy.hpp"            // GEMM 调度策略
#include "fa_block.h"                                  // FlashAttention 块定义
#include "catlass/gemm/gemm_type.hpp"                  // GemmType：元素类型+布局的组合类型
#include "catlass/layout/layout.hpp"                   // 布局定义（RowMajor/ColumnMajor）

// ---- 本项目 v3 头文件 ----
#include "kernel_common.hpp"   // 核函数公共定义：常量、枚举、FAIKernelParams、GetQNBlockTile 等
#include "kernel_operator.h"    // AscendC 算子编程框架头文件
#include "tilingdata.h"        // FAInferTilingData 结构体定义（Host 传给 Device 的分块参数）

using namespace Catlass;        // CATLASS 命名空间
using namespace KernelCommon;  // 核函数公共定义命名空间（常量、枚举等）

namespace SplitFuse {
    // ============================================================================
    // FAInferKernel 类模板
    // ----------------------------------------------------------------------------
    // FlashAttention v3 推理核类，承载 Device 侧真正的计算逻辑。
    // 采用 Cube（矩阵乘）+ Vector（softmax/rescale）双引擎流水线设计，
    // 通过 HardEvent 信号量在两个引擎间同步数据。
    //
    // 模板参数说明：
    //   BlockMmadQK           —— Q×K^T 矩阵乘块（Cube 引擎），封装了 Q/K 的数据类型、
    //                            布局、分块形状与计算策略
    //   BlockMmadPV           —— P×V 矩阵乘块（Cube 引擎），封装了 P/V 的数据类型、
    //                            布局、分块形状与计算策略
    //   EpilogueOnlineSoftmax —— 在线 softmax 后处理（Vector 引擎），负责对 S=QK^T
    //                            做 scale、减最大值、exp、求和，并生成 P=softmax(S)
    //   EpilogueRescaleO      —— 输出 rescale 后处理（Vector 引擎），负责将累加的
    //                            O 缩放并写入最终输出 O 与 logsumexp(lse)
    //   PAGED_CACHE_FLAG      —— 是否启用分页 KV Cache（Paged Attention），影响 K/V
    //                            的地址计算方式（通过 block_table 间接寻址）
    //   MASK_TYPE             —— 掩码类型：NO_MASK(无)/MASK_CAUSAL(因果)/MASK_SPEC(自定义)
    //   INPUT_LAYOUT          —— 输入布局：BSND(标准 batch) / TND(变长，紧凑排列)
    // ============================================================================
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        bool PAGED_CACHE_FLAG,
        FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK,
        FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND>
    class FAInferKernel {
    public:
        // ---- 从 BlockMmadQK 提取 Q/K/S 的类型与布局（编译期类型推导）----
        using ArchTag = typename BlockMmadQK::ArchTag;            // 架构标签（如 AtlasA2）
        using L1TileShape = typename BlockMmadQK::L1TileShape;     // L1 缓存分块形状
        using ElementQ = typename BlockMmadQK::ElementA;          // Q 的元素类型（如 half）
        using LayoutQ = typename BlockMmadQK::LayoutA;            // Q 的内存布局（RowMajor）
        using ElementK = typename BlockMmadQK::ElementB;          // K 的元素类型
        using LayoutK = typename BlockMmadQK::LayoutB;            // K 的内存布局（ColumnMajor）
        using ElementS = typename BlockMmadQK::ElementC;           // S=QK^T 的元素类型（如 float）
        using LayoutS = typename BlockMmadQK::LayoutC;            // S 的内存布局

        // ---- 从 BlockMmadPV 提取 P/V 的类型与布局 ----
        using ElementP = typename BlockMmadPV::ElementA;          // P=softmax(S) 的元素类型
        using LayoutP = typename BlockMmadPV::LayoutA;            // P 的内存布局
        using ElementV = typename BlockMmadPV::ElementB;          // V 的元素类型
        using LayoutV = typename BlockMmadPV::LayoutB;            // V 的内存布局

        // ---- 从 EpilogueOnlineSoftmax 提取 Mask 的类型与布局 ----
        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;  // 掩码元素类型（int8_t）
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;    // 掩码布局

        // ---- 从 EpilogueRescaleO 提取输出 O / 临时 O / lse / update 的类型与布局 ----
        using ElementO = typename EpilogueRescaleO::ElementOutput;     // 最终输出 O 的元素类型
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;        // O 的内存布局
        using ElementOTmp = typename EpilogueRescaleO::ElementInput;    // 中间结果 OTmp 的元素类型
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;      // OTmp 的内存布局
        using ElementLse = typename EpilogueRescaleO::ElementLse;       // logsumexp 的元素类型（float）
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;          // lse 的内存布局
        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;  // update 缓冲区的元素类型
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;    // update 的内存布局

        // LSE 模式：NONE(不输出 lse) / OUTPUT(输出 lse)，编译期常量
        static constexpr Epilogue::LseModeT LSE_MODE = EpilogueRescaleO::LSE_MODE;

        // 默认构造函数（__aicore__ inline 表示在 AI Core 上内联执行）
        __aicore__ inline
        FAInferKernel() {}

        // ========================================================================
        // operator() —— 核函数主入口
        // ------------------------------------------------------------------------
        // 每个 AI Core 调用一次，负责处理分配给该核的所有 task。
        // 一个 task = 一个 Q 的 N 维分块 × 一个 Q 的 S 维分块（即一组 query token
        // 对全部 KV 的注意力计算）。核内通过 stride 遍历 batch，跨核通过 coreIdx 步进。
        //
        // 整体流程分为以下阶段：
        //   阶段1：从 tiling 数据中提取计算参数
        //   阶段2：绑定 GlobalTensor（GM 全局内存地址）
        //   阶段3：Cube 引擎初始化（HardEvent 信号量 + L1 分块计算 + BlockMmad 实例化）
        //   阶段4：Vector 引擎初始化（HardEvent 信号量 + Epilogue 实例化）
        //   阶段5：计算 stride、偏移量与首个 batch 的分块参数
        //   阶段6：主任务循环（遍历所有 task，每个 task 内部循环 KV 分块）
        //   阶段7：等待所有 HardEvent 完成 + PipeBarrier 同步
        // ========================================================================
        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            // ---- 阶段1：从 tiling 数据中提取计算参数 ----
            // fATilingData 是 Host 侧通过 FAInferTilingData 填充后传入 Device 的分块参数
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            uint64_t mm1OutSize = fATilingData->mm1OutSize;             // QK^T 工作区大小（字节）
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;   // softmax 工作区大小（字节）
            uint64_t mm2OutSize = fATilingData->mm2OutSize;             // PV 工作区大小（字节）
            uint32_t batch = fATilingData->batch;                       // batch 数
            uint32_t qHeads = fATilingData->numHeads;                   // Q 头数
            uint32_t kvHeads = fATilingData->kvHeads;                   // KV 头数（MQA/GQA 时 < qHeads）
            uint32_t embed = fATilingData->embeddingSize;               // Q/K 的 head 维度（如 128）
            uint32_t embedV = fATilingData->embeddingSizeV;             // V 的 head 维度（可与 embed 不同）
            uint32_t pagedBlockSize = fATilingData->blockSize;          // 分页块大小（每个 block 的 KV token 数）
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;  // 每 batch 最大 block 数
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;        // 第 0 个 batch 的 task 数
            uint32_t totalTaskNum = fATilingData->totalTaskNum;                  // 所有 batch 的 task 总数
            uint32_t blockSize = fATilingData->blockSize;                       // 分页块大小（同 pagedBlockSize）
            uint32_t maskType = fATilingData->maskType;                        // 掩码类型（运行期值）
            float scaleValue = fATilingData->scaleValue;                       // softmax 缩放因子（1/sqrt(d)）

            // ---- 阶段2：绑定 GlobalTensor（GM 全局内存地址）----
            // 将 params 中的裸指针封装为 AscendC::GlobalTensor，便于后续按元素访问
            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);              // Query 输入
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)params.k);              // Key 输入
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)params.v);              // Value 输入
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);     // 注意力掩码输入
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));  // 分页块表（block->物理页映射）
            AscendC::GlobalTensor<int32_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);  // 实际 Q 序列长度数组
            AscendC::GlobalTensor<int32_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen); // 实际 KV 序列长度数组
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);              // 最终输出 O
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);        // logsumexp 输出
            // 工作区划分为 4 个区域，依次存放 S、P、OTmp、OUpdate
            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace));    // S=QK^T 工作区（mm1 区域）
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize));  // P=softmax(S) 工作区
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize + smOnlineOutSize));  // PV 中间结果
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace +
                mm1OutSize + smOnlineOutSize + mm2OutSize));  // rescale update 缓冲区

            // 获取当前核的索引与总核数（用于 task 分配）
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_CUBE__
            // ---- 阶段3：Cube 引擎初始化 ----
            // Cube 引擎负责矩阵乘（Q×K^T 和 P×V）。下面先设置各流水线事件标志位，
            // 使 Cube 与 MTE1/MTE2（数据搬运）之间的依赖关系可以异步等待。
            // SetFlag 预置信号量，后续 WaitFlag 时才会阻塞等待对应事件完成。
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);   // Cube(M) -> MTE1：矩阵乘完成后通知 MTE1 搬运
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);     // FIX(向量转标量) -> Cube
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0); // MTE1 -> MTE2：L1 到 L0 搬运完成通知
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            
            // ---- L1 缓存分块大小动态计算 ----
            // 目标：在 L1 缓存容量限制内，最大化每次矩阵乘的 N 维分块，减少循环次数。
            uint32_t kDynNum = RoundUp(embed, NUM_128);                 // embed 向上对齐到 128 的倍数
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;            // 至少 256（保证 L0 性能）
            // 预留 V 的 L1 空间后，剩余空间用于 Q 和 K
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);  // Q 分块占用空间
            // 计算最大可容纳的 N 维分块数（考虑双缓冲 DOUBLE_BUFFER）
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            // N 维分块不超过硬件上限 L1_MAX_N_NUM(128)，且需对齐
            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);  // QK 的 L1 占用
            // 实例化 Q×K^T 矩阵乘块（Cube 引擎），传入动态分块参数
            BlockMmadQK blockMmadQK(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;  // PV 的 K 维分块
            // 实例化 P×V 矩阵乘块（Cube 引擎），复用 L1_QK_SIZE 空间
            BlockMmadPV blockMmadPV(resource, nDynNum, kPVDynNum, MAX_KV_STACK_LEN, L1_QK_SIZE);
#endif
#ifdef __DAV_C220_VEC__
            // ---- 阶段4：Vector 引擎初始化 ----
            // Vector 引擎负责 softmax 和 rescale 等向量运算。设置流水线事件标志位。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);    // MTE3(GM->UB) -> Vector
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE3 -> MTE2
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);    // Vector -> MTE2
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            // 实例化在线 softmax 与 rescale 后处理（Vector 引擎）
            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
            EpilogueRescaleO epilogueRescaleO(resource);

            // Vector 核的 coreIdx 需除以子块数，映射到逻辑核索引（与 Cube 核对齐）
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            // ---- 阶段5：计算 stride、偏移量与首个 batch 的分块参数 ----
            // stride：每个 token 在 GM 中的跨步（单位：元素数），用于地址计算
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);    // Q: 跨过一个 token 的元素数
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);   // O: 同上（V 维度可能不同）
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);   // K: 跨过一个 token 的元素数
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV); // V: 同上
            uint32_t embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);   // embed 对齐到 16（BLOCK_SIZE=16）
            uint32_t embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE); // embedV 对齐到 16
            uint32_t groupSize = qHeads / kvHeads;                        // GQA 分组大小（每个 KV 头对应几个 Q 头）

            // 各 batch 的 GM 偏移量（随 batch 推进累加）
            uint64_t qBOffset = 0;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t oBOffset = 0;
            uint64_t lseBOffset = 0;
            uint64_t blockBOffset = 0;

            // 遍历 batch 用的游标
            uint32_t preTotalTaskNum = 0;   // 前面所有 batch 累计的 task 数
            uint32_t curBatch = 0;          // 当前 batch 索引
            uint32_t totalQTokens = 0;       // TND 布局下所有 batch 的 Q token 总数（用于 lse 布局）
            // 初始化首个 batch 的序列长度
            uint32_t qSeqlen = fATilingData->maxQSeqlen;
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
            if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                // TND（变长）布局：Q/KV 按序列紧凑排列，通过前缀和数组定位每个 batch 的起始
                totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch));  // 末尾元素即总 token 数
                uint32_t prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch + 1)) - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    // 非分页模式下 KV 也是紧凑排列，需通过前缀和计算长度
                    uint32_t prevKvSeqlenSum = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch + 1)) - prevKvSeqlenSum;
                }
            }
            // 计算首个 batch 的分块参数
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);  // N 维分块大小（考虑 GQA）
            uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile); // 每组 Q 头的 N 分块数
            uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;            // 该 batch 的 N 分块总数
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);               // S 维分块大小（固定 128）
            uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);        // 该 batch 的 S 分块数
            uint32_t curTotalTaskNum = firstBatchTaskNum;                    // 累计 task 数（含当前 batch）
            // ---- 阶段6：主任务循环 ----
            // 跨核步进：每个核处理 taskIdx = coreIdx, coreIdx+coreNum, coreIdx+2*coreNum, ...
            // Go through each task.
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                // 当 taskIdx 超出当前 batch 的累计 task 数时，推进到下一个 batch
                // Get the offset of each core on the GM.
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;
                    preTotalTaskNum = curTotalTaskNum;
                    // 累加各张量的 batch 偏移
                    qBOffset += qSeqlen * strideQ;
                    if constexpr (!PAGED_CACHE_FLAG) {
                        // 非分页：K/V 按 kvSeqlen 累加偏移
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                    } else {
                        // 分页：通过 block_table 偏移（每 batch 跳过 maxNumBlocksPerBatch 个块表项）
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
                    }
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);
                    lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);

                    // 读取新 batch 的序列长度并重新计算分块参数
                    qSeqlen = fATilingData->maxQSeqlen;
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        uint32_t prevQSeqlenSum = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                        qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch + 1)) - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                            kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch + 1)) - prevKvSeqlenSum;
                        }
                    }
                    curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                    curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                    curQSBlockTile = GetQSBlockTile(kvSeqlen);
                    curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                    curTotalTaskNum += curQNBlockNum * curQSBlockNum;  // 该 batch 的 task 数 = N分块数 × S分块数
                }
                // 将全局 taskIdx 分解为当前 batch 内的 (S分块索引, N分块索引)
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;        // S 维分块索引
                uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;  // N 维分块索引
                uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;       // 组内 N 分块索引

                // 由 N 分块索引推导出 KV 头索引与 Q 头起始索引（GQA 映射）
                uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;          // 对应的 KV 头索引
                uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;  // Q 头起始索引
                uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;  // lse 的 token 偏移

                // 计算各张量在 GM 中的绝对偏移
                uint64_t gmOffsetQ = qBOffset +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                    static_cast<uint64_t>(qNStartIdx * embed);
                uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);
                uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);
                uint64_t gmOffsetO = oBOffset +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                    static_cast<uint64_t>(qNStartIdx * embedV);
                uint64_t gmOffsetLse = lseBOffset +
                    static_cast<uint64_t>(lseTokenOffset + qNStartIdx);

                // 计算当前分块的实际尺寸（最后一个分块可能不足一个完整 tile）
                uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;  // S 维实际行数
                uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;  // N 维实际列数
                uint32_t rowNum = qSBlockSize * qNBlockSize;                 // 矩阵乘的 M 维（行数）
                uint32_t rowNumRound = RoundUp(rowNum, FaiKenel::BLOCK_SIZE); // 对齐到 16

                // 因果掩码优化：计算无需处理的 KV 起始位置，跳过全零区域
                uint32_t noSkipKvS = kvSeqlen;
                if (maskType != 0U) {
                    uint32_t diffS = kvSeqlen - qSeqlen;                     // KV 与 Q 的长度差
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;  // 当前 Q 分块对应的 KV 上界
                    noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, noSkipKvS);
                }
                uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);  // KV 外层循环次数
 	 
                uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;  // 每次迭代的分页块数
                uint32_t stackSeqTile = MAX_KV_STACK_LEN;   // 当前迭代的 KV 序列长度
                uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;  // 对齐后的长度
                uint32_t preKVNum = PRE_LAUNCH;              // 预取缓冲数（流水线深度，=2）
                int32_t stackSeqCount = 0;                   // KV 迭代计数器（用于预取/双缓冲管理）

#ifdef __DAV_C220_CUBE__
                // Cube 侧：定义布局临时变量并预加载 Q 到 L1
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, stackSeqTile);
                LayoutV layoutVTemp(stackSeqTile, strideV);
                blockMmadQK.resetBlockStart();
                blockMmadPV.resetBlockStart();
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);  // 预加载 Q
#endif
                // ====================================================================
                // KV 内层循环（软件流水线）
                // ----------------------------------------------------------------
                // 循环总次数 = kvSLoopNumTotal + preKVNum，其中：
                //   - 前 preKVNum 次：仅执行 QK+softmax（填充流水线）
                //   - 中间：同时执行 QK+softmax 与 PV+rescale（稳态流水）
                //   - 后 preKVNum 次：仅执行 PV+rescale（排空流水线）
                // 通过 PRE_LAUNCH(=2) 预取实现 Cube/Vector 双引擎重叠执行。
                // ====================================================================
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx ++) {
                    // ---- 第一阶段：Q×K^T 矩阵乘 + 在线 softmax（仅前 kvSLoopNumTotal 次）----
                    if (kvSIdx < kvSLoopNumTotal) {
                        // 计算当前迭代的 KV 序列长度（最后一次可能不足 MAX_KV_STACK_LEN）
                        if (kvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                        } else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        // 双缓冲槽位索引：在 PRE_LAUNCH+1 个缓冲区间轮转
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                        // S 的工作区偏移：每个核独占一段，按缓冲槽位偏移
                        uint64_t gmOffsetS =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};  // QK 矩阵乘的实际形状
                        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                        // Cube 侧：执行 Q×K^T 矩阵乘，结果写入 gS 工作区
                        if constexpr (PAGED_CACHE_FLAG) {
                            // 分页模式：传入 block_table 起始地址，K 通过块表间接寻址
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK],
                                gS[gmOffsetS],
                                gBlockTable[blockBOffset],
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        } else {
                            // 非分页模式：K 直接连续寻址
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK],
                                gS[gmOffsetS],
                                gBlockTable,
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        }
                        // 通知 Vector 引擎：QK^T 计算完成，S 数据已就绪
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                        // Vector 侧：对 S=QK^T 执行在线 softmax，生成 P=softmax(S)
                        LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                        LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                        uint64_t gmOffsetP = gmOffsetS;  // P 与 S 复用同一工作区偏移
                        // 因果掩码的上下界（用于判断当前分块是否需要应用三角掩码）
                        uint32_t triUp = noSkipKvS - qSBlockSize;
                        uint32_t triDown = noSkipKvS;
                        uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                        bool doTriUMask = triUp < kvSEndIdx - 1;  // 是否需要应用上三角掩码
                        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                            // 因果掩码分支
                            if (doTriUMask) {
                                // 当前分块跨越掩码边界，需要应用三角掩码
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gMask,
                                    layOutP,
                                    layOutS,
                                    layOutMask,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),   // 是否首次迭代（初始化 max/sum）
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    qkReady,                 // 等待 Cube 的 QK 完成信号
                                    triUp,
                                    triDown,
                                    kvSStartIdx,
                                    kvSEndIdx);
                            } else {
                                // 当前分块完全在掩码下方，无需掩码（可跳过 mask 读取）
                                uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                Arch::CrossCoreWaitFlag(qkReady);
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    (stackSeqCount == noMaskStackSeqNum - 1),  // 是否掩码区最后一次
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod);
                            }
                        } else {
                            // 无掩码或自定义掩码分支
                            Arch::CrossCoreWaitFlag(qkReady);
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                0,
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod);
                        }
                        // 通知 Cube 引擎：softmax 完成，P 数据已就绪
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                    }
                    // ---- 第二阶段：P×V 矩阵乘 + 输出 rescale（延迟 preKVNum 次后开始）----
                    if (kvSIdx >= preKVNum) {
                        // nowkvSIdx 是对应的 QK 迭代索引（因延迟了 preKVNum 步）
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;
                        if (nowkvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * MAX_KV_STACK_LEN;
                        } 
                        else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        // 缓冲槽位索引（与 QK 阶段对齐，延迟 PRE_LAUNCH 步）
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                        // OTmp 的工作区偏移
                        uint64_t gmOffsetOTmp =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};  // PV 矩阵乘的实际形状
                        LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                        // Cube 侧：执行 P×V 矩阵乘，结果写入 gOTmp 工作区
                        LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                        uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;;
                        if constexpr (PAGED_CACHE_FLAG) {
                            // 分页模式：V 通过 block_table 间接寻址
                            blockMmadPV(
                                gP[gmOffsetP],
                                gV[gmOffsetV],
                                gOTmp[gmOffsetOTmp],
                                gBlockTable[blockBOffset],
                                layoutPTemp,
                                layoutVTemp,
                                layoutOTmp,
                                actualBlockShapePV,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady);   // 等待 Vector 的 softmax 完成信号
                        } else {
                            // 非分页模式：V 直接连续寻址
                            blockMmadPV(
                                gP[gmOffsetP],
                                gV[gmOffsetV],
                                gOTmp[gmOffsetOTmp],
                                gBlockTable,
                                layoutPTemp,
                                layoutVTemp,
                                layoutOTmp,
                                actualBlockShapePV,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady);
                        }
                        // 通知 Vector 引擎：PV 计算完成，OTmp 数据已就绪
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                        // Vector 侧：对 OTmp 执行 rescale，累加到最终输出 O 并更新 lse
                        LayoutO layoutO(qSeqlen, embed * qHeads);
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                        Arch::CrossCoreWaitFlag(pvReady);  // 等待 Cube 的 PV 完成信号
                        epilogueRescaleO(
                            gO[gmOffsetO],           // 最终输出 O
                            gOTmp[gmOffsetOTmp],      // 本次 PV 中间结果
                            gOUpdate[gmOffsetUpdate], // rescale 更新缓冲区
                            gLse[gmOffsetLse],        // logsumexp 输出
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            qNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),    // 是否首次迭代（初始化 O）
                            nowkvSIdx + 1 >= kvSLoopNumTotal,     // 是否最后一次迭代（写回最终 O）
                            curStackTileMod);
#endif
                    }
                    stackSeqCount++;  // KV 迭代计数器递增
                }
            }
            // ---- 阶段7：等待所有 HardEvent 完成 + PipeBarrier 同步 ----
            // 确保核函数退出前所有异步流水线操作都已结束，避免资源泄漏或数据竞争
#ifdef __DAV_C220_CUBE__
            // Cube 侧：等待所有预置的 M_MTE1 / FIX_M / MTE1_MTE2 事件完成
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            // Vector 侧：等待所有预置的 MTE3_V / MTE3_MTE2 / V_MTE2 事件完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();  // 全流水线屏障：确保所有核同步到此
        }

    private:
        // ---- 私有成员：硬件资源与跨核同步信号量 ----
        Arch::Resource<ArchTag> resource;                          // 硬件资源管理（L1/L0/UB 缓冲区）
        Arch::CrossCoreFlag qkReady{QK_READY_ID};                  // QK^T 完成信号（Cube -> Vector）
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};        // softmax 完成信号（Vector -> Cube）
        Arch::CrossCoreFlag pvReady{PV_READY_ID};                  // PV 完成信号（Cube -> Vector）
    };
}

namespace SplitFuse {
    // ============================================================================
    // FAInfer —— FlashAttention v3 推理核函数入口（全局核函数）
    // ----------------------------------------------------------------------------
    // 这是 Host 侧通过 AscendCL LaunchKernel 调用的核函数入口。
    // 职责：根据模板参数组装 CATLASS 类型（元素类型、布局、分块形状、调度策略），
    //       构造 FAInferKernel 实例并调用其 operator() 执行真正的注意力计算。
    //
    // 模板参数说明（共 7 个，组合出 16 种特化 = 2(dtype) × 2(paged) × 2(causal) × 2(varlen)）：
    //   InputDtypeQ      —— Q 的输入数据类型（默认 half/bf16）
    //   InputDtypeKv      —— KV 的输入数据类型（默认 half/bf16）
    //   IntermCalcPrec   —— 中间计算精度（默认 float，用于 softmax/rescale 累加）
    //   PagedCacheFlag   —— 是否启用分页 KV Cache
    //   maskCategory     —— 掩码类型：NO_MASK / MASK_CAUSAL / MASK_SPEC
    //   inLayout         —— 输入布局：BSND(标准) / TND(变长)
    //   lseMode          —— LSE 模式：NONE(不输出) / OUTPUT(输出 logsumexp)
    // ============================================================================
    template <
        typename InputDtypeQ = half,
        typename InputDtypeKv = half,
        typename IntermCalcPrec = float,
        bool PagedCacheFlag = false,
        FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
        FaiKenel::inputLayout inLayout = FaiKenel::inputLayout::TND,
        Epilogue::LseModeT lseMode = Epilogue::LseModeT::NONE>
    __global__ __aicore__ void FAInfer(
        uint64_t fftsAddr,       // FFTS（快速任务调度）同步基地址，用于跨核信号量通信
        GM_ADDR q,               // Query 输入 GM 地址
        GM_ADDR k,               // Key 输入 GM 地址
        GM_ADDR v,               // Value 输入 GM 地址
        GM_ADDR mask,            // 注意力掩码 GM 地址
        GM_ADDR blockTables,     // 分页块表 GM 地址（block -> 物理页映射）
        GM_ADDR o,               // 输出 O GM 地址
        GM_ADDR lse,             // logsumexp 输出 GM 地址
        GM_ADDR actualQseqlen,   // 实际 Q 序列长度数组 GM 地址
        GM_ADDR actualKvseqlen,  // 实际 KV 序列长度数组 GM 地址
        GM_ADDR workspace,       // 工作区 GM 地址（存放 S/P/OTmp/OUpdate）
        GM_ADDR tiling)          // 分块参数 GM 地址（FAInferTilingData）
    {
        // 设置 FFTS 同步基地址，使跨核信号量（CrossCoreSetFlag/WaitFlag）可用
        AscendC::SetSyncBaseAddr(fftsAddr);

        // ---- 定义元素类型与内存布局 ----
        using ArchTag = Arch::AtlasA2;              // 目标架构：Atlas A2（Ascend 910B）
        using ElementQ = InputDtypeQ;               // Q 元素类型
        using LayoutQ = layout::RowMajor;            // Q 布局：行主序
        using ElementK = InputDtypeKv;              // K 元素类型
        using LayoutK = layout::ColumnMajor;         // K 布局：列主序（便于 Q×K^T）
        using ElementV = InputDtypeKv;               // V 元素类型
        using LayoutV = layout::RowMajor;            // V 布局：行主序
        using ElementS = IntermCalcPrec;            // S=QK^T 元素类型（高精度）
        using LayoutS = layout::RowMajor;            // S 布局：行主序
        using ElementP = InputDtypeQ;                // P=softmax(S) 元素类型（与 Q 同精度）
        using LayoutP = layout::RowMajor;            // P 布局：行主序
        using ElementO = InputDtypeQ;                // 输出 O 元素类型（与 Q 同精度）
        using LayoutO = layout::RowMajor;            // O 布局：行主序
        using ElementLse = float;                    // logsumexp 元素类型（始终 float）
        using LayoutLse = layout::RowMajor;          // lse 布局：行主序
        using ElementMask = int8_t;                  // 掩码元素类型（int8_t）
        using LayoutMask = layout::RowMajor;         // 掩码布局：行主序
        using ElementOTmp = IntermCalcPrec;          // PV 中间结果元素类型（高精度）
        using LayoutOTmp = layout::RowMajor;         // OTmp 布局：行主序
        using ElementUpdate = IntermCalcPrec;        // update 缓冲区元素类型（高精度）
        using LayoutUpdate = layout::RowMajor;       // update 布局：行主序

        // ---- 构造 Q×K^T 矩阵乘块类型（BlockMmadQK）----
        using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;   // L1 分块：M=128(Q_TILE_CEIL), N=128, K=128
        using L0TileShapeQK = GemmShape<128, 128, 128>;           // L0 分块：128×128×128
        using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<PagedCacheFlag, false>;  // QK 调度策略
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;          // Q 的类型+布局组合
        using KType = Gemm::GemmType<ElementK, LayoutK>;          // K 的类型+布局组合
        using SType = Gemm::GemmType<ElementS, LayoutS>;          // S 的类型+布局组合
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                                   QType, KType, SType>;  // QK 矩阵乘块

        // ---- 构造在线 softmax 后处理类型（EpilogueOnlineSoftmax）----
        using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<lseMode, IntermCalcPrec>;
        using PType = Gemm::GemmType<ElementP, LayoutP>;          // P 的类型+布局组合
        using maskType = Gemm::GemmType<ElementMask, LayoutMask>;  // mask 的类型+布局组合
        using EpilogueOnlineSoftmax =
            Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;  // softmax 块

        // ---- 构造 P×V 矩阵乘块类型（BlockMmadPV）----
        using L1TileShapePV = GemmShape<128, 128, 256>;           // L1 分块：M=128, N=128, K=256
        using L0TileShapePV = GemmShape<128, 128, 128>;           // L0 分块：128×128×128
        using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PagedCacheFlag, false>;  // PV 调度策略
        using VType = Gemm::GemmType<ElementV, LayoutV>;          // V 的类型+布局组合
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;  // OTmp 的类型+布局组合
        using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                                   PType, VType, OTmpType>;  // PV 矩阵乘块

        // ---- 构造输出 rescale 后处理类型（EpilogueRescaleO）----
        using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<lseMode, IntermCalcPrec>;
        using OType = Gemm::GemmType<ElementO, LayoutO>;            // O 的类型+布局组合
        using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;  // update 的类型+布局组合
        using LseType = Gemm::GemmType<ElementLse, LayoutLse>;      // lse 的类型+布局组合
        using EpilogueRescaleO =
            Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;  // rescale 块

        // ---- 实例化 FAInferKernel 并执行 ----
        // 将上述 4 个块类型 + 3 个编译期标志传入 FAInferKernel 模板
        using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                                            PagedCacheFlag, maskCategory, inLayout>;
        // 打包所有 GM 地址为 FAIKernelParams
        FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
        FAInferKernel flashAttnInfer;  // 构造核类实例
        flashAttnInfer(params);        // 调用 operator() 执行 FlashAttention 计算
    }
}