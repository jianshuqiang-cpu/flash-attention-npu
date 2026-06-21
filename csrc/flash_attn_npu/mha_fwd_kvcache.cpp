/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// mha_fwd_kvcache.cpp 是 FlashAttention NPU 前向推理 kernel 的核心实现文件。
//
// 本文件包含两个命名空间作用域内的 SplitFuse 组件：
//
//   1) class FAInferKernel<...>（L34-L579）:
//      双异构核（Cube + Vector）协作的 FlashAttention 前向推理 kernel 类。
//      在 Atlas A2 (C220) 架构上，每个 AI Core 包含：
//        - 1 个 Cube 计算单元（执行矩阵乘 MMAD）
//        - 2 个 Vector 子核（执行逐元素运算：softmax/mask/rescale 等）
//      编译器会对同一个 kernel 入口编译两次，通过 __DAV_C220_CUBE__ / __DAV_C220_VEC__
//      宏选择各自的代码路径，两侧通过 CrossCoreFlag 做生产者-消费者同步，
//      形成三段式深度流水：
//        Cube(Q*K^T) ──qkReady──> Vector(softmax+mask) ──softmaxReady──>
//        Cube(P*V)    ──pvReady──> Vector(O rescale/累加/归一化)
//
//   2) __global__ void FAInfer<...>(...)（L583-L666）:
//      全局 kernel 入口函数，负责：
//        - 设置 CrossCoreFlag 的同步基地址（SetSyncBaseAddr）
//        - 通过 using 声明组装所有模板类型（数据类型、布局、tile shape、dispatch policy）
//        - 构造 FAIKernelParams 打包 11 个 GM 地址
//        - 实例化 FAInferKernel 并调用 operator()
//
// 关键执行流程（operator() 内）：
//   ① 从 tiling 读取标量参数，绑定所有 GlobalTensor
//   ② 初始化核内事件 flag（灌泡，让第一轮循环不阻塞）
//   ③ Cube 侧：计算 L1 tile 大小、构造 BlockMmadQK/BlockMmadPV
//      Vector 侧：构造三个 Epilogue（OnlineSoftmax/RescaleO/InitOut）
//   ④ 计算 stride、embed 对齐值、GQA groupSize
//   ⑤ 若为 TND 布局，重新计算 totalTaskNum（变长前缀差分）
//   ⑥ 进入 task 循环（多核任务分配：taskIdx = coreIdx; taskIdx += coreNum）
//      - while 循环跨 batch 推进，累加 GM offset
//      - 解码 taskIdx → qSBlockIdx / qNBlockIdx → GM offset
//      - 计算 qSBlockSize/qNBlockSize（尾块处理）
//      - 因果 mask 下计算 noSkipKvS，得到 kvSLoopNumTotal
//      - 零 KV 边界：Vector 侧 InitOut 初始化 O=0/LSE=+inf
//      - Cube 侧 loadQGM：Q 只在 KV 循环外加载一次到 L1
//      - KV 外层循环（含 PRE_LAUNCH=2 预发射/排空）：
//          · QK matmul (Cube) → SetFlag(qkReady)
//          · Online softmax + causal mask (Vector) → SetFlag(softmaxReady)
//          · PV matmul (Cube, 落后 PRE_LAUNCH 轮) → SetFlag(pvReady)
//          · Rescale O / online softmax update (Vector, 落后 PRE_LAUNCH 轮)
//   ⑦ 等待所有 in-flight 操作完成（排空）
//   ⑧ PipeBarrier<PIPE_ALL>() 全局屏障后退出

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "online_softmax_low_prec.hpp"    // softmax/rescale 的 half 精度版本（当前未实例化，预留）
#include "online_softmax.hpp"            // softmax/rescale 的 float 精度版本（默认使用）
#include "rescale_o_low_prec.hpp"
#include "rescale_o.hpp"
#include "init_outputs.hpp"              // O/LSE 初始化 epilogue（零 KV 边界使用）
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "pv_matmul.hpp"                 // P*V 矩阵乘（Cube 侧）
#include "qk_matmul.hpp"                 // Q*K^T 矩阵乘（Cube 侧）
#include "catlass/gemm/dispatch_policy.hpp"
#include "fa_block.h"                    // 前向 epilogue/gemm 的 dispatch policy 定义
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_common.hpp"             // 公共常量、枚举、FAIKernelParams、tile 计算函数
#include "kernel_operator.h"
#include "tilingdata.h"                  // FAInferTilingData 宿主机 tiling 结构体
using namespace Catlass;
using namespace KernelCommon;

namespace SplitFuse {
    // FAInferKernel：FlashAttention 前向推理 kernel 主类。
    //
    // 模板参数：
    //   BlockMmadQK          - Q*K^T 矩阵乘类型（Cube 侧 BlockMmad 实例）
    //   BlockMmadPV          - P*V 矩阵乘类型（Cube 侧 BlockMmad 实例）
    //   EpilogueOnlineSoftmax - online softmax epilogue 类型（Vector 侧）
    //   EpilogueRescaleO     - O 重缩放/累加/归一化 epilogue 类型（Vector 侧）
    //   EpilogueInitOut      - O/LSE 初始化 epilogue 类型（Vector 侧，零 KV 边界）
    //   PAGED_CACHE_FLAG     - 是否启用 Paged KV cache（true 时通过 blockTable 做页表寻址）
    //   MASK_TYPE            - Mask 类型（NO_MASK / MASK_CAUSAL），编译期 if constexpr 裁剪
    //   INPUT_LAYOUT         - 输入布局（BSND / TND），编译期 if constexpr 裁剪
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        class EpilogueInitOut,
        bool PAGED_CACHE_FLAG,
        FaiKenel::MaskType MASK_TYPE = FaiKenel::MaskType::NO_MASK,
        FaiKenel::inputLayout INPUT_LAYOUT = FaiKenel::inputLayout::BSND>
    class FAInferKernel {
    public:
        // ===== 从模板参数提取类型别名，统一后续代码使用 =====
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        using ElementQ = typename BlockMmadQK::ElementA;     // Q 元素类型（half/bf16）
        using LayoutQ = typename BlockMmadQK::LayoutA;       // Q 布局（RowMajor）
        using ElementK = typename BlockMmadQK::ElementB;     // K 元素类型
        using LayoutK = typename BlockMmadQK::LayoutB;       // K 布局（ColumnMajor，即 K^T 视角）
        using ElementS = typename BlockMmadQK::ElementC;     // S=QK^T 输出类型（中间精度，通常 float）
        using LayoutS = typename BlockMmadQK::LayoutC;       // S 布局（RowMajor）

        using ElementP = typename BlockMmadPV::ElementA;     // P=softmax(S) 类型（half/bf16）
        using LayoutP = typename BlockMmadPV::LayoutA;       // P 布局（RowMajor）
        using ElementV = typename BlockMmadPV::ElementB;     // V 元素类型（与 K 相同）
        using LayoutV = typename BlockMmadPV::LayoutB;       // V 布局（RowMajor）

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;  // Mask 元素类型（int8）
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

        using ElementO = typename EpilogueRescaleO::ElementOutput;       // O 输出类型（half/bf16）
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;     // PV 输出 OTmp 类型（中间精度 float）
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;        // LSE 类型（float）
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;  // online softmax update 中间量（float）
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        static constexpr Epilogue::LseModeT LSE_MODE = EpilogueRescaleO::LSE_MODE;

        __aicore__ inline
        FAInferKernel() {}

        // kernel 主函数：在每个 AI Core 的 Cube 核和 Vector 子核上都被调用（双编译）。
        // 通过 __DAV_C220_CUBE__ / __DAV_C220_VEC__ 宏选择各自代码路径。
        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            // ===== ① 从 tiling 结构体读取标量参数 =====
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            uint64_t mm1OutSize = fATilingData->mm1OutSize;           // QK matmul workspace 大小（字节）
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize; // softmax 输出 workspace 大小
            uint64_t mm2OutSize = fATilingData->mm2OutSize;           // PV matmul workspace 大小
            uint32_t batch = fATilingData->batch;                     // batch size
            uint32_t qHeads = fATilingData->numHeads;                 // Q 头数
            uint32_t kvHeads = fATilingData->kvHeads;                 // KV 头数（MHA时=qHeads; GQA/MQA时<qHeads）
            uint32_t embed = fATilingData->embeddingSize;             // Q/K head_dim
            uint32_t embedV = fATilingData->embeddingSizeV;           // V head_dim（可与 Q/K 不同）
            uint32_t pagedBlockSize = fATilingData->blockSize;        // paged KV cache 每块 token 数
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch; // 每 batch 最大页数
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;       // 首批任务数
            uint32_t totalTaskNum = fATilingData->totalTaskNum;       // 总任务数
            uint32_t blockSize = fATilingData->blockSize;             // 同 pagedBlockSize
            uint32_t maskType = fATilingData->maskType;               // 0=无mask, 1=causal
            float scaleValue = fATilingData->scaleValue;              // softmax 缩放因子（1/sqrt(d)）
            float softcapValue = fATilingData->softcapValue;          // softcap 值（>0 启用）

            // ===== 绑定所有 GM 全局张量指针 =====
            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int32_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int32_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);

            // workspace 切分为四个区域（按 mm1/sm/mm2 顺序排列）：
            //   gS       = Q*K^T 输出 S（attention scores）
            //   gP       = softmax(S) 输出 P（attention probabilities）
            //   gOTmp    = P*V 临时输出 OTmp
            //   gOUpdate = online softmax 的 rowmax/rowsum/update 中间量
            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace));
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace +
                mm1OutSize + smOnlineOutSize + mm2OutSize));

            // ===== 多核任务分配索引 =====
            uint32_t coreIdx = AscendC::GetBlockIdx();     // 物理执行单元编号
            uint32_t coreNum = AscendC::GetBlockNum();     // 物理执行单元总数

            // ===== ② Cube 侧初始化：灌泡事件 flag + L1 tile 计算 + 构造 GEMM 对象 =====
#ifdef __DAV_C220_CUBE__
            // 灌泡（prefill）：在进入 KV 循环前，将所有双缓冲/多缓冲事件置为 signaled，
            // 模拟"上一轮已经完成、buffer 空闲可用"的状态，确保第一轮 WaitFlag 不阻塞。
            // 这些 SetFlag 与函数结尾的 WaitFlag 一一对应。
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

            // L1 tile 大小动态计算：
            //   1) kDynNum：embed 维度向上对齐到 128，最小 256
            //   2) 总 L1=512KB，先扣除 V 矩阵空间（双缓冲，MAX_KV_STACK_LEN=512 行）
            //   3) 剩余给 Q+K，扣除 Q 固定占用（Q_TILE_CEIL=128 行）
            //   4) 剩余空间除以 kDynNum、DOUBLE_BUFFER、对齐到 32 得到 nDynNum（KV 序列维 tile 大小）
            //   5) nDynNum 再受 L1_MAX_N_NUM=128 硬件上限约束
            uint32_t kDynNum = RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
            BlockMmadQK blockMmadQK(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            // PV 的 K 维度（KV序列方向）需要与 QK 的 N 维度匹配：
            //   kPVDynNum = nDynNum * kDynNum / M（M=128）
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            BlockMmadPV blockMmadPV(resource, nDynNum, kPVDynNum, MAX_KV_STACK_LEN, L1_QK_SIZE);
#endif
            // ===== ②' Vector 侧初始化：灌泡事件 flag + 构造 Epilogue 对象 =====
#ifdef __DAV_C220_VEC__
            // 同理，灌泡 Vector 侧的所有 event flag
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            // 构造三个 Vector epilogue：
            //   epilogueOnlineSoftmax：scale/mask/exp/rowmax/rowsum（传入 scale 和 softcap）
            //   epilogueRescaleO：O 重缩放 + 累加 + 归一化
            //   epilogueInitOut：零 KV 边界时初始化 O=0, LSE=+inf
            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue, softcapValue);
            EpilogueRescaleO epilogueRescaleO(resource);
            EpilogueInitOut epilogueInitOut(resource);
            // Vector 侧有 2 个 sub-core，需要将 sub-core 编号除以 GetSubBlockNum() 映射到 AI Core 逻辑编号，
            // 与 Cube 侧 coreIdx 对齐，保证同一核的 Cube/Vector 共享 workspace
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            // ===== ④ 计算 stride 和对齐值 =====
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);    // Q 张量：每个 token 跨 head 步长
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);   // O 张量步长
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);   // K 张量步长
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);  // V 张量步长
            uint32_t embedRound = RoundUp(embed, FaiKenel::BLOCK_SIZE);      // embed 对齐到 16（DMA 对齐要求）
            uint32_t embedRoundV = RoundUp(embedV, FaiKenel::BLOCK_SIZE);
            uint32_t groupSize = qHeads / kvHeads;                          // GQA group size：每个 KV head 对应多少 Q head

            // batch 维度 GM offset 累积变量
            uint64_t qBOffset = 0;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t oBOffset = 0;
            uint64_t lseBOffset = 0;
            uint64_t blockBOffset = 0;

            // ===== ⑤ 初始化当前 batch 状态 =====
            uint32_t preTotalTaskNum = 0;    // 之前所有 batch 的任务数累计
            uint32_t curBatch = 0;           // 当前处理的 batch 编号
            uint32_t totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));
            uint32_t qSeqlen = fATilingData->maxQSeqlen;
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));

            // TND（packed varlen）布局：每个 batch 序列长度不同，需要通过 cumulative seqlen
            // 数组做前缀差分重新计算 totalTaskNum 和每个 batch 的 qSeqlen/kvSeqlen
            if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                totalTaskNum = 0;
                firstBatchTaskNum = 0;
                for (int32_t curBatch = 0; curBatch < batch; curBatch++) {
                    // cumulative seqlen 差分：当前 batch 实际长度 = cum[curBatch+1] - cum[curBatch]
                    uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch + 1));
                    uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch + 1));
                    uint32_t prevQSeqlenSum = (curBatch == 0) ?
                        0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                    qSeqlen = qSeqlen - prevQSeqlenSum;
                    if constexpr (!PAGED_CACHE_FLAG) {
                        uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                            0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                        kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                    }
                    // 动态计算每个 batch 的 tile 大小和任务数
                    uint64_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
                    uint64_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                    uint64_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
                    uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
                    uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
                    if (curBatch == 0) {
                        firstBatchTaskNum = curTaskNum;
                    }
                    totalTaskNum += curTaskNum;
                }
                // 为第一个 task（curBatch=0）设置正确的 qSeqlen/kvSeqlen
                totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch));
                qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch + 1));
                kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch + 1));
                uint32_t prevQSeqlenSum = (curBatch == 0) ?
                    0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                        0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }

            // ===== 为当前 batch 计算 tile 参数（初始值，进入 task 循环后可能随 batch 切换更新）=====
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
            uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;   // head 维总块数
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile); // 序列维总块数
            uint32_t curTotalTaskNum = firstBatchTaskNum;

            // ===== ⑥ 任务循环：多核按 coreIdx 步长分配任务 =====
            // Go through each task.
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                // Get the offset of each core on the GM.
                // 跨 batch 推进：当 taskIdx 超过当前 batch 任务数上限时，跳到下一个 batch，
                // 累加 GM 偏移并重新计算 tile 参数
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;
                    preTotalTaskNum = curTotalTaskNum;
                    qBOffset += qSeqlen * strideQ;
                    if constexpr (!PAGED_CACHE_FLAG) {
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                    } else {
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);  // paged 下累积页表偏移
                    }
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);
                    lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);

                    qSeqlen = fATilingData->maxQSeqlen;
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    // TND 布局下需要前缀差分得到真实序列长度
                    if constexpr(INPUT_LAYOUT == FaiKenel::inputLayout::TND) {
                        qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch + 1));
                        kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch + 1));
                        uint32_t prevQSeqlenSum = (curBatch == 0) ?
                            0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));;
                        qSeqlen = qSeqlen - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                            kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                        }
                    }
                    // 更新当前 batch 的 tile 参数
                    curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                    curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                    curQSBlockTile = GetQSBlockTile(kvSeqlen);
                    curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                    curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                }
                // 将 taskIdx 解码为当前 batch 内的坐标
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;                // 序列维块索引
                uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;   // head 维块索引
                uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;        // group 内的 head 块索引

                // KV head 索引和 Q head 起始索引
                uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;                    // 当前处理的 KV head 索引
                uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;  // Q head 起始索引
                uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;       // LSE token 偏移

                // ===== 计算各张量在 GM 上的起始偏移 =====
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

                // ===== 计算当前 task 的实际块大小（尾块处理）=====
                uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;   // 序列维实际行数（尾块不足128）
                uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile; // head 维实际 head 数
                uint32_t rowNum = qSBlockSize * qNBlockSize;    // 本 task 总"行"数（序列行 × head 数）
                uint32_t rowNumRound = RoundUp(rowNum, FaiKenel::BLOCK_SIZE);  // 向上对齐到 16

                // ===== 因果 mask 下计算需要处理的 KV 长度（跳过全被 mask 的部分）=====
                int64_t noSkipKvS = static_cast<int64_t>(kvSeqlen);
                if (maskType != 0U) {
                    // diffS = max(kvSeqlen - qSeqlen, 0)：KV 前缀长度（decode/extend 场景）
                    int64_t diffS = kvSeqlen - qSeqlen;
                    diffS = (diffS < 0) ? 0 : diffS;
                    // 当前 Q block 最后一个 token 能看到的最大 KV 位置
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
                    noSkipKvS = AscendC::Std::min(static_cast<int64_t>(kvSeqlen), noSkipKvS);
                }
                uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);  // KV 外层循环次数

                // paged KV cache 参数
                uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;  // 每个 stack 包含的页数
                uint32_t stackSeqTile = MAX_KV_STACK_LEN;        // 当前 KV stack tile 大小（512，尾块时缩小）
                uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;     // KV stack tile 对齐大小
                uint32_t preKVNum = PRE_LAUNCH;                  // 预发射深度 = 2
                int32_t stackSeqCount = 0;                      // 已发射的 KV stack 计数器

                // ===== 零 KV 边界处理（kvSLoopNumTotal <= 0） =====
#ifdef __DAV_C220_VEC__
                if (kvSLoopNumTotal <= 0) {
                    LayoutO layoutO(qSeqlen, embed * qHeads);
                    LayoutLse layoutLse(totalQTokens, qHeads);
                    epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutO, layoutLse, qSBlockSize, qNBlockSize);
                }
#endif
                // ===== Cube 侧：KV 循环前加载 Q 到 L1（Q 只加载一次）=====
#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, stackSeqTile);
                LayoutV layoutVTemp(stackSeqTile, strideV);
                blockMmadQK.resetBlockStart();
                blockMmadPV.resetBlockStart();
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
#endif
                // ===== KV 外层循环（含 PRE_LAUNCH=2 预发射和排空）=====
                // 实际循环次数 = kvSLoopNumTotal + preKVNum = kvSLoopNumTotal + 2
                // 前 preKVNum 轮只做 QK+Softmax（发射阶段，填满流水线）
                // 中间轮同时做 QK+Softmax+PV+RescaleO（稳态流水）
                // 最后 preKVNum 轮只做 PV+RescaleO（排空阶段，流水线中剩余数据）
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx ++) {
                    // ---- 前 kvSLoopNumTotal 轮：发射 QK matmul + Softmax ----
                    if (kvSIdx < kvSLoopNumTotal) {
                        // 计算当前 KV stack tile 的实际大小（尾块处理）
                        if (kvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                        } else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        // 三缓冲槽位：stackSeqCount % 3
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                        // workspace 中 S/P 区域的 GM 偏移（每核独立、每槽独立）
                        uint64_t gmOffsetS =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};  // QK 实际计算形状
                        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);

                        // ===== Cube 侧：Q*K^T 矩阵乘 =====
#ifdef __DAV_C220_CUBE__
                        if constexpr (PAGED_CACHE_FLAG) {
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK],
                                gS[gmOffsetS],
                                gBlockTable[blockBOffset],   // paged 模式下传入页表
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,                      // 当前 KV stack 索引
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        } else {
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK],
                                gS[gmOffsetS],
                                gBlockTable,                 // 非 paged 传入占位
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);  // 通知 Vector 核：QK 完成
#endif
                        // ===== Vector 侧：online softmax + causal mask =====
#ifdef __DAV_C220_VEC__
                        LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                        LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                        uint64_t gmOffsetP = gmOffsetS;  // P 写回 S 同一位置（覆盖）
                        // causal mask 区域计算
                        uint32_t triUp = noSkipKvS - qSBlockSize;          // 三角线上界（无需 mask 的 KV 位置）
                        uint32_t triDown = noSkipKvS;                      // 三角线下界
                        uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;  // 当前 KV stack 起始位置
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;   // 当前 KV stack 结束位置
                        bool doTriUMask = triUp < kvSEndIdx - 1;           // 是否跨越因果对角线
                        if constexpr (MASK_TYPE == FaiKenel::MaskType::MASK_CAUSAL) {
                            if (doTriUMask) {
                                // 跨越对角线：需要传入 mask 张量和三角线参数
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gMask,                       // 预生成的 2048x2048 上三角 mask
                                    layOutP,
                                    layOutS,
                                    layOutMask,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),        // 是否首块（首块需要初始化 m/l）
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    qkReady,                     // 等待 QK 完成
                                    triUp, triDown,
                                    kvSStartIdx, kvSEndIdx);
                            } else {
                                // 完全在对角线下方：无需 mask，走快速路径
                                uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                Arch::CrossCoreWaitFlag(qkReady);  // 等待 QK 完成
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    (stackSeqCount == noMaskStackSeqNum - 1),  // 最后一块无 mask 块
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod);
                            }
                        } else {
                            // 无 mask：直接 softmax
                            Arch::CrossCoreWaitFlag(qkReady);
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                0,                          // 无 mask 特殊标记
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);  // 通知 Cube 核：softmax 完成
#endif
                    }
                    // ---- 落后 PRE_LAUNCH(=2) 轮后开始消费：PV matmul + RescaleO ----
                    if (kvSIdx >= preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;   // 当前消费的 KV stack 索引
                        // 计算消费轮的 stack tile 大小（尾块处理）
                        if (nowkvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * MAX_KV_STACK_LEN;
                        }
                        else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        // 消费轮使用落后 PRE_LAUNCH 的槽位：(stackSeqCount - PRE_LAUNCH) % 3
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetOTmp =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};  // PV 实际形状
                        LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);

                        // ===== Cube 侧：P*V 矩阵乘 =====
#ifdef __DAV_C220_CUBE__
                        LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                        uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;;
                        if constexpr (PAGED_CACHE_FLAG) {
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
                                softmaxReady);             // 等待 softmax 完成
                        } else {
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
                                softmaxReady);             // 等待 softmax 完成
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);  // 通知 Vector 核：PV 完成
#endif
                        // ===== Vector 侧：O 重缩放 + 累加 + 最终归一化 =====
#ifdef __DAV_C220_VEC__
                        LayoutO layoutO(qSeqlen, embed * qHeads);
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                        Arch::CrossCoreWaitFlag(pvReady);   // 等待 PV 完成
                        epilogueRescaleO(
                            gO[gmOffsetO],                  // 输出 O（GM 上累加）
                            gOTmp[gmOffsetOTmp],            // PV 临时输出
                            gOUpdate[gmOffsetUpdate],       // online softmax 中间量（m/l/dm）
                            gLse[gmOffsetLse],              // LSE 输出
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            qNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),       // 是否首块（首块直接赋值 O = OTmp）
                            nowkvSIdx + 1 >= kvSLoopNumTotal,        // 是否末块（末块做最终归一化 O = O / rowsum）
                            curStackTileMod);
#endif
                    }
                    stackSeqCount++;
                }
            }
            // ===== ⑦ 排空：等待所有 in-flight 操作完成 =====
#ifdef __DAV_C220_CUBE__
            // Cube 侧：等待所有 DMA 搬运和 Cube 计算完成，确保数据正确写回 GM
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
            // Vector 侧：等待所有 Vector 计算和 DMA 写回完成
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
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
            // ⑧ 全 pipe 屏障，确保所有核、所有 pipe 操作完成后 kernel 退出
            AscendC::PipeBarrier<PIPE_ALL>();
        }

    private:
        Arch::Resource<ArchTag> resource;                        // 核上硬件资源管理（TM/UB/L1/L0 等）
        Arch::CrossCoreFlag qkReady{QK_READY_ID};                // Cube→Vector：QK 完成信号
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};      // Vector→Cube：softmax 完成信号
        Arch::CrossCoreFlag pvReady{PV_READY_ID};                // Cube→Vector：PV 完成信号
    };
}

namespace SplitFuse {
    // FAInfer：全局 __global__ kernel 入口函数。
    //
    // 模板参数：
    //   InputDtypeQ     - Q 数据类型（half / bfloat16_t）
    //   InputDtypeKv    - K/V 数据类型（half / bfloat16_t）
    //   IntermCalcPrec  - 中间计算精度（float 高精度 / half 低精度预留）
    //   PagedCacheFlag  - 是否启用 Paged KV cache
    //   maskCategory    - Mask 类型
    //   inLayout        - 输入布局
    //   lseMode         - LSE 输出模式（NONE 不输出 / OUT_ONLY 输出 LSE）
    //
    // 函数参数（共 12 个 GM 地址）：
    //   fftsAddr       - FFTS C2C 同步基地址（CrossCoreFlag 共享内存）
    //   q,k,v          - Q/K/V 张量 GM 地址
    //   mask           - 预生成 causal mask GM 地址
    //   blockTables    - Paged KV cache 页表 GM 地址
    //   o              - 输出 O GM 地址
    //   lse            - LSE 输出 GM 地址
    //   actualQseqlen  - Q cumulative seqlen 数组 GM 地址
    //   actualKvseqlen - KV cumulative seqlen 数组 GM 地址
    //   workspace      - 临时 workspace GM 地址
    //   tiling         - FAInferTilingData 标量参数 GM 地址
    template <
        typename InputDtypeQ = half,
        typename InputDtypeKv = half,
        typename IntermCalcPrec = float,
        bool PagedCacheFlag = false,
        FaiKenel::MaskType maskCategory = FaiKenel::MaskType::NO_MASK,
        FaiKenel::inputLayout inLayout = FaiKenel::inputLayout::TND,
        Epilogue::LseModeT lseMode = Epilogue::LseModeT::NONE>
    __global__ __aicore__ void FAInfer(
        uint64_t fftsAddr,
        GM_ADDR q,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR mask,
        GM_ADDR blockTables,
        GM_ADDR o,
        GM_ADDR lse,
        GM_ADDR actualQseqlen,
        GM_ADDR actualKvseqlen,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        // 设置跨核同步基地址（FFTS C2C 控制地址），CrossCoreFlag 通过此地址做 Cube↔Vector 信号量
        AscendC::SetSyncBaseAddr(fftsAddr);

        // ===== 类型组装：为 FAInferKernel 指定所有模板实参 =====
        using ArchTag = Arch::AtlasA2;                    // 目标架构：Atlas A2 (C220)
        using ElementQ = InputDtypeQ;
        using LayoutQ = layout::RowMajor;
        using ElementK = InputDtypeKv;
        using LayoutK = layout::ColumnMajor;              // K 以列主序加载（即 K^T 视角）
        using ElementV = InputDtypeKv;
        using LayoutV = layout::RowMajor;
        using ElementS = IntermCalcPrec;                  // S=QK^T 用中间精度（float）
        using LayoutS = layout::RowMajor;
        using ElementP = InputDtypeQ;                     // P=softmax(S) 回到 half/bf16
        using LayoutP = layout::RowMajor;
        using ElementO = InputDtypeQ;
        using LayoutO = layout::RowMajor;
        using ElementLse = float;                         // LSE 始终用 float
        using LayoutLse = layout::RowMajor;
        using ElementMask = int8_t;                       // causal mask 为 int8 上三角矩阵
        using LayoutMask = layout::RowMajor;
        using ElementOTmp = IntermCalcPrec;               // PV 临时输出用中间精度
        using LayoutOTmp = layout::RowMajor;
        using ElementUpdate = IntermCalcPrec;             // online softmax update 中间量用中间精度
        using LayoutUpdate = layout::RowMajor;

        // ===== GEMM 类型组装（Cube 侧 QK 和 PV 矩阵乘）=====
        using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;   // QK L1 tile: M=128, N=128(dyn), K=128
        using L0TileShapeQK = GemmShape<128, 128, 128>;           // QK L0 tile（Cube 单元计算块）
        using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<PagedCacheFlag, false>;  // QK 专属 dispatch policy
        using QType = Gemm::GemmType<ElementQ, LayoutQ>;
        using KType = Gemm::GemmType<ElementK, LayoutK>;
        using SType = Gemm::GemmType<ElementS, LayoutS>;
        using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                                   QType, KType, SType>;

        // ===== Epilogue 类型组装（Vector 侧 online softmax）=====
        using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<lseMode, IntermCalcPrec>;
        using PType = Gemm::GemmType<ElementP, LayoutP>;
        using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
        using EpilogueOnlineSoftmax =
            Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

        // ===== PV 矩阵乘类型组装 =====
        using L1TileShapePV = GemmShape<128, 128, 256>;           // PV L1 tile: M=128, N=128, K=256
        using L0TileShapePV = GemmShape<128, 128, 128>;
        using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PagedCacheFlag, false>;  // PV 专属 dispatch policy
        using VType = Gemm::GemmType<ElementV, LayoutV>;
        using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
        using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                                   PType, VType, OTmpType>;

        // ===== RescaleO 和 InitOut epilogue 类型组装 =====
        using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT<lseMode, IntermCalcPrec>;
        using DispatchPolicyInitOutWhenZero = Epilogue::EpilogueAtlasA2InitOutWhenZero<lseMode>;
        using OType = Gemm::GemmType<ElementO, LayoutO>;
        using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
        using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
        using EpilogueInitOut = Epilogue::Block::BlockEpilogue<DispatchPolicyInitOutWhenZero, OType, LseType>;
        using EpilogueRescaleO =
            Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

        // ===== 实例化 kernel 类并执行 =====
        using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, EpilogueInitOut,
                                            PagedCacheFlag, maskCategory, inLayout>;
        FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
        FAInferKernel flashAttnInfer;
        flashAttnInfer(params);
    }
}
