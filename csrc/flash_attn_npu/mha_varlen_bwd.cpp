/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

// mha_varlen_bwd.cpp 是 FlashAttention NPU 反向传播（backward）kernel 的核心实现文件，
// 对应 Python 接口中的 varlen_bwd（支持变长/varlen/packed 输入）。
//
// 本文件包含两个命名空间作用域内的 FAG (FlashAttention Gradient) 组件：
//
//   1) class FAGKernel<...>（L46-L338）:
//      双异构核（Cube + Vector）协作的 FlashAttention 反向 kernel 类。
//      反向需要计算三个梯度：
//        dQ = dS * K
//        dK = dS^T * Q
//        dV = P^T * dOut
//      其中 dS（softmax 输入的梯度）通过 softmax gradient 公式计算：
//        dS = P * (dP - D)，其中 dP = dOut * V^T，D = rowsum(dP * P)
//      Cube 核负责 3 个矩阵乘（Cube1/Cube2/Cube3），Vector 核负责 4 个 epilogue：
//        - Pre：清零 dQ/dK/dV workspace
//        - Sfmg (SoftmaxGrad pre-reduction)：预计算 D = rowsum(dOut * P)
//        - Op：重算 P=Softmax(scale*S+mask)，计算 dS = P*(dP - D)
//        - Post：scale + fp32->half cast + 写回 dQ/dK/dV
//      Cube 与 Vector 通过 CUBE2VEC(ID=7)、VEC2CUBE(ID=8)、CUBE2POST(ID=9) 跨核信号流水。
//
//   2) __global__ void FAG<...>(...)（L340-L448）:
//      全局 kernel 入口，设置 FFTS 同步基地址，声明 kernel 类型为 MIX_AIC_1_2
//      （1个Cube核 : 2个Vector子核），组装所有模板类型并调用 FAGKernel。
//
// 数学公式：
//   S = scale * Q * K^T         (Cube1, 第一次调用)
//   P = softmax(mask(S))        (VEC_Op 重算)
//   dP = dOut * V^T             (Cube1, 第二次调用)
//   D = rowsum(dP * P)          (VEC_Sfmg 预归约 + VEC_Op 完成)
//   dS = P * (dP - D)           (VEC_Op)
//   dQ = dS * K                 (Cube2)
//   dV = P^T * dOut             (Cube3, 第一次)
//   dK = dS^T * Q               (Cube3, 第二次)
//
// 流水线结构（while taskId 循环）：
//   - Cube1（S/dP 生产者）处理 taskId 轮，写入 slot[taskId%2]
//   - Vector Op 消费 slot[taskId%2]，计算 P/dS
//   - Cube2/Cube3（dQ/dK/dV 消费者）处理 taskId-1 轮，读 slot[(taskId-1)%2]
//   - 双缓冲 (taskId%2) 让生产和消费并行，形成 Cube1→Vec→Cube2/3 的深度流水
//
// 注意：反向 kernel 不保存前向的 S 和 P，而是在反向中重新计算 S（重算forward），
// 这是 Flash Attention 反向的标准"重算 forward"策略，节省大量显存。

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "fag_epilogue_pre.hpp"     // Pre：dQ/dK/dV fp32 workspace 清零
#include "fag_epilogue_sfmg.hpp"    // Sfmg：softmax gradient 逐行归约辅助项 D = rowsum(dP*P)
#include "fag_epilogue_op.hpp"      // Op：重算 P=Softmax(S)，计算 dS=P*(dP-D)
#include "fag_epilogue_post.hpp"    // Post：scale、cast、写回 dQ/dK/dV
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "fag_mmad_cube1.hpp"       // Cube1: Q*K^T=S 和 dOut*V^T=dP（A=RowMajor, B=ColumnMajor）
#include "fag_mmad_cube2.hpp"       // Cube2: dQ=dS*K（A=RowMajor, B=RowMajor）
#include "fag_mmad_cube3.hpp"       // Cube3: dV=P^T*dOut 和 dK=dS^T*Q（A=ColumnMajor, B=RowMajor）
#include "catlass/gemm/dispatch_policy.hpp"
#include "fag_block.h"              // FAG dispatch policy 定义
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_operator.h"
#include "fag_common/common_header.h"  // FAG 公共常量/枚举/AddrInfo 结构体/tiling 索引
#include "fag_common/cube_addr.h"      // Cube 侧地址生成器
#include "fag_common/vector_addr.h"    // Vector 侧地址生成器
using namespace Catlass;

namespace FAG {

    // FAGKernel：FlashAttention 反向 kernel 主类。
    //
    // 模板参数：
    //   BlockMmadFAGCube1_ - Cube1 矩阵乘类型（Q*K^T 和 dOut*V^T，A不转置 B转置）
    //   BlockMmadFAGCube2_ - Cube2 矩阵乘类型（dQ = dS*K，A不转置 B不转置）
    //   BlockMmadFAGCube3_ - Cube3 矩阵乘类型（dV=P^T*dOut, dK=dS^T*Q，A转置 B不转置）
    //   EpilogueFAGPre_    - Workspace 清零 epilogue
    //   EpilogueFAGSfmg_   - SoftmaxGrad 归约 epilogue
    //   EpilogueFAGOp_     - 重算P + 计算dS epilogue
    //   EpilogueFAGPost_   - scale+cast+写回 epilogue
    //   maskType           - Mask 类型（NO_MASK / MASK_CAUSAL），编译期裁剪
    //   inputLayout        - 输入布局（TND / BSND），编译期裁剪
    template <
        class BlockMmadFAGCube1_,
        class BlockMmadFAGCube2_,
        class BlockMmadFAGCube3_,
        class EpilogueFAGPre_,
        class EpilogueFAGSfmg_,
        class EpilogueFAGOp_,
        class EpilogueFAGPost_,
        MaskType maskType = MaskType::NO_MASK,
        InputLayout inputLayout = InputLayout::TND
    >
    class FAGKernel {
    public:
        // ===== 类型别名：从三个 Cube GEMM 模板参数提取元素/布局类型 =====
        using BlockMmadFAGCube1 = BlockMmadFAGCube1_;
        using ArchTag = typename BlockMmadFAGCube1::ArchTag;
        using L1TileShape = typename BlockMmadFAGCube1::L1TileShape;
        using ElementA1 = typename BlockMmadFAGCube1::ElementA;    // Q/dOut 元素类型（half/bf16）
        using LayoutA1 = typename BlockMmadFAGCube1::LayoutA;      // RowMajor
        using ElementB1 = typename BlockMmadFAGCube1::ElementB;    // K/V 元素类型
        using LayoutB1 = typename BlockMmadFAGCube1::LayoutB;      // ColumnMajor
        using ElementC1 = typename BlockMmadFAGCube1::ElementC;    // S/dP 输出（float）
        using LayoutC1 = typename BlockMmadFAGCube1::LayoutC;      // RowMajor

        using BlockMmadFAGCube2 = BlockMmadFAGCube2_;
        using ElementA2 = typename BlockMmadFAGCube2::ElementA;    // dS（half/bf16）
        using LayoutA2 = typename BlockMmadFAGCube2::LayoutA;      // RowMajor
        using ElementB2 = typename BlockMmadFAGCube2::ElementB;    // K
        using LayoutB2 = typename BlockMmadFAGCube2::LayoutB;      // RowMajor
        using ElementC2 = typename BlockMmadFAGCube2::ElementC;    // dQ（float）
        using LayoutC2 = typename BlockMmadFAGCube2::LayoutC;      // RowMajor

        using BlockMmadFAGCube3 = BlockMmadFAGCube3_;
        using ElementA3 = typename BlockMmadFAGCube3::ElementA;    // P/dS（half/bf16）
        using LayoutA3 = typename BlockMmadFAGCube3::LayoutA;      // ColumnMajor（转置）
        using ElementB3 = typename BlockMmadFAGCube3::ElementB;    // dOut/Q
        using LayoutB3 = typename BlockMmadFAGCube3::LayoutB;      // RowMajor
        using ElementC3 = typename BlockMmadFAGCube3::ElementC;    // dV/dK（float）
        using LayoutC3 = typename BlockMmadFAGCube3::LayoutC;      // RowMajor

        using EpilogueFAGPre = EpilogueFAGPre_;
        using EpilogueFAGSfmg = EpilogueFAGSfmg_;
        using EpilogueFAGOp = EpilogueFAGOp_;
        using EpilogueFAGPost = EpilogueFAGPost_;



        /// Params：Device 侧 kernel 参数打包结构体（类似前向的 FAIKernelParams）
        struct Params {
            GM_ADDR q;             // Q 矩阵
            GM_ADDR k;             // K 矩阵
            GM_ADDR v;             // V 矩阵
            GM_ADDR dout;          // 输出梯度 dOut
            GM_ADDR q_right;       // （预留）Q 右矩阵
            GM_ADDR k_right;       // （预留）K 右矩阵
            GM_ADDR pse_shift;     // （预留）位置编码偏移
            GM_ADDR drop_mask;     // （预留）dropout mask
            GM_ADDR padding_mask;  // padding mask
            GM_ADDR atten_mask;    // attention mask（causal/dense）
            GM_ADDR row_lse;       // 前向 LSE（当前反向重算forward，未使用）
            GM_ADDR row_in;        // （预留）
            GM_ADDR out;           // 前向输出 O = P*V（Sfmg 用它近似计算 D=rowsum(dout*O)）
            GM_ADDR prefix;        // prefix 参数
            GM_ADDR cu_seq_qlen;   // Q cumulative seqlen 数组（TND 布局用）
            GM_ADDR cu_seq_kvlen;  // KV cumulative seqlen 数组
            GM_ADDR q_start_idx;   // Q 起始索引
            GM_ADDR kv_start_idx;  // KV 起始索引
            GM_ADDR dq;            // 输出：dQ
            GM_ADDR dk;            // 输出：dK
            GM_ADDR dv;            // 输出：dV
            GM_ADDR workspace;     // 临时 workspace（含 dq/dk/dv fp32 累加区、S/dP/P/dS/mm 中间结果）
            GM_ADDR tiling_data;   // 宿主机打包的 tiling 标量参数
            GM_ADDR ptrDump;       // dump 调试指针

            CATLASS_DEVICE
            Params() {}

            CATLASS_DEVICE
            Params(
                GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR dout_,
                GM_ADDR q_right_, GM_ADDR k_right_, GM_ADDR pse_shift_,
                GM_ADDR drop_mask_, GM_ADDR padding_mask_, GM_ADDR atten_mask_,
                GM_ADDR row_lse_, GM_ADDR row_in_,
                GM_ADDR out_, GM_ADDR prefix_, GM_ADDR cu_seq_qlen_,
                GM_ADDR cu_seq_kvlen_, GM_ADDR q_start_idx_, GM_ADDR kv_start_idx_,
                GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR workspace_, GM_ADDR tiling_data_, GM_ADDR ptrDump_
            ) : q(q_), k(k_), v(v_), dout(dout_),
                q_right(q_right_), k_right(k_right_), pse_shift(pse_shift_),
                drop_mask(drop_mask_), padding_mask(padding_mask_), atten_mask(atten_mask_),
                row_lse(row_lse_), row_in(row_in_),
                out(out_), prefix(prefix_), cu_seq_qlen(cu_seq_qlen_),
                cu_seq_kvlen(cu_seq_kvlen_), q_start_idx(q_start_idx_), kv_start_idx(kv_start_idx_),
                dq(dq_), dk(dk_), dv(dv_), workspace(workspace_), tiling_data(tiling_data_), ptrDump(ptrDump_)
            {
            }
        };

        CATLASS_DEVICE
        FAGKernel() {}

        // kernel 主函数：分别在 Cube 核和 Vector 子核上通过 #ifdef 双编译。
        CATLASS_DEVICE
        void operator()(Params const &params)
        {
// ============================================================
// Cube 核代码路径：执行 3 个矩阵乘（Cube1/Cube2/Cube3）
// ============================================================
#ifdef __DAV_C220_CUBE__
            // ===== 从 tiling 读取标量参数 =====
            AscendC::GlobalTensor<uint64_t> tilingData;
            tilingData.SetGlobalBuffer((__gm__ uint64_t *)params.tiling_data);
            AscendC::GlobalTensor<uint32_t> tilingDataU32;
            tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)params.tiling_data);

            int64_t batch = tilingData.GetValue(TILING_B);              // batch size
            int64_t g = tilingData.GetValue(TILING_G);                  // GQA 分组数
            int64_t nheads_k = tilingData.GetValue(TILING_N2);          // KV 头数
            int64_t nheads = nheads_k * g;                              // Q 头数
            int64_t headdim = tilingData.GetValue(TILING_D);            // head_dim
            // workspace 各区域字节偏移
            int64_t dqWorkSpaceOffset = tilingData.GetValue(TILING_DQ_WORKSPACE_OFFSET);
            int64_t dkWorkSpaceOffset = tilingData.GetValue(TILING_DK_WORKSPACE_OFFSET);
            int64_t dvWorkSpaceOffset = tilingData.GetValue(TILING_DV_WORKSPACE_OFFSET);
            int64_t mm1WorkspaceOffset = tilingData.GetValue(TILING_MM1_WORKSPACE_OFFSET);  // dP = dOut*V^T
            int64_t mm2WorkspaceOffset = tilingData.GetValue(TILING_MM2_WORKSPACE_OFFSET);  // S = Q*K^T
            int64_t pWorkSpaceOffset = tilingData.GetValue(TILING_P_WORKSPACE_OFFSET);      // P (softmax 概率)
            int64_t dsWorkSpaceOffset = tilingData.GetValue(TILING_DS_WORKSPACE_OFFSET);    // dS

            uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);   // 总物理核数
            int64_t mixCoreNum = (coreNum + 1) / 2;                                // 逻辑 Cube 核数（1:2 比例）
            uint32_t seq_q_len = 0;
            uint32_t seq_k_len = 0;
            struct CubeAddrInfo cubeAddrInfo[2];    // 双缓冲地址表（slot0/slot1）
            int32_t taskId = 0;                     // 当前任务轮次
            bool running = true;
            __gm__ uint8_t * actucal_seq_q_addr = params.cu_seq_qlen;
            __gm__ uint8_t * actucal_seq_k_addr = params.cu_seq_kvlen;

            // TND 布局：cu_seq_qlen/cu_seq_kvlen 第一个元素是0，跳过；BSND 布局从tiling读固定长度
            if constexpr(inputLayout == InputLayout::TND) {
                actucal_seq_q_addr = (__gm__ uint8_t *)((__gm__ int32_t *)params.cu_seq_qlen + 1);
                actucal_seq_k_addr = (__gm__ uint8_t *)((__gm__ int32_t *)params.cu_seq_kvlen + 1);
            } else {
                seq_q_len = tilingData.GetValue(TILING_T1) / batch;
                seq_k_len = tilingData.GetValue(TILING_T2) / batch;
            }

            // 初始化 Cube 侧地址生成器
            CubeAddr<maskType, inputLayout> cubeAddr;
            cubeAddr.init(batch, nheads, g, headdim, GetBlockIdx(), seq_q_len, seq_k_len, actucal_seq_q_addr, actucal_seq_k_addr, mixCoreNum);

            // ping-pong 计数器（传给 GEMM 做 L1/L0 双缓冲轮转）
            uint32_t pingpongFlagL1A = 0;
            uint32_t pingpongFlagL1B = 0;
            uint32_t pingpongFlagL0A = 0;
            uint32_t pingpongFlagL0B = 0;
            uint32_t pingpongFlagC = 0;

            // 构造三个 Cube 矩阵乘对象
            BlockMmadFAGCube1 blockMmadFAGCube1(resource, nheads, nheads_k, headdim);
            BlockMmadFAGCube2 blockMmadFAGCube2(resource, nheads, nheads_k, headdim);
            BlockMmadFAGCube3 blockMmadFAGCube3(resource, nheads, nheads_k, headdim);

            // ===== Cube 核主循环：按 taskId 逐轮推进 =====
            // 每轮：
            //   1. Cube1 计算 S=Q*K^T 和 dP=dOut*V^T（生产，写 slot[taskId%2]）
            //   2. 跨核通知 Vector CUBE2VEC
            //   3. 若 taskId>0，等待 Vector 完成上一轮 VEC2CUBE 后，
            //      Cube2 计算 dQ=dS*K；Cube3 计算 dV=P^T*dOut 和 dK=dS^T*Q（消费，读 slot[(taskId-1)%2]）
            while (running) {
                // 生成当前轮 Cube 地址表（最多16个128x128子块）
                cubeAddrInfo[taskId % 2].taskId = taskId;
                cubeAddr.addr_mapping(&cubeAddrInfo[taskId % 2]);
                if (cubeAddrInfo[taskId % 2].blockLength > 0) {
                    SetFlag();    // 灌泡内部 MTE1/MTE2/M 事件
                    // Cube1 第一次调用：S = Q * K^T → workspace[mm2WorkspaceOffset]
                    CubeAddrInfo addrs = cubeAddrInfo[taskId % 2];
                    blockMmadFAGCube1(cubeAddrInfo[taskId % 2], (__gm__ ElementA1*)(params.q), (__gm__ ElementB1 *)(params.k), (__gm__ float*)(params.workspace + mm2WorkspaceOffset),
                        pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                    // Cube1 第二次调用：dP = dOut * V^T → workspace[mm1WorkspaceOffset]
                    blockMmadFAGCube1(cubeAddrInfo[taskId % 2], (__gm__ ElementA1*)(params.dout), (__gm__ ElementB1*)(params.v), (__gm__ float*)(params.workspace + mm1WorkspaceOffset),
                        pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                    WaitFlag();   // 等待本轮 S/dP 全部写回 workspace
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);  // 通知 Vector：本轮 S/dP 就绪
                }
                // 消费上一轮（taskId-1）：等待 Vector 完成 P/dS 计算后做 dQ/dK/dV
                if (taskId > 0 && cubeAddrInfo[(taskId - 1) % 2].blockLength > 0) {
                    AscendC::WaitEvent(VEC2CUBE);   // 等待 Vector 完成上一轮 Op
                    SetFlag();
                    // Cube2：dQ = dS * K → dq workspace（fp32 atomic add）
                    blockMmadFAGCube2(cubeAddrInfo[(taskId - 1) % 2], (__gm__ ElementA2*)(params.workspace + dsWorkSpaceOffset), (__gm__ ElementB2*)(params.k), (__gm__ float*)(params.workspace + dqWorkSpaceOffset),
                        pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B);
                    WaitFlag();
                    SetFlag();
                    // Cube3 第一次：dV = P^T * dOut → dv workspace（fp32 atomic add）
                    blockMmadFAGCube3(cubeAddrInfo[(taskId - 1) % 2], (__gm__ ElementA3*)(params.workspace + pWorkSpaceOffset), (__gm__ ElementB3*)(params.dout), (__gm__ float*)(params.workspace + dvWorkSpaceOffset),
                        pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                    WaitFlag();
                    SetFlag();
                    // Cube3 第二次：dK = dS^T * Q → dk workspace（fp32 atomic add）
                    blockMmadFAGCube3(cubeAddrInfo[(taskId - 1) % 2], (__gm__ ElementA3*)(params.workspace + dsWorkSpaceOffset), (__gm__ ElementB3*)(params.q), (__gm__ float*)(params.workspace + dkWorkSpaceOffset),
                        pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B, pingpongFlagC);
                    WaitFlag();
                }
                // blockLength==0 表示全局遍历结束（addr_mapping 不再产生有效子块）
                if (cubeAddrInfo[taskId % 2].blockLength == 0) {
                    running = false;
                }
                taskId++;
            }
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);  // 通知 Vector Post：所有 Cube 任务完成
#endif

// ============================================================
// Vector 子核代码路径：执行 4 个 epilogue（Pre / Sfmg / Op / Post）
// ============================================================
#ifdef __DAV_C220_VEC__

            AscendC::GlobalTensor<uint64_t> tilingData;
            tilingData.SetGlobalBuffer((__gm__ uint64_t *)params.tiling_data);
            AscendC::GlobalTensor<uint32_t> tilingDataU32;
            tilingDataU32.SetGlobalBuffer((__gm__ uint32_t *)params.tiling_data);

            int64_t batch = tilingData.GetValue(TILING_B);
            int64_t g = tilingData.GetValue(TILING_G);
            int64_t nheads_k = tilingData.GetValue(TILING_N2);
            int64_t nheads = nheads_k * g;
            int64_t headdim = tilingData.GetValue(TILING_D);
            uint32_t coreNum = tilingDataU32.GetValue(TILING_CORE_NUM * CONST_2);
            int64_t mixCoreNum = (coreNum + 1) / 2;

            uint32_t seq_q_len = 0;
            uint32_t seq_k_len = 0;
            __gm__ uint8_t * actucal_seq_q_addr = params.cu_seq_qlen;
            __gm__ uint8_t * actucal_seq_k_addr = params.cu_seq_kvlen;

            if constexpr(inputLayout == InputLayout::TND) {
                actucal_seq_q_addr = (__gm__ uint8_t *)((__gm__ int32_t *)params.cu_seq_qlen + 1);
                actucal_seq_k_addr = (__gm__ uint8_t *)((__gm__ int32_t *)params.cu_seq_kvlen + 1);
            } else {
                seq_q_len = tilingData.GetValue(TILING_T1) / batch;
                seq_k_len = tilingData.GetValue(TILING_T2) / batch;
            }

            struct VecAddrInfo vecAddrInfo;

            // ===== Vector 阶段 1/4：Pre（在独立 TPipe 上执行，清零 dQ/dK/dV 的 fp32 workspace）=====
            AscendC::TPipe pipePre;
            EpilogueFAGPre epilogueFagPre(resource, &pipePre, params.dq, params.dk, params.dv, params.workspace, params.tiling_data);
            epilogueFagPre();
            pipePre.Destroy();

            // ===== Vector 阶段 2/4：Sfmg（SoftmaxGrad pre-reduction）=====
            // 逐行计算 D = rowsum(dOut * O)，作为 softmax gradient 的辅助项
            AscendC::TPipe pipeSoftmaxGrad;
            EpilogueFAGSfmg epilogueFagSfmg(resource, &pipeSoftmaxGrad, params.dout, params.out, actucal_seq_q_addr, params.workspace, batch, params.tiling_data);
            epilogueFagSfmg();
            pipeSoftmaxGrad.Destroy();

            AscendC::SyncAll();   // 所有 Vector 核完成 Pre+Sfmg 后再进入 Op 循环

            // ===== Vector 阶段 3/4：Op（核心循环，与 Cube 流水）=====
            // 每轮等待 CUBE2VEC，读取 Cube1 的 S 和 dP 结果：
            //   1) SubGrapA：S*scale + mask → P = Softmax(S)（重算forward）
            //   2) SubGrapB：dS = P * (dP - D)
            // 然后发 VEC2CUBE 通知 Cube2/Cube3 可以消费
            AscendC::TPipe pipeVec;
            EpilogueFAGOp epilogueFagOp(resource, &pipeVec, params.row_lse,
                params.atten_mask, actucal_seq_q_addr, actucal_seq_k_addr, params.workspace, batch, params.tiling_data);

            VectorAddr<maskType, inputLayout> vector_addr;
            vector_addr.init(batch, nheads, g, headdim, GetBlockIdx() / 2, seq_q_len, seq_k_len, actucal_seq_q_addr, actucal_seq_k_addr, mixCoreNum);
            int32_t taskId = 0;
            bool running = true;
            while (running) {
                vector_addr.addr_mapping(&vecAddrInfo);
                if (vecAddrInfo.blockLength > 0) {
                    AscendC::WaitEvent(CUBE2VEC);    // 等待 Cube 完成本轮 S/dP
                    vecAddrInfo.taskId = taskId;
                    epilogueFagOp(vecAddrInfo);      // 重算 P、计算 dS
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);  // 通知 Cube：本轮 P/dS 就绪
                }
                if (vecAddrInfo.blockLength == 0) {
                    running = false;
                }
                taskId++;
            }
            pipeVec.Destroy();
            AscendC::WaitEvent(CUBE2POST);  // 等待所有 Cube 任务（dQ/dK/dV）完成
            AscendC::SyncAll();             // 所有 Vector 核同步

            // ===== Vector 阶段 4/4：Post（scale + fp32→half cast + 写回 dq/dk/dv）=====
            AscendC::TPipe pipePost;
            EpilogueFAGPost epilogueFagPost(resource, &pipePost, params.dq, params.dk, params.dv, params.workspace, params.tiling_data);
            epilogueFagPost();
            pipePost.Destroy();
#endif
        }

        // SetFlag：Cube 核内部灌泡/双缓冲同步。
        // 在每次 Cube matmul 调用前置位 MTE1→MTE2 和 M→MTE1 方向上的事件，
        // 模拟"buffer 初始空闲"状态，避免第一轮 WaitFlag 阻塞。
        CATLASS_DEVICE
        void SetFlag()
        {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        }

        // WaitFlag：Cube 核内部排空，等待所有 MTE1→MTE2 数据搬运和 MMAD 计算完成，
        // 确保结果正确写回 workspace 后再发跨核信号。
        CATLASS_DEVICE
        void WaitFlag()
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        }

private:
    Arch::Resource<ArchTag> resource;    // 核上硬件资源管理
};

// FAG：全局 __global__ kernel 入口函数。
//
// 模板参数：
//   InputDtype - 数据类型（half / bfloat16_t）
//   maskType   - Mask 类型（NO_MASK / MASK_CAUSAL）
//   inputLayout- 输入布局（TND / BSND）
//
// 注意：反向中间计算精度固定为 float（不提供 low_prec 版本）。
template <
    typename InputDtype = half,
    MaskType maskType = MaskType::NO_MASK,
    InputLayout inputLayout = InputLayout::TND>
__global__ __aicore__
void FAG(uint64_t fftsAddr,
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dout,
        GM_ADDR q_right, GM_ADDR k_right,
        GM_ADDR pse_shift, GM_ADDR drop_mask, GM_ADDR padding_mask,
        GM_ADDR atten_mask, GM_ADDR row_lse, GM_ADDR row_in,
        GM_ADDR out, GM_ADDR prefix, GM_ADDR cu_seq_qlen,
        GM_ADDR cu_seq_kvlen, GM_ADDR q_start_idx, GM_ADDR kv_start_idx,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
        GM_ADDR workspace, GM_ADDR tiling_data, GM_ADDR ptrDump)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // 声明混合 AIC 类型：1 Cube : 2 Vector

#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif

    using ArchTag = Arch::AtlasA2;

    // ===== Cube1 类型组装：Q*K^T=S, dOut*V^T=dP（A不转置 B转置）=====
    using ElementA1 = InputDtype;
    using LayoutA1 = layout::RowMajor;
    using ElementB1 = InputDtype;
    using LayoutB1 = layout::ColumnMajor;
    using ElementC1 = float;
    using LayoutC1 = layout::RowMajor;
    using A1Type = Catlass::Gemm::GemmType<ElementA1, LayoutA1>;
    using B1Type = Catlass::Gemm::GemmType<ElementB1, LayoutB1>;
    using C1Type = Catlass::Gemm::GemmType<ElementC1, LayoutC1>;
    using DispatchPolicyCube1 = Catlass::Gemm::MmadAtlasA2FAGCube1;
    using L1TileShapeCube1 = GemmShape<256, 128, 256>;   // M=256, N=128, K=256
    using L0TileShapeCube1 = L1TileShapeCube1;
    using BlockMmadFAGCube1 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube1, L1TileShapeCube1, L0TileShapeCube1, A1Type, B1Type, C1Type>;

    // ===== Cube2 类型组装：dQ = dS * K（A不转置 B不转置）=====
    using ElementA2 = InputDtype;
    using LayoutA2 = layout::RowMajor;
    using ElementB2 = InputDtype;
    using LayoutB2 = layout::RowMajor;
    using ElementC2 = float;
    using LayoutC2 = layout::RowMajor;

    using A2Type = Catlass::Gemm::GemmType<ElementA2, LayoutA2>;
    using B2Type = Catlass::Gemm::GemmType<ElementB2, LayoutB2>;
    using C2Type = Catlass::Gemm::GemmType<ElementC2, LayoutC2>;
    using DispatchPolicyCube2 = Catlass::Gemm::MmadAtlasA2FAGCube2;
    using L1TileShapeCube2 = GemmShape<128, 128, 128>;   // M=128, N=128, K=128
    using L0TileShapeCube2 = L1TileShapeCube2;

    using BlockMmadFAGCube2 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube2, L1TileShapeCube2, L0TileShapeCube2, A2Type, B2Type, C2Type>;

    // ===== Cube3 类型组装：dV=P^T*dOut, dK=dS^T*Q（A转置 B不转置）=====
    using ElementA3 = InputDtype;
    using LayoutA3 = layout::ColumnMajor;
    using ElementB3 = InputDtype;
    using LayoutB3 = layout::RowMajor;
    using ElementC3 = float;
    using LayoutC3 = layout::RowMajor;

    using A3Type = Catlass::Gemm::GemmType<ElementA3, LayoutA3>;
    using B3Type = Catlass::Gemm::GemmType<ElementB3, LayoutB3>;
    using C3Type = Catlass::Gemm::GemmType<ElementC3, LayoutC3>;
    using DispatchPolicyCube3 = Catlass::Gemm::MmadAtlasA2FAGCube3;
    using L1TileShapeCube3 = GemmShape<256, 128, 256>;   // M=256, N=128, K=256
    using L0TileShapeCube3 = L1TileShapeCube3;

    using BlockMmadFAGCube3 = Catlass::Gemm::Block::BlockMmad<DispatchPolicyCube3, L1TileShapeCube3, L0TileShapeCube3, A3Type, B3Type, C3Type>;

    // ===== Epilogue 类型组装 =====
    using ElementVecDtype = InputDtype;

    using EpilogueAtlasA2FAGPre = Catlass::Epilogue::EpilogueAtlasA2FAGPre;
    using EpilogueFAGPre = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPre, ElementVecDtype>;

    using EpilogueAtlasA2FAGSfmg = Catlass::Epilogue::EpilogueAtlasA2FAGSfmg;
    using EpilogueFAGSfmg = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGSfmg, ElementVecDtype, std::integral_constant<InputLayout, inputLayout>>;

    using EpilogueAtlasA2FAGOp = Catlass::Epilogue::EpilogueAtlasA2FAGOp;
    using EpilogueFAGOp = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGOp, ElementVecDtype, std::integral_constant<InputLayout, inputLayout>>;

    using EpilogueAtlasA2FAGPost = Catlass::Epilogue::EpilogueAtlasA2FAGPost;
    using EpilogueFAGPost = Catlass::Epilogue::Block::BlockEpilogue<EpilogueAtlasA2FAGPost, ElementVecDtype>;


    // ===== 实例化 kernel 并执行 =====
    using FAGKernel = FAGKernel<BlockMmadFAGCube1, BlockMmadFAGCube2, BlockMmadFAGCube3, EpilogueFAGPre, EpilogueFAGSfmg, EpilogueFAGOp, EpilogueFAGPost, maskType, inputLayout>;
    typename FAGKernel::Params params{
        q, k, v, dout,
        q_right, k_right, pse_shift,
        drop_mask, padding_mask, atten_mask,
        row_lse, row_in,
        out, prefix, cu_seq_qlen,
        cu_seq_kvlen, q_start_idx, kv_start_idx,
        dq, dk, dv, workspace, tiling_data, ptrDump};

    FAGKernel fag;
    fag(params);
}
}
