/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE3_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE3_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "fag_block.h"
#include "fag_common/common_header.h"

////////////////////////////////////////////////////////////////////
namespace Catlass::Gemm::Block
{
    ////////////////////////////////////////////////////////////////////

    // FAG 反向中的 Cube3 MMAD 实现。
    // 该模板处理“左矩阵转置、右矩阵不转置”的矩阵乘，核心用于两类梯度：
    // 1. dV = P^T * dOut，其中 left 是 P workspace，right 是 dOut，out 是 fp32 dv workspace。
    // 2. dK = dS^T * Q，其中 left 是 dS workspace，right 是 Q，out 是 fp32 dk workspace。
    template <
        class L1TileShape_,
        class L0TileShape_,
        class AType_,
        class BType_,
        class CType_,
        class BiasType_,
        class TileCopy_,
        class TileMmad_>
    struct BlockMmad<
        MmadAtlasA2FAGCube3,
        L1TileShape_,
        L0TileShape_,
        AType_,
        BType_,
        CType_,
        BiasType_,
        TileCopy_,
        TileMmad_>
    {
    public:
        // Type Aliases
        using DispatchPolicy = MmadAtlasA2FAGCube3;
        using ArchTag = typename DispatchPolicy::ArchTag;
        using L1TileShape = L1TileShape_;
        using L0TileShape = L0TileShape_;
        using ElementA = typename AType_::Element;
        using LayoutA = typename AType_::Layout;
        using ElementB = typename BType_::Element;
        using LayoutB = typename BType_::Layout;
        using ElementC = typename CType_::Element;
        using LayoutC = typename CType_::Layout;
        using TileMmad = TileMmad_;
        using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
        using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
        using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
        using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
        using ElementAccumulator =
            typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
        using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
        using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
        using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
        using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
        using LayoutCInL0 = layout::zN;

        using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
        using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

        static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
        static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
        static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
        static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
        static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
        static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
        static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
        static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
        static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

        static const uint32_t C0_SIZE = 16;
        static const uint32_t SIZE_16 = 16;
        static const uint32_t SIZE_32 = 32;
        static const uint32_t SIZE_64 = 64;
        static const uint32_t SIZE_128 = 128;
        static const uint32_t SIZE_256 = 256;
        static const uint32_t SIZE_LONG_BLOCK = 16384;
        static const uint32_t SIZE_384 = 384;
        static const uint32_t SIZE_ONE_K = 1024;
        // 与 Cube1/Vector 约定一致：每个 AIC 有两个 task slot，每个 slot 最多保存 16 个 128x128 中间块。
        static const uint32_t BLOCK_WORKSPACE = 16 * 128 * 128;

        /// Construct
        CATLASS_DEVICE
        BlockMmad(Arch::Resource<ArchTag> &resource, uint64_t nheadsIn, uint64_t nheadsKIn, uint64_t headDimIn)
        {
            nheads = nheadsIn;
            nheads_k = nheadsKIn;
            headdim = headDimIn;
            globalBlockOffset = GetBlockIdx() * 16 * 128 * 128;

            // padding 值为 0，保证尾块 C0 对齐时无效元素不会贡献到 dK/dV。
            AscendC::SetLoadDataPaddingValue<uint64_t>(0);
            uint64_t config = 0x1;
            AscendC::SetNdParaImpl(config);

            // L1 预切 A/B ping-pong 缓冲：A 放 P/dS workspace 子块，B 放 Q/dOut 子块。
            l1_a_ping_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1_a_pong_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_128 * SIZE_ONE_K);
            l1_b_ping_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_256 * SIZE_ONE_K);
            l1_b_pong_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_384 * SIZE_ONE_K);

            // L0A/L0B/L0C 用于承接 L1 子块、执行 MMAD 并暂存 fp32 累加结果。
            l0_a_ping_tensor = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0_a_pong_tensor = resource.l0ABuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);
            l0_b_ping_tensor = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0_b_pong_tensor = resource.l0BBuf.template GetBufferByByte<ElementB>(SIZE_32 * SIZE_ONE_K);

            l0_c_ping_tensor = resource.l0CBuf.template GetBufferByByte<float>(0);
            l0_c_pong_tensor = resource.l0CBuf.template GetBufferByByte<float>(SIZE_64 * SIZE_ONE_K);
        }

        /// Destructor
        CATLASS_DEVICE
        ~BlockMmad()
        {
        }

        CATLASS_DEVICE
        void operator()(const CubeAddrInfo &addrs, __gm__ ElementA *left, __gm__ ElementB *right, __gm__ float *out,
                        uint32_t &pingpongFlagL1A, uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
                        uint32_t &pingpongFlagL0B, uint32_t &pingpongFlagC)
        {
            // 读取 Vector 已完成的上一拍 P/dS，因此使用 addrs.taskId % 2 对应的 workspace slot。
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                pingpongFlagL0B = 0;
                int32_t ping_pong_flag_l0_b_last = 0;
                // P/dS workspace 会压缩跳过 causal 无效块，skip_num 用来修正连续存储下的读取偏移。
                int32_t skip_num = 0;
                auto &shapeInfo = addrs.addrInfo[i];

                // left 的逻辑形状是 (kx, ky)，但 LayoutA 为 ColumnMajor，相当于参与 P^T/dS^T。
                // right 的逻辑形状是 (ky, headdim)，输出是 (kx, headdim)。
                // 因此 Cube3 的归约维是 Q 方向 ky，输出行方向是 K 方向 kx。
                uint32_t kn = shapeInfo.kx;
                uint32_t km = shapeInfo.ky;
                uint32_t lineStride = shapeInfo.lineStride;

                uint32_t l1_m_size = km;
                uint32_t l1_n_size = kn;
                uint32_t l1_k_size = headdim;

                uint32_t l1_m_size_align = RoundUp<C0_SIZE>(l1_m_size);
                uint32_t l1_n_size_align = RoundUp<C0_SIZE>(l1_n_size);
                uint32_t l1_m_block_size_tail = (l1_m_size % 128) == 0 ? 128 : (l1_m_size % 128);
                uint32_t l1_n_block_size_tail = (l1_n_size % 128) == 0 ? 128 : (l1_n_size % 128);
                uint32_t l1_m_block_size_align_tail = (l1_m_size_align % 128) == 0 ? 128 : (l1_m_size_align % 128);
                uint32_t l1_n_block_size_align_tail = (l1_n_size_align % 128) == 0 ? 128 : (l1_n_size_align % 128);

                uint32_t m_loop = CeilDiv<SIZE_128>(km);
                uint32_t n_loop = CeilDiv<SIZE_128>(kn);

                // A 从 P/dS workspace 读取，shapeInfo.out 是当前 attention block 的 workspace 基址。
                // B 从原始 dOut/Q 读取，C 写到 dv/dk workspace 的 KV 位置。
                __gm__ ElementA* gm_a = left + (shapeInfo.out + globalBlockOffset);
                __gm__ ElementB* gm_b = right + shapeInfo.left;
                __gm__ float* gm_out = out + shapeInfo.right;

                AscendC::GlobalTensor<ElementA> gLeft;
                gLeft.SetGlobalBuffer((__gm__ ElementA *)gm_a);

                AscendC::GlobalTensor<ElementB> gRight;
                gRight.SetGlobalBuffer((__gm__ ElementB *)gm_b);

                AscendC::GlobalTensor<ElementC> gOut;
                gOut.SetGlobalBuffer((__gm__ ElementC *)gm_out);

                // CubeAddr 中的标志表示该块是否处于边界；这里取反后表示“该方向存在可跳过的 causal 无效区域”。
                bool lowerLeft = !shapeInfo.lowerLeft;
                bool upperRight = !shapeInfo.upperRight;

                LocalTensor<ElementB> *l1_b_buf_tensor = pingpongFlagL1B ? &l1_b_pong_tensor : &l1_b_ping_tensor;
                
                LayoutA layoutA(km, kn, 128);
                // B 是 Q/dOut，逻辑形状为 (ky, headdim)，stride 使用 nheads * headdim，因为 Q/dOut 按 query head 排布。
                LayoutB layoutB(km, headdim, nheads * headdim);
                // 右矩阵 Q/dOut 在整个 Cube3 block 内复用，先一次性搬到 L1B。
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);

                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(l1_m_size, l1_k_size));
                LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(l1_m_size, l1_k_size);
                copyGmToL1B(*l1_b_buf_tensor, gRight, layoutBInL1, layoutTileB);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                
                for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
                    if (n_loop_index == 0) {
                        // B=Q/dOut 只在 n_loop_index == 0 时从 L1B 搬到 L0B。
                        // 后续 n_loop_index 复用同一批 L0B 数据，沿 K token 输出方向循环计算。
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_last + 2 + FLAG_SHIFT);

                        for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                            int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                            int32_t l1_b_buf_offset = (m_loop_index == 0) ? 0 : 128 * 16;
                            int32_t l0_b_buf_offset = (m_loop_index == 0) ? 0 : 128 * 128;

                            LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(m_remain, l1_k_size);
                            copyL1ToL0B(l0_b_ping_tensor[l0_b_buf_offset], (*l1_b_buf_tensor)[l1_b_buf_offset], layoutBInL0, layoutBInL1);

                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);
                            pingpongFlagL0B = 1 - pingpongFlagL0B;
                        }
                    }

                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        LocalTensor<ElementA>* l1_a_buf_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;
                        LocalTensor<ElementA>* l0_a_buf_tensor = pingpongFlagL0A ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                        LocalTensor<ElementB>* l0_b_buf_tensor = m_loop_index ? &l0_b_pong_tensor : &l0_b_ping_tensor;
                        LocalTensor<float>* l0_c_buf_tensor = pingpongFlagC ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                        uint32_t real_m = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
                        uint32_t real_n = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : 128;
                        uint32_t real_m_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                        uint32_t real_n_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : 128;

                        // 对固定 n_loop_index，沿 m_loop_index 累加归约维 ky。
                        // 第一个 M 子块初始化 L0C，最后一个 M 子块输出 L0C。
                        bool init_c = (m_loop_index == 0);
                        bool out_c = (m_loop_index == (m_loop - 1));
                        // 若最后一个 K 方向块的右上角无效且 m=0 被跳过，则 m=1 需要重新初始化 L0C。
                        if (m_loop_index != 0 && n_loop_index == n_loop - 1 && upperRight) {
                            init_c = true;
                        }

                        bool is_skip = false;
                        // causal 右上角无效：最后一个 n 块、首个 m 块不参与 P^T/dS^T 计算。
                        if (n_loop_index == (n_loop - 1) && m_loop_index == 0 && upperRight)
                        {
                            is_skip = true;
                        }

                        // causal 左下角无效：首个 n 块、最后一个 m 块不参与 P^T/dS^T 计算。
                        if (n_loop_index == 0 && m_loop_index == (m_loop - 1) && lowerLeft)
                        {
                            is_skip = true;
                        }

                        if (is_skip) {
                            skip_num++;
                        }

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);
                        if (!is_skip) {
                            // A=P/dS workspace 子块从 GM 搬到 L1A。
                            // workspace 中只连续保存有效块，因此物理偏移需要减去已经跳过的 skip_num。
                            uint32_t dstNzC0Stride = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(real_n, real_m));
                            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(real_n, real_m);
                            copyGmToL1A(*l1_a_buf_tensor, gLeft[(n_loop_index * m_loop + m_loop_index - skip_num) * SIZE_128 * SIZE_128], layoutAInL1, layoutTileA);
                        }

                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);
                        if (!is_skip) {
                            // A=P^T/dS^T 的一个子块从 L1A 搬到 L0A。
                            uint32_t m_c0_loop = (real_m + 15) / 16;
                            uint32_t n_c0_loop = (real_n + 15) / 16;
                            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(real_m, real_n);
                            LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(real_m, real_n);
                            copyL1ToL0A((*l0_a_buf_tensor), (*l1_a_buf_tensor), layoutAInL0, layoutAInL1);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);
                        
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);

                        if (!is_skip) {
                            // MMAD: P^T/dS^T(real_n, real_m) x dOut/Q(real_m, headdim) -> dV/dK(real_n, headdim)。
                            // unit flag 为 2 表示继续累加，为 3 表示当前 n 子块的 M 维归约完成并可输出 L0C。
                            int unit_flag = 0b10;
                            if (out_c) {
                                unit_flag = 0b11;
                            }
                            tileMmad(*l0_c_buf_tensor, *l0_a_buf_tensor, *l0_b_buf_tensor, real_n, l1_k_size, real_m, init_c, unit_flag);
                        }

                        if (out_c && n_loop_index == n_loop - 1) {
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ping_pong_flag_l0_b_last + 2 + FLAG_SHIFT);
                            ping_pong_flag_l0_b_last = 1 - ping_pong_flag_l0_b_last;
                        }
                        
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);

                        // 当前 n 子块完成所有 M 子块归约后，才把 L0C 写回 dk/dv workspace。
                        if (!is_skip && out_c) {
                            // 不同 Q block 可能累加到同一 dK/dV 行，因此写回使用 fp32 atomic add。
                            AscendC::SetAtomicType<float>();

                            auto blockShape = MakeCoord(real_n, l1_k_size);
                            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
                            LayoutC layoutC(real_n, l1_k_size, nheads_k * headdim);
                            copyL0CToGm(gOut[n_loop_index * 128 * nheads_k * headdim], *l0_c_buf_tensor, layoutC, layoutInL0C, 3);
                            AscendC::SetAtomicNone();
                        }
                        pingpongFlagL1A = 1 - pingpongFlagL1A;
                        pingpongFlagL0A = 1 - pingpongFlagL0A;
                    }
                    pingpongFlagC = 1 - pingpongFlagC;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);
                pingpongFlagL1B = 1 - pingpongFlagL1B;
            }
        }

    protected:
        /// Data members
        // TileCopy/TileMmad 封装 GM/L1/L0 数据搬运和 Cube MMAD 指令。
        TileMmad tileMmad;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;

        LocalTensor<ElementA> l1_a_ping_tensor;
        LocalTensor<ElementA> l1_a_pong_tensor;
        LocalTensor<ElementB> l1_b_ping_tensor;
        LocalTensor<ElementB> l1_b_pong_tensor;

        // L0A L0B
        LocalTensor<ElementA> l0_a_ping_tensor;
        LocalTensor<ElementA> l0_a_pong_tensor;
        LocalTensor<ElementB> l0_b_ping_tensor;
        LocalTensor<ElementB> l0_b_pong_tensor;

        // L0C
        LocalTensor<float> l0_c_ping_tensor;
        LocalTensor<float> l0_c_pong_tensor;

        // FLAG_SHIFT 把同类事件拆成不同编号，避免 L0A/L0B/L1A 的事件槽互相覆盖。
        uint32_t FLAG_SHIFT = 3;

        uint64_t nheads;
        uint64_t nheads_k;
        uint64_t headdim;

        // pingPongIdx 选择 P/dS workspace 的 task slot，globalBlockOffset 定位当前 AIC 的私有 workspace 区域。
        uint32_t pingPongIdx = 0;
        uint64_t globalBlockOffset = 0;
    };
} // namespace Catlass::Gemm::Block

#endif  // ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE3_HPP
