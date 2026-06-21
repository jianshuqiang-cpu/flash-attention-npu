/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE1_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE1_HPP

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

    // FAG 反向中的 Cube1 MMAD 实现。
    // 该模板专门处理“左矩阵不转置、右矩阵转置”的 128x128 分块矩阵乘，
    // 在 mha_varlen_bwd.cpp 中会被复用为 Q * K^T 和 dOut * V^T 两类计算。
    // CubeAddrInfo 负责告诉本模块每个 attention block 的 GM 输入/输出偏移、真实块大小以及 causal 边界信息。
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
        MmadAtlasA2FAGCube1,
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
        using DispatchPolicy = MmadAtlasA2FAGCube1;
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

        static const uint32_t C0_SIZE = 16;
        static const uint32_t SIZE_16 = 16;
        static const uint32_t SIZE_32 = 32;
        static const uint32_t SIZE_64 = 64;
        static const uint32_t SIZE_128 = 128;
        static const uint32_t SIZE_256 = 256;
        static const uint32_t SIZE_LONG_BLOCK = 16384;
        static const uint32_t SIZE_384 = 384;
        static const uint32_t SIZE_ONE_K = 1024;
        // 每个 Cube task 最多携带 16 个 128x128 输出块；同一个 AIC 再按 taskId 奇偶使用两份 workspace 做乒乓。
        static const uint32_t BLOCK_WORKSPACE = 16 * 128 * 128;

        /// Construct
        CATLASS_DEVICE
        BlockMmad(Arch::Resource<ArchTag> &resource, uint64_t nheadsIn, uint64_t nheadsKIn, uint64_t headDimIn)
        {
            cube1Cnt = 0;
            nheads = nheadsIn;
            nheads_k = nheadsKIn;
            headdim = headDimIn;
            globalBlockOffset = GetBlockIdx() * 16 * 128 * 128;

            // padding 值设为 0，尾块不足 C0 对齐时不会把无效元素带入 MMAD。
            AscendC::SetLoadDataPaddingValue<uint64_t>(0);
            uint64_t config = 0x1;
            AscendC::SetNdParaImpl(config);

            // L1 中预切出 A/B 的 ping-pong 缓冲，尽量让 GM->L1 与 L1->L0/MMAD 流水重叠。
            l1_a_ping_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1_a_pong_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_128 * SIZE_ONE_K);
            l1_b_ping_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_256 * SIZE_ONE_K);
            l1_b_pong_tensor = resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_384 * SIZE_ONE_K);

            // L0A/L0B/L0C 同样预切 ping-pong 缓冲，L0C 保存 fp32 累加结果后再写回 GM workspace。
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
        void cube1_base_matmul(
            LocalTensor<ElementA> *l1_a_tensor, LocalTensor<ElementB> *l1_b_tensor,
            GlobalTensor<ElementC> *gOut, uint32_t &pingpongFlagL1A,
            uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
            uint32_t &pingpongFlagL0B, uint32_t &pingpongFlagC,
            int32_t l1_m_size, int32_t l1_n_size, bool upper_right_flag)
        {
            // 输入已经位于 L1：A 形状约为 (M, D)，B 形状约为 (D, N)。
            // 本函数继续切成 128x128x128 的 L0 计算块，并把 C(M,N) 的每个有效 128x128 子块写到 workspace。
            int32_t l1_m_size_ = l1_m_size;
            int32_t l1_n_size_ = l1_n_size;

            uint32_t l1_m_size_align_ = RoundUp<C0_SIZE>(l1_m_size_);
            uint32_t l1_n_size_align_ = RoundUp<C0_SIZE>(l1_n_size_);

            uint32_t m0_ = 128;
            uint32_t n0_ = 128;
            // Cube1 的 K 维是 headdim，当前 tiling 约束下按 128 进入 L0/MMAD。
            uint32_t k0_ = 128;

            uint32_t m_mad_ = 128;
            uint32_t n_mad_ = 128;
            uint32_t k_mad_ = 128;

            int32_t dst_n_size_ = 128;

            for (int n_offset = 0; n_offset < l1_n_size_; n_offset += 128) {
                // 外层遍历 N 维，即 attention block 的 K/token 方向；尾块按真实长度参与计算。
                n_mad_ = Min((l1_n_size_ - n_offset), 128);
                n0_ = RoundUp<C0_SIZE>(n_mad_);

                LocalTensor<ElementB>* l0_b_tensor = pingpongFlagL0B ? &l0_b_pong_tensor : &l0_b_ping_tensor;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + pingpongFlagL0B + 2);
                LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(k0_, n_mad_);
                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(k0_, n_mad_);
                // B 在 L1 中按 Cube 友好的排列存放；每次搬一个 N 子块到 L0B。
                copyL1ToL0B(*l0_b_tensor, (*l1_b_tensor)[n_offset * SIZE_16], layoutBInL0, layoutBInL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);

                for (int m_offset = 0; m_offset < l1_m_size_; m_offset += SIZE_128) {
                    // 内层遍历 M 维，即 attention block 的 Q/token 方向。
                    m_mad_ = Min((l1_m_size_ - m_offset), 128);
                    m0_ = RoundUp<C0_SIZE>(m_mad_);

                    // causal 下三角中，右上角块不需要计算；这里跳过该 n 子块中的第一个 m 子块。
                    bool l0_skip_flag = (upper_right_flag && m_offset == 0);
                    LocalTensor<ElementA>* l0_a_tensor = pingpongFlagL0A ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                    LocalTensor<float>* l0_c_tensor = pingpongFlagC ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + pingpongFlagL0A);
                    if (!l0_skip_flag) {
                        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1_m_size_align_, k0_);
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(m_mad_, k0_);
                        // A 从 L1 搬到 L0A，尾块按 m_mad_ 描述真实有效行数。
                        copyL1ToL0A(*l0_a_tensor, (*l1_a_tensor)[m_offset * SIZE_16], layoutAInL0, layoutAInL1);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                    if (!l0_skip_flag) {
                        AscendC::MmadParams commonMadParams {
                            BASE_BLOCK_LENGTH,
                            BASE_BLOCK_LENGTH,
                            BASE_BLOCK_LENGTH,
                            3,
                            false,
                            true
                        };

                        // MMAD 不接受 m=1 的特殊形状，这里把 1 行提升为 2 行，额外行来自 padding，不影响有效输出。
                        uint16_t m_modify = (m_mad_ == 1) ? 2 : m_mad_;
                        tileMmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, m_modify, n_mad_, k_mad_, true, 3);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3 + pingpongFlagL0A);
                    // cube1Cnt 是当前 CubeAddrInfo 条目内的有效输出块序号；跳过 causal 无效块时不递增。
                    auto out_offset = cube1Cnt * SIZE_LONG_BLOCK;

                    if (!l0_skip_flag) {
                        auto blockShape = MakeCoord(m_mad_, n_mad_);
                        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
                        LayoutC layoutC(m_mad_, n_mad_, 128);
                        // L0C 中的 fp32 C 块写入 GM workspace，后续 Vector epilogue 按相同块序读取。
                        copyL0CToGm((*gOut)[out_offset], *l0_c_tensor, layoutC, layoutInL0C, 3);
                        cube1Cnt++;
                    }

                    pingpongFlagC = 1 - pingpongFlagC;
                    pingpongFlagL0A = 1 - pingpongFlagL0A;
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3 +  pingpongFlagL0B + 2);
                pingpongFlagL0B = 1 - pingpongFlagL0B;
            }
        }

        CATLASS_DEVICE
        void operator()(const CubeAddrInfo &addrs, __gm__ ElementA *left, __gm__ ElementB *right, __gm__ float *out,
                        uint32_t &pingpongFlagL1A, uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
                        uint32_t &pingpongFlagL0B, uint32_t &pingpongFlagC)
        {
            // Cube 和 Vector 交替消费两个 task slot；同一 AIC 的 workspace 也按 taskId 奇偶切分，避免读写冲突。
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;
            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                // 每个 AddrInfo 独立从第 0 个有效 128x128 输出块开始写。
                cube1Cnt = 0;
                auto &shapeInfo = addrs.addrInfo[i];
                // ky/kx 分别表示当前 attention 子矩阵的 Q 方向长度和 K 方向长度，尾块可能小于 128。
                uint32_t km = shapeInfo.ky;
                uint32_t kn = shapeInfo.kx;
                int32_t lineStride = shapeInfo.lineStride;

                // Cube1 的归约维是 headDim；A=(km, headdim)，B^T=(headdim, kn)，C=(km, kn)。
                uint32_t src_k_size_ = headdim;

                uint32_t l1_m_size_ = km;
                uint32_t l1_n_size_ = kn;
                uint32_t l1_k_size_ = src_k_size_;
                uint32_t l1_m_size_align_ = RoundUp<C0_SIZE>(l1_m_size_);
                uint32_t l1_n_size_align_ = RoundUp<C0_SIZE>(l1_n_size_);

                uint32_t m_loop = CeilDiv<SIZE_256>(km);
                uint32_t n_loop = CeilDiv<SIZE_128>(kn);

                // CubeAddr 已经把 batch/head/layout/GQA 等因素折算成线性 GM 偏移，这里只做指针绑定。
                auto gm_a = left + shapeInfo.left;
                auto gm_b = right + shapeInfo.right;
                auto gm_c = out + shapeInfo.out + globalBlockOffset;

                AscendC::GlobalTensor<ElementA> gLeft;
                gLeft.SetGlobalBuffer((__gm__ ElementA *)gm_a);

                AscendC::GlobalTensor<ElementB> gRight;
                gRight.SetGlobalBuffer((__gm__ ElementB *)gm_b);

                AscendC::GlobalTensor<ElementC> gOut;
                gOut.SetGlobalBuffer((__gm__ ElementC *)gm_c);

                // shapeInfo 中的标志来自地址生成器；这里取反后作为“是否需要处理 causal 边界跳过”的判断条件。
                bool upperRight = !shapeInfo.upperRight;
                bool lowerLeft = !shapeInfo.lowerLeft;

                // A 的行间 stride 跨过所有 Q heads，B 的列/head stride 跨过 KV heads；GQA/MQA 已在地址侧选定 KV head。
                LayoutA layoutA(km, headdim, nheads * headdim);
                LayoutB layoutB(headdim, kn, nheads_k * headdim);
                LayoutC layoutC(km, kn, 128);
                for (int m_index = 0; m_index < m_loop; m_index++) {
                    LocalTensor<ElementA>* l1_a_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(km, static_cast<uint32_t>(headdim)));
                    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1_m_size_, headdim);                                        
                    // 当前 AddrInfo 的 A 矩阵整体搬入 L1A；随后在 cube1_base_matmul 中按 128 行切到 L0A。
                    copyGmToL1A(*l1_a_tensor, gLeft, layoutAInL1, layoutTileA);

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);

                    for (int n_index = 0; n_index < n_loop; n_index++) {
                        // B 按 N 维每 128 列分批搬入 L1B，最后一个 n block 可能是尾块。
                        l1_n_size_ = (n_index == n_loop - 1) ? (kn - n_index * 128) : 128;
                        l1_n_size_align_ = RoundUp<C0_SIZE>(l1_n_size_);
                        // 只有最后一个 N 子块可能对应 causal 的右上无效区域。
                        bool upper_right_flag = (upperRight && n_index == n_loop - 1);

                        LocalTensor<ElementB>* l1_b_tensor = pingpongFlagL1B ? &l1_b_pong_tensor : &l1_b_ping_tensor;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);
                        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(l1_k_size_, l1_n_size_));
                        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(l1_k_size_, l1_n_size_);
                        // right 以原始 (token, head, dim) 方式连续存储；偏移跨过 n_index 个 128-token KV 子块。
                        copyGmToL1B(*l1_b_tensor, gRight[n_index * 128 * src_k_size_ * nheads_k], layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);

                        // 完成当前 A 与当前 B 子块的 Cube 计算，并将有效 C 子块写入 ping-pong workspace。
                        cube1_base_matmul(l1_a_tensor, l1_b_tensor, &gOut, pingpongFlagL1A, pingpongFlagL0A, pingpongFlagL1B, pingpongFlagL0B,
                            pingpongFlagC, l1_m_size_, l1_n_size_, upper_right_flag);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);
                        pingpongFlagL1B = 1 - pingpongFlagL1B;
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);
                    pingpongFlagL1A = 1 - pingpongFlagL1A;
                }
            }
        }

    protected:
        /// Data members
        // TileCopy/TileMmad 是 CATLASS 对数据搬运和 Cube 矩阵乘指令的封装。
        TileMmad tileMmad;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;

        // 当前 AddrInfo 内已经写出的有效 128x128 C 块数量，用于压缩跳过后的 workspace 布局。
        uint32_t cube1Cnt = 0;
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

        uint64_t nheads;
        uint64_t nheads_k;
        uint64_t headdim;

        // pingPongIdx 选择 task slot，globalBlockOffset 选择当前 AIC 在全局 workspace 中的私有区域。
        uint32_t pingPongIdx = 0;
        uint64_t globalBlockOffset = 0;
    };

    ////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif  // ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE1_HPP
