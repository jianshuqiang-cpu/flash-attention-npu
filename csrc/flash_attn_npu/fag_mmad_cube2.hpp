/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE2_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE2_HPP

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

    // FAG 反向中的 Cube2 MMAD 实现。
    // 该模板处理“左矩阵不转置、右矩阵不转置”的矩阵乘，核心公式是 dQ = dS * K。
    // left 通常来自 Vector epilogue 写出的 dS workspace，right 是原始 K，out 是 fp32 dq workspace。
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
        MmadAtlasA2FAGCube2,
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
        using DispatchPolicy = MmadAtlasA2FAGCube2;
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
        using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
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

            // padding 值为 0，保证尾块 C0 对齐时无效元素不会贡献到 dQ。
            AscendC::SetLoadDataPaddingValue<uint64_t>(0);
            uint64_t config = 0x1;
            AscendC::SetNdParaImpl(config);

            // L1 预切 A/B ping-pong 缓冲：A 放 dS 子块，B 放 K 子块。
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
        void operator()(
            const CubeAddrInfo &addrs, __gm__ ElementA *left, __gm__ ElementB *right, __gm__ float *out,
            uint32_t &pingpongFlagL1A, uint32_t &pingpongFlagL0A, uint32_t &pingpongFlagL1B,
            uint32_t &pingpongFlagL0B)
        {
            // 读取 Vector 已完成的上一拍 dS，因此使用 addrs.taskId % 2 对应的 workspace slot。
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                auto &shapeInfo = addrs.addrInfo[i];

                // A 从 dS workspace 读取，shapeInfo.out 是当前 attention block 的 dS 基址。
                // B 从原始 K 读取，C 写到 dq workspace 的 Q 位置；多个 K block 对同一 dQ 行累加。
                auto gm_a = left + (shapeInfo.out + globalBlockOffset);
                auto gm_b = right + shapeInfo.right;
                auto gm_c = out + shapeInfo.left;

                AscendC::GlobalTensor<ElementA> gLeft;
                gLeft.SetGlobalBuffer((__gm__ ElementA *)gm_a);

                AscendC::GlobalTensor<ElementB> gRight;
                gRight.SetGlobalBuffer((__gm__ ElementB *)gm_b);

                AscendC::GlobalTensor<ElementC> gOut;
                gOut.SetGlobalBuffer((__gm__ ElementC *)gm_c);

                // dS 的逻辑形状是 (ky, kx)，K 的逻辑形状是 (kx, headdim)，输出 dQ 是 (ky, headdim)。
                // 这里 km=Q 方向长度，kn=K 方向长度，kn 同时也是矩阵乘归约维。
                uint32_t kn = shapeInfo.kx;
                uint32_t km = shapeInfo.ky;
                uint32_t lineStride = shapeInfo.lineStride;

                int32_t l1_m_size = km;
                int32_t l1_n_size = kn;
                int32_t l1_k_size = headdim;

                int32_t l1_m_size_align = RoundUp<C0_SIZE>(l1_m_size);
                int32_t l1_n_size_align = RoundUp<C0_SIZE>(l1_n_size);
                int32_t l1_m_block_size_tail = (l1_m_size % 128) == 0 ? 128 : (l1_m_size % 128);
                int32_t l1_n_block_size_tail = (l1_n_size % 128) == 0 ? 128 : (l1_n_size % 128);
                int32_t l1_m_block_size_align_tail = (l1_m_size_align % 128) == 0 ? 128 : (l1_m_size_align % 128);
                int32_t l1_n_block_size_align_tail = (l1_n_size_align % 128) == 0 ? 128 : (l1_n_size_align % 128);

                int32_t m_loop = CeilDiv<SIZE_128>(km);
                int32_t n_loop = CeilDiv<SIZE_128>(kn);
                // causal 右上角块在 dS workspace 中被压缩跳过，读取 dS 时需要同步跳过这些无效块。
                bool upperRight = !shapeInfo.upperRight;

                LayoutA layoutA(km, kn, 128);
                int32_t skip_num = 0;
                for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
                    // 外层遍历归约维 K/token，每次处理一个 dS 的 N 子块和 K 的 token 子块。
                    int32_t n_remain = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : 128;
                    int32_t n_remain_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : 128;
                    // 第一个 K 子块初始化 L0C，后续 K 子块继续累加同一个 dQ 子块。
                    bool l0_c_init_flag = (n_loop_index == 0);

                    LayoutB layoutB(headdim, n_remain, nheads_k * headdim);

                    // 将当前 K 子块从 GM 搬到 L1B，逻辑形状为 (n_remain, headdim)。
                    AscendC::LocalTensor<ElementB>* l1_b_buf_tensor = pingpongFlagL1B ? &l1_b_pong_tensor : &l1_b_ping_tensor;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);

                    auto layoutTileB = layoutB.GetTileLayout(MakeCoord(static_cast<uint32_t>(n_remain), static_cast<uint32_t>(headdim)));
                    LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(n_remain, headdim);
                    // gRight 的偏移跨过 n_loop_index 个 128-token KV 子块；stride 中包含 nheads_k。
                    copyGmToL1B(*l1_b_buf_tensor, gRight[n_loop_index * nheads_k * 128 * headdim], layoutBInL1, layoutTileB);

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);

                    // 预取当前 K 子块对应的所有 dS(M,N) 子块到 L1A。
                    pingpongFlagL1A = 0;
                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        AscendC::LocalTensor<ElementA>* l1_a_buf_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;
                        int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
                        int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                        bool is_skip = false;

                        // 与 Cube1 写 dS workspace 的压缩规则保持一致：causal 右上无效块不占 workspace。
                        if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                            skip_num++;
                            is_skip = true;
                        } 

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);

                        if (!is_skip) {
                            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(n_remain)));
                            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(m_remain, n_remain);
                            // dS workspace 中只存有效块，所以读取偏移要减去 skip_num。
                            copyGmToL1A(*l1_a_buf_tensor, gLeft[(m_loop * n_loop_index + m_loop_index - skip_num) * 128 * 128], layoutAInL1, layoutTileA);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                        pingpongFlagL1A = 1 - pingpongFlagL1A;
                    }

                    // 当前 K 子块只需搬一次到 L0B，随后在所有 M 子块计算中常驻复用。
                    AscendC::LocalTensor<ElementB>* l0_b_buf_tensor = pingpongFlagL0B ? &l0_b_pong_tensor : &l0_b_ping_tensor;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1B);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0B + 2 + FLAG_SHIFT);

                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(n_remain, headdim);
                    copyL1ToL0B(*l0_b_buf_tensor, *l1_b_buf_tensor, layoutBInL0, layoutBInL1);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1B);

                    pingpongFlagL1A = 0;
                    pingpongFlagL0A = 0;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0B + 2);

                    // L0B 常驻当前 K 子块，沿 M/Q 方向逐块计算 dQ。
                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        AscendC::LocalTensor<ElementA>* l1_a_buf_tensor = pingpongFlagL1A ? &l1_a_pong_tensor : &l1_a_ping_tensor;
                        AscendC::LocalTensor<ElementA>* l0_a_buf_tensor = pingpongFlagL0A ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                        // m_loop_index 为 0 时初始化/使用 ping C，后续子块使用 pong C，配合 l0_c_init_flag 做 K 维累加。
                        AscendC::LocalTensor<float>* l0_c_buf_tensor = m_loop_index ? &l0_c_pong_tensor : &l0_c_ping_tensor;

                        int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
                        int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                        bool is_skip = false;

                        if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                            is_skip = true;
                        }
                        // dS 的 causal 右上无效块不参与 dQ 计算。
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlagL1A + 2);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);
                        if (!is_skip) {
                            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(m_remain, n_remain);
                            LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(m_remain, n_remain);
                            // A=dS(M,N) 从 L1A 搬到 L0A。
                            copyL1ToL0A(*l0_a_buf_tensor, *l1_a_buf_tensor, layoutAInL0, layoutAInL1);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlagL1A + 2);

                        // MMAD: dS(m_remain, n_remain) x K(n_remain, headdim) -> dQ(m_remain, headdim)。
                        bool last_k = false;
                        // 如果右上角最后一个 K 子块被跳过，则倒数第二个 K 子块才是该 M 子块的最后一次累加。
                        last_k = (m_loop_index == 0 && upperRight) ? n_loop_index == n_loop - 2 : n_loop_index == n_loop - 1;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlagL0A);
                        if (!is_skip) {
                            // MMAD 不接受 m=1 的特殊形状，提升为 2 行后依赖 padding 保持有效输出不变。
                            uint16_t m_modify = (m_remain == 1) ? 2 : m_remain;

                            // unit flag 为 2 表示继续累加，为 3 表示本轮 K 维结束并可输出 L0C。
                            tileMmad(*l0_c_buf_tensor, *l0_a_buf_tensor, *l0_b_buf_tensor, m_modify, headdim, n_remain, l0_c_init_flag, last_k ? 3 : 2);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0A + FLAG_SHIFT);

                        // 只有当前 M 子块完成所有 K 子块累加后，才把 L0C 写回 dq workspace。
                        if (!is_skip && last_k) {
                            // 不同 KV block 可能累加到同一 dQ 行，因此写回使用 fp32 atomic add。
                            AscendC::SetAtomicType<float>();
                            auto blockShape = MakeCoord(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(headdim));
                            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
                            LayoutC layoutC(m_remain, headdim, nheads * headdim);
                            copyL0CToGm((gOut)[m_loop_index * nheads * 128 * headdim], *l0_c_buf_tensor, layoutC, layoutInL0C, 3);
                            AscendC::SetAtomicNone();
                        }
                        pingpongFlagL1A = 1 - pingpongFlagL1A;
                        pingpongFlagL0A = 1 - pingpongFlagL0A;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlagL0B + 2 + FLAG_SHIFT);

                    pingpongFlagL0B = 1 - pingpongFlagL0B;
                    pingpongFlagL1B = 1 - pingpongFlagL1B;
                }
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

        // pingPongIdx 选择 dS workspace 的 task slot，globalBlockOffset 定位当前 AIC 的私有 workspace 区域。
        uint32_t pingPongIdx = 0;
        uint64_t globalBlockOffset = 0;
    };

////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif  // ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_CUBE2_HPP
