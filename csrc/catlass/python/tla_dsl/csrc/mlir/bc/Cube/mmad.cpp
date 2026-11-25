#include "catlass/catlass.hpp"
#include "../common.h"

namespace Catlass::Gemm {

template <class ElementA, class ElementB, class ElementC>
__aicore__ __attribute__((always_inline)) void Mmad(
    __ca__ ElementA* a, __cb__ ElementB* b, __cc__ ElementC* c, uint32_t m, uint32_t n, uint32_t k, bool initC = true,
    uint8_t unitFlag = 0)
{
    AscendC::LocalTensor<ElementA> l0a{
        AscendC::TPosition::A2, (uint32_t) reinterpret_cast<int64_t>(a), (uint32_t)(m * k)};
    AscendC::LocalTensor<ElementB> l0b{
        AscendC::TPosition::B2, (uint32_t) reinterpret_cast<int64_t>(b), (uint32_t)(n * k)};
    AscendC::LocalTensor<ElementC> l0c{
        AscendC::TPosition::CO1, (uint32_t) reinterpret_cast<int64_t>(c), (uint32_t)(m * n)};

    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.unitFlag = unitFlag;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.disableGemv = true;

    AscendC::Mmad(l0c, l0a, l0b, mmadParams);

    const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
    if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

} // namespace Catlass::Gemm

extern "C" {
__aicore__
    __attribute__((always_inline))
    // MLIR lowering passes memref<?xT, strided<[?], offset:?>, #ca/cb/cc> (rank 1); only GM uses rank 2.
    // Cube MMAD on dav-c310 uses fp32 L0C (__cc__); f16/bf16 outputs use copy_l0c_to_gm_row_major_* stubs.
    void
    _mlir_ciface_mmad_float_float_float(
        memref_t<__ca__ float, 1>* a, memref_t<__cb__ float, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m, int64_t n,
        int64_t k, bool initC = true, uint8_t unitFlag = 0)
{
    Catlass::Gemm::Mmad<float, float, float>(
        a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);
}

__aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_half_half_float(
    memref_t<__ca__ half, 1>* a, memref_t<__cb__ half, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m, int64_t n,
    int64_t k, bool initC = true, uint8_t unitFlag = 0)
{
    Catlass::Gemm::Mmad<half, half, float>(
        a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);
}

__aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_bf16_bf16_float(
    memref_t<__ca__ bfloat16_t, 1>* a, memref_t<__cb__ bfloat16_t, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m,
    int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)
{
    Catlass::Gemm::Mmad<bfloat16_t, bfloat16_t, float>(
        a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);
}
}
