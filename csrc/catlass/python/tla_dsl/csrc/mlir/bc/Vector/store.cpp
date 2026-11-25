#include "../common.h"
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

extern "C" {

// store with block stride
// strideConfig[31:16] = (uint16_t)block_stride
// strideConfig[15:0]  = (uint16_t)repeat_stride
#define REGISTER_VSSTB(Dtype, dtype)                                                                     \
    __aiv__ __attribute__((always_inline)) void _mlir_ciface_store_with_stride_##dtype(                  \
        VectorReg<Dtype> srcReg, memref_t<__ubuf__ Dtype, 1>* dstUb, int32_t blockStride, ave_preg preg) \
    {                                                                                                    \
        __ubuf__ Dtype* dstAddr = dstUb->aligned + dstUb->offset;                                        \
        int32_t strideConfig = blockStride << 16; /* repeat_stride=0 */                                  \
        vector_bool mask = convertAVEPregToVecBool(preg);                                                \
        vsstb(srcReg, dstAddr, strideConfig, mask);                                                      \
    }

REGISTER_VSSTB(float, float)
REGISTER_VSSTB(half, half)
REGISTER_VSSTB(bfloat16_t, bf16)
}

#endif
