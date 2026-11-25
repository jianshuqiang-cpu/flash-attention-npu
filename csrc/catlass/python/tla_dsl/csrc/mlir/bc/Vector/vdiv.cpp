#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

// Mirror AscendNPU-IR RegBase/Vector/Vdiv.cpp::ave_vdiv: integer VFDiv is
// lowered to _mlir_ciface_vdiv_{int16,int32,uint16,uint32}_t by
// ConvertHIVMAVEToStandard; slim DSL AIV bc must provide those symbols via CCE
// vdiv instead of linking the CANN toolkit template.
template <typename T>
__aiv__ __attribute__((always_inline)) VectorReg<T> ave_vdiv(VectorReg<T> src0, VectorReg<T> src1, ave_preg preg)
{
    vector_bool p = convertAVEPregToVecBool(preg);
    VectorReg<T> sret;
    vdiv(sret, src0, src1, p, MODE_ZEROING);
    return sret;
}

#define DECLARE_DIV_VV(op_name, dtype)                                                        \
    __aiv__ __attribute__((always_inline)) VectorReg<dtype> _mlir_ciface_##op_name##_##dtype( \
        VectorReg<dtype> src0, VectorReg<dtype> src1, ave_preg preg)

#define REGISTE_DIV_VV(op_name, dtype)            \
    DECLARE_DIV_VV(op_name, dtype)                \
    {                                             \
        return ave_vdiv<dtype>(src0, src1, preg); \
    }

} // namespace

extern "C" {
REGISTE_DIV_VV(vdiv, int16_t);
REGISTE_DIV_VV(vdiv, int32_t);
REGISTE_DIV_VV(vdiv, uint16_t);
REGISTE_DIV_VV(vdiv, uint32_t);
}

#endif
