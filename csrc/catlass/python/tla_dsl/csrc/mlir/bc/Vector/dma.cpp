#include "../dma_common.h"

#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToUB(
    memref_t<__gm__ T, 2>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToGM(
    memref_t<__ubuf__ T, 1>* src, memref_t<__gm__ T, 2>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToL1(
    memref_t<__ubuf__ T, 1>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_UB(NameSrc, NameDst, EnumSrc, EnumDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##NameSrc##_to_ub_##NameDst##_##DType(          \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::EnumSrc, LayoutTag::EnumDst, DType>(                          \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_UB_TO_GM(NameSrc, NameDst, EnumSrc, EnumDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##NameSrc##_to_gm_##NameDst##_##DType(          \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::EnumSrc, LayoutTag::EnumDst, DType>(                          \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_UB_TO_L1(NameSrc, NameDst, EnumSrc, EnumDst, DType)                                         \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##NameSrc##_to_l1_##NameDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),            \
        DESC_ABI_PARAMS(dst))                                                                                \
    {                                                                                                        \
        copyUBToL1<Catlass::Arch::Ascend950, LayoutTag::EnumSrc, LayoutTag::EnumDst, DType>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                             \
    }

REGISTER_GM_TO_UB(row_major, row_major, RowMajor, RowMajor, float)
REGISTER_GM_TO_UB(row_major, row_major, RowMajor, RowMajor, half)
REGISTER_GM_TO_UB(row_major, row_major, RowMajor, RowMajor, bf16)
REGISTER_GM_TO_UB(row_major, row_major, RowMajor, RowMajor, int32_t)
REGISTER_GM_TO_UB(row_major, row_major, RowMajor, RowMajor, int16_t)
REGISTER_GM_TO_UB(row_major, row_major, RowMajor, RowMajor, int8_t)

REGISTER_UB_TO_GM(row_major, row_major, RowMajor, RowMajor, float)
REGISTER_UB_TO_GM(row_major, row_major, RowMajor, RowMajor, half)
REGISTER_UB_TO_GM(row_major, row_major, RowMajor, RowMajor, bf16)
REGISTER_UB_TO_GM(row_major, row_major, RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_GM(row_major, row_major, RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_GM(row_major, row_major, RowMajor, RowMajor, int8_t)

REGISTER_UB_TO_L1(row_major, zN, RowMajor, zN, float)
REGISTER_UB_TO_L1(row_major, zN, RowMajor, zN, half)
REGISTER_UB_TO_L1(row_major, zN, RowMajor, zN, bf16)
REGISTER_UB_TO_L1(zN, zN, zN, zN, float)
REGISTER_UB_TO_L1(zN, zN, zN, zN, half)
REGISTER_UB_TO_L1(zN, zN, zN, zN, bf16)

#endif
}
