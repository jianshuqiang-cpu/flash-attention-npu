#include "../dma_common.h"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToL1(
    memref_t<__gm__ T, 2>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename T>
CATLASS_DEVICE void copyL1ToL0A(
    memref_t<__cbuf__ T, 1>* src, memref_t<__ca__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL0ATensor<LayoutTag::zN, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename T>
CATLASS_DEVICE void copyL1ToL0B(
    memref_t<__cbuf__ T, 1>* src, memref_t<__cb__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL0BTensor<LayoutTag::nZ, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst>
CATLASS_DEVICE void copyL0CToGM(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__gm__ ElementDst, 2>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0C, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, unitFlag);
}

template <
    class ArchTag, LayoutTag DstLayout, Catlass::Gemm::Tile::CopyL0CToUBMode Mode, typename ElementSrc,
    typename ElementDst>
CATLASS_DEVICE void copyL0CToUB(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__ubuf__ ElementDst, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag, uint8_t subBlockId)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0C, ElementSrc>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, ElementDst>(dst, dstDesc);
    if constexpr (Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(srcTensor), decltype(dstTensor), Mode>{}(
            dstTensor, srcTensor, (bool)subBlockId, unitFlag);
    } else if constexpr (
        Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M ||
        Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(srcTensor), decltype(dstTensor), Mode>{}(
            dstTensor, srcTensor, unitFlag);
    }
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_L1(NameSrc, NameDst, EnumSrc, EnumDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##NameSrc##_to_l1_##NameDst##_##DType(          \
        memref_t<__gm__ DType, 2>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToL1<Catlass::Arch::Ascend950, LayoutTag::EnumSrc, LayoutTag::EnumDst, DType>(                          \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

REGISTER_GM_TO_L1(row_major, zN, RowMajor, zN, float)
REGISTER_GM_TO_L1(row_major, zN, RowMajor, zN, half)
REGISTER_GM_TO_L1(row_major, zN, RowMajor, zN, bf16)
REGISTER_GM_TO_L1(column_major, nZ, ColumnMajor, nZ, float)
REGISTER_GM_TO_L1(column_major, nZ, ColumnMajor, nZ, half)
REGISTER_GM_TO_L1(column_major, nZ, ColumnMajor, nZ, bf16)

#define REGISTER_L1_TO_L0A(NameSrc, EnumSrc, DType)                                                                   \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_##NameSrc##_to_l0a_zN_##DType(                  \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__ca__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyL1ToL0A<Catlass::Arch::Ascend950, LayoutTag::EnumSrc, DType>(                                             \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

REGISTER_L1_TO_L0A(zN, zN, float)
REGISTER_L1_TO_L0A(zN, zN, half)
REGISTER_L1_TO_L0A(zN, zN, bf16)
REGISTER_L1_TO_L0A(nZ, nZ, float)
REGISTER_L1_TO_L0A(nZ, nZ, half)
REGISTER_L1_TO_L0A(nZ, nZ, bf16)

#define REGISTER_L1_TO_L0B(NameSrc, EnumSrc, DType)                                                                   \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_##NameSrc##_to_l0b_nZ_##DType(                  \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__cb__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyL1ToL0B<Catlass::Arch::Ascend950, LayoutTag::EnumSrc, DType>(                                             \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

REGISTER_L1_TO_L0B(zN, zN, float)
REGISTER_L1_TO_L0B(zN, zN, half)
REGISTER_L1_TO_L0B(zN, zN, bf16)
REGISTER_L1_TO_L0B(nZ, nZ, float)
REGISTER_L1_TO_L0B(nZ, nZ, half)
REGISTER_L1_TO_L0B(nZ, nZ, bf16)

#define REGISTER_L0C_TO_GM(DTypeSrc, DTypeDst)                                                      \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_gm_row_major_##DTypeDst(  \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src), \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag)                                                     \
    {                                                                                               \
        copyL0CToGM<Catlass::Arch::Ascend950, LayoutTag::RowMajor, DTypeSrc, DTypeDst>(             \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag);                          \
    }

REGISTER_L0C_TO_GM(float, float)
REGISTER_L0C_TO_GM(float, half)
REGISTER_L0C_TO_GM(float, bf16)

#define REGISTER_L0C_TO_UB(NameDst, EnumDst, mode, MODE, DTypeSrc, DTypeDst)                                    \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_ub_##NameDst##_##mode##_##DTypeDst(   \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),           \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId)                                             \
    {                                                                                                           \
        copyL0CToUB<                                                                                            \
            Catlass::Arch::Ascend950, LayoutTag::EnumDst, Catlass::Gemm::Tile::CopyL0CToUBMode::MODE, DTypeSrc, \
            DTypeDst>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, subBlockId);                \
    }

REGISTER_L0C_TO_UB(row_major, RowMajor, nosplit, NO_SPLIT, float, float)
REGISTER_L0C_TO_UB(row_major, RowMajor, nosplit, NO_SPLIT, float, half)
REGISTER_L0C_TO_UB(row_major, RowMajor, nosplit, NO_SPLIT, float, bf16)
// split mode src=dst
REGISTER_L0C_TO_UB(row_major, RowMajor, splitm, SPLIT_M, float, float)
REGISTER_L0C_TO_UB(row_major, RowMajor, splitn, SPLIT_N, float, float)
REGISTER_L0C_TO_UB(column_major, ColumnMajor, nosplit, NO_SPLIT, float, float)
REGISTER_L0C_TO_UB(column_major, ColumnMajor, nosplit, NO_SPLIT, float, half)
REGISTER_L0C_TO_UB(column_major, ColumnMajor, nosplit, NO_SPLIT, float, bf16)

#endif
}
