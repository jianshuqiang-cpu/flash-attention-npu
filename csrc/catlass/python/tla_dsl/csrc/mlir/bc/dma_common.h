#pragma once

#include "common.h"

#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using bf16 = bfloat16_t;

/// Layout tags for the bc-layer DMA helpers. Mirrors the IR-side
/// `TensorLayoutTag` enum (same enumerators, same order)
enum class LayoutTag
{
    Unknown,
    RowMajor,
    ColumnMajor,
    zN,
    zZ,
    nZ,
    L0C,
    zNUnAlign,
};

/// Unified 4D (12-field) tensor descriptor. Linear layouts (RowMajor/
/// ColumnMajor) carry shape2=shape3=stride2=stride3=1, so a single descriptor
/// shape serves both Linear and NZFamily endpoints.
struct TensorDesc {
    uint32_t shape0;
    uint32_t shape1;
    uint32_t shape2;
    uint32_t shape3;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    int64_t stride3;
    uint32_t coord0;
    uint32_t coord1;
    uint32_t originShape0;
    uint32_t originShape1;
};

template <typename T, size_t Dim>
CATLASS_DEVICE T* basePtr(memref_t<T, Dim>* memref)
{
    return memref->aligned + memref->offset;
}

template <typename T, size_t Dim>
CATLASS_DEVICE uint32_t localAddr(memref_t<T, Dim>* memref)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(basePtr(memref)));
}

template <typename T, size_t Dim>
CATLASS_DEVICE uint32_t elementCount(memref_t<T, Dim>* memref)
{
    // Dynamic ptr-backed on-chip memrefs deliberately carry zero as an unknown
    // allocation-capacity sentinel. LocalTensor consumers use the address plus
    // the separately supplied TensorDesc shape/layout; this count is not part of
    // the copy operation's logical-shape contract.
    uint32_t n = 1;
    for (size_t i = 0; i < Dim; ++i)
        n *= static_cast<uint32_t>(memref->sizes[i]);
    return n;
}

template <typename T>
CATLASS_DEVICE AscendC::GlobalTensor<T> makeGlobalTensor(__gm__ T* ptr)
{
    AscendC::GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(ptr);
    return tensor;
}

template <typename DescT>
CATLASS_DEVICE auto makeTlaTileCoord(const DescT& desc)
{
    return tla::MakeCoord(desc.coord0, desc.coord1);
}

template <typename T>
CATLASS_DEVICE auto makeRowMajorTlaLayout(const TensorDesc& desc)
{
    return tla::MakeLayout(
        tla::MakeShape(desc.shape0, desc.shape1), tla::MakeStride(desc.stride0, tla::Int<1>{}),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

CATLASS_DEVICE auto makeColumnMajorTlaLayout(const TensorDesc& desc)
{
    return tla::MakeLayout(
        tla::MakeShape(desc.shape0, desc.shape1), tla::MakeStride(tla::Int<1>{}, desc.stride1),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makezNTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal = Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{}, tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makezNUnAlignTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal = Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(desc.shape0 * desc.shape1, tla::Int<1>{}),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{}, desc.stride3), tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makenZTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal = Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<1>{}, desc.stride1),
            tla::MakeStride(tla::Int<eleNumPerC0>{}, tla::Int<eleNumPerFractal>{})),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeL0CTlaLayout(const TensorDesc& desc)
{
    constexpr uint32_t eleNumPerFractal = 256;
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

// ---------------------------------------------------------------------------
// Position-tagged tensor constructors. LayoutTag is a non-type template
// parameter; each Position selects the layout constructor appropriate to that
// memory (e.g. UB zN uses the UnAlign variant, L1/L0A zN uses the aligned one).
// ---------------------------------------------------------------------------

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeGMTensor(memref_t<__gm__ T, 2>* memref, const TensorDesc& desc)
{
    if constexpr (Tag == LayoutTag::RowMajor) {
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref)), makeRowMajorTlaLayout<T>(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else if constexpr (Tag == LayoutTag::ColumnMajor) {
        return tla::MakeTensor(
            makeGlobalTensor(basePtr(memref)), makeColumnMajorTlaLayout(desc), makeTlaTileCoord(desc),
            Catlass::Arch::PositionGM{});
    } else {
        static_assert(
            Tag == LayoutTag::RowMajor || Tag == LayoutTag::ColumnMajor,
            "GM tensor supports RowMajor/ColumnMajor only");
    }
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeUBTensor(memref_t<__ubuf__ T, 1>* memref, const TensorDesc& desc)
{
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::VECCALC, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::RowMajor) {
        return tla::MakeTensor(
            tensor, makeRowMajorTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else if constexpr (Tag == LayoutTag::ColumnMajor) {
        return tla::MakeTensor(
            tensor, makeColumnMajorTlaLayout(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else if constexpr (Tag == LayoutTag::zN || Tag == LayoutTag::zNUnAlign) {
        return tla::MakeTensor(
            tensor, makezNUnAlignTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionUB{});
    } else {
        static_assert(
            Tag == LayoutTag::RowMajor || Tag == LayoutTag::ColumnMajor || Tag == LayoutTag::zN ||
                Tag == LayoutTag::zNUnAlign,
            "UB tensor supports RowMajor/ColumnMajor/zN only");
    }
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL1Tensor(memref_t<__cbuf__ T, 1>* memref, const TensorDesc& desc)
{
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::A1, localAddr(memref), elementCount(memref));
    if constexpr (Tag == LayoutTag::zN) {
        return tla::MakeTensor(tensor, makezNTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else if constexpr (Tag == LayoutTag::nZ) {
        return tla::MakeTensor(tensor, makenZTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
    } else {
        static_assert(Tag == LayoutTag::zN || Tag == LayoutTag::nZ, "L1 tensor supports zN/nZ only");
    }
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0ATensor(memref_t<__ca__ T, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::zN, "L0A tensor supports zN only");
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::A2, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makezNTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0A{});
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0BTensor(memref_t<__cb__ T, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::nZ, "L0B tensor supports nZ only");
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::B2, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makenZTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0B{});
}

template <LayoutTag Tag, typename T>
CATLASS_DEVICE auto makeL0CTensor(memref_t<__cc__ T, 1>* memref, const TensorDesc& desc)
{
    static_assert(Tag == LayoutTag::L0C, "L0C tensor supports L0C only");
    AscendC::LocalTensor<T> tensor(AscendC::TPosition::CO1, localAddr(memref), elementCount(memref));
    return tla::MakeTensor(tensor, makeL0CTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0C{});
}

// ---------------------------------------------------------------------------
// C-ABI helpers for the unified 12-field descriptor. DESC_ABI_PARAMS declares
// the 12 int64 descriptor args in a ciface signature; TENSOR_DESC_12 builds the
// matching TensorDesc from them. Both take a prefix token (src/dst).
// ---------------------------------------------------------------------------
#define DESC_ABI_PARAMS(P)                                                                                \
    int64_t P##Shape0, int64_t P##Shape1, int64_t P##Shape2, int64_t P##Shape3, int64_t P##Stride0,       \
        int64_t P##Stride1, int64_t P##Stride2, int64_t P##Stride3, int64_t P##Coord0, int64_t P##Coord1, \
        int64_t P##OrgShape0, int64_t P##OrgShape1

#define TENSOR_DESC_12(P)                                                                                            \
    TensorDesc                                                                                                       \
    {                                                                                                                \
        (uint32_t) P##Shape0, (uint32_t)P##Shape1, (uint32_t)P##Shape2, (uint32_t)P##Shape3, P##Stride0, P##Stride1, \
            P##Stride2, P##Stride3, (uint32_t)P##Coord0, (uint32_t)P##Coord1, (uint32_t)P##OrgShape0,                \
            (uint32_t)P##OrgShape1                                                                                   \
    }
