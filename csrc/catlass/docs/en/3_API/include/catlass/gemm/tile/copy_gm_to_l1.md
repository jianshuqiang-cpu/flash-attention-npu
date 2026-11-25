# Basic Template for Copying GM to L1
>
> [Code location](../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## CopyGmToL1

### Description

### Prototype

- Structure template

```
template <
    class ArchTag,          // Architecture tag
    class GmType,           // GEMM type of the operand on the GM
    class L1Type = void     // GEMM type of the operand on the L1
>
struct CopyGmToL1
```

- Partial specialization implementation

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::RowMajor>  | Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::RowMajor>  |  Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  | Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  | Gemm::GemmType<Element, layout::VectorLayout>   |  Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::NDC1HWC0, AscendC::TPosition::GM>  |  - |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::KDC1KHKWN1N0C0, AscendC::TPosition::GM>  |  - |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  Gemm::GemmType<Element, layout::nN, AscendC::TPosition::B1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::RowMajor>  |  Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1> |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::RowMajor>  |  - |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::ColumnMajor>  |  - |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::zN>  |  - |
|<class ArchTag, class Element>|  ArchTag  |   Gemm::GemmType<Element, layout::nZ> |  - |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::PaddingRowMajor>  |  - |
|\<class Element\>|  Arch::AtlasA2  |   Gemm::GemmType<Element, layout::PaddingColumnMajor>  |  - |
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::RowMajor>  |  Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1> |
|<class ArchTag, class Element>|  ArchTag  |  Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>  |  Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1> |

- Invocation

```
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // Destination operand (LocalTensor)
    AscendC::GlobalTensor<Element> const &srcTensor,    // Source operand (GlobalTensor)
    LayoutDst const &layoutDst,         // Destination operand layout
    LayoutSrc const &layoutSrc          // Source operand layout
)
```

## CopyGmToL1IntervalDataCopy

### Description

### Prototype

- Structure template

```
template <
    class ArchTag,          // Architecture tag
    class GmType,           // GEMM type of the operand on the GM
    class L1Type = void     // GEMM type of the operand on the L1
>
struct CopyGmToL1IntervalDataCopy
```

- Partial specialization implementation

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::RowMajor>| -|
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::PaddingRowMajor>| -|
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::ColumnMajor>| -|
|-|  Arch::AtlasA2  | Gemm::GemmType<half, layout::PaddingColumnMajor>| -|

## CopyGmToL1GMMPTD

### Description

### Prototype

- Structure template

```
template <
    class ArchTag,          // Architecture tag
    class GmType,           // GEMM type of the operand on the GM
    class L1Type = void     // GEMM type of the operand on the L1
>
struct CopyGmToL1GMMPTD
```

- Partial specialization implementation

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|\<class Element\>|  Arch::AtlasA2  | Gemm::GemmType<Element, layout::RowMajor>| -|

## CopyGmToL1DynamicOptimized

### Description

### Prototype

- Structure template

```
template <
    class ArchTag,          // Architecture tag
    class GmType,           // GEMM type of the operand on the GM
    class L1Type = void     // GEMM type of the operand on the L1
>
struct CopyGmToL1DynamicOptimized
```

- Partial specialization implementation

|template| ArchTag  | GmType | L1Type |
| :------ | :------: |------: |------: |
|\<class Element\>|  Arch::AtlasA2  | Gemm::GemmType<Element, layout::RowMajor>| -|
|\<class Element\>|  Arch::AtlasA2  | Gemm::GemmType<Element, layout::ColumnMajor>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::zN>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::nZ>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::PaddingRowMajor>| -|
|\<class Element\>|  Arch::AtlasA2  |  Gemm::GemmType<Element, layout::PaddingColumnMajor>| -|

## TileCopyTla

### Description

### Prototype

- Structure template

```
template <
    class ElementSrc,   // Data type of the source operand
    class ElementDst,   // Data type of the destination operand
    class LayoutSrc,    // Layout of the source operand
    class LayoutDst,    // Layout of the destination operand
    class CoordSrc,     // Coordinates of the source operand in the tensor
    class CoordDst      // Coordinates of the destination operand in the tensor
    >
struct TileCopyTla<
    Arch::AtlasA2,                                  // Architecture tag
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>,
        LayoutSrc,
        CoordSrc,
        AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>,   // Tensor structure of the source operand
        LayoutDst,
        CoordDst,
        AscendC::TPosition::A1>,                    // Tensor structure of the destination operand
    std::enable_if_t<cond0 && cond1>              // Judgment conditions. For cond0 and cond1, see the following partial specialization implementation.
    >
```

- Partial specialization implementation

| cond0 | cond1 |
|------: |------: |
| tla::detail::isRowMajor\<LayoutSrc\>::value|tla::detail::iszN<ElementDst, LayoutDst>::value|
| tla::detail::isColumnMajor\<LayoutSrc\>::value|tla::detail::isnZ<ElementDst, LayoutDst>::value|
| tla::detail::iszN\<LayoutSrc\>::value|tla::detail::iszN<ElementDst, LayoutDst>::value|
| tla::detail::isnZ\<LayoutSrc\>::value|tla::detail::isnZ<ElementDst, LayoutDst>::value|

## TileCopyTlaExt

### Description

### Prototype

- Structure template

```
template <
    class ElementSrc,   // Data type of the source operand
    class ElementDst,   // Data type of the destination operand
    class LayoutSrc,    // Layout of the source operand
    class LayoutDst,    // Layout of the destination operand
    class CoordSrc,     // Coordinates of the source operand in the tensor
    class CoordDst      // Coordinates of the destination operand in the tensor
    >
struct TileCopyTla<
    Arch::AtlasA2,                                  // Architecture tag
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>,
        LayoutSrc,
        CoordSrc,
        AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>,   // Tensor structure of the source operand
        LayoutDst,
        CoordDst,
        AscendC::TPosition::A1>,                    // Tensor structure of the destination operand
    cond0,          // See the following partial specialization implementation
    cond1,          // See the following partial specialization implementation
    >
```

- Partial specialization implementation

| cond0 | cond1 |
|------: |------: |
| layout::RowMajor|layout::zN|
| layout::PaddingRowMajor|layout::zN|
| layout::ColumnMajor|layout::nZ|
| layout::PaddingColumnMajor|layout::nZ|
| layout::zN|layout::zN|
| layout::nZ|layout::nZ|
