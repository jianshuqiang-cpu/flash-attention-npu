# CopyL0CToDst (Shared Infrastructure)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp)

[TOC]

## Overview

The `copy_l0c_to_dst` module defines the shared infrastructure required for L0C data movement, including quantization mode mapping, scale granularity enumeration, Unified Buffer movement mode enumeration, and TLA template declarations. It is included and referenced by the [copy_l0c_to_gm](../copy_l0c_to_gm/README.md) and [copy_l0c_to_ub](../copy_l0c_to_ub.md) modules and does not contain operator calling APIs.

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyL0CToDstQuantMode](./copy_l0c_to_dst.md) | — | Ascend 950| Quantization mode mapping|
| [ScaleGranularity](./copy_l0c_to_dst.md#scalegranularity) | — | Atlas A2/Ascend 950| Scale granularity enumeration|
| [CopyL0CToUBMode](./copy_l0c_to_dst.md) | — | Ascend 950| Unified Buffer movement mode enumeration|

## Dependencies

```
copy_l0c_to_dst (infrastructure)
    ├── copy_l0c_to_gm  (L0C→GM: CopyL0CToGm + CopyL0CToGmTla)
    └── copy_l0c_to_ub  (L0C→UB: CopyL0CToUBTla)
```

## Applicable Hardware Models

| Architecture| Supported or Not|
| :------ | :------ |
| Atlas A2 (`CATLASS_ARCH == 2201`)| `ScaleGranularity` + equivalent `CopyL0CToGmQuantMode` (in `atlasa2/copy_l0c_to_gm.hpp`)|
| Ascend 950 (`CATLASS_ARCH == 3510`)| `CopyL0CToDstQuantMode` + `CopyL0CToUBMode` + Template declaration|
