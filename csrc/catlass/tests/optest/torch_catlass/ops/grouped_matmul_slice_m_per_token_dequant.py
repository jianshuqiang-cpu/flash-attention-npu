import torch
from torch import Tensor

def grouped_matmul_slice_m_per_token_dequant(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    scale: Tensor,
    perTokenScale: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS grouped matmul Slice-M with per-token dequant.

    Source: example 07_grouped_slice_m_per_token_dequant_moe.

    int8 A @ int8 B, dequantized by scale and perTokenScale, output fp16.

    Args:
        mat1: int8 ``(M, K)``.
        mat2: int8 ``(G, K, N)``.
        groupList: int32 group boundaries on NPU.
        scale: per-column scale.
        perTokenScale: per-token scale.
        outDType: Output dtype.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output ``(M, N)``.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.grouped_matmul_slice_m_per_token_dequant_moe(
        mat1, mat2, groupList, scale, perTokenScale, outDType,
        transA, transB, useNzA, useNzB
    )


def grouped_matmul_slice_m_per_token_dequant_multistage(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    scale: Tensor,
    perTokenScale: Tensor,
    outDType: str | torch.dtype = torch.bfloat16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS grouped matmul Slice-M with per-token dequant (multistage).

    Source: example 10_grouped_matmul_slice_m_per_token_dequant.

    int8 A @ int8 B, dequantized by scale and perTokenScale, output bf16.
    Uses multistage workspace with callback dispatch.

    Args:
        mat1: int8 ``(M, K)``.
        mat2: int8 ``(G, K, N)``.
        groupList: int64 group boundaries on NPU.
        scale: per-column scale (bf16).
        perTokenScale: per-token scale (bf16).
        outDType: Output dtype.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output ``(M, N)``.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.grouped_matmul_slice_m_per_token_dequant(
        mat1, mat2, groupList, scale, perTokenScale, outDType,
        transA, transB, useNzA, useNzB
    )


def grouped_matmul_slice_k_per_token_dequant(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    scale: Tensor,
    perTokenScale: Tensor,
    outDType: str | torch.dtype = torch.bfloat16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS grouped matmul Slice-K with per-token dequant.

    Source: example 11_grouped_matmul_slice_k_per_token_dequant.

    int8 A @ int8 B, dequantized by scale and perTokenScale, output bf16.
    A is ColumnMajor, pass as ``(K, M)`` RowMajor with transA=True.

    Args:
        mat1: int8 ``(M, K)`` (or ``(K, M)`` if transA).
        mat2: int8 ``(K, N)``.
        groupList: int64 group boundaries on NPU.
        scale: per-group per-column scale.
        perTokenScale: per-group per-token scale.
        outDType: Output dtype.
        transA: Whether to read ``mat1`` as transposed (default ``False``).
        transB: Whether to read ``mat2`` as transposed.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output ``(G, M, N)``.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.grouped_matmul_slice_k_per_token_dequant(
        mat1, mat2, groupList, scale, perTokenScale, outDType,
        transA, transB, useNzA, useNzB
    )
