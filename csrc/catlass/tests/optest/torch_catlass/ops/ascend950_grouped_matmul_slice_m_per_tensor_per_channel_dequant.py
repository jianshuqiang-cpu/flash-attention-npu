import torch
from torch import Tensor


def ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    perChannelScale: Tensor,
    scalePerTensor: float = 1.0,
    quantMode: int = 0,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 grouped matmul Slice-M with fixpipe per-tensor/per-channel dequant.

    Source: example 48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant.

    int8 A @ int8 B, dequantized via fixpipe with per-tensor or per-channel scale.

    Args:
        mat1: int8 ``(M, K)``.
        mat2: int8 ``(G, K, N)``.
        groupList: int64 group boundaries on NPU.
        perChannelScale: per-channel scale (uint64 encoded fp32, shape ``(N,)``).
        scalePerTensor: per-tensor scale (float scalar).
        quantMode: 0 = per_tensor, 1 = per_channel.
        outDType: Output dtype.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output ``(M, N)`` fp16.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant(
        mat1, mat2, groupList, perChannelScale,
        scalePerTensor, quantMode, outDType,
        transA, transB, useNzA, useNzB
    )
