import torch
from torch import Tensor

def gemv_aiv(
    matA: Tensor, vecX: Tensor,
    outDType: str | torch.dtype = torch.float32,
    alpha: float = 1.0, beta: float = 0.0,
) -> Tensor:
    if isinstance(outDType, str):
        dt = outDType.lower()
        outDType = getattr(torch, dt, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.gemv_aiv(
        matA, vecX, outDType, alpha, beta, False, False, False, False)
