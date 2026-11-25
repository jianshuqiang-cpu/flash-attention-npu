#!/usr/bin/env python3
"""Generate Python wrapper for a torch_catlass op.

Usage: python gen_python.py <name>
"""

import sys

WRAPPER_TEMPLATE = """import torch
from torch import Tensor

def {func_name}(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    \"\"\"Run CATLASS {readable_name} on NPU tensors.

    Source: example {example_id}.

    Args:
        mat1: Left input matrix. Shape ``(M, K)`` unless ``transA`` is true.
        mat2: Right input matrix. Shape ``(K, N)`` unless ``transB`` is true.
        outDType: Output dtype. Accepted strings are ``float16``, ``float32``
            and ``bf16``/``bfloat16``.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is stored in the CATLASS NZ block format.
        formatB: Whether ``mat2`` is stored in the CATLASS NZ block format.

    Returns:
        Output tensor with shape ``(M, N)`` on the active NPU device.
    \"\"\"
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{{outDType}} is not a data type of torch")
    return torch.ops.catlass.{func_name}(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
"""


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <nn> <name>")
        sys.exit(1)

    nn = sys.argv[1]
    name = sys.argv[2]
    readable = name.replace("_", " ")
    print(WRAPPER_TEMPLATE.format(
        func_name=name,
        readable_name=readable,
        example_id=f"{nn}_{name}",
    ))


if __name__ == "__main__":
    main()
