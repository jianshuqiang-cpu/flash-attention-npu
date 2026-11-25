#!/usr/bin/env python3
"""Generate pytest for a basic matmul example.

Usage: python gen_test.py <nn> <name> [--padding] [--transB] [--small]
"""

import sys

TEST_TEMPLATE = """import pytest
import torch
import torch_npu

pytestmark = pytest.mark.skipif(
    torch_npu.npu.device_count() <= 0,
    reason="torch-catlass integration tests require an available Ascend NPU",
)


def test_{func_name}():
    import torch_catlass

    m, n, k = {M}, {N}, {K}
    a = torch.randn({a_shape}, dtype=torch.float16, device="npu")
    b = torch.randn({b_shape}, dtype=torch.float16, device="npu")

    result = torch_catlass.{func_name}(a, b, "float16", {transA}, {transB}, False, False)
    expected = {reference}

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
"""


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <nn> <name> [--transB] [--padding] [--small]")
        sys.exit(1)

    nn = sys.argv[1]
    name = sys.argv[2]
    opts = set(sys.argv[3:])

    transA = "False"
    transB = "False"
    a_shape = "m, k"
    b_shape = "k, n"
    reference = "torch.matmul(a, b)"

    if "--transB" in opts:
        transB = "True"
        b_shape = "n, k"
        reference = "torch.matmul(a, b.T)"

    if "--small" in opts:
        M, N, K = 64, 128, 128
    elif "--padding" in opts:
        M, N, K = 128, 256, 64
    else:
        M, N, K = 256, 256, 256

    print(TEST_TEMPLATE.format(
        func_name=name,
        M=M, N=N, K=K,
        a_shape=a_shape, b_shape=b_shape,
        transA=transA, transB=transB,
        reference=reference,
    ))


if __name__ == "__main__":
    main()
