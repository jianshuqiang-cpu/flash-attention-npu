import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("transB", [True, False])
def test_strided_batched_matmul_tla(transA, transB):
    """Compare the CATLASS strided batched matmul (TLA) wrapper against a reference computation.

    Golden logic (from examples/45_strided_batched_matmul_tla/strided_batched_matmul_tla.cpp):
    For each batch b: C[b] = A[b] @ B[b]
    where A[b] is [M, K] and B[b] is [K, N], with strided memory access
    between batches.
    """
    batch, m, n, k = 2, 256, 256, 513

    a = torch.randn(batch, m, k, dtype=torch.float16, device="npu")
    b = torch.randn(batch, k, n, dtype=torch.float16, device="npu")
    expected = torch.matmul(a, b)

    a = a.permute(0, 2, 1).contiguous() if transA else a
    b = b.permute(0, 2, 1).contiguous() if transB else b

    result = torch_catlass.strided_batched_matmul_tla(
        a, b, "float16", transA=transA, transB=transB
    )


    assert result.shape == (batch, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result - expected).abs().max().item()}"
    )

@only_on_2201
@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("transB", [True, False])
def test_transposed_strided_batched_matmul_tla(transA, transB):    
    """Compare the CATLASS transposed strided batched matmul (TLA) wrapper against a reference computation.

    The transposed batched matmul produces equivalent results to:
    ```python
    ret = torch.matmul(a.permute(1, 0, 2), b)
    ```
    """
    batch, m, n, k = 2, 256, 256, 513

    a = torch.randn(batch, m, k, dtype=torch.float16, device="npu")
    b = torch.randn(batch, k, n, dtype=torch.float16, device="npu")
    expected = torch.matmul(a, b)

    a = a.permute(1, 0, 2).contiguous()
    a = a.permute(2, 1, 0).contiguous() if transA else a
    b = b.permute(0, 2, 1).contiguous() if transB else b

    result = torch_catlass.strided_batched_matmul_tla(
        a, b, "float16", transA=transA, transB=transB,
        batchTransA=True, batchTransB=False
    )

    assert result.shape == (batch, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
