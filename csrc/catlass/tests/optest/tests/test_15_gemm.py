import pytest
import torch
import torch_npu

from common import only_on_2201

pytestmark = [only_on_2201, pytest.mark.skipif(
    torch_npu.npu.device_count() <= 0,
    reason="torch-catlass integration tests require an available Ascend NPU",
)]


def test_gemm():
    import torch_catlass

    m, n, k = 128, 128, 128
    dtype = torch.float32

    a = torch.randn(m, k, dtype=dtype, device="npu")
    b = torch.randn(k, n, dtype=dtype, device="npu")

    result = torch_catlass.gemm(a, b, "float32", 1.0, 0.0, False, False, False, False)
    expected = torch.matmul(a, b)

    assert result.shape == (m, n)
    assert result.dtype == dtype
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
