import pytest
import torch
import torch_npu

from common import only_on_2201

pytestmark = [only_on_2201, pytest.mark.skipif(
    torch_npu.npu.device_count() <= 0,
    reason="torch-catlass integration tests require an available Ascend NPU",
)]


def test_gemv_aiv():
    import torch_catlass

    m, n = 128, 512

    a = torch.randn(m, n, dtype=torch.float32, device="npu")
    x = torch.randn(n, dtype=torch.float32, device="npu")

    result = torch_catlass.gemv_aiv(a, x, "float32")
    expected = torch.matmul(a, x)

    assert result.shape == (m,)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
