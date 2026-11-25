import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_batched_matmul():
    B, m, n, k = 4, 128, 64, 256
    a = torch.randn(B, m, k, dtype=torch.float16, device="npu")
    b = torch.randn(B, k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.batched_matmul(a, b)
    expected = torch.bmm(a, b)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
