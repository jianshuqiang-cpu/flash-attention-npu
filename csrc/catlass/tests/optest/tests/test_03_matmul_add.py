import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_matmul_add():
    m, n, k = 256, 128, 64
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    x = torch.randn(m, n, dtype=torch.float16, device="npu")
    x_copy = x.clone()  # save reference before in-place modification

    result = torch_catlass.matmul_add(a, b, x, "float16", False, False, False, False)
    expected = torch.matmul(a, b) + x_copy

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
