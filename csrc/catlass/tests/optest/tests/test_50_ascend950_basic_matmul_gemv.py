import pytest
import torch
import torch_npu


from common import only_on_3510


@only_on_3510
def test_ascend950_basic_matmul_gemv():
    """Compare CATLASS Ascend950 basic matmul GEMV against torch.matmul."""
    import torch_catlass

    m, n, k = 1, 128, 127
    a = torch.randn(m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_basic_matmul_gemv(a, b, "float32", False, False, False, False)
    expected = torch.matmul(a, b)

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
