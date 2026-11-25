import pytest
import torch
import torch_npu


from common import only_on_3510


@only_on_3510
def test_ascend950_matmul_fixpipe_opti():
    """Compare CATLASS Ascend950 matmul fixpipe opti against torch.matmul."""
    import torch_catlass

    m, n, k = 128, 128, 128
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.ascend950_matmul_fixpipe_opti(a, b, "float32", False, False, False, False)
    expected = torch.matmul(a, b).float()

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
