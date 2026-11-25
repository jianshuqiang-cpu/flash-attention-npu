import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_single_core_splitk_matmul():
    m, n, k = 256, 128, 256
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(n, k, dtype=torch.float16, device="npu")

    result = torch_catlass.single_core_splitk_matmul(a, b, "float16", False, True, False, False)
    expected = torch.matmul(a, b.T)

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
