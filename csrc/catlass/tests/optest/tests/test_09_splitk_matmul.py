import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_splitk_matmul():
    m, n, k = 256, 128, 512
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    r = torch_catlass.splitk_matmul(a, b)
    e = torch.matmul(a, b)
    assert r.shape == (m, n)
    assert torch.allclose(r, e, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
