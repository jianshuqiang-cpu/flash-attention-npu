import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_quant_matmul():
    m, n, k = 128, 64, 256
    a = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="npu")
    b = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="npu")
    s = torch.randn(n, dtype=torch.float16, device="npu").abs() * 0.1
    ps = torch.randn(m, dtype=torch.float16, device="npu").abs() * 0.1
    r = torch_catlass.quant_matmul(a, b, s, ps)
    e = torch.matmul(a.float(), b.float()) * s * ps.unsqueeze(1)
    assert r.shape == (m, n)
    assert torch.allclose(r.float(), e, rtol=1e-1, atol=1e-1)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
