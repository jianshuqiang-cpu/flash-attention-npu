import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_grouped_matmul_slice_m():
    G = 4
    M_total = 512
    n = 128
    k = 64
    group_sizes = [128, 128, 128, 128]
    group_list = torch.tensor([128, 256, 384, 512], dtype=torch.int64, device="npu")

    a = torch.randn(M_total, k, dtype=torch.float16, device="npu")
    b = torch.randn(G, k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.grouped_matmul_slice_m(a, b, group_list)
    expected = []
    offset = 0
    for i, size in enumerate(group_sizes):
        part = torch.matmul(a[offset:offset+size], b[i])
        expected.append(part)
        offset += size
    expected = torch.cat(expected, dim=0)

    assert result.shape == (M_total, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
