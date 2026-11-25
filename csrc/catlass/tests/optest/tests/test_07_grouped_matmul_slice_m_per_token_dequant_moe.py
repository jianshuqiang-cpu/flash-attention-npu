import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_grouped_matmul_slice_m_per_token_dequant_moe():
    G = 4
    M_total = 512
    n = 128
    k = 64
    group_list = torch.tensor([128, 256, 384, 512], dtype=torch.int32, device="npu")
    group_sizes = [128, 128, 128, 128]

    a = torch.randint(-8, 8, (M_total, k), dtype=torch.int8, device="npu")
    b = torch.randint(-8, 8, (G, k, n), dtype=torch.int8, device="npu")
    scale = torch.randn(G, n, dtype=torch.float32, device="npu").abs() * 0.1
    per_token_scale = torch.randn(M_total, dtype=torch.float32, device="npu").abs() * 0.1

    result = torch_catlass.grouped_matmul_slice_m_per_token_dequant(
        a, b, group_list, scale, per_token_scale
    )
    expected_list = []
    offset = 0
    for i, size in enumerate(group_sizes):
        part = torch.matmul(a[offset:offset+size].float(), b[i].float())
        part = part * scale[i] * per_token_scale[offset:offset+size].unsqueeze(1)
        expected_list.append(part)
        offset += size
    expected = torch.cat(expected_list, dim=0)

    assert result.shape == (M_total, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.float(), expected, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
