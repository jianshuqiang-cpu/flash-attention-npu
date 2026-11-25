import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_grouped_matmul_slice_m_per_token_dequant_multistage():
    G = 2
    M_total = 256
    n = 64
    k = 32
    group_list = torch.tensor([128, 256], dtype=torch.int64, device="npu")
    group_sizes = [128, 128]

    a = torch.randint(-4, 4, (M_total, k), dtype=torch.int8, device="npu")
    b = torch.randint(-4, 4, (G, k, n), dtype=torch.int8, device="npu")
    scale = torch.randn(G, n, dtype=torch.bfloat16, device="npu").abs() * 0.1
    per_token_scale = torch.randn(M_total, dtype=torch.bfloat16, device="npu").abs() * 0.1

    result = torch_catlass.grouped_matmul_slice_m_per_token_dequant_multistage(
        a, b, group_list, scale, per_token_scale
    )
    expected_list = []
    offset = 0
    for i, size in enumerate(group_sizes):
        part = torch.matmul(a[offset:offset+size].float(), b[i].float())
        part = part * scale[i].float() * per_token_scale[offset:offset+size].float().unsqueeze(1)
        expected_list.append(part)
        offset += size
    expected = torch.cat(expected_list, dim=0)

    assert result.shape == (M_total, n)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "npu"
    assert torch.allclose(result.float(), expected, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
