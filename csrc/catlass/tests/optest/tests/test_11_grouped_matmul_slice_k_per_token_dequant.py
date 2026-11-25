import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_grouped_matmul_slice_k_per_token_dequant():
    G = 4
    m = 128
    n = 128
    K_total = 256
    group_list = torch.tensor([64, 128, 192, 256], dtype=torch.int64, device="npu")
    group_sizes = [64, 64, 64, 64]

    a = torch.randint(-8, 8, (K_total, m), dtype=torch.int8, device="npu")
    b = torch.randint(-8, 8, (K_total, n), dtype=torch.int8, device="npu")
    scale = torch.randn(G, n, dtype=torch.bfloat16, device="npu").abs() * 0.1
    per_token_scale = torch.randn(G, m, dtype=torch.bfloat16, device="npu").abs() * 0.1

    result = torch_catlass.grouped_matmul_slice_k_per_token_dequant(
        a, b, group_list, scale, per_token_scale, transA=True
    )
    expected_list = []
    offset = 0
    for i, size in enumerate(group_sizes):
        part = torch.matmul(a[offset:offset+size].T.float(), b[offset:offset+size].float())
        part = part * scale[i].float() * per_token_scale[i].float().unsqueeze(1)
        expected_list.append(part)
        offset += size
    expected = torch.stack(expected_list, dim=0)

    assert result.shape == (G, m, n)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "npu"
    assert torch.allclose(result.float(), expected, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
