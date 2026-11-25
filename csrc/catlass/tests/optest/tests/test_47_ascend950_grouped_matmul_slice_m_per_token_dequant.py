import re

import pytest
import torch
import torch_npu
import torch_catlass

from common import only_on_3510

GROUP_SIZES = (128, 128)
M_TOTAL = sum(GROUP_SIZES)
N = 128
K = 256

def _prefix_sum_group_list(group_sizes: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(group_sizes, dtype=torch.int64).cumsum(0).npu()

def _grouped_per_token_dequant_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_sizes: tuple[int, ...],
) -> torch.Tensor:
    expected = []
    offset = 0
    for group_id, group_size in enumerate(group_sizes):
        end = offset + group_size
        group_result = a[offset:end].float() @ b[group_id].float()
        group_result = group_result * scale[group_id].float()
        group_result = group_result * per_token_scale[offset:end].float().unsqueeze(1)
        expected.append(group_result)
        offset = end
    return torch.cat(expected, dim=0)

@only_on_3510
def test_ascend950_grouped_matmul_slice_m_per_token_dequant():
    a = torch.randint(-4, 4, (M_TOTAL, K), dtype=torch.int8)
    b = torch.randint(-4, 4, (len(GROUP_SIZES), K, N), dtype=torch.int8)
    scale = torch.randn(len(GROUP_SIZES), N, dtype=torch.float32).abs() * 0.1
    per_token_scale = torch.randn(M_TOTAL, dtype=torch.float32).abs() * 0.1

    result = torch_catlass.ascend950_grouped_matmul_slice_m_per_token_dequant(
        a.npu(),
        b.npu(),
        _prefix_sum_group_list(GROUP_SIZES),
        scale.npu().reshape(-1),
        per_token_scale.npu(),
    )

    expected = _grouped_per_token_dequant_reference(a, b, scale, per_token_scale, GROUP_SIZES)

    assert result.shape == (M_TOTAL, N)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.float().cpu(), expected, rtol=1e-1, atol=1e-1), (
        f"max diff = {(result.float().cpu() - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
