import re

import pytest
import torch
import torch_npu
import torch_catlass

from common import only_on_3510

GROUP_SIZES = (128, 128, 128, 128)
M_TOTAL = sum(GROUP_SIZES)
N = 256
K = 256

def _prefix_sum_group_list(group_sizes: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(group_sizes, dtype=torch.int64).cumsum(0).npu()

def _grouped_matmul_reference(
    a: torch.Tensor, b: torch.Tensor, group_sizes: tuple[int, ...]
) -> torch.Tensor:
    expected = []
    offset = 0
    for group_id, group_size in enumerate(group_sizes):
        end = offset + group_size
        expected.append(a[offset:end].float() @ b[group_id].float())
        offset = end
    return torch.cat(expected, dim=0)

@only_on_3510
def test_ascend950_grouped_matmul_slice_m():
    a = torch.randn(M_TOTAL, K, dtype=torch.float16)
    b = torch.randn(len(GROUP_SIZES), K, N, dtype=torch.float16)

    result = torch_catlass.ascend950_grouped_matmul_slice_m(
        a.npu(), b.npu(), _prefix_sum_group_list(GROUP_SIZES)
    )
    expected = _grouped_matmul_reference(a, b, GROUP_SIZES)

    assert result.shape == (M_TOTAL, N)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.cpu().float(), expected, rtol=1e-2, atol=1e-2), (
        f"max diff = {(result.cpu().float() - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
