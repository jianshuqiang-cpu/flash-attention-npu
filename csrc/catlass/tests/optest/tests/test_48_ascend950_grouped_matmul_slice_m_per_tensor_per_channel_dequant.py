import re
import struct

import pytest
import torch
import torch_npu
import torch_catlass

from common import only_on_3510

GROUP_SIZES = (128, 128)
M_TOTAL = sum(GROUP_SIZES)
N = 128
K = 256
SCALE_PER_TENSOR = 0.5

def _fp32_to_uint64_tensor(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Cast fp32 scale values to uint64 representation (fp32 bits zero-extended)."""
    buf = scale_fp32.float().cpu().numpy().tobytes()
    uint64_vals = []
    for i in range(0, len(buf), 4):
        fp32_bits = struct.unpack("<I", buf[i : i + 4])[0]
        uint64_vals.append(fp32_bits)
    return torch.tensor(uint64_vals, dtype=torch.int64, device=scale_fp32.device)


def _prefix_sum_group_list(group_sizes: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(group_sizes, dtype=torch.int64).cumsum(0).npu()


def _grouped_dequant_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    group_sizes: tuple[int, ...],
    quant_mode: int,
    per_channel_scale: torch.Tensor,
) -> torch.Tensor:
    expected = []
    offset = 0
    for group_id, group_size in enumerate(group_sizes):
        end = offset + group_size
        group_result = a[offset:end].float() @ b[group_id].float()
        if quant_mode == 0:
            group_result = group_result * SCALE_PER_TENSOR
        else:
            group_result = group_result * per_channel_scale.float().unsqueeze(0)
        expected.append(group_result)
        offset = end
    return torch.cat(expected, dim=0)

@only_on_3510
@pytest.mark.parametrize("quant_mode", [0, 1], ids=["per_tensor", "per_channel"])
def test_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant(quant_mode):
    a = torch.randint(-4, 4, (M_TOTAL, K), dtype=torch.int8)
    b = torch.randint(-4, 4, (len(GROUP_SIZES), K, N), dtype=torch.int8)

    per_channel_scale_fp32 = torch.randn(N, dtype=torch.float32).abs() * 0.1
    per_channel_scale = _fp32_to_uint64_tensor(per_channel_scale_fp32).npu()

    result = torch_catlass.ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant(
        a.npu(),
        b.npu(),
        _prefix_sum_group_list(GROUP_SIZES),
        per_channel_scale,
        scalePerTensor=SCALE_PER_TENSOR,
        quantMode=quant_mode,
    )

    expected = _grouped_dequant_reference(
        a, b, GROUP_SIZES, quant_mode, per_channel_scale_fp32
    )

    assert result.shape == (M_TOTAL, N)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.float().cpu(), expected, rtol=1e-1, atol=1e-1), (
        f"quant_mode={quant_mode}: max diff = {(result.float().cpu() - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
