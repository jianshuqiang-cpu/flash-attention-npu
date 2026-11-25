import pytest
import torch
import torch_npu


from common import only_on_3510


def _ceildiv(a, b):
    return (a + b - 1) // b


@only_on_3510
def test_ascend950_quant_matmul_per_group_per_block_tla():
    """Compare CATLASS Ascend950 quant matmul per-group per-block against reference."""
    import torch_catlass

    m, n, k = 128, 128, 128
    gs = 128
    bs = 128

    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="npu")
    b = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="npu")

    x1_scale = torch.rand(m, _ceildiv(k, gs), dtype=torch.float32, device="npu") * 0.01
    x2_scale = torch.rand(_ceildiv(k, bs), _ceildiv(n, bs), dtype=torch.float32, device="npu") * 0.01

    a_fp32 = a.cpu().float()
    b_fp32 = b.cpu().float()
    x1_cpu = x1_scale.cpu()
    x2_cpu = x2_scale.cpu()

    x1_scale_expanded = x1_cpu.unsqueeze(-1).expand(-1, -1, gs).reshape(m, -1)[:, :k]
    x2_scale_expanded = (
        x2_cpu.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)[:k, :n]
    )
    expected = torch.matmul(a_fp32 * x1_scale_expanded, b_fp32 * x2_scale_expanded).half()

    result = torch_catlass.ascend950_quant_matmul_per_group_per_block_tla(
        a, b, x1_scale, x2_scale, "float16", False, False, False, False
    )

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.cpu(), expected.cpu(), rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
