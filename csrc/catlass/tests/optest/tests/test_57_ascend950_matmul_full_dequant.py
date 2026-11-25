import pytest
import torch
import torch_npu


from common import only_on_3510


@pytest.mark.parametrize(
    "x1_quant_mode,x2_quant_mode,has_bias",
    [
        ("per_token", "per_tensor", False),
        ("per_token", "per_channel", False),
        ("per_tensor", "per_channel", False),
        ("default", "per_channel", False),
        ("per_token", "per_tensor", True),
        ("per_token", "per_channel", True),
        ("default", "per_tensor", True),
        ("default", "per_channel", True),
    ],
    ids=[
        "pt_pt_nobias",
        "pt_pc_nobias",
        "pts_pc_nobias",
        "def_pc_nobias",
        "pt_pt_bias",
        "pt_pc_bias",
        "def_pt_bias",
        "def_pc_bias",
    ],
)
@only_on_3510
def test_ascend950_matmul_full_dequant(x1_quant_mode, x2_quant_mode, has_bias):
    """Compare CATLASS Ascend950 matmul full dequant against reference for all quant mode combos."""
    import torch_catlass

    m, n, k = 64, 64, 64

    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="npu")
    b = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="npu")

    x1_scale = None
    if x1_quant_mode == "per_tensor":
        x1_scale = torch.rand(1, dtype=torch.float32, device="npu") * 0.01
    elif x1_quant_mode == "per_token":
        x1_scale = torch.rand(m, dtype=torch.float32, device="npu") * 0.01

    x2_scale = None
    if x2_quant_mode == "per_tensor":
        x2_scale = torch.rand(1, dtype=torch.float32, device="npu") * 0.01
    elif x2_quant_mode == "per_channel":
        x2_scale = torch.rand(n, dtype=torch.float32, device="npu") * 0.01

    bias = None
    if has_bias:
        bias = torch.rand(n, dtype=torch.float32, device="npu") * 0.01

    a_fp32 = a.cpu().float()
    b_fp32 = b.cpu().float()
    expected = torch.matmul(a_fp32, b_fp32)

    if x2_quant_mode == "per_tensor":
        expected = expected * x2_scale.cpu().item()
    elif x2_quant_mode == "per_channel":
        expected = expected * x2_scale.cpu().unsqueeze(0)

    if x1_quant_mode == "per_tensor":
        expected = expected * x1_scale.cpu().item()
    elif x1_quant_mode == "per_token":
        expected = expected * x1_scale.cpu().unsqueeze(1)

    if has_bias:
        expected = expected + bias.cpu().unsqueeze(0)

    expected = expected.half()

    result = torch_catlass.ascend950_matmul_full_dequant(
        a, b, x1_scale, x2_scale, bias,
        "float16", False, False, False, False,
        x1_quant_mode, x2_quant_mode
    )

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.cpu(), expected.cpu(), rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
