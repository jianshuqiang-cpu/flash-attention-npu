import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_w8a16_matmul():
    """Compare the CATLASS W8A16 matmul wrapper against a reference computation.

    Golden logic (from examples/30_w8a16_matmul/w8a16_matmul.cpp):
    A is fp16, B is int8 dequantized to fp16 via prologue (identity: val * 1.0 + 0.0).
    Then C = A @ B_fp16
    """
    m, n, k = 256, 256, 256

    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="npu")

    result = torch_catlass.w8a16_matmul(
        a, b, "float16", transA=False, transB=False
    )

    a_f32 = a.float()
    b_f32 = b.float()
    expected = torch.matmul(a_f32, b_f32).to(torch.float16).npu()

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    rtol = 1e-1
    atol = 1e-1
    assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
