import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_a2_fp8_e4m3_matmul():
    """Compare the CATLASS FP8 E4M3 matmul wrapper against a reference computation.

    Golden logic (from examples/29_a2_fp8_e4m3_matmul/gen_data.py):
    A and B are FP8 E4M3 values (stored as int8), dequantized to fp16 via
    prologue, then matmul with fp32 accumulator. Reference:
      C = fp16(fp8_A) @ fp16(fp8_B)  in fp32 precision.
    """
    m, n, k = 256, 256, 256

    a_f32 = torch.randn(m, k, dtype=torch.float32)
    b_f32 = torch.randn(k, n, dtype=torch.float32)

    a_fp8 = a_f32.to(torch.float8_e4m3fn)
    b_fp8 = b_f32.to(torch.float8_e4m3fn)

    a_int8 = a_fp8.view(torch.int8).npu()
    b_int8 = b_fp8.view(torch.int8).npu()

    result = torch_catlass.a2_fp8_e4m3_matmul(
        a_int8, b_int8, "float16", transA=False, transB=False
    )

    a_fp16 = a_fp8.to(torch.float16)
    b_fp16 = b_fp8.to(torch.float16)
    expected = torch.matmul(a_fp16.float(), b_fp16.float()).to(torch.float16).npu()

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
