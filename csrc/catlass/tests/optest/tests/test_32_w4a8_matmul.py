import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


def _pack_int4(data: torch.Tensor) -> torch.Tensor:
    """Pack int8 values into int4 format, matching gen_data.py logic.

    Each pair of int8 values (v0, v1) in row-major order is packed into one byte:
      packed = ((v1 & 0x0F) << 4) | (v0 & 0x0F)
    If n is odd, a zero column is appended before packing.

    Args:
        data: int8 tensor of shape (k, n).

    Returns:
        Packed int8 tensor of shape (k, (n + 1) // 2).
    """
    k, n = data.shape
    if n % 2 != 0:
        data = torch.cat([data, torch.zeros(k, 1, dtype=torch.int8)], dim=1)
    pairs = data.reshape(-1, 2)
    packed = ((pairs[:, 1].int() & 0x0F) << 4) | (pairs[:, 0].int() & 0x0F)
    return packed.to(torch.int8).reshape(k, -1)


@only_on_2201
def test_w4a8_matmul():
    """Compare the CATLASS W4A8 matmul wrapper against a reference computation.

    Golden logic (from examples/32_w4a8_matmul/gen_data.py):
    A is int8, B is int4 (packed as int8 via gen_data_int4), dequantized to int8
    via prologue, then per-tensor scalar dequantization:
        C = 1.5 * (A_int8 @ B_deq_int8)
    Output is fp16.

    The golden uses the original (unpacked) int8 B values (B_orig), NOT the
    packed int4 data passed to the kernel.
    """
    m, n, k = 256, 256, 256

    a = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="npu")
    b_orig = torch.randint(-8, 8, (k, n), dtype=torch.int8)
    b_packed = _pack_int4(b_orig).npu()

    result = torch_catlass.w4a8_matmul(
        a, b_packed, "float16", transA=False, transB=False
    )

    a_f32 = a.float().cpu()
    b_orig_f32 = b_orig.float()
    expected = torch.matmul(a_f32, b_orig_f32) * 1.5
    expected = expected.to(torch.float16).npu()

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
