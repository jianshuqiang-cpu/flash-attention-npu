import pytest
import torch
import torch_npu
import torch_catlass


from common import only_on_2201


@only_on_2201
def test_grouped_matmul_slice_k():
    G = 4
    m = 128
    n = 128
    K_total = 256
    group_list = torch.tensor([64, 128, 192, 256], dtype=torch.int64, device="npu")
    group_sizes = [64, 64, 64, 64]

    # Slice-K kernel expects ColumnMajor A. Pass RowMajor A as (K,M) with
    # transA=True: adapter reads dim0=K, dim1=M. Kernel sees A as ColumnMajor
    # (M,K) from the RowMajor (K,M) storage buffer, i.e. A^T.
    a = torch.randn(K_total, m, dtype=torch.float16, device="npu")
    b = torch.randn(K_total, n, dtype=torch.float16, device="npu")

    result = torch_catlass.grouped_matmul_slice_k(a, b, group_list, transA=True)
    expected = []
    offset = 0
    for size in group_sizes:
        part = torch.matmul(a[offset:offset+size, :].T, b[offset:offset+size])
        expected.append(part)
        offset += size
    expected = torch.stack(expected, dim=0)

    assert result.shape == (G, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
