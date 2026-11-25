import pytest
import torch
import torch_npu

from common import only_on_2201

pytestmark = [only_on_2201, pytest.mark.skipif(
    torch_npu.npu.device_count() <= 0,
    reason="torch-catlass integration tests require an available Ascend NPU",
)]


def test_group_gemm():
    import torch_catlass

    groupCnt = 3
    m, n = 64, 128
    kList = [64, 128, 192]
    kTotal = sum(kList)
    groupList = torch.tensor(kList, dtype=torch.int64).cumsum(dim=0).to("npu")

    a = torch.randn(kTotal, m, dtype=torch.float16, device="npu")
    b = torch.randn(kTotal, n, dtype=torch.float16, device="npu")

    result = torch_catlass.group_gemm(a, b, groupList, "float16", transA=True, transB=False, formatA=False, formatB=False)

    assert result.shape == (groupCnt, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    offset = 0
    for i in range(groupCnt):
        ki = kList[i]
        a_i = a[offset:offset + ki, :].T
        b_i = b[offset:offset + ki, :]
        expected_i = torch.matmul(a_i, b_i)
        rtol = 1e-2
        atol = 1e-2
        assert torch.allclose(result[i], expected_i, rtol=rtol, atol=atol), (
            f"Group {i} not close: max diff = {(result[i] - expected_i).abs().max().item()}"
        )
        offset += ki


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
