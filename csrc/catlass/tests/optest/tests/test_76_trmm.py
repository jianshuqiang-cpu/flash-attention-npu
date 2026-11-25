# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_npu

from common import only_on_2201

pytestmark = [
    only_on_2201,
    pytest.mark.skipif(
        torch_npu.npu.device_count() <= 0,
        reason="torch-catlass integration tests require an available Ascend NPU",
    ),
]


def _triangular(x, uplo):
    return torch.tril(x) if uplo == 0 else torch.triu(x)


def _rand_triangular(size, dtype, uplo):
    return _triangular(torch.randn(size, size, dtype=dtype), uplo)


@pytest.mark.parametrize(
    "side,uplo,trans,m,n",
    [
        (0, 0, 0, 128, 96),
        (0, 1, 1, 128, 64),
        (1, 0, 0, 96, 128),
        (1, 1, 1, 64, 128),
        (0, 0, 1, 128, 96),
        (0, 1, 0, 128, 64),
        (1, 0, 1, 96, 128),
        (1, 1, 0, 64, 128),
    ],
)
@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_trmm(side, uplo, trans, m, n, alpha):
    import torch_catlass

    dtype = torch.float32

    if side == 0:
        tri_cpu = _rand_triangular(m, dtype, uplo)
        dense_cpu = torch.randn(m, n, dtype=dtype)
        tri = tri_cpu.to("npu")
        dense = dense_cpu.to("npu")
        result = torch_catlass.trmm(tri, dense, dtype, side, uplo, trans, 0, alpha)
        tri_op = tri_cpu.T if trans else tri_cpu
        expected = alpha * torch.matmul(tri_op, dense_cpu)
    else:
        dense_cpu = torch.randn(m, n, dtype=dtype)
        tri_cpu = _rand_triangular(n, dtype, uplo)
        dense = dense_cpu.to("npu")
        tri = tri_cpu.to("npu")
        result = torch_catlass.trmm(dense, tri, dtype, side, uplo, trans, 0, alpha)
        tri_op = tri_cpu.T if trans else tri_cpu
        expected = alpha * torch.matmul(dense_cpu, tri_op)

    assert result.shape == (m, n)
    assert result.dtype == dtype
    assert result.device.type == "npu"
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, expected, rtol=1e-2, atol=1e-2), (
        f"Results not close: max diff = {(result_cpu - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
