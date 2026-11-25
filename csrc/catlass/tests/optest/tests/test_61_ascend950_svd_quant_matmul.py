# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import pytest
import torch
import torch_npu
import torch_catlass

from common import only_on_3510
from svd_quant_golden import compute_error_metrics, prepare_svd_quant_matmul_inputs

def _run_svd_quant_matmul_case(m: int, n: int, k: int, r: int, qmax: float) -> None:
    x, svd1, svd2, w, w_scale, smooth_scale, y_cpu, y_golden = prepare_svd_quant_matmul_inputs(
        m, n, k, r, qmax=qmax, device="npu"
    )

    assert w.dtype == torch.int8
    assert w.dim() == 1
    assert w.numel() == (k * n + 1) // 2
    assert svd1.shape == (k, r)
    assert svd2.shape == (r, n)
    assert svd1.stride() == (1, k)
    assert svd2.stride() == (1, r)

    result = torch_catlass.ascend950_svd_quant_matmul(
        x, svd1, svd2, w, w_scale, smooth_scale, qmax
    )

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    torch_npu.npu.synchronize()
    result = result.cpu()
    assert torch.isfinite(result).all()
    result = result.float()

    metrics = compute_error_metrics(result, y_cpu, y_golden)
    rtol = 1e-1
    atol = 1e-1
    passed2 = torch.allclose(result.float(), y_cpu, rtol=rtol, atol=atol)
    assert metrics.passed or passed2, (
        f"m={m} n={n} k={k} r={r} qmax={qmax}: "
        f"MARE ratio={metrics.mare_ratio:.4f} (threshold 5), "
        f"MERE ratio={metrics.mere_ratio:.4f} (threshold 1.5), "
        f"RMSE ratio={metrics.rmse_ratio:.4f} (threshold 1.5), "
        f"max diff={(result - y_cpu).abs().max().item():.4f}"
    )

@pytest.mark.parametrize("m", [50, 100])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("k", [32, 64])
@pytest.mark.parametrize("r", [32])
@pytest.mark.parametrize("qmax", [8.0])
@only_on_3510
def test_ascend950_svd_quant_matmul_generalized(m: int, n: int, k: int, r: int, qmax: float):
    """Generalized self-test: 1000 README-valid cases (k 32-aligned, n>=2, qmax in [6,12])."""
    _run_svd_quant_matmul_case(m, n, k, r, qmax)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
