"""S1/S2 staging boundaries: L1 transfers, L0 subtiles and online updates."""

import pytest
import torch

from tests.test_flash_attn_npu_v3 import test_fa_kvcache_ops as _check_kvcache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_len", [512, 513, 1023, 1024, 1025, 2049])
@pytest.mark.parametrize("mode", ["dense", "causal", "paged-local"])
def test_staged_s2_boundaries(dtype, kv_len, mode):
    paged = mode == "paged-local"
    _check_kvcache(
        dtype, 2, 4, 2, 257, kv_len, 128, int(paged), 128,
        mode == "causal", "TND" if paged else "BSND", paged,
        129 if paged else -1, 65 if paged else -1,
        2.0 if paged else 0.0, 0, False,
    )
