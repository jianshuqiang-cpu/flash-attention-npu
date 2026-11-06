"""Q tile and UB lifetime regressions; use ASCEND_RT_VISIBLE_DEVICES to isolate a device."""
import pytest
import torch
import torch_npu

from tests.common.compare import assert_fa_close


@pytest.fixture(autouse=True)
def require_950():
    if not torch.npu.is_available() or 'Ascend950' not in torch.npu.get_device_name():
        pytest.skip('requires Ascend950')
    torch.manual_seed(950128)
    torch.set_num_threads(8)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mask', ['none', 'causal', 'swa'])
@pytest.mark.parametrize('nq,d,dv,page,heads,nk', [
    (63, 128, 160, 128, 4, 513), (64, 192, 192, 256, 4, 513),
    (65, 128, 192, 128, 4, 513), (127, 192, 256, 512, 4, 513),
    (128, 256, 192, 128, 4, 513), (129, 256, 256, 1024, 4, 513),
    (17, 160, 192, 128, 16, 513), (3, 192, 192, 128, 16, 513),
    (129, 128, 256, 128, 4, 33), (257, 160, 192, 128, 4, 17),
    (65, 192, 128, 256, 4, 513), (129, 128, 128, 512, 4, 513),
    (256, 128, 160, 256, 64, 1025), (256, 192, 192, 512, 64, 1025),
    (256, 256, 256, 1024, 64, 1025), (129, 128, 192, 128, 128, 513),
    (128, 128, 256, 256, 128, 513),
    (513, 128, 256, 512, 64, 257),
])
def test_q128_paged_reference(dtype, mask, nq, d, dv, page, heads, nk):
    from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata
    kvheads = 2
    npages = (nk + page - 1) // page
    q_cpu = torch.randn(nq, heads, d, dtype=dtype)
    k_cpu = torch.randn(npages, page, kvheads, d, dtype=dtype)
    v_cpu = torch.randn(npages, page, kvheads, dv, dtype=dtype)
    # Reverse page mapping, independently padded physical K/V, nonzero offset.
    ks = torch.empty(npages * 2, page, kvheads, d, dtype=dtype, device='npu')
    vs = torch.empty(npages * 3, page, kvheads, dv, dtype=dtype, device='npu')
    k, v = ks[::2], vs[1::3]
    k.copy_(k_cpu.flip(0)); v.copy_(v_cpu.flip(0))
    table = torch.arange(npages - 1, -1, -1, dtype=torch.int32, device='npu')[None]
    lengths = torch.tensor([nk], dtype=torch.int32, device='npu')
    cu = torch.tensor([0, nq], dtype=torch.int32, device='npu')
    q = q_cpu.npu()
    window = (173, 19) if mask == 'swa' else (-1, -1)
    if mask == 'swa' and heads >= 64:
        window = (641, 63)
    metadata = get_scheduler_metadata(
        batch_size=1, max_seqlen_q=nq, max_seqlen_k=nk,
        num_heads_q=heads, num_heads_kv=kvheads, headdim=d, headdim_v=dv,
        cache_seqlens=lengths, qkv_dtype=dtype, cu_seqlens_q=cu,
        page_size=page, causal=mask == 'causal', window_size=window, num_splits=1)
    kwargs = dict(cache_seqlens=lengths, page_table=table, cu_seqlens_q=cu,
                  max_seqlen_q=nq, causal=mask == 'causal', window_size=window,
                  scheduler_metadata=metadata, num_splits=1, return_softmax_lse=True)
    out, lse, *_ = flash_attn_with_kvcache(q, k, v, **kwargs)
    out_ref = out.clone()
    for _ in range(8):
        repeated, *_ = flash_attn_with_kvcache(q, k, v, **kwargs)
        torch.testing.assert_close(repeated, out_ref, atol=0, rtol=0)
    kk = k_cpu.reshape(-1, kvheads, d)[:nk].repeat_interleave(heads // kvheads, dim=1).permute(1, 0, 2)
    vv = v_cpu.reshape(-1, kvheads, dv)[:nk].repeat_interleave(heads // kvheads, dim=1).permute(1, 0, 2)
    qq = q_cpu.permute(1, 0, 2)
    row = torch.arange(nq)[:, None] + nk - nq
    col = torch.arange(nk)[None]
    invalid = torch.zeros(nq, nk, dtype=torch.bool)
    if mask == 'causal':
        invalid = col > row
    elif mask == 'swa':
        invalid = (col < row - window[0]) | (col > row + window[1])
    scores = (qq.double() @ kk.double().transpose(-1, -2)) * d ** -.5
    scores.masked_fill_(invalid, -torch.inf)
    ref = (scores.softmax(-1).nan_to_num() @ vv.double()).permute(1, 0, 2)
    score_pt = (qq.float() @ kk.float().transpose(-1, -2)).to(dtype).float() * d ** -.5
    score_pt.masked_fill_(invalid, -torch.inf)
    pt = (score_pt.softmax(-1).nan_to_num().to(dtype).float() @ vv.float()).to(dtype).permute(1, 0, 2)
    assert_fa_close(out, ref, pt, name='q128 output')
    expected_lse = scores.logsumexp(-1)
    expected_lse.masked_fill_(invalid.all(-1), torch.inf)
    torch.testing.assert_close(lse.cpu().double(), expected_lse, atol=3e-3, rtol=3e-3)
