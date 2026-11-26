"""Validate compact physical-page conversion and FA3 prefill on Ascend 950."""

import pytest
import torch
import torch_npu

if not torch_npu.npu.is_available() or "Ascend950" not in torch_npu.npu.get_device_name():
    pytest.skip("FP8 KV staging requires Ascend950", allow_module_level=True)

from flash_attn_npu_3 import flash_attn_with_kvcache, prepare_fp8_kv_staging


def make_inputs(q_heads, kv_heads, layout, window):
    rng = torch.Generator().manual_seed(82)
    q = torch.randn((647, q_heads, 192), generator=rng).to(torch.bfloat16).npu()
    shape = (18, 128, kv_heads, 192)
    ks = (torch.rand(shape[:-1], generator=rng) + 0.25).npu()
    vs = (torch.rand(shape[:-1], generator=rng) + 0.5).npu()
    k, v = (torch.randn(shape, generator=rng).to(torch.float8_e4m3fn) for _ in range(2))
    if layout == "BNBD":
        k, v = (cache.transpose(1, 2).contiguous().npu().transpose(1, 2) for cache in (k, v))
    else:
        k, v = k.npu(), v.npu()
    # A gapped page axis with a nonzero storage offset in both layouts.
    k, v = k[1::2], v[1::2]
    assert k.stride(1) == (192 if layout == "BNBD" else kv_heads * 192)
    # Ragged queries with an empty slot, repeated pages within and across
    # requests, and tail lengths inside a physical page.
    table = torch.tensor(
        [[8, 7, 5, 2, 5, 3, 0], [6, 7, 2, 0, 0, 0, 0], [8, 3, 2, 0, 0, 0, 0]],
        dtype=torch.int32, device="npu",
    )
    table = torch.nn.functional.pad(table, (0, 32), value=-1)
    used = torch.tensor([896, 311, 257], dtype=torch.int32, device="npu")
    prefix = torch.tensor([0, 513, 513, 647], dtype=torch.int32, device="npu")
    args = dict(page_table=table, cache_seqlens=used, cu_seqlens_q=prefix,
                max_seqlen_q=513, window_size=window)
    return q, k, v, ks[1::2].contiguous(), vs[1::2].contiguous(), args


def assert_equal(a, b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


@pytest.mark.parametrize("q_heads,kv_heads,layout,window,softcap", [
    (5, 1, "BBND", (127, 0), 0.0),
    (16, 4, "BBND", (127, 0), 0.0),
    (10, 2, "BNBD", (511, 0), 0.0),
    (8, 8, "BBND", (-1, 0), 2.0),
])
def test_fp8_kv_staging(q_heads, kv_heads, layout, window, softcap):
    torch.set_num_threads(4)
    q, k, v, ks, vs, args = make_inputs(q_heads, kv_heads, layout, window)
    plan = prepare_fp8_kv_staging(q, k, softcap=softcap, **args)
    assert plan._page_table.shape == (3, 7)
    physical = [5, 2, 3, 0, 8] if window == (127, 0) else [8, 7, 5, 2, 3, 0]
    assert plan.num_staged_pages == len(physical)
    assert plan.workspace_bytes == len(physical) * 128 * kv_heads * 192 * 4
    staged = plan._stage(k, v, ks, vs)
    native = []
    for scratch, cache, scale in zip(staged, (k, v), (ks, vs)):
        reference = (cache.cpu().float() * scale.cpu().unsqueeze(-1)).bfloat16()
        assert_equal(scratch, reference[physical])
        native.append(reference.npu())
    expected = flash_attn_with_kvcache(
        q, *native, causal=True, softcap=softcap, return_softmax_lse=True, **args,
    )
    mixed = flash_attn_with_kvcache(
        q, k, v, k_descale=ks, v_descale=vs, causal=True, softcap=softcap, return_softmax_lse=True, **args,
    )
    # The plan owns its remapped table and immutable sequence-length snapshot.
    args["page_table"].fill_(-1)
    args["cache_seqlens"].zero_()
    args["cu_seqlens_q"].zero_()
    actual = plan.forward(q, k, v, k_descale=ks, v_descale=vs, return_softmax_lse=True)
    for actual_tensor, reference, mixed_tensor in zip(actual[:2], expected[:2], mixed[:2]):
        assert_equal(actual_tensor, reference)
        torch.testing.assert_close(actual_tensor.cpu(), mixed_tensor.cpu(), rtol=5e-3, atol=5e-3)
    assert_equal(plan.forward(q, k, v, k_descale=ks, v_descale=vs), expected[0])


def test_fp8_kv_staging_replays_with_new_scales():
    q, k, v, ks, vs, args = make_inputs(5, 1, "BBND", (127, 0))
    plan = prepare_fp8_kv_staging(q, k, **args)
    workspace = plan.allocate_workspace()
    expected = plan.forward(q, k, v, k_descale=ks, v_descale=vs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = plan.forward(q, k, v, k_descale=ks, v_descale=vs, workspace=workspace)
    graph.replay()
    assert_equal(captured, expected)
    vs.mul_(0.75)
    new_expected = plan.forward(q, k, v, k_descale=ks, v_descale=vs)
    graph.replay()
    assert_equal(captured, new_expected)
    assert not torch.equal(expected.cpu(), new_expected.cpu())
    with pytest.raises(RuntimeError, match="separate storage"):
        plan.forward(q, k, v, k_descale=ks, v_descale=vs, workspace=(workspace[0], workspace[0]))


def test_fp8_kv_staging_empty_kv_and_invalid_visible_page():
    q, k, v, ks, vs, args = make_inputs(5, 1, "BBND", (17, 0))
    args["page_table"].fill_(-1)
    with pytest.raises(ValueError, match="Visible physical page"):
        prepare_fp8_kv_staging(q, k, **args)
    args["cache_seqlens"].zero_()
    plan = prepare_fp8_kv_staging(q, k, **args)
    assert plan.num_staged_pages == 0
    out = plan.forward(q, k, v, k_descale=ks, v_descale=vs)
    assert torch.count_nonzero(out.cpu()).item() == 0


@pytest.mark.parametrize("q_len,max_q,window", [
    (1, 1, (511, 0)),
    (4, 128, (511, 0)),
    (16, 16, (-1, 0)),
])
def test_fp8_decode_uses_original_path(q_len, max_q, window, monkeypatch):
    q, k, v, ks, vs, _ = make_inputs(16, 1, "BBND", window)
    q = q[:q_len]
    table = torch.tensor([[8, 3, 7, 1, 5, 4, 2, 0, 6], [0] * 9],
                         dtype=torch.int32, device="npu")
    lengths = torch.tensor([1151, 0], dtype=torch.int32, device="npu")
    prefixes = torch.tensor([0, q_len, q_len], dtype=torch.int32, device="npu")
    options = dict(page_table=table, cache_seqlens=lengths, cu_seqlens_q=prefixes,
                   max_seqlen_q=max_q, window_size=window)

    from flash_attn_npu_3 import fp8_kv_staging

    def unexpected_staging(*args, **kwargs):
        pytest.fail("Decode entered BF16 staging")

    monkeypatch.setattr(fp8_kv_staging, "_compact_page_table", unexpected_staging)
    monkeypatch.setattr(fp8_kv_staging.flash_attn_npu_3_950, "_stage_fp8_kv", unexpected_staging)
    plan = prepare_fp8_kv_staging(q, k, **options)
    assert plan.num_staged_pages == plan.workspace_bytes == 0
    assert plan.allocate_workspace() is None
    assert_equal(plan._page_table, table)
    expected = flash_attn_with_kvcache(
        q, k, v, k_descale=ks, v_descale=vs, causal=True, return_softmax_lse=True, **options,
    )
    actual = plan.forward(q, k, v, k_descale=ks, v_descale=vs, return_softmax_lse=True)
    for result, reference in zip(actual[:2], expected[:2]):
        assert_equal(result, reference)


def test_fp8_staging_above_decode_query_limit():
    q, k, v, ks, vs, _ = make_inputs(5, 1, "BBND", (511, 0))
    q = q[:17]
    args = dict(
        page_table=torch.tensor([[8, 3]], dtype=torch.int32, device="npu"),
        cache_seqlens=torch.tensor([245], dtype=torch.int32, device="npu"),
        cu_seqlens_q=torch.tensor([0, 17], dtype=torch.int32, device="npu"),
        max_seqlen_q=17,
    )
    plan = prepare_fp8_kv_staging(q, k, **args)
    assert plan.num_staged_pages == 2
    workspace = plan.allocate_workspace()
    assert workspace is not None
    reference = flash_attn_with_kvcache(
        q, k, v, k_descale=ks, v_descale=vs, causal=True, window_size=(511, 0), **args,
    )
    torch.testing.assert_close(
        plan.forward(q, k, v, k_descale=ks, v_descale=vs, workspace=workspace).cpu(),
        reference.cpu(), rtol=5e-3, atol=5e-3,
    )
