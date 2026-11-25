import os, sys, glob, csv, time, statistics, torch, torch_npu
sys.path.insert(0, '/tmp/opencode/pr466')
from flash_attn_npu_3.flash_attn_npu_interface_950 import flash_attn_with_kvcache
D, QH, KVH, PS = 192, 16, 1, 128
WD = '/tmp/opencode/pr466/kprof2'

def build(qtok, batch, sq, kv, window, mode):
    """mode: 'bf16' baseline (no quant) or 'fp8' (KV quantized per token/head)."""
    g = torch.Generator().manual_seed(20261120)
    q = torch.randn(qtok, QH, D, generator=g).to(torch.bfloat16)
    pages = (kv + PS - 1) // PS
    # logical (pre-quantization) KV in bf16
    kref = (torch.randn(batch * pages, PS, KVH, D, generator=g)).to(torch.bfloat16)
    vref = (torch.randn(batch * pages, PS, KVH, D, generator=g)).to(torch.bfloat16)
    table = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    common = dict(cache_seqlens=torch.full((batch,), kv, dtype=torch.int32),
                  page_table=table, cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32) * sq,
                  max_seqlen_q=sq, causal=True, window_size=window, num_splits=0,
                  return_softmax_lse=True)
    if mode == 'bf16':
        kw = dict(q=q, k_cache=kref, v_cache=vref, **common)
    elif mode == 'fp16':
        kw = dict(q=q.to(torch.float16), k_cache=kref.to(torch.float16), v_cache=vref.to(torch.float16), **common)
    else:
        ks = kref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
        vs = vref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
        k8 = (kref.float() / ks).to(torch.float8_e4m3fn)
        v8 = (vref.float() / vs).to(torch.float8_e4m3fn)
        kw = dict(q=q, k_cache=k8, v_cache=v8, k_descale=ks.squeeze(-1).contiguous(),
                  v_descale=vs.squeeze(-1).contiguous(), **common)
    return {k: (v.to('npu:0') if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}

def call(kw):
    out = flash_attn_with_kvcache(**kw)
    return (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)

def wall(kw, reps=8):
    for _ in range(3): call(kw)
    torch.npu.synchronize(); s = []
    for _ in range(reps):
        a = time.perf_counter(); call(kw); torch.npu.synchronize(); s.append((time.perf_counter() - a) * 1e3)
    return statistics.median(s)

def kern(kw, reps=5):
    for _ in range(2): call(kw)
    torch.npu.synchronize()
    for f in glob.glob(WD + '/**/op_summary_*.csv', recursive=True): os.unlink(f)
    os.environ['ASCEND_WORK_PATH'] = WD
    from torch_npu.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as p:
        for _ in range(reps): call(kw)
        torch.npu.synchronize()
    p.export_chrome_trace(WD + '/trace.json')
    tot = 0.0
    for f in glob.glob(WD + '/**/op_summary_*.csv', recursive=True):
        for r in csv.DictReader(open(f)):
            if r.get('Op Name', '').startswith('cann_bench_'): continue
            try: tot += float(r.get('Task Duration(us)') or 0)
            except Exception: pass
    return tot / reps / 1e3

cases = [('prefill q4096/kv8192 SWA512',   (4096, 1, 4096),  8192,  (511, 0)),
         ('prefill q4096/kv8192 full',     (4096, 1, 4096),  8192,  (-1, -1)),
         ('decode 12x4/kv8192 SWA512',     (48, 12, 4),      8192,  (511, 0)),
         ('decode 12x4/kv131072 SWA512',   (48, 12, 4),      131072, (511, 0)),
         ('decode 12x4/kv131072 full',     (48, 12, 4),      131072, (-1, -1))]
print(f"{'case':<32} {'path':<5} {'wall ms':>9} {'kernel ms':>10}")
for name, shp, kv, win in cases:
    try:
        kb = build(*shp, kv, win, 'bf16'); k8 = build(*shp, kv, win, 'fp8'); kh = build(*shp, kv, win, 'fp16')
        ob = call(kb)[0].float(); o8 = call(k8)[0].float(); torch.npu.synchronize()
        rel = float((o8 - ob).abs().mean() / ob.abs().mean().clamp_min(1e-9))
        wh = wall(kh); kh_ = kern(kh)
        wb, w8 = wall(kb), wall(k8); kb_, k8_ = kern(kb), kern(k8)
        print(f"{name:<32} {'bf16':<5} {wb:9.2f} {kb_:10.2f}")
        print(f"{name:<32} {'fp16':<5} {wh:9.2f} {kh_:10.2f}")
        print(f"{name:<32} {'fp8':<5} {w8:9.2f} {k8_:10.2f}   fp8/bf16 wall={w8/wb:4.2f}x kernel={k8_/kb_:4.2f}x | fp8/fp16 wall={w8/wh:4.2f}x kernel={k8_/kh_:4.2f}x  rel_err={rel:.2e}", flush=True)
    except Exception as e:
        print(f"{name:<32} FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
