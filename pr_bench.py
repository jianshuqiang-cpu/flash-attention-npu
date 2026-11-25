import os, sys, glob, csv, time, statistics, torch, torch_npu
sys.path.insert(0, '/tmp/opencode/pr466')
from flash_attn_npu_3.flash_attn_npu_interface_950 import flash_attn_with_kvcache

D, QH, KVH, PS, WIN = 192, 16, 1, 128, (511, 0)
WD = '/tmp/opencode/pr466/kprof'

def make(shapes, dtype_kv, kv):
    g = torch.Generator().manual_seed(20261120)
    q_tok, batch, sq = shapes
    q = torch.randn(q_tok, QH, D, generator=g).to(torch.bfloat16)
    pages = (kv + PS - 1) // PS
    kf = (torch.randn(batch * pages, PS, KVH, D, generator=g) * 4).to(torch.float8_e4m3fn)
    vf = (torch.randn(batch * pages, PS, KVH, D, generator=g) * 4).to(torch.float8_e4m3fn)
    ks = 0.2 + torch.rand(batch * pages, PS, KVH, generator=g)
    vs = 0.2 + torch.rand(batch * pages, PS, KVH, generator=g)
    kd = (kf.float() * ks[..., None]).to(torch.bfloat16)
    vd = (vf.float() * vs[..., None]).to(torch.bfloat16)
    table = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    cu = torch.arange(batch + 1, dtype=torch.int32) * sq
    common = dict(cache_seqlens=torch.full((batch,), kv, dtype=torch.int32),
                  page_table=table, cu_seqlens_q=cu, max_seqlen_q=sq,
                  causal=True, window_size=WIN, num_splits=0, return_softmax_lse=True)
    if dtype_kv == 'fp8':
        kw = dict(q=q, k_cache=kf, v_cache=vf, k_descale=ks, v_descale=vs, **common)
    elif dtype_kv == 'bf16':
        kw = dict(q=q, k_cache=kd, v_cache=vd, **common)
    else:  # fp16 baseline: same values in fp16
        kw = dict(q=q.to(torch.float16), k_cache=kd.to(torch.float16), v_cache=vd.to(torch.float16), **common)
    return {k: (v.to('npu:0') if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}

def call(kw):
    out = flash_attn_with_kvcache(**kw)
    return (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)

def wall(kw, reps=6):
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

cases = [('prefill q4096/kv8192', (4096, 1, 4096), 8192),
         ('decode 12x4/kv8192', (48, 12, 4), 8192),
         ('decode 12x4/kv131072', (48, 12, 4), 131072)]
print(f"{'case':<24} {'path':<6} {'wall ms':>9} {'kernel ms':>10}")
res = {}
for name, shp, kv in cases:
    kw = {d: make(shp, d, kv) for d in ('fp8', 'bf16', 'fp16')}
    for d in ('fp16', 'bf16', 'fp8'):
        try:
            o = call(kw[d])[0]; torch.npu.synchronize()
            print(f"{name:<24} {d:<6} {wall(kw[d]):9.2f} {kern(kw[d]):10.2f}", flush=True)
        except Exception as e:
            print(f"{name:<24} {d:<6} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
    # accuracy: fp8 vs bf16 reference
    o8 = call(kw['fp8'])[0].float(); ob = call(kw['bf16'])[0].float(); torch.npu.synchronize()
    den = ob.abs().mean().clamp_min(1e-6)
    print(f"{name:<24} acc    rel_err(fp8 vs bf16)={float(((o8-ob).abs().mean()/den)):.3e}", flush=True)
