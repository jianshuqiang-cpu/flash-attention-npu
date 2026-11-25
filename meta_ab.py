import sys, time, statistics, torch, torch_npu
sys.path.insert(0,'/tmp/opencode/pr466')
from pr_bench2 import build
from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata
def md_for(kw, sq, kv):
    cs = kw['cache_seqlens']; cu = kw['cu_seqlens_q']; pt = kw['page_table']
    return get_scheduler_metadata(1 if cu.numel()==2 else int(cu.numel()-1), sq, pt.shape[1]*128, kw['q'].shape[1],
                                  kw['k_cache'].shape[2], 192, cs, qkv_dtype=torch.bfloat16,
                                  kv_dtype=torch.float8_e4m3fn, cu_seqlens_q=cu, page_size=128,
                                  causal=True, window_size=tuple(kw['window_size']), num_splits=0)
def call(kw, md=None):
    if md is not None: kw = dict(kw, scheduler_metadata=md)
    o = flash_attn_with_kvcache(**kw); return o
def wall(kw, md=None, reps=8):
    for _ in range(3): call(kw, md)
    torch.npu.synchronize(); s=[]
    for _ in range(reps):
        a=time.perf_counter(); call(kw, md); torch.npu.synchronize(); s.append((time.perf_counter()-a)*1e3)
    return statistics.median(s)
cases=[('prefill 4096/8192 SWA512',(4096,1,4096),8192,(511,0)),
       ('prefill 4096/8192 full',(4096,1,4096),8192,(-1,-1)),
       ('decode 12x4/8192 SWA512',(48,12,4),8192,(511,0)),
       ('decode 12x4/131072 SWA512',(48,12,4),131072,(511,0)),
       ('decode 12x4/131072 full',(48,12,4),131072,(-1,-1))]
print(f"{'case':<28}{'host tiling':>12}{'+metadata':>11}{'提升':>8}")
for name,shp,kv,win in cases:
    kw=build(*shp,kv,win,'fp8'); md=md_for(kw, shp[2], kv)
    th=wall(kw,None); tm=wall(kw,md)
    print(f"{name:<28}{th:>12.3f}{tm:>11.3f}{th/tm:>7.2f}x", flush=True)
