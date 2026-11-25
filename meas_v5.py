import sys, time, statistics, torch, torch_npu
root=sys.argv[1]; sys.path.insert(0, root)
from pr_bench2 import build, kern
from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata, prepare_fd_metadata
cases=(("prefill 4096/8192 SWA512",(4096,1,4096),8192,(511,0)),
       ("prefill 4096/8192 full",(4096,1,4096),8192,(-1,-1)),
       ("decode 12x4/8192 SWA512",(48,12,4),8192,(511,0)),
       ("decode 12x4/131072 SWA512",(48,12,4),131072,(511,0)),
       ("decode 12x4/32768 full",(48,12,4),32768,(-1,-1)),
       ("decode 12x4/131072 full",(48,12,4),131072,(-1,-1)))
def wall(kw,reps=10):
    f=lambda: flash_attn_with_kvcache(**kw)
    for _ in range(4): f()
    torch.npu.synchronize(); s=[]
    for _ in range(reps):
        t=time.perf_counter(); f(); torch.npu.synchronize(); s.append((time.perf_counter()-t)*1e3)
    return statistics.median(s)
for name,shp,kv,win in cases:
    kwb=build(*shp,kv,win,'bf16'); kwf=build(*shp,kv,win,'fp8')
    cs,cu,pt=kwf['cache_seqlens'],kwf['cu_seqlens_q'],kwf['page_table']
    md=get_scheduler_metadata(int(cs.numel()),shp[2],pt.shape[1]*128,16,1,192,cs,qkv_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,cu_seqlens_q=cu,page_size=128,causal=True,window_size=win,num_splits=0)
    md=prepare_fd_metadata(md, kwf['q'], kwf['k_cache'], kwf['v_cache'], cs, page_table=pt, cu_seqlens_q=cu,
        max_seqlen_q=shp[2], causal=True, window_size=win, return_softmax_lse=True, num_splits=0)
    kw=dict(kwf, scheduler_metadata=md)
    ob=flash_attn_with_kvcache(**kwb)[0]; ov=flash_attn_with_kvcache(**kw)[0]; torch.npu.synchronize()
    rel=float((ob.float()-ov.float()).abs().mean()/ob.float().abs().mean())
    maxd=float((ob.float()-ov.float()).abs().max())
    w=wall(kw); k=min(kern(kw),kern(kw))
    print(f"V5 {name:<26} wall={w:6.3f} kern={k:6.3f} rel={rel:.2e} maxabs={maxd:.2e} fd={getattr(md,'flash_decode',False)}", flush=True)
