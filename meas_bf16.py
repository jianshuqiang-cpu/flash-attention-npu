import sys, os, time, statistics, torch, torch_npu
root=sys.argv[1]; sys.path.insert(0, root)
from pr_bench2 import build, kern
from flash_attn_npu_3.flash_attn_npu_interface_950 import flash_attn_with_kvcache
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
    kw=build(*shp,kv,win,'bf16')
    w=wall(kw); k=min(kern(kw),kern(kw))
    print(f"BASE {name:<26} wall={w:6.3f} kern={k:6.3f}", flush=True)
