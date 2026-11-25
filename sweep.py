import sys, time, statistics, torch, torch_npu
sys.path.insert(0,'/tmp/opencode/pr466')
from pr_bench2 import build, call
def wall(kw, reps=8):
    for _ in range(3): call(kw)
    torch.npu.synchronize(); s=[]
    for _ in range(reps):
        a=time.perf_counter(); call(kw); torch.npu.synchronize(); s.append((time.perf_counter()-a)*1e3)
    return statistics.median(s)
cases=[('prefill-full',(4096,1,4096),8192,(-1,-1)),
       ('prefill-swa',(4096,1,4096),8192,(511,0)),
       ('decode-full',(48,12,4),131072,(-1,-1)),
       ('decode-swa',(48,12,4),131072,(511,0))]
for name,shp,kv,win in cases:
    base=build(*shp,kv,win,'fp8')
    print(f"== {name} (fp8)", flush=True)
    for ns in (0,1,2,4,8):
        kw=dict(base); kw['num_splits']=ns
        try: print(f"   num_splits={ns}: {wall(kw):7.2f} ms", flush=True)
        except Exception as e: print(f"   num_splits={ns}: FAIL {str(e)[:70]}", flush=True)
