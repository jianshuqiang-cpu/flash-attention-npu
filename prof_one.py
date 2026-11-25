import sys, torch, torch_npu
sys.path.insert(0,'/tmp/opencode/pr466')
from pr_bench2 import build, call
mode, case = sys.argv[1], sys.argv[2]
if case == 'prefill-full':   kw = build(4096,1,4096,8192,(-1,-1),mode)
elif case == 'prefill-swa':  kw = build(4096,1,4096,8192,(511,0),mode)
elif case == 'dec-full':     kw = build(48,12,4,131072,(-1,-1),mode)
else:                        kw = build(48,12,4,131072,(511,0),mode)
for _ in range(3): call(kw)
torch.npu.synchronize(); print("done", mode, case)
