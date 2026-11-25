#!/usr/bin/env python3
"""FP8(E4M3) KV paged attention — PyTorch 调用样例（Ascend 950 / FA3，PR #466）。

覆盖两种用法：
  acc  : 精度自检。小规模用例与 CPU 高精度参考对比，并给出
         「量化误差（fp8 vs 原始 bf16）」与「实现误差（NPU vs CPU，同一份量化 KV）」。
  perf : 性能自检。三个 step5 典型场景下 fp8 KV 与 bf16 KV 的耗时对比。

接口约束（不满足会直接报错）：
  * Q 必须是 bfloat16（fp16 的 Q 走不了混存路径）；
  * K/V 必须是 torch.float8_e4m3fn，且必须是**分页缓存** (num_blocks, 128, num_kv_heads, 192)；
  * Dqk = Dv = 192，page_size = 128；
  * k_descale / v_descale：FP32、同设备、contiguous，形状 (num_blocks, 128, num_kv_heads)，
    每个元素缩放「一个物理 token × 一个 KV 头」；
  * 不支持 append-KV（边写边算），也不支持 dense 布局的 fp8 KV（用 flash_attn_with_kvcache）。

用法：
  python fp8_kv_attention_demo.py acc
  python fp8_kv_attention_demo.py perf
  python fp8_kv_attention_demo.py all --device npu:0
"""

import argparse
import math
import statistics
import time

import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_with_kvcache

D = 192          # head_dim（QK 与 PV 都是 192）
PS = 128         # page size
QH = 16          # query 头数
KVH = 1          # kv 头数（GQA 16:1，和 step5 一致）
WIN = (511, 0)   # 滑窗 512（step5 FA3 的 SWA 配置）


# --------------------------------------------------------------------------- #
# 造数：分页 KV，同时给出「原始 bf16」和「fp8 + per-token-per-head scale」两份
# --------------------------------------------------------------------------- #
def make_paged(batch, sq, kv, heads_q=QH, heads_kv=KVH, window=WIN, seed=0, device="npu:0"):
    g = torch.Generator().manual_seed(seed)
    pages = (kv + PS - 1) // PS
    q = torch.randn(batch * sq, heads_q, D, generator=g).to(torch.bfloat16)
    k_bf = torch.randn(batch * pages, PS, heads_kv, D, generator=g).to(torch.bfloat16)
    v_bf = torch.randn(batch * pages, PS, heads_kv, D, generator=g).to(torch.bfloat16)

    # per-token-per-head 对称量化：scale = amax / 448（e4m3 的最大值）
    ks = k_bf.float().abs().amax(-1, keepdim=True).clamp_min(1e-6) / 448.0
    vs = v_bf.float().abs().amax(-1, keepdim=True).clamp_min(1e-6) / 448.0
    k_fp8 = (k_bf.float() / ks).to(torch.float8_e4m3fn)
    v_fp8 = (v_bf.float() / vs).to(torch.float8_e4m3fn)
    ks = ks.squeeze(-1).contiguous()      # (num_blocks, 128, kv_heads) fp32
    vs = vs.squeeze(-1).contiguous()

    table = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    cu_q = torch.arange(batch + 1, dtype=torch.int32) * sq
    common = dict(
        cache_seqlens=torch.full((batch,), kv, dtype=torch.int32),
        page_table=table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=sq,
        causal=True,
        window_size=window,
        num_splits=0,                 # 0 = 让 host 选 tiling（mixed KV 强制 host tiling）
        return_softmax_lse=True,
    )

    def to_dev(d):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

    kws_fp8 = to_dev(dict(q=q, k_cache=k_fp8, v_cache=v_fp8,
                          k_descale=ks, v_descale=vs, **common))
    kws_bf16 = to_dev(dict(q=q, k_cache=k_bf, v_cache=v_bf, **common))
    cpu = dict(q=q, k_bf=k_bf, v_bf=v_bf, k_fp8=k_fp8, v_fp8=v_fp8, ks=ks, vs=vs,
               sq=sq, kv=kv, window=window, batch=batch, heads_q=heads_q, heads_kv=heads_kv)
    return kws_fp8, kws_bf16, cpu


def call_a(kwargs):
    out = flash_attn_with_kvcache(**kwargs)
    return (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)


# --------------------------------------------------------------------------- #
# CPU 参考：把 KV 按页展开成 (kv, heads_kv, D)，GQA 复制到 heads_q，做因果 + 滑窗 softmax
# --------------------------------------------------------------------------- #
def cpu_reference(q, k_pages, v_pages, kv, window, heads_kv, causal=True):
    sq, heads_q, _ = q.shape
    k = k_pages.reshape(-1, heads_kv, D)[:kv].repeat_interleave(heads_q // heads_kv, dim=1).float()
    v = v_pages.reshape(-1, heads_kv, D)[:kv].repeat_interleave(heads_q // heads_kv, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) / math.sqrt(D)
    anchor = (kv - sq + torch.arange(sq))[:, None]      # 右对齐因果
    keys = torch.arange(kv)[None, :]
    if causal:
        scores = scores.masked_fill((keys > anchor)[None], -torch.inf)
    if window[0] >= 0:
        scores = scores.masked_fill((keys < anchor - window[0])[None], -torch.inf)
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def rel_err(a, b):
    return float((a - b).abs().mean() / b.abs().mean().clamp_min(1e-9))


# --------------------------------------------------------------------------- #
# acc：精度自检
# --------------------------------------------------------------------------- #
def run_acc(device):
    print("== 精度自检（batch=1, sq=128, kv=1024, 16:1 GQA, 因果 + 滑窗 512）")
    kws_fp8, kws_bf16, c = make_paged(1, 128, 1024, seed=7, device=device)
    o_fp8 = call_a(kws_fp8)[0].float().cpu()
    o_bf16 = call_a(kws_bf16)[0].float().cpu()
    torch.npu.synchronize()

    # 未量化 KV 的 CPU 参考、量化后 KV 的 CPU 参考
    ref_orig = cpu_reference(c["q"], c["k_bf"], c["v_bf"], c["kv"], c["window"], c["heads_kv"])
    k_deq = (c["k_fp8"].float() * c["ks"].unsqueeze(-1)).to(torch.bfloat16)
    v_deq = (c["v_fp8"].float() * c["vs"].unsqueeze(-1)).to(torch.bfloat16)
    ref_quant = cpu_reference(c["q"], k_deq, v_deq, c["kv"], c["window"], c["heads_kv"])

    print(f"  fp8  输出 vs bf16 输出（原始 KV）      : {rel_err(o_fp8, o_bf16):.3e}   <- 量化误差")
    print(f"  fp8  输出 vs CPU 参考（量化后 KV）     : {rel_err(o_fp8, ref_quant):.3e}   <- 实现误差")
    print(f"  bf16 输出 vs CPU 参考（未量化 KV）     : {rel_err(o_bf16, ref_orig):.3e}   <- bf16 自身误差")
    print("  说明：量化误差来自 e4m3 KV（约 3~4%）；实现误差约 0.2%，两者相互独立。")

    # 数值规模参考（确认不是空输出）
    print(f"  输出形状 {tuple(o_fp8.shape)}，平均绝对值 {float(o_fp8.abs().mean()):.3f}")


# --------------------------------------------------------------------------- #
# perf：性能对比
# --------------------------------------------------------------------------- #
def _time(kwargs, reps=6, warmup=3):
    for _ in range(warmup):
        call_a(kwargs)
    torch.npu.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call_a(kwargs)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def run_perf(device):
    cases = [
        ("prefill q4096/kv8192 SWA512", 1, 4096, 8192, WIN),
        ("prefill q4096/kv8192 full", 1, 4096, 8192, (-1, -1)),
        ("decode 12x4/kv8192 SWA512", 12, 4, 8192, WIN),
        ("decode 12x4/kv131072 SWA512", 12, 4, 131072, WIN),
        ("decode 12x4/kv131072 full", 12, 4, 131072, (-1, -1)),
    ]
    print("== 性能对比（交替 A/B，6 轮中位数；同一份逻辑 KV）")
    print(f"{'场景':<30}{'bf16(ms)':>10}{'fp8(ms)':>10}{'fp8/bf16':>10}")
    for name, batch, sq, kv, win in cases:
        kws_fp8, kws_bf16, _ = make_paged(batch, sq, kv, window=win, device=device)
        tb = _time(kws_bf16)
        tf = _time(kws_fp8)
        print(f"{name:<30}{tb:>10.3f}{tf:>10.3f}{tf / tb:>9.2f}x")
    print("（fp8/bf16 < 1 表示 fp8 更快；SWA 场景在启动地板，比值仅作参考）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("acc", "perf", "all"), default="all", nargs="?")
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()
    torch.npu.set_device(args.device)
    if args.mode in ("acc", "all"):
        run_acc(args.device)
    if args.mode in ("perf", "all"):
        if args.mode == "all":
            print()
        run_perf(args.device)


if __name__ == "__main__":
    main()
