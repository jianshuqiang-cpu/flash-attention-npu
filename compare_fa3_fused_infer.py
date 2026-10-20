"""
FA3 flash_attn_with_kvcache vs torch_npu.npu_fused_infer_attention_score

覆盖 causal prefill / causal decode / 非 mask，三种场景均为 BSND+TND。
采集口径对齐 b.py：warmup 2 次，第 3 次迭代侵入式 profiler，解析 kernel_details.csv Duration。
另用 NPU Event 对 FA3 / fused 分别采样，取端到端耗时中位数。

默认从 HBM 装得下的全笛卡尔积里按 FLOPs 均衡抽 50 组；--full 跑全部可装下的组合。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import sys
from statistics import mean, median

script_dir = sys.path.pop(0)
import torch
import torch_npu

_FA_ROOT = os.path.dirname(os.path.abspath(__file__))
if _FA_ROOT not in sys.path:
    sys.path.insert(0, _FA_ROOT)

flash_attn_with_kvcache = None
analyse = None


def _load_runtime():
    global flash_attn_with_kvcache, analyse
    if flash_attn_with_kvcache is not None:
        return
    from torch_npu.profiler.profiler import analyse as _analyse
    from flash_attn_npu_3 import flash_attn_with_kvcache as _fa3
    flash_attn_with_kvcache = _fa3
    analyse = _analyse

K = 1024
M = 1024 * 1024
DTYPE = torch.bfloat16
TARGET_ITER = 3
E2E_WARMUP = 5
E2E_ITERS = 10
NUM_CASES = 50
PROFILE_ROOT = "./batchtest/profile_fa3_fused_infer"
RESULT_CSV = "./batchtest/compare_fa3_fused_infer.csv"
ATTN_MASK_SIZE = 2048
SPARSE_MODE_CAUSAL = 3
HBM_RATIO = 0.45
ELEM_BYTES = 2

AUX_PAT = re.compile(r"Fill|ReduceSum|Cast", re.I)
FA3_PAT = re.compile(r"SplitFuse|FAInfer|FACombine|FlashDecode", re.I)
FUSED_PAT = re.compile(
    r"FusedInferAttentionScore|PromptFlashAttention|IncreFlashAttention|"
    r"IFAKernel|PFAKernel|FusedInfer",
    re.I,
)

B_PREFILL = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 256, 512, 1024]
B_OTHER = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96]
GQA_CAUSAL = [(32, 2), (20, 20), (16, 2), (8, 1), (5, 1), (4, 1)]
GQA_NONMASK = [(40, 40), (24, 24), (6, 6), (5, 5), (4, 4), (8, 1)]
S_EQUAL = [128, 256, 512, 1024, 2048, 4096, 16 * K, 32 * K, 64 * K, 128 * K, 256 * K, M]
QS_DECODE = [1, 2, 3, 4, 8]
D_CAUSAL = [128, 256]
D_NONMASK = [64, 128]
NONMASK_PAIRS = [
    ("qS1", 2 * K, 16 * K),
    ("qS1", 4 * K, 32 * K),
    ("qS1", 8 * K, 64 * K),
    ("qS1", 16 * K, 128 * K),
    ("qS1", 32 * K, 256 * K),
    ("qS1", 64 * K, M),
    ("qS2", 8 * K, 16 * K),
    ("qS2", 16 * K, 32 * K),
    ("qS2", 32 * K, 64 * K),
    ("qS2", 64 * K, 128 * K),
    ("qS2", 128 * K, 256 * K),
    ("qS2", 512 * K, M),
    ("qS3", 16 * K, 16 * K),
    ("qS3", 32 * K, 32 * K),
    ("qS3", 64 * K, 64 * K),
    ("qS3", 128 * K, 128 * K),
    ("qS3", 256 * K, 256 * K),
    ("qS3", M, M),
]


def parse_len(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip().lower().replace("_", "")
    if text.endswith("m"):
        return int(float(text[:-1]) * M)
    if text.endswith("k"):
        return int(float(text[:-1]) * K)
    return int(text)


def fmt_len(n: int) -> str:
    if n >= M and n % M == 0:
        return f"{n // M}m"
    if n >= K and n % K == 0:
        return f"{n // K}k"
    return str(n)


def profile_dirs(profile_dir: str) -> set[str]:
    return {
        path
        for path in glob.glob(os.path.join(profile_dir, "*_ascend_pt"))
        if os.path.isdir(path)
    }


def tensor_bytes(batch: int, nheads: int, nkv: int, q_s: int, kv_s: int, head_dim: int) -> int:
    """Q + K + V + O, bf16/fp16."""
    q_elems = batch * q_s * nheads * head_dim
    kv_elems = batch * kv_s * nkv * head_dim
    return ELEM_BYTES * (2 * q_elems + 2 * kv_elems)


def flops(batch: int, nheads: int, q_s: int, kv_s: int, head_dim: int, causal: bool) -> int:
    """QK + PV matmul FLOPs；causal prefill 按下三角约一半。"""
    qk_pv = 4 * batch * nheads * q_s * kv_s * head_dim
    if causal and q_s == kv_s:
        return qk_pv // 2
    return qk_pv


def make_case(
    scene: str,
    layout: str,
    causal: bool,
    batch: int,
    nheads: int,
    nkv: int,
    q_s: int,
    kv_s: int,
    head_dim: int,
    family: str = "",
) -> dict:
    scale = head_dim ** -0.5
    name = (
        f"{scene}_{layout}_B{batch}_N{nheads}x{nkv}_"
        f"q{fmt_len(q_s)}_kv{fmt_len(kv_s)}_d{head_dim}"
    )
    return {
        "scene": scene,
        "layout": layout,
        "causal": causal,
        "batch": batch,
        "nheads": nheads,
        "nkv": nkv,
        "q_s": q_s,
        "kv_s": kv_s,
        "head_dim": head_dim,
        "pair_family": family,
        "scale": scale,
        "name": name,
        "total_q": batch * q_s,
        "total_kv": batch * kv_s,
        "bytes": tensor_bytes(batch, nheads, nkv, q_s, kv_s, head_dim),
        "flops": flops(batch, nheads, q_s, kv_s, head_dim, causal),
    }


def iter_all_cases(scenes: set[str]) -> list[dict]:
    cases = []
    if "causal_prefill" in scenes:
        for batch in B_PREFILL:
            for nheads, nkv in GQA_CAUSAL:
                for seq in S_EQUAL:
                    for head_dim in D_CAUSAL:
                        for layout in ("BSND", "TND"):
                            cases.append(
                                make_case(
                                    "causal_prefill", layout, True,
                                    batch, nheads, nkv, seq, seq, head_dim,
                                )
                            )
    if "causal_decode" in scenes:
        for batch in B_OTHER:
            for nheads, nkv in GQA_CAUSAL:
                for q_s in QS_DECODE:
                    for kv_s in S_EQUAL:
                        for head_dim in D_CAUSAL:
                            for layout in ("BSND", "TND"):
                                cases.append(
                                    make_case(
                                        "causal_decode", layout, True,
                                        batch, nheads, nkv, q_s, kv_s, head_dim,
                                    )
                                )
    if "nonmask" in scenes:
        for batch in B_OTHER:
            for nheads, nkv in GQA_NONMASK:
                for family, q_s, kv_s in NONMASK_PAIRS:
                    for head_dim in D_NONMASK:
                        for layout in ("BSND", "TND"):
                            cases.append(
                                make_case(
                                    "nonmask", layout, False,
                                    batch, nheads, nkv, q_s, kv_s, head_dim, family,
                                )
                            )
    return cases


def device_hbm_bytes(device: str) -> int:
    idx = 0
    if ":" in device:
        idx = int(device.split(":")[-1])
    return int(torch.npu.get_device_properties(idx).total_memory)


def filter_hbm(cases: list[dict], limit_bytes: int) -> tuple[list[dict], int]:
    kept = [c for c in cases if c["bytes"] <= limit_bytes]
    return kept, len(cases) - len(kept)


def _pick_even(pool: list[dict], want: int) -> list[dict]:
    if want <= 0 or not pool:
        return []
    pool = sorted(pool, key=lambda c: c["flops"])
    want = min(want, len(pool))
    if want == 1:
        indices = [len(pool) // 2]
    else:
        indices = [int(round(j * (len(pool) - 1) / (want - 1))) for j in range(want)]
    seen = set()
    picked = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            picked.append(pool[idx])
    fill = 0
    while len(picked) < want and fill < len(pool):
        if fill not in seen:
            picked.append(pool[fill])
            seen.add(fill)
        fill += 1
    return picked


def sample_cases(cases: list[dict], n: int) -> list[dict]:
    if n <= 0 or n >= len(cases):
        return list(cases)
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for case in cases:
        by_bucket.setdefault((case["scene"], case["layout"]), []).append(case)
    buckets = list(by_bucket.keys())
    if not buckets:
        return []
    base, extra = divmod(n, len(buckets))
    picked = []
    for i, key in enumerate(buckets):
        want = base + (1 if i < extra else 0)
        picked.extend(_pick_even(by_bucket[key], want))
    return picked[:n]


def prefix_sums(length: int, batch: int) -> list[int]:
    return [length * (i + 1) for i in range(batch)]


def make_atten_mask(device: str) -> torch.Tensor:
    return torch.triu(
        torch.ones(ATTN_MASK_SIZE, ATTN_MASK_SIZE, dtype=torch.bool, device=device),
        diagonal=1,
    )


def select_fused_mask(case: dict, default_mask: torch.Tensor) -> torch.Tensor | None:
    if not case["causal"]:
        return None
    # qS=1 且 kvS>2048：
    #   BSND 走 IFA，压缩 2048 mask 非法，必须不传 mask（sparse_mode=3 仍生效）
    #   TND 走 split-fuse，必须传 [2048,2048] 压缩 mask
    if case["q_s"] == 1 and case["kv_s"] > ATTN_MASK_SIZE and case["layout"] == "BSND":
        return None
    return default_mask


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)


def build_inputs(case: dict, device: str, input_seed: int | None = None) -> dict:
    gen = None
    if input_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(input_seed)
    batch, nheads, nkv = case["batch"], case["nheads"], case["nkv"]
    q_s, kv_s, head_dim = case["q_s"], case["kv_s"], case["head_dim"]
    layout = case["layout"]
    kw = dict(dtype=DTYPE, device=device, generator=gen)
    if layout == "BSND":
        q = torch.randn(batch, q_s, nheads, head_dim, **kw)
        k = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        cache_seqlens = torch.full((batch,), kv_s, dtype=torch.int32, device=device)
        cu_q = None
        max_q = None
        actual_q = [q_s] * batch
        actual_kv = [kv_s] * batch
    else:
        q = torch.randn(batch * q_s, nheads, head_dim, **kw)
        k = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        cache_seqlens = torch.full((batch,), kv_s, dtype=torch.int32, device=device)
        cu_q = torch.tensor([0] + prefix_sums(q_s, batch), dtype=torch.int32, device=device)
        max_q = q_s
        actual_q = prefix_sums(q_s, batch)
        actual_kv = prefix_sums(kv_s, batch)
    return {
        "q": q,
        "k": k,
        "v": v,
        "cache_seqlens": cache_seqlens,
        "cu_q": cu_q,
        "max_q": max_q,
        "actual_q": actual_q,
        "actual_kv": actual_kv,
    }


def run_fa3(inp: dict, case: dict) -> None:
    _load_runtime()
    kwargs = dict(
        q=inp["q"],
        k_cache=inp["k"],
        v_cache=inp["v"],
        cache_seqlens=inp["cache_seqlens"],
        softmax_scale=case["scale"],
        causal=case["causal"],
        window_size=(-1, -1),
        num_splits=0,
    )
    if case["layout"] == "TND":
        kwargs["cu_seqlens_q"] = inp["cu_q"]
        kwargs["max_seqlen_q"] = inp["max_q"]
    flash_attn_with_kvcache(**kwargs)


def run_fused(inp: dict, case: dict, atten_mask: torch.Tensor) -> None:
    torch_npu.npu_fused_infer_attention_score(
        inp["q"],
        inp["k"],
        inp["v"],
        num_heads=case["nheads"],
        num_key_value_heads=case["nkv"],
        input_layout=case["layout"],
        scale=case["scale"],
        pre_tokens=2147483647,
        next_tokens=0 if case["causal"] else 2147483647,
        sparse_mode=SPARSE_MODE_CAUSAL if case["causal"] else 0,
        inner_precise=0,
        atten_mask=select_fused_mask(case, atten_mask),
        actual_seq_lengths=inp["actual_q"],
        actual_seq_lengths_kv=inp["actual_kv"],
    )


def _row_ms(row: dict) -> float:
    return float(row["Duration(us)"]) / 1000.0


def parse_iter3_ms(ascend_pt: str) -> tuple[float | None, float | None]:
    csv_paths = glob.glob(os.path.join(ascend_pt, "**/kernel_details.csv"), recursive=True)
    if not csv_paths:
        raise FileNotFoundError(f"kernel_details.csv not found under current profile: {ascend_pt}")
    csv_path = max(csv_paths, key=os.path.getmtime)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fa3_ms = fused_ms = 0.0
    fa3_hit = fused_hit = False
    for row in rows:
        name, typ = row.get("Name", ""), row.get("Type", "")
        if AUX_PAT.search(name) or AUX_PAT.search(typ):
            continue
        blob = f"{name} {typ}"
        if FA3_PAT.search(blob):
            fa3_ms += _row_ms(row)
            fa3_hit = True
        elif FUSED_PAT.search(blob):
            fused_ms += _row_ms(row)
            fused_hit = True
    return (fa3_ms if fa3_hit else None, fused_ms if fused_hit else None)


def e2e_median_ms(fn, warmup: int, iters: int) -> float:
    """NPU Event 端到端耗时中位数，单位 ms。warmup 与计时迭代都在 device 同步之后。"""
    torch.npu.synchronize()
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def run_case(
    case: dict,
    device: str,
    profile_root: str,
    atten_mask: torch.Tensor,
    input_seed: int | None = None,
    e2e_warmup: int = E2E_WARMUP,
    e2e_iters: int = E2E_ITERS,
) -> dict:
    inp = build_inputs(case, device, input_seed=input_seed)
    fa3_e2e_ms = fused_e2e_ms = None
    if e2e_iters > 0:
        fa3_e2e_ms = e2e_median_ms(lambda: run_fa3(inp, case), e2e_warmup, e2e_iters)
        fused_e2e_ms = e2e_median_ms(
            lambda: run_fused(inp, case, atten_mask), e2e_warmup, e2e_iters,
        )
    e2e_delta_ms = (
        fa3_e2e_ms - fused_e2e_ms
        if fa3_e2e_ms is not None and fused_e2e_ms is not None
        else None
    )
    e2e_ratio = fa3_e2e_ms / fused_e2e_ms if fa3_e2e_ms and fused_e2e_ms else None
    e2e_degrade_pct = (e2e_ratio - 1.0) * 100 if e2e_ratio else None
    prof_dir = os.path.join(profile_root, case["label"])
    os.makedirs(prof_dir, exist_ok=True)
    profiles_before = profile_dirs(prof_dir)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
        l2_cache=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir, analyse_flag=False),
    )

    for step in range(1, TARGET_ITER + 2):
        if step == TARGET_ITER:
            prof.start()
            run_fa3(inp, case)
            run_fused(inp, case, atten_mask)
            prof.step()
            prof.stop()
            break
        run_fa3(inp, case)
        run_fused(inp, case, atten_mask)

    profiles_after = profile_dirs(prof_dir)
    new_profiles = profiles_after - profiles_before
    if len(new_profiles) != 1:
        raise RuntimeError(
            f"Expected exactly one profile generated for {case['label']}, "
            f"found {len(new_profiles)}: {sorted(new_profiles)}"
        )
    ascend_pt = new_profiles.pop()
    _load_runtime()
    analyse(profiler_path=ascend_pt, max_process_number=16, export_type="text")
    fa3_ms, fused_ms = parse_iter3_ms(ascend_pt)
    delta_ms = fa3_ms - fused_ms if fa3_ms is not None and fused_ms is not None else None
    ratio = fa3_ms / fused_ms if fa3_ms and fused_ms else None
    degrade_pct = (ratio - 1.0) * 100 if ratio else None
    return {
        **case,
        "fa3_ms": fa3_ms,
        "fused_ms": fused_ms,
        "delta_ms": delta_ms,
        "ratio": ratio,
        "degrade_pct": degrade_pct,
        "fa3_e2e_ms": fa3_e2e_ms,
        "fused_e2e_ms": fused_e2e_ms,
        "e2e_delta_ms": e2e_delta_ms,
        "e2e_ratio": e2e_ratio,
        "e2e_degrade_pct": e2e_degrade_pct,
    }


def analyze_results(results: list[dict]) -> None:
    ok = [r for r in results if r.get("fa3_ms") and r.get("fused_ms")]
    e2e_ok = [r for r in results if r.get("fa3_e2e_ms") and r.get("fused_e2e_ms")]
    if not ok and not e2e_ok:
        print("无有效结果")
        return

    def stat(vals):
        return mean(vals), median(vals), min(vals), max(vals)

    if ok:
        ratios = [r["ratio"] for r in ok]
        deltas = [r["delta_ms"] for r in ok]
        r_mean, r_med, r_min, r_max = stat(ratios)
        d_mean, d_med, d_min, d_max = stat(deltas)

        print("\n" + "=" * 96)
        print("FA3 flash_attn_with_kvcache 相对 npu_fused_infer_attention_score")
        print("第 3 次迭代 kernel Duration 合计")
        print("=" * 96)
        print(f"有效用例: {len(ok)}/{len(results)}")
        print(f"ratio  (fa3/fused): mean={r_mean:.3f}x  median={r_med:.3f}x  min={r_min:.3f}x  max={r_max:.3f}x")
        print(f"delta  (fa3-fused): mean={d_mean:.3f}ms median={d_med:.3f}ms min={d_min:.3f}ms max={d_max:.3f}ms")
        print(f"劣化百分比 (ratio-1): mean={(r_mean - 1) * 100:.1f}%  median={(r_med - 1) * 100:.1f}%")

        for scene in ("causal_prefill", "causal_decode", "nonmask"):
            sub = [r for r in ok if r["scene"] == scene]
            if not sub:
                continue
            br = [r["ratio"] for r in sub]
            bd = [r["delta_ms"] for r in sub]
            print(
                f"\n[{scene}] n={len(sub)} "
                f"ratio mean={mean(br):.3f}x median={median(br):.3f}x | "
                f"delta mean={mean(bd):.3f}ms median={median(bd):.3f}ms"
            )

        worst = sorted(ok, key=lambda r: r["ratio"], reverse=True)[:5]
        best = sorted(ok, key=lambda r: r["ratio"])[:5]
        print("\nFA3 相对最慢 TOP5 (ratio):")
        for r in worst:
            print(
                f"  {r['label']} {r['name']} fa3={r['fa3_ms']:.3f} fused={r['fused_ms']:.3f} ratio={r['ratio']:.3f}x"
            )
        print("\nFA3 相对最快 TOP5 (ratio):")
        for r in best:
            print(
                f"  {r['label']} {r['name']} fa3={r['fa3_ms']:.3f} fused={r['fused_ms']:.3f} ratio={r['ratio']:.3f}x"
            )

    if e2e_ok:
        e2e_ratios = [r["e2e_ratio"] for r in e2e_ok]
        e2e_deltas = [r["e2e_delta_ms"] for r in e2e_ok]
        er_mean, er_med, er_min, er_max = stat(e2e_ratios)
        ed_mean, ed_med, ed_min, ed_max = stat(e2e_deltas)
        print("\n" + "=" * 96)
        print("端到端 NPU Event 中位数（warmup 后逐次同步计时）")
        print("=" * 96)
        print(f"有效用例: {len(e2e_ok)}/{len(results)}")
        print(
            f"e2e ratio  (fa3/fused): mean={er_mean:.3f}x  median={er_med:.3f}x  "
            f"min={er_min:.3f}x  max={er_max:.3f}x"
        )
        print(
            f"e2e delta  (fa3-fused): mean={ed_mean:.3f}ms median={ed_med:.3f}ms "
            f"min={ed_min:.3f}ms max={ed_max:.3f}ms"
        )
        print(f"劣化百分比 (ratio-1): mean={(er_mean - 1) * 100:.1f}%  median={(er_med - 1) * 100:.1f}%")
        for scene in ("causal_prefill", "causal_decode", "nonmask"):
            sub = [r for r in e2e_ok if r["scene"] == scene]
            if not sub:
                continue
            br = [r["e2e_ratio"] for r in sub]
            bd = [r["e2e_delta_ms"] for r in sub]
            print(
                f"\n[{scene}] n={len(sub)} "
                f"e2e ratio mean={mean(br):.3f}x median={median(br):.3f}x | "
                f"e2e delta mean={mean(bd):.3f}ms median={median(bd):.3f}ms"
            )
    print("=" * 96)


def save_csv(results: list[dict], path: str) -> None:
    fields = [
        "case_id", "label", "scene", "layout", "name", "causal",
        "batch", "nheads", "nkv", "q_s", "kv_s", "head_dim", "pair_family",
        "total_q", "total_kv", "bytes", "flops",
        "fa3_ms", "fused_ms", "delta_ms", "ratio", "degrade_pct",
        "fa3_e2e_ms", "fused_e2e_ms", "e2e_delta_ms", "e2e_ratio", "e2e_degrade_pct",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})
    print(f"\n结果 CSV: {path}")


def parse_scenes(raw: str) -> set[str]:
    alias = {
        "all": {"causal_prefill", "causal_decode", "nonmask"},
        "prefill": {"causal_prefill"},
        "decode": {"causal_decode"},
        "causal_prefill": {"causal_prefill"},
        "causal_decode": {"causal_decode"},
        "nonmask": {"nonmask"},
    }
    out: set[str] = set()
    for part in raw.split(","):
        key = part.strip().lower()
        if key not in alias:
            raise ValueError(f"未知 --scene {part!r}，可选 all,causal_prefill,causal_decode,nonmask")
        out |= alias[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FA3 flash_attn_with_kvcache vs npu_fused_infer_attention_score (b.py 口径)"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--scene", default="all", help="all / causal_prefill,causal_decode,nonmask")
    parser.add_argument(
        "--layout",
        default="all",
        help="all / BSND / TND，或逗号组合如 BSND,TND",
    )
    parser.add_argument("--num-cases", type=int, default=NUM_CASES, help="均衡抽样条数；--full 时忽略")
    parser.add_argument("--full", action="store_true", help="跑 HBM 过滤后的全部组合")
    parser.add_argument(
        "--name",
        default=None,
        help="只跑指定用例，逗号分隔，匹配 case name 的精确名或子串，"
             "例如 causal_decode_BSND_B1_N4x1_q8_kv1m_d256",
    )
    parser.add_argument("--list-only", action="store_true", help="只打印用例，不跑 NPU")
    parser.add_argument("--e2e-warmup", type=int, default=E2E_WARMUP, help="端到端计时前的预热次数")
    parser.add_argument("--e2e-iters", type=int, default=E2E_ITERS, help="端到端采样次数，取中位数；0 表示跳过")
    parser.add_argument("--profile-root", default=PROFILE_ROOT)
    parser.add_argument("--result-csv", default=RESULT_CSV)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hbm-ratio", type=float, default=HBM_RATIO, help="QKV+O 占卡内存上限比例")
    parser.add_argument("--max-bytes", type=int, default=None, help="覆盖自动 HBM 上限，单位字节")
    parser.add_argument(
        "--long",
        action="store_true",
        help="只保留长序列：kvS>=16k（可用 --min-kv-s 覆盖）",
    )
    parser.add_argument("--min-kv-s", default=None, help="kvS 下限，如 16k / 64k / 1m / 65536")
    parser.add_argument("--max-kv-s", default=None, help="kvS 上限")
    parser.add_argument("--min-q-s", default=None, help="qS 下限")
    parser.add_argument("--max-q-s", default=None, help="qS 上限")
    parser.add_argument("--max-batch", type=int, default=None, help="B 上限，长序列建议 1 或 8")
    args = parser.parse_args()

    scenes = parse_scenes(args.scene)
    layouts = {p.strip().upper() for p in args.layout.split(",") if p.strip()}
    if "ALL" in layouts:
        layouts = {"BSND", "TND"}
    if layouts - {"BSND", "TND"}:
        raise ValueError(f"未知 --layout {args.layout!r}，可选 all,BSND,TND")
    universe = [c for c in iter_all_cases(scenes) if c["layout"] in layouts]
    min_kv = parse_len(args.min_kv_s) or (16 * K if args.long else None)
    max_kv = parse_len(args.max_kv_s)
    min_q = parse_len(args.min_q_s)
    max_q = parse_len(args.max_q_s)
    if min_kv is not None:
        universe = [c for c in universe if c["kv_s"] >= min_kv]
    if max_kv is not None:
        universe = [c for c in universe if c["kv_s"] <= max_kv]
    if min_q is not None:
        universe = [c for c in universe if c["q_s"] >= min_q]
    if max_q is not None:
        universe = [c for c in universe if c["q_s"] <= max_q]
    if args.max_batch is not None:
        universe = [c for c in universe if c["batch"] <= args.max_batch]
    if min_kv or max_kv or min_q or max_q or args.max_batch is not None:
        print(
            f"序列过滤: kvS=[{fmt_len(min_kv) if min_kv else '-'}, {fmt_len(max_kv) if max_kv else '-'}] "
            f"qS=[{fmt_len(min_q) if min_q else '-'}, {fmt_len(max_q) if max_q else '-'}] "
            f"B<={args.max_batch if args.max_batch is not None else '-'} → {len(universe)} 组"
        )
    if not universe:
        raise RuntimeError("序列过滤后无用例，请放宽 --min-kv-s / --max-batch")
    if args.name:
        keys = [k.strip() for k in args.name.split(",") if k.strip()]
        matched = [
            c for c in universe
            if any(k == c["name"] or k in c["name"] for k in keys)
        ]
        if not matched:
            raise RuntimeError(f"--name 未命中任何用例: {keys}")
        universe = matched
        print(f"--name 命中 {len(universe)} 组: {[c['name'] for c in universe]}")
    print(
        f"场景 {sorted(scenes)} 候选 {len(universe)} 组 | "
        f"dtype=bf16 | 第{TARGET_ITER}次迭代 kernel Duration | "
        f"e2e median warmup={args.e2e_warmup} iters={args.e2e_iters}"
    )

    hbm_limit = args.max_bytes
    npu_ok = torch.npu.is_available()
    if not args.list_only and not npu_ok:
        raise RuntimeError("未检测到可用 NPU 设备")
    if npu_ok:
        torch.npu.set_device(args.device)
        if hbm_limit is None:
            hbm_limit = int(device_hbm_bytes(args.device) * args.hbm_ratio)
    if hbm_limit is None:
        feasible, skipped = universe, 0
        print("未做 HBM 过滤（无 NPU 且未指定 --max-bytes）")
    else:
        print(f"HBM 过滤: Q+K+V+O <= {hbm_limit / (1024 ** 3):.2f} GiB ({args.hbm_ratio:.0%} 卡容量)")
        feasible, skipped = filter_hbm(universe, hbm_limit)
        print(f"可装下 {len(feasible)} 组，跳过 {skipped} 组（估算会 OOM）")

    if not feasible:
        raise RuntimeError("HBM 过滤后无可用用例，请增大 --hbm-ratio 或缩小场景")

    cases = feasible if (args.full or args.name) else sample_cases(feasible, args.num_cases)
    for i, case in enumerate(cases, 1):
        prefix = {"causal_prefill": "cp", "causal_decode": "cd", "nonmask": "nm"}[case["scene"]]
        case["case_id"] = i
        case["label"] = f"{prefix}_{i:03d}"

    print(f"实际执行 {len(cases)} 组" + ("（--full）" if args.full else f"（抽样 --num-cases {args.num_cases}）"))
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
        set_global_seed(args.seed)

    for case in cases:
        print(
            f"  {case['label']} {case['name']}  "
            f"bytes={case['bytes'] / (1024 ** 3):.3f}GiB  flops={case['flops']:.3e}"
        )

    if args.list_only:
        return

    _load_runtime()
    atten_mask = make_atten_mask(args.device)
    results = []
    for case in cases:
        print(
            f"\n>>> [{case['case_id']:03d}/{len(cases)}] {case['label']} {case['name']}"
        )
        try:
            input_seed = None if args.seed is None else args.seed + case["case_id"] * 100_003
            row = run_case(
                case,
                args.device,
                args.profile_root,
                atten_mask,
                input_seed=input_seed,
                e2e_warmup=args.e2e_warmup,
                e2e_iters=args.e2e_iters,
            )
            results.append(row)
            if row.get("fa3_e2e_ms") is not None and row.get("fused_e2e_ms") is not None:
                print(
                    f"    e2e fa3={row['fa3_e2e_ms']:.3f}ms fused={row['fused_e2e_ms']:.3f}ms "
                    f"delta={row['e2e_delta_ms']:+.3f}ms ratio={row['e2e_ratio']:.3f}x "
                    f"degrade={row['e2e_degrade_pct']:.1f}%"
                )
            if row["fa3_ms"] is None or row["fused_ms"] is None:
                print(f"    PARSE MISS fa3={row['fa3_ms']} fused={row['fused_ms']}")
            else:
                print(
                    f"    kernel fa3={row['fa3_ms']:.3f}ms fused={row['fused_ms']:.3f}ms "
                    f"delta={row['delta_ms']:+.3f}ms ratio={row['ratio']:.3f}x degrade={row['degrade_pct']:.1f}%"
                )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            results.append({**case, "fa3_ms": None, "fused_ms": None})
        finally:
            torch.npu.empty_cache()

    save_csv(results, args.result_csv)
    print("\n" + "-" * 148)
    print(
        f"{'id':>3} {'label':>7} {'scene':<16} {'layout':<5} {'name':<40} "
        f"{'k_fa3':>8} {'k_fused':>8} {'k_ratio':>7} "
        f"{'e2e_fa3':>8} {'e2e_fus':>8} {'e2e_ratio':>9}"
    )
    print("-" * 148)
    for row in results:
        if not row.get("fa3_ms") and not row.get("fa3_e2e_ms"):
            continue
        k_fa3 = f"{row['fa3_ms']:.3f}" if row.get("fa3_ms") else "-"
        k_fused = f"{row['fused_ms']:.3f}" if row.get("fused_ms") else "-"
        k_ratio = f"{row['ratio']:.3f}x" if row.get("ratio") else "-"
        e_fa3 = f"{row['fa3_e2e_ms']:.3f}" if row.get("fa3_e2e_ms") else "-"
        e_fused = f"{row['fused_e2e_ms']:.3f}" if row.get("fused_e2e_ms") else "-"
        e_ratio = f"{row['e2e_ratio']:.3f}x" if row.get("e2e_ratio") else "-"
        print(
            f"{row['case_id']:>3} {row['label']:>7} {row['scene']:<16} {row['layout']:<5} "
            f"{row['name']:<40.40} {k_fa3:>8} {k_fused:>8} {k_ratio:>7} "
            f"{e_fa3:>8} {e_fused:>8} {e_ratio:>9}"
        )
    analyze_results(results)


if __name__ == "__main__":
    main()
