"""
FA3 no sink、FIA no sink、FIA v2 no sink、FIA v2 sink 的 L2 对照。

四路：
  fa3         flash_attn_npu_3.flash_attn_with_kvcache，无 sink
  fia         torch_npu.npu_fused_infer_attention_score，无 sink
  fia_v2      torch_npu.npu_fused_infer_attention_score_v2，无 sink
  fia_v2_sink torch_npu.npu_fused_infer_attention_score_v2，learnable_sink

默认 shape 对齐 compare_fa4_fused_infer_sink_l2.py：
  TND、B=8、N=4x4、d=64、nonmask、q=64K，kv=128K/160K/192K/224K/256K。
四路分开 profile。FA3 与 FIA 的 kernel 名都会落到 FAInfer / FusedInferAttentionScore，
同一次 profile 无法按名字拆开。

四路都请求 LSE 输出，走 LSEOUT tiling：
  fa3         return_softmax_lse=True
  fia         softmax_lse_flag=True
  fia_v2      return_softmax_lse=True
  fia_v2_sink return_softmax_lse=True

profiler：l2_cache=True。
  L2Cache 指标组给 read hit/miss；
  Memory 指标组给 main_mem_read_bw。910B 上 l2_read_bw 常为 0。
无 sink 的三路额外比一次 out；v2 sink 相对 v2 nosink 只记录数值差，公式不同。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from statistics import median

sys.path.pop(0)
import torch
import torch_npu

_FA_ROOT = os.path.dirname(os.path.abspath(__file__))
if _FA_ROOT not in sys.path:
    sys.path.insert(0, _FA_ROOT)

flash_attn_with_kvcache = None
analyse = None

IMPLS = ("fa3", "fia", "fia_v2", "fia_v2_sink")
IMPL_HELP = {
    "fa3": "FA3 no sink",
    "fia": "FIA no sink (npu_fused_infer_attention_score)",
    "fia_v2": "FIA no sink (npu_fused_infer_attention_score_v2)",
    "fia_v2_sink": "FIA sink (npu_fused_infer_attention_score_v2)",
}


def _load_runtime():
    global flash_attn_with_kvcache, analyse
    if flash_attn_with_kvcache is not None:
        return
    from torch_npu.profiler.profiler import analyse as _analyse
    from flash_attn_npu_3 import flash_attn_with_kvcache as _fa3
    flash_attn_with_kvcache = _fa3
    analyse = _analyse


K = 1024
DTYPE = torch.bfloat16
ELEM_BYTES = 2
Q_S = 64 * K
KV_SWEEP = [128 * K, 160 * K, 192 * K, 224 * K, 256 * K]
KNEE_KV = 192 * K
BATCH = 8
NHEADS = 4
NKV = 4
HEAD_DIM = 64
LAYOUT = "TND"
TARGET_ITER = 3
E2E_WARMUP = 2
E2E_ITERS = 3
ATTN_MASK_SIZE = 2048
SPARSE_MODE_CAUSAL = 3
PROFILE_ROOT = "./batchtest/profile_fa3_fia_sink_l2_lse"
RESULT_CSV = "./batchtest/compare_fa3_fia_sink_l2_lse.csv"
AUX_KEYS = ("fill", "reducesum", "cast")


def parse_len(raw: str | int) -> int:
    if isinstance(raw, int):
        return raw
    text = str(raw).strip().lower().replace("_", "")
    if text.endswith("m"):
        return int(float(text[:-1]) * K * K)
    if text.endswith("k"):
        return int(float(text[:-1]) * K)
    return int(text)


def fmt_len(n: int) -> str:
    if n >= K and n % K == 0:
        return f"{n // K}k"
    return str(n)


def parse_len_list(raw: str) -> list[int]:
    return [parse_len(part) for part in raw.split(",") if part.strip()]


def profile_dirs(profile_dir: str) -> set[str]:
    return {
        path
        for path in glob.glob(os.path.join(profile_dir, "*_ascend_pt"))
        if os.path.isdir(path)
    }


def flops(batch: int, nheads: int, q_s: int, kv_s: int, head_dim: int, causal: bool) -> int:
    qk_pv = 4 * batch * nheads * q_s * kv_s * head_dim
    if causal and q_s == kv_s:
        return qk_pv // 2
    return qk_pv


def kv_bytes(batch: int, nkv: int, kv_s: int, head_dim: int) -> int:
    return ELEM_BYTES * 2 * batch * kv_s * nkv * head_dim


def make_case(
    batch: int,
    nheads: int,
    nkv: int,
    q_s: int,
    kv_s: int,
    head_dim: int,
    layout: str,
    causal: bool,
) -> dict:
    name = (
        f"{'causal' if causal else 'nonmask'}_{layout}_B{batch}_N{nheads}x{nkv}_"
        f"q{fmt_len(q_s)}_kv{fmt_len(kv_s)}_d{head_dim}"
    )
    return {
        "name": name,
        "layout": layout,
        "causal": causal,
        "batch": batch,
        "nheads": nheads,
        "nkv": nkv,
        "q_s": q_s,
        "kv_s": kv_s,
        "head_dim": head_dim,
        "scale": head_dim ** -0.5,
        "kv_bytes": kv_bytes(batch, nkv, kv_s, head_dim),
        "flops": flops(batch, nheads, q_s, kv_s, head_dim, causal),
    }


def prefix_sums(length: int, batch: int) -> list[int]:
    return [length * (i + 1) for i in range(batch)]


def build_inputs(case: dict, device: str, seed: int) -> dict:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    batch, nheads, nkv = case["batch"], case["nheads"], case["nkv"]
    q_s, kv_s, head_dim = case["q_s"], case["kv_s"], case["head_dim"]
    kw = dict(dtype=DTYPE, device=device, generator=gen)
    sink = torch.randn(nheads, dtype=torch.bfloat16, device=device, generator=gen)
    if case["layout"] == "BSND":
        q = torch.randn(batch, q_s, nheads, head_dim, **kw)
        k = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        cu_q = None
        max_q = None
        actual_q = [q_s] * batch
        actual_kv = [kv_s] * batch
    else:
        q = torch.randn(batch * q_s, nheads, head_dim, **kw)
        k = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        cu_q = torch.tensor([0] + prefix_sums(q_s, batch), dtype=torch.int32, device=device)
        max_q = q_s
        actual_q = prefix_sums(q_s, batch)
        actual_kv = prefix_sums(kv_s, batch)
    cache_seqlens = torch.full((batch,), kv_s, dtype=torch.int32, device=device)
    return {
        "q": q, "k": k, "v": v, "sink": sink,
        "cu_q": cu_q, "max_q": max_q,
        "actual_q": actual_q, "actual_kv": actual_kv,
        "cache_seqlens": cache_seqlens,
    }


def make_atten_mask(device: str) -> torch.Tensor:
    return torch.triu(
        torch.ones(ATTN_MASK_SIZE, ATTN_MASK_SIZE, dtype=torch.bool, device=device),
        diagonal=1,
    )


def select_fused_mask(case: dict, default_mask: torch.Tensor | None) -> torch.Tensor | None:
    if not case["causal"] or default_mask is None:
        return None
    if case["q_s"] == 1 and case["kv_s"] > ATTN_MASK_SIZE and case["layout"] == "BSND":
        return None
    return default_mask


def run_fa3(inp: dict, case: dict):
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
        return_softmax_lse=True,
    )
    if case["layout"] == "TND":
        kwargs["cu_seqlens_q"] = inp["cu_q"]
        kwargs["max_seqlen_q"] = inp["max_q"]
    return flash_attn_with_kvcache(**kwargs)


def run_fia(inp: dict, case: dict, atten_mask: torch.Tensor | None):
    return torch_npu.npu_fused_infer_attention_score(
        inp["q"], inp["k"], inp["v"],
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
        softmax_lse_flag=True,
    )


def run_fia_v2(inp: dict, case: dict, atten_mask: torch.Tensor | None, *, with_sink: bool):
    kwargs = dict(
        num_query_heads=case["nheads"],
        num_key_value_heads=case["nkv"],
        input_layout=case["layout"],
        softmax_scale=case["scale"],
        pre_tokens=2147483647,
        next_tokens=0 if case["causal"] else 2147483647,
        sparse_mode=SPARSE_MODE_CAUSAL if case["causal"] else 0,
        inner_precise=0,
        atten_mask=select_fused_mask(case, atten_mask),
        actual_seq_qlen=inp["actual_q"],
        actual_seq_kvlen=inp["actual_kv"],
        return_softmax_lse=True,
    )
    if with_sink:
        kwargs["learnable_sink"] = inp["sink"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["k"], inp["v"], **kwargs)


def runners_for(inp: dict, case: dict, atten_mask: torch.Tensor | None) -> dict:
    return {
        "fa3": lambda: run_fa3(inp, case),
        "fia": lambda: run_fia(inp, case, atten_mask),
        "fia_v2": lambda: run_fia_v2(inp, case, atten_mask, with_sink=False),
        "fia_v2_sink": lambda: run_fia_v2(inp, case, atten_mask, with_sink=True),
    }


def as_out(result) -> torch.Tensor:
    if isinstance(result, (tuple, list)):
        result = result[0]
    while result.dim() > 0 and result.shape[-1] == 1 and result.dim() > 3:
        result = result.squeeze(-1)
    return result


def precision_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    if tuple(a.shape) != tuple(b.shape):
        return {"max_abs": None, "cosine": None, "shape": f"{tuple(a.shape)} vs {tuple(b.shape)}"}
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = (af - bf).abs()
    denom = (af.norm() * bf.norm()).item()
    cosine = (af @ bf).item() / denom if denom > 0 else float("nan")
    return {"max_abs": diff.max().item(), "cosine": cosine, "shape": None}


def compare_outs(runners: dict, impls: tuple[str, ...]) -> dict:
    outs = {name: as_out(runners[name]()) for name in impls}
    torch.npu.synchronize()
    wanted = {
        "fa3_fia": ("fa3", "fia"),
        "fa3_v2": ("fa3", "fia_v2"),
        "fia_v2": ("fia", "fia_v2"),
        "v2_sink": ("fia_v2", "fia_v2_sink"),
    }
    pairs = {
        key: precision_metrics(outs[left], outs[right])
        for key, (left, right) in wanted.items()
        if left in outs and right in outs
    }
    flat = {}
    for key, metric in pairs.items():
        flat[f"{key}_max_abs"] = metric["max_abs"]
        flat[f"{key}_cosine"] = metric["cosine"]
        flat[f"{key}_shape"] = metric["shape"]
    return flat


def e2e_median_ms(fn, warmup: int, iters: int) -> float | None:
    if iters <= 0:
        return None
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


def _is_aux(name: str) -> bool:
    blob = name.lower()
    return any(key in blob for key in AUX_KEYS)


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if raw == "" or raw.upper() in {"N/A", "NA", "NAN"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _matching_cols(fieldnames: list[str], needle: str) -> list[str]:
    key = needle.lower()
    return [name for name in fieldnames if key in name.lower()]


def _sum_cols(row: dict, cols: list[str]) -> float | None:
    total = 0.0
    found = False
    for name in cols:
        value = _num(row.get(name))
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _latest_csv(ascend_pt: str, filename: str) -> str | None:
    paths = glob.glob(os.path.join(ascend_pt, "**", filename), recursive=True)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _read_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def parse_main_kernel(ascend_pt: str) -> tuple[dict | None, list[str]]:
    path = _latest_csv(ascend_pt, "kernel_details.csv")
    if path is None:
        return None, []
    fields, rows = _read_csv(path)
    best = None
    best_us = -1.0
    for row in rows:
        blob = f"{row.get('Name', '')} {row.get('Type', '')}"
        if _is_aux(blob):
            continue
        dur = _num(row.get("Duration(us)"))
        if dur is None or dur <= best_us:
            continue
        best = row
        best_us = dur
    return best, fields


def _counter(rows: list[dict], fields: list[str], needles: tuple[str, ...]) -> float | None:
    seen = []
    for needle in needles:
        for name in _matching_cols(fields, needle):
            if name not in seen:
                seen.append(name)
    total = 0.0
    found = False
    for row in rows:
        value = _sum_cols(row, seen)
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def parse_l2_counters(rows: list[dict], fields: list[str]) -> dict:
    read_hit = _counter(rows, fields, ("read_local_l2_hit", "r0_read_cache_hit", "r1_read_cache_hit"))
    read_miss = _counter(rows, fields, ("read_local_l2_miss", "r0_read_cache_miss", "r1_read_cache_miss"))
    write_miss = _counter(rows, fields, ("write_local_l2_miss", "write_cache_miss"))
    l2_read_bw = _counter(rows, fields, ("l2_read_bw",))
    hbm_read_bw = _counter(rows, fields, ("main_mem_read_bw",))
    derived_hit = None
    if read_hit is not None and read_miss is not None and (read_hit + read_miss) > 0:
        derived_hit = read_hit / (read_hit + read_miss)
    return {
        "read_hit": read_hit,
        "read_miss": read_miss,
        "write_miss": write_miss,
        "l2_read_bw": l2_read_bw,
        "hbm_read_bw": hbm_read_bw,
        "derived_hit_rate": derived_hit,
    }


def parse_soc_hit_rate(ascend_pt: str, kernel_name: str) -> tuple[float | None, float | None]:
    path = _latest_csv(ascend_pt, "l2_cache.csv")
    if path is None:
        return None, None
    _, rows = _read_csv(path)
    hits, victims = [], []
    for row in rows:
        op_name = row.get("Op Name", "")
        if _is_aux(op_name):
            continue
        if kernel_name and op_name not in kernel_name and kernel_name not in op_name:
            continue
        hit = _num(row.get("Hit Rate"))
        victim = _num(row.get("Victim Rate"))
        if hit is not None:
            hits.append(hit)
        if victim is not None:
            victims.append(victim)
    hit_rate = sum(hits) / len(hits) if hits else None
    victim_rate = sum(victims) / len(victims) if victims else None
    return hit_rate, victim_rate


def profile_impl(fn, prof_dir: str, aic_metrics) -> str:
    os.makedirs(prof_dir, exist_ok=True)
    before = profile_dirs(prof_dir)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=aic_metrics,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
        l2_cache=True,
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
            fn()
            torch.npu.synchronize()
            prof.step()
            prof.stop()
            break
        fn()
        torch.npu.synchronize()
    created = profile_dirs(prof_dir) - before
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one new profile under {prof_dir}, found {len(created)}: {sorted(created)}"
        )
    ascend_pt = created.pop()
    _load_runtime()
    analyse(profiler_path=ascend_pt, max_process_number=16, export_type="text")
    return ascend_pt


def _fill_missing(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if dst.get(key) is None and value is not None:
            dst[key] = value


def run_case(
    case: dict,
    device: str,
    profile_root: str,
    atten_mask: torch.Tensor | None,
    seed: int,
    e2e_warmup: int,
    e2e_iters: int,
    metric_passes: tuple[str, ...],
    impls: tuple[str, ...],
    skip_precision: bool,
    skip_profile: bool,
) -> tuple[list[dict], dict]:
    inp = build_inputs(case, device, seed)
    runners = runners_for(inp, case, atten_mask)
    prec = {} if skip_precision else compare_outs(runners, impls)
    metric_map = {
        "l2": torch_npu.profiler.AiCMetrics.L2Cache,
        "memory": torch_npu.profiler.AiCMetrics.Memory,
    }
    rows = []
    for impl in impls:
        fn = runners[impl]
        row = {
            **case,
            "impl": impl,
            "kernel_ms": None,
            "kernel_name": None,
            "e2e_ms": e2e_median_ms(fn, e2e_warmup, e2e_iters),
            "hit_rate": None,
            "victim_rate": None,
            "derived_hit_rate": None,
            "read_hit": None,
            "read_miss": None,
            "write_miss": None,
            "l2_read_bw": None,
            "hbm_read_bw": None,
            **prec,
        }
        if not skip_profile:
            for pass_name in metric_passes:
                label = f"{case['name']}_{impl}_{pass_name}"
                ascend_pt = profile_impl(fn, os.path.join(profile_root, label), metric_map[pass_name])
                kernel, fields = parse_main_kernel(ascend_pt)
                if kernel is None:
                    print(f"    PARSE MISS {impl} {pass_name}: no kernel row, header={fields[:8]}")
                    continue
                kernel_name = kernel.get("Name", "")
                counters = parse_l2_counters([kernel], fields)
                soc_hit, victim = parse_soc_hit_rate(ascend_pt, kernel_name)
                parsed = {
                    "kernel_ms": (_num(kernel.get("Duration(us)")) or 0.0) / 1000.0,
                    "kernel_name": kernel_name,
                    "hit_rate": soc_hit,
                    "victim_rate": victim,
                    **counters,
                }
                if row["kernel_ms"] is None:
                    row["kernel_ms"] = parsed["kernel_ms"]
                    row["kernel_name"] = parsed["kernel_name"]
                _fill_missing(row, parsed)
        rows.append(row)
    return rows, prec


def _fmt(value: float | None, spec: str) -> str:
    if value is None:
        return "-"
    return format(value, spec)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _tflops(row: dict) -> float | None:
    ms = row.get("kernel_ms")
    if not ms:
        return None
    return row["flops"] / (ms * 1e-3) / 1e12


def _read_bw(row: dict) -> float | None:
    l2 = row.get("l2_read_bw")
    hbm = row.get("hbm_read_bw")
    if l2 not in (None, 0, 0.0):
        return l2
    return hbm


def _step_drop(prev: float | None, cur: float | None) -> float | None:
    if prev is None or cur is None or prev == 0:
        return None
    return (prev - cur) / prev


def knee_notes(rows: list[dict]) -> list[str]:
    notes = []
    for impl in IMPLS:
        series = sorted(
            [row for row in rows if row["impl"] == impl and row.get("kernel_ms")],
            key=lambda row: row["kv_s"],
        )
        if len(series) < 3:
            continue
        by_kv = {row["kv_s"]: row for row in series}
        eff = {kv: _tflops(row) for kv, row in by_kv.items()}
        miss = {
            kv: row["read_miss"] / kv
            for kv, row in by_kv.items()
            if row.get("read_miss") is not None and kv
        }
        kv_list = [row["kv_s"] for row in series]
        edges = [
            (left, right, _step_drop(eff.get(left), eff.get(right)))
            for left, right in zip(kv_list, kv_list[1:])
        ]

        def touches(left: int, right: int) -> bool:
            return left <= KNEE_KV <= right

        baseline = [drop for left, right, drop in edges if drop is not None and not touches(left, right)]
        baseline_drop = median(baseline) if baseline else 0.0
        flagged = False
        for left, right, drop in edges:
            if not touches(left, right) or drop is None:
                continue
            miss_jump = _step_drop(miss.get(left), miss.get(right))
            miss_up = -miss_jump if miss_jump is not None else None
            worse = drop >= 0.10 and drop >= baseline_drop + 0.08
            miss_worse = miss_up is not None and miss_up >= 0.25
            if worse or miss_worse:
                flagged = True
                notes.append(
                    f"[{impl}] kv {fmt_len(left)}→{fmt_len(right)} "
                    f"算力下跌 {drop * 100:.1f}%（此前中位 {baseline_drop * 100:.1f}%）"
                    + (f"，read miss/token 上升 {miss_up * 100:.1f}%" if miss_up is not None else "")
                )
        if not flagged:
            notes.append(f"[{impl}] kv≈{fmt_len(KNEE_KV)} 附近没有明显算力或 read miss 拐点")
    return notes


def _pick(rows: list[dict], kv_s: int, impl: str) -> dict | None:
    return next((row for row in rows if row["kv_s"] == kv_s and row["impl"] == impl), None)


def print_table(rows: list[dict]) -> None:
    kv_list = sorted({row["kv_s"] for row in rows})
    print("\n" + "-" * 132)
    print(
        f"{'kv':>6} {'fa3':>10} {'fia':>10} {'fia_v2':>10} {'v2_sink':>10} "
        f"{'fa3/sink':>9} {'fia/sink':>9} {'v2/sink':>9}"
    )
    print("-" * 132)
    for kv_s in kv_list:
        got = {impl: _pick(rows, kv_s, impl) for impl in IMPLS}
        sink_ms = got["fia_v2_sink"].get("kernel_ms") if got["fia_v2_sink"] else None
        cells = []
        for impl in IMPLS:
            row = got[impl]
            cells.append(_fmt(row.get("kernel_ms") if row else None, "10.1f"))
        ratios = [
            _fmt(_ratio(got[impl].get("kernel_ms") if got[impl] else None, sink_ms), "8.3f")
            for impl in ("fa3", "fia", "fia_v2")
        ]
        print(f"{fmt_len(kv_s):>6} {' '.join(cells)} {' '.join(ratios)}")
    print("-" * 132)
    print(
        f"{'kv':>6} {'impl':<12} {'hit':>8} {'hbmGB/s':>8} {'rd_miss':>12} {'tflops':>8} {'e2e':>10}"
    )
    for kv_s in kv_list:
        for impl in IMPLS:
            row = _pick(rows, kv_s, impl)
            if row is None:
                continue
            hit = row.get("hit_rate")
            if hit is None:
                hit = row.get("derived_hit_rate")
            print(
                f"{fmt_len(kv_s):>6} {impl:<12} {_fmt(hit, '8.4f')} "
                f"{_fmt(_read_bw(row), '8.2f')} {_fmt(row.get('read_miss'), '12.3g')} "
                f"{_fmt(_tflops(row), '8.2f')} {_fmt(row.get('e2e_ms'), '10.1f')}"
            )
    prec_rows = [row for row in rows if row.get("fa3_v2_max_abs") is not None]
    seen = set()
    if prec_rows:
        print("-" * 132)
        print(f"{'kv':>6} {'fa3-fia':>12} {'fa3-v2':>12} {'fia-v2':>12} {'v2-sink':>12}")
        for row in prec_rows:
            if row["kv_s"] in seen:
                continue
            seen.add(row["kv_s"])
            print(
                f"{fmt_len(row['kv_s']):>6} "
                f"{_fmt(row.get('fa3_fia_max_abs'), '12.3g')} "
                f"{_fmt(row.get('fa3_v2_max_abs'), '12.3g')} "
                f"{_fmt(row.get('fia_v2_max_abs'), '12.3g')} "
                f"{_fmt(row.get('v2_sink_max_abs'), '12.3g')}"
            )


def analyze_results(rows: list[dict]) -> None:
    print("\n" + "=" * 96)
    print("L2 容量拐点（单位算力 TFLOP/s，以及 read miss / kv token）")
    print(f"判定点 kv≈{fmt_len(KNEE_KV)}：相对前一段，算力下跌 ≥10% 且高于基线 8 个百分点，或 read miss/token 上升 ≥25%")
    print("=" * 96)
    notes = knee_notes(rows)
    if not notes:
        print("有效点不足，无法判断拐点")
    for note in notes:
        print(note)
    if any("下跌" in note or "上升" in note for note in notes):
        print("结论: 出现拐点的路径在 kv≈192K 附近可基本确认 L2 容量问题。")
    elif notes:
        print("结论: 这次扫描没有在 kv≈192K 附近形成足够大的拐点。")
    print("=" * 96)


def save_csv(rows: list[dict], path: str) -> None:
    fields = [
        "name", "impl", "layout", "causal",
        "batch", "nheads", "nkv", "q_s", "kv_s", "head_dim",
        "kv_bytes", "flops",
        "kernel_ms", "e2e_ms", "kernel_name",
        "hit_rate", "victim_rate", "derived_hit_rate",
        "read_hit", "read_miss", "write_miss",
        "l2_read_bw", "hbm_read_bw",
        "fa3_fia_max_abs", "fa3_fia_cosine",
        "fa3_v2_max_abs", "fa3_v2_cosine",
        "fia_v2_max_abs", "fia_v2_cosine",
        "v2_sink_max_abs", "v2_sink_cosine",
        "fa3_fia_shape", "fa3_v2_shape", "fia_v2_shape", "v2_sink_shape",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(f"\n结果 CSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FA3 no sink vs FIA v1/v2 no sink vs FIA v2 sink"
    )
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--nheads", type=int, default=NHEADS)
    parser.add_argument("--nkv", type=int, default=NKV)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--layout", default=LAYOUT, choices=["TND", "BSND"])
    parser.add_argument("--q", default=fmt_len(Q_S))
    parser.add_argument("--kv", default=",".join(fmt_len(n) for n in KV_SWEEP))
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--impls", default=",".join(IMPLS), help="fa3,fia,fia_v2,fia_v2_sink")
    parser.add_argument("--metrics", default="l2,memory")
    parser.add_argument("--e2e-warmup", type=int, default=E2E_WARMUP)
    parser.add_argument("--e2e-iters", type=int, default=E2E_ITERS)
    parser.add_argument("--skip-precision", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--profile-root", default=PROFILE_ROOT)
    parser.add_argument("--result-csv", default=RESULT_CSV)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    q_s = parse_len(args.q)
    kv_list = parse_len_list(args.kv)
    if not kv_list:
        raise RuntimeError("--kv 为空")
    impls = tuple(part.strip() for part in args.impls.split(",") if part.strip())
    unknown = [name for name in impls if name not in IMPLS]
    if unknown or not impls:
        raise RuntimeError(f"--impls 只能是 {','.join(IMPLS)}，未知 {unknown}")
    metric_passes = tuple(part.strip().lower() for part in args.metrics.split(",") if part.strip())
    if not args.skip_profile and (not metric_passes or set(metric_passes) - {"l2", "memory"}):
        raise RuntimeError("--metrics 只能是 l2,memory 的组合")
    if args.layout == "BSND" and "fia_v2_sink" in impls:
        print("FIA learnable_sink 仅支持 TND；BSND 上 fia_v2_sink 可能失败")

    cases = [
        make_case(args.batch, args.nheads, args.nkv, q_s, kv_s, args.head_dim, args.layout, args.causal)
        for kv_s in kv_list
    ]
    row_bytes = ELEM_BYTES * 2 * args.nkv * args.head_dim
    print(
        f"q={fmt_len(q_s)} kv={[fmt_len(n) for n in kv_list]} | "
        f"{args.layout} B={args.batch} N={args.nheads}x{args.nkv} d={args.head_dim} "
        f"{'causal' if args.causal else 'nonmask'}"
    )
    print("对比: " + " | ".join(f"{name}={IMPL_HELP[name]}" for name in impls))
    print(
        f"单行 K+V {row_bytes} B/token；192MiB 约 {192 * 1024 * 1024 / row_bytes / K:.1f}K tokens。"
        f"拐点观察点 kv≈{fmt_len(KNEE_KV)}。"
    )
    print(
        f"LSE 输出=True | profiler l2_cache=True | metrics={','.join(metric_passes)} | "
        f"第{TARGET_ITER}次迭代 | e2e warmup={args.e2e_warmup} iters={args.e2e_iters} | device={args.device}"
    )
    for case in cases:
        print(f"  {case['name']}  K+V={case['kv_bytes'] / (1024 ** 3):.3f}GiB  flops={case['flops']:.3e}")
    if args.list_only:
        return
    if not torch.npu.is_available():
        raise RuntimeError("未检测到可用 NPU 设备")

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    _load_runtime()
    atten_mask = make_atten_mask(args.device) if args.causal else None
    results = []
    for index, case in enumerate(cases, 1):
        print(f"\n>>> [{index:02d}/{len(cases)}] {case['name']}")
        try:
            seed = args.seed + index * 100_003
            rows, prec = run_case(
                case, args.device, args.profile_root, atten_mask, seed,
                args.e2e_warmup, args.e2e_iters, metric_passes, impls,
                args.skip_precision, args.skip_profile,
            )
            results.extend(rows)
            if prec:
                print(
                    f"    nosink max|Δ| fa3-fia={_fmt(prec.get('fa3_fia_max_abs'), '.3g')} "
                    f"fa3-v2={_fmt(prec.get('fa3_v2_max_abs'), '.3g')} "
                    f"fia-v2={_fmt(prec.get('fia_v2_max_abs'), '.3g')} "
                    f"v2-sink={_fmt(prec.get('v2_sink_max_abs'), '.3g')}"
                )
            for row in rows:
                hit = row.get("hit_rate")
                if hit is None:
                    hit = row.get("derived_hit_rate")
                print(
                    f"    {row['impl']}: kernel={_fmt(row.get('kernel_ms'), '.3f')}ms "
                    f"e2e={_fmt(row.get('e2e_ms'), '.3f')}ms "
                    f"hit={_fmt(hit, '.4f')} hbm_bw={_fmt(_read_bw(row), '.2f')} "
                    f"read_miss={_fmt(row.get('read_miss'), '.3g')}"
                )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            for impl in impls:
                results.append({**case, "impl": impl, "kernel_ms": None})
        finally:
            torch.npu.empty_cache()

    save_csv(results, args.result_csv)
    print_table(results)
    analyze_results(results)


if __name__ == "__main__":
    main()
