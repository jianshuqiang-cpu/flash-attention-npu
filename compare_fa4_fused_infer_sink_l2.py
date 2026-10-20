"""
FA4 vs npu_fused_infer_attention_score_v2 的 L2 容量扫描。

固定 q=64K，扫描 kv=128K、160K、192K、224K、256K。
默认 shape 对齐最近的 nonmask 对照：TND、B=8、N=4x4、d=64、causal=False。
同一组 shape 再跑一轮 learnable_sink=None。

profiler 相对 compare_fa4_fused_infer_sink.py 的改动：
  l2_cache=True，导出 l2_cache.csv 的 Hit Rate。
  另采两套 AiCMetrics，因为一次 profile 只能选一个指标组：
    L2Cache → read/write hit 与 miss 计数
    Memory  → l2_read_bw、main_mem_read_bw
FA4 与 fused 分开 profile，避免同一步里前一个 kernel 把 L2 预热给后一个。

若单位算力或 read miss 在 kv≈192K 附近出现拐点，可基本确认是 L2 容量问题。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
from statistics import median

sys.path.pop(0)
import torch
import torch_npu

_FA_ROOT = os.path.dirname(os.path.abspath(__file__))
if _FA_ROOT not in sys.path:
    sys.path.insert(0, _FA_ROOT)

flash_attn_func = None
flash_attn_varlen_func = None
analyse = None


def _load_runtime():
    global flash_attn_func, flash_attn_varlen_func, analyse
    if flash_attn_func is not None:
        return
    from torch_npu.profiler.profiler import analyse as _analyse
    from flash_attn_npu_4 import flash_attn_func as _fa4
    from flash_attn_npu_4 import flash_attn_varlen_func as _fa4_varlen
    flash_attn_func = _fa4
    flash_attn_varlen_func = _fa4_varlen
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
HEAD_DIM = 128
LAYOUT = "TND"
TARGET_ITER = 3
E2E_WARMUP = 2
E2E_ITERS = 3
ATTN_MASK_SIZE = 2048
SPARSE_MODE_CAUSAL = 3
PROFILE_ROOT = "./batchtest/profile_fa4_fused_infer_sink_l2"
RESULT_CSV = "./batchtest/compare_fa4_fused_infer_sink_l2.csv"
# CANN file ageing 在目标盘可用空间低于 20MiB 时直接丢弃 device/host 数据。
_CANN_MIN_FREE = 20 * 1024 * 1024
# 一组 L2+Memory profile 远大于这个门槛，留 1GiB 避免采到一半写满。
_PROFILE_FREE_FLOOR = 1024 * 1024 * 1024

FA4_KEYS = ("splitfuse", "fainfer", "facombine", "flashdecode")
FUSED_KEYS = (
    "fusedinferattentionscore",
    "promptflashattention",
    "increflashattention",
    "ifakernel",
    "pfakernel",
    "fusedinfer",
)
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


def _dir_free(path: str) -> int:
    parent = path if os.path.isdir(path) else (os.path.dirname(os.path.abspath(path)) or ".")
    os.makedirs(parent, exist_ok=True)
    return shutil.disk_usage(parent).free


def place_on_writable_fs(path: str, fallback_name: str, floor: int) -> str:
    """目标盘空间不够时改写到 /dev/shm，否则 CANN 不落 device trace。"""
    try:
        free = _dir_free(path)
    except OSError:
        free = 0
    if free >= floor:
        return path
    fallback = os.path.join("/dev/shm", fallback_name)
    try:
        shm_free = _dir_free(fallback)
    except OSError as exc:
        raise RuntimeError(
            f"{path} 剩余 {free / (1024 ** 2):.1f} MiB，且 /dev/shm 不可用: {exc}"
        ) from exc
    if shm_free < _CANN_MIN_FREE:
        raise RuntimeError(
            f"{path} 剩余 {free / (1024 ** 2):.1f} MiB，/dev/shm 剩余 {shm_free / (1024 ** 2):.1f} MiB，"
            "都低于 CANN profiler 的 20MiB 落盘门槛"
        )
    print(
        f"{path} 所在磁盘剩余 {free / (1024 ** 2):.1f} MiB，"
        f"CANN 在可用空间低于 20MiB 时不采集。"
        f"输出改到 {fallback}（剩余 {shm_free / (1024 ** 3):.1f} GiB）"
    )
    return fallback


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
    use_sink: bool,
) -> dict:
    sink_tag = "sink" if use_sink else "nosink"
    name = (
        f"{'causal' if causal else 'nonmask'}_{layout}_B{batch}_N{nheads}x{nkv}_"
        f"q{fmt_len(q_s)}_kv{fmt_len(kv_s)}_d{head_dim}_{sink_tag}"
    )
    return {
        "name": name,
        "layout": layout,
        "causal": causal,
        "use_sink": use_sink,
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
    sink = None
    if case["use_sink"]:
        sink = torch.randn(nheads, dtype=torch.bfloat16, device=device, generator=gen)
    if case["layout"] == "BSND":
        q = torch.randn(batch, q_s, nheads, head_dim, **kw)
        k = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        cu_q = cu_k = None
        max_q = max_kv = None
        actual_q = [q_s] * batch
        actual_kv = [kv_s] * batch
    else:
        q = torch.randn(batch * q_s, nheads, head_dim, **kw)
        k = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        cu_q = torch.tensor([0] + prefix_sums(q_s, batch), dtype=torch.int32, device=device)
        cu_k = torch.tensor([0] + prefix_sums(kv_s, batch), dtype=torch.int32, device=device)
        max_q, max_kv = q_s, kv_s
        actual_q = prefix_sums(q_s, batch)
        actual_kv = prefix_sums(kv_s, batch)
    return {
        "q": q, "k": k, "v": v, "sink": sink,
        "cu_q": cu_q, "cu_k": cu_k, "max_q": max_q, "max_kv": max_kv,
        "actual_q": actual_q, "actual_kv": actual_kv,
    }


def make_atten_mask(device: str) -> torch.Tensor:
    return torch.triu(
        torch.ones(ATTN_MASK_SIZE, ATTN_MASK_SIZE, dtype=torch.bool, device=device),
        diagonal=1,
    )


def run_fa4(inp: dict, case: dict):
    _load_runtime()
    if case["layout"] == "BSND":
        return flash_attn_func(
            inp["q"], inp["k"], inp["v"],
            softmax_scale=case["scale"],
            causal=case["causal"],
            window_size=(-1, -1),
            learnable_sink=inp["sink"],
        )
    return flash_attn_varlen_func(
        inp["q"], inp["k"], inp["v"],
        cu_seqlens_q=inp["cu_q"],
        cu_seqlens_k=inp["cu_k"],
        max_seqlen_q=inp["max_q"],
        max_seqlen_k=inp["max_kv"],
        softmax_scale=case["scale"],
        causal=case["causal"],
        window_size=(-1, -1),
        learnable_sink=inp["sink"],
    )


def run_fused(inp: dict, case: dict, atten_mask: torch.Tensor | None):
    mask = atten_mask if case["causal"] else None
    kwargs = dict(
        num_query_heads=case["nheads"],
        num_key_value_heads=case["nkv"],
        input_layout=case["layout"],
        softmax_scale=case["scale"],
        pre_tokens=2147483647,
        next_tokens=0 if case["causal"] else 2147483647,
        sparse_mode=SPARSE_MODE_CAUSAL if case["causal"] else 0,
        inner_precise=0,
        atten_mask=mask,
        actual_seq_qlen=inp["actual_q"],
        actual_seq_kvlen=inp["actual_kv"],
    )
    if inp["sink"] is not None:
        kwargs["learnable_sink"] = inp["sink"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["k"], inp["v"], **kwargs)


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


def _op_kind(name: str) -> str | None:
    blob = name.lower()
    if any(key in blob for key in AUX_KEYS):
        return None
    if any(key in blob for key in FA4_KEYS):
        return "fa4"
    if any(key in blob for key in FUSED_KEYS):
        return "fused"
    return None


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


def parse_kernel_rows(ascend_pt: str, impl: str) -> tuple[float | None, list[dict], list[str]]:
    path = _latest_csv(ascend_pt, "kernel_details.csv")
    if path is None:
        return None, [], []
    fields, rows = _read_csv(path)
    matched = []
    duration_us = 0.0
    hit = False
    for row in rows:
        kind = _op_kind(row.get("Name", "") + " " + row.get("Type", ""))
        if kind != impl:
            continue
        dur = _num(row.get("Duration(us)"))
        if dur is not None:
            duration_us += dur
            hit = True
        matched.append(row)
    ms = duration_us / 1000.0 if hit else None
    return ms, matched, fields


def _counter(rows: list[dict], fields: list[str], needles: tuple[str, ...]) -> float | None:
    cols = []
    for needle in needles:
        cols.extend(_matching_cols(fields, needle))
    # 去重，避免一个列被两个 needle 命中后加两次。
    seen = []
    for name in cols:
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


def parse_soc_hit_rate(ascend_pt: str, impl: str) -> tuple[float | None, float | None]:
    path = _latest_csv(ascend_pt, "l2_cache.csv")
    if path is None:
        return None, None
    _, rows = _read_csv(path)
    hits, victims = [], []
    for row in rows:
        if _op_kind(row.get("Op Name", "")) != impl:
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
    # 不传 schedule 时默认每步都是 RECORD。stop 停在 RECORD 只会告警，
    # device/host 目录为空，解析不到 kernel_details.csv。
    # active=1、repeat=1 让 start 处于 RECORD_AND_SAVE，随后 step 转到 NONE 并落盘。
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_modules=False,
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=1, skip_first=0,
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir, analyse_flag=False),
    )
    for step in range(1, TARGET_ITER + 2):
        if step == TARGET_ITER:
            prof.start()
            fn()
            # fused 的 launch 是异步的。不在 step 前同步时，device kernel 不会进入 trace。
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
) -> list[dict]:
    inp = build_inputs(case, device, seed)
    runners = {
        "fa4": lambda: run_fa4(inp, case),
        "fused": lambda: run_fused(inp, case, atten_mask),
    }
    metric_map = {
        "l2": torch_npu.profiler.AiCMetrics.L2Cache,
        "memory": torch_npu.profiler.AiCMetrics.Memory,
    }
    rows = []
    for impl, fn in runners.items():
        row = {
            **case,
            "impl": impl,
            "kernel_ms": None,
            "e2e_ms": e2e_median_ms(fn, e2e_warmup, e2e_iters),
            "hit_rate": None,
            "victim_rate": None,
            "derived_hit_rate": None,
            "read_hit": None,
            "read_miss": None,
            "write_miss": None,
            "l2_read_bw": None,
            "hbm_read_bw": None,
        }
        for pass_name in metric_passes:
            label = f"{case['name']}_{impl}_{pass_name}"
            prof_dir = os.path.join(profile_root, label)
            ascend_pt = profile_impl(fn, prof_dir, metric_map[pass_name])
            kernel_ms, kernel_rows, fields = parse_kernel_rows(ascend_pt, impl)
            counters = parse_l2_counters(kernel_rows, fields)
            soc_hit, victim = parse_soc_hit_rate(ascend_pt, impl)
            parsed = {
                "kernel_ms": kernel_ms,
                "hit_rate": soc_hit,
                "victim_rate": victim,
                **counters,
            }
            if row["kernel_ms"] is None:
                row["kernel_ms"] = parsed["kernel_ms"]
            _fill_missing(row, parsed)
            if not kernel_rows:
                print(f"    PARSE MISS {impl} {pass_name}: no kernel row, header={fields[:12]}")
        rows.append(row)
    return rows


def _fmt(value: float | None, spec: str) -> str:
    if value is None:
        return "-"
    return format(value, spec)


def _read_bw(row: dict) -> float | None:
    """l2_read_bw 在 910B 的 Memory 组里常为 0，这时用 main_mem_read_bw。"""
    l2 = row.get("l2_read_bw")
    hbm = row.get("hbm_read_bw")
    if l2 not in (None, 0, 0.0):
        return l2
    return hbm


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _tflops(row: dict) -> float | None:
    ms = row.get("kernel_ms")
    if not ms:
        return None
    return row["flops"] / (ms * 1e-3) / 1e12


def _series(rows: list[dict], impl: str, use_sink: bool) -> list[dict]:
    picked = [row for row in rows if row["impl"] == impl and row["use_sink"] is use_sink and row.get("kernel_ms")]
    return sorted(picked, key=lambda row: row["kv_s"])


def _step_drop(prev: float | None, cur: float | None) -> float | None:
    if prev is None or cur is None or prev == 0:
        return None
    return (prev - cur) / prev


def knee_notes(rows: list[dict]) -> list[str]:
    notes = []
    for use_sink in (True, False):
        for impl in ("fa4", "fused"):
            series = _series(rows, impl, use_sink)
            if len(series) < 3:
                continue
            by_kv = {row["kv_s"]: row for row in series}
            eff = {kv: _tflops(row) for kv, row in by_kv.items()}
            miss = {}
            for kv, row in by_kv.items():
                if row.get("read_miss") is not None and kv:
                    miss[kv] = row["read_miss"] / kv
            edges = []
            kv_list = [row["kv_s"] for row in series]
            for left, right in zip(kv_list, kv_list[1:]):
                edges.append((left, right, _step_drop(eff.get(left), eff.get(right))))
            def _touches(left: int, right: int) -> bool:
                return left <= KNEE_KV <= right
            baseline = [drop for left, right, drop in edges if drop is not None and not _touches(left, right)]
            baseline_drop = median(baseline) if baseline else 0.0
            sink_tag = "sink" if use_sink else "nosink"
            flagged = False
            for left, right, drop in edges:
                touches = _touches(left, right)
                if not touches or drop is None:
                    continue
                miss_jump = _step_drop(miss.get(left), miss.get(right))
                # miss 增加时 drop 为负，取相反数看涨幅。
                miss_up = -miss_jump if miss_jump is not None else None
                worse = drop >= 0.10 and drop >= baseline_drop + 0.08
                miss_worse = miss_up is not None and miss_up >= 0.25
                if worse or miss_worse:
                    flagged = True
                    notes.append(
                        f"[{sink_tag}/{impl}] kv {fmt_len(left)}→{fmt_len(right)} "
                        f"算力下跌 {drop * 100:.1f}%（此前中位 {baseline_drop * 100:.1f}%）"
                        + (f"，read miss/token 上升 {miss_up * 100:.1f}%" if miss_up is not None else "")
                    )
            if not flagged:
                notes.append(f"[{sink_tag}/{impl}] kv≈{fmt_len(KNEE_KV)} 附近没有明显算力或 read miss 拐点")
    return notes


def print_table(rows: list[dict]) -> None:
    print("\n" + "-" * 168)
    print(
        f"{'sink':<7} {'kv':>6} {'impl':<5} {'kernel':>10} {'e2e':>10} "
        f"{'soc_hit':>8} {'rd_hit':>8} {'l2_rGB/s':>10} {'hbm_rGB/s':>10} "
        f"{'rd_miss':>12} {'wr_miss':>12} {'K+V GiB':>8}"
    )
    print("-" * 168)
    ordered = sorted(rows, key=lambda row: (not row["use_sink"], row["kv_s"], row["impl"]))
    for row in ordered:
        print(
            f"{'sink' if row['use_sink'] else 'nosink':<7} {fmt_len(row['kv_s']):>6} {row['impl']:<5} "
            f"{_fmt(row.get('kernel_ms'), '10.3f')} {_fmt(row.get('e2e_ms'), '10.3f')} "
            f"{_fmt(row.get('hit_rate'), '8.4f')} {_fmt(row.get('derived_hit_rate'), '8.4f')} "
            f"{_fmt(row.get('l2_read_bw'), '10.2f')} {_fmt(row.get('hbm_read_bw'), '10.2f')} "
            f"{_fmt(row.get('read_miss'), '12.3g')} {_fmt(row.get('write_miss'), '12.3g')} "
            f"{row['kv_bytes'] / (1024 ** 3):8.3f}"
        )
    print("-" * 168)
    print(
        f"{'sink':<7} {'kv':>6} {'k_ratio':>8} {'hit_f/u':>8} {'bw_f/u':>8} {'miss_f/u':>8}"
    )
    keys = sorted({(row["use_sink"], row["kv_s"]) for row in rows})
    for use_sink, kv_s in keys:
        fa4 = next((row for row in rows if row["use_sink"] is use_sink and row["kv_s"] == kv_s and row["impl"] == "fa4"), None)
        fused = next((row for row in rows if row["use_sink"] is use_sink and row["kv_s"] == kv_s and row["impl"] == "fused"), None)
        if fa4 is None or fused is None:
            continue
        hit_fa4 = fa4.get("hit_rate") if fa4.get("hit_rate") is not None else fa4.get("derived_hit_rate")
        hit_fused = fused.get("hit_rate") if fused.get("hit_rate") is not None else fused.get("derived_hit_rate")
        bw_fa4 = _read_bw(fa4)
        bw_fused = _read_bw(fused)
        k_ratio = _ratio(fa4.get("kernel_ms"), fused.get("kernel_ms"))
        print(
            f"{'sink' if use_sink else 'nosink':<7} {fmt_len(kv_s):>6} "
            f"{(_fmt(k_ratio, '7.3f') + 'x') if k_ratio is not None else '-':>8} "
            f"{_fmt(_ratio(hit_fa4, hit_fused), '8.3f')} "
            f"{_fmt(_ratio(bw_fa4, bw_fused), '8.3f')} "
            f"{_fmt(_ratio(fa4.get('read_miss'), fused.get('read_miss')), '8.3f')}"
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
    touched = [note for note in notes if "下跌" in note or "上升" in note]
    if touched:
        print("结论: kv≈192K 附近出现性能或 miss 拐点，可基本确认 L2 容量问题。")
    elif notes:
        print("结论: 这次扫描没有在 kv≈192K 附近形成足够大的拐点。")
    print("=" * 96)


def save_csv(rows: list[dict], path: str) -> None:
    fields = [
        "name", "impl", "use_sink", "layout", "causal",
        "batch", "nheads", "nkv", "q_s", "kv_s", "head_dim",
        "kv_bytes", "flops",
        "kernel_ms", "e2e_ms",
        "hit_rate", "victim_rate", "derived_hit_rate",
        "read_hit", "read_miss", "write_miss",
        "l2_read_bw", "hbm_read_bw",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(f"\n结果 CSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FA4 vs fused L2 hit/bandwidth/miss sweep at q=64K")
    parser.add_argument("--device", default="npu:6")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--nheads", type=int, default=NHEADS)
    parser.add_argument("--nkv", type=int, default=NKV)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--layout", default=LAYOUT, choices=["TND", "BSND"])
    parser.add_argument("--q", default=fmt_len(Q_S))
    parser.add_argument("--kv", default=",".join(fmt_len(n) for n in KV_SWEEP), help="逗号分隔，如 128k,160k,192k,224k,256k")
    parser.add_argument("--causal", action="store_true", help="默认 nonmask")
    parser.add_argument("--sink", default="both", choices=["both", "sink", "none"])
    parser.add_argument("--metrics", default="l2,memory", help="l2=miss/hit 计数，memory=read bandwidth，逗号组合")
    parser.add_argument("--e2e-warmup", type=int, default=E2E_WARMUP)
    parser.add_argument("--e2e-iters", type=int, default=E2E_ITERS, help="0 跳过端到端计时")
    parser.add_argument("--profile-root", default=PROFILE_ROOT)
    parser.add_argument("--result-csv", default=RESULT_CSV)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    q_s = parse_len(args.q)
    kv_list = parse_len_list(args.kv)
    if not kv_list:
        raise RuntimeError("--kv 为空")
    metric_passes = tuple(part.strip().lower() for part in args.metrics.split(",") if part.strip())
    if not metric_passes or set(metric_passes) - {"l2", "memory"}:
        raise RuntimeError("--metrics 只能是 l2,memory 的组合")
    sink_modes = {"both": (True, False), "sink": (True,), "none": (False,)}[args.sink]
    if args.layout == "BSND" and any(sink_modes):
        print("FIA learnable_sink 仅支持 TND；BSND 下 fused 仍会跑，sink 传入可能失败")

    cases = []
    for kv_s in kv_list:
        for use_sink in sink_modes:
            cases.append(make_case(
                args.batch, args.nheads, args.nkv, q_s, kv_s, args.head_dim,
                args.layout, args.causal, use_sink,
            ))

    if not args.list_only:
        args.profile_root = place_on_writable_fs(
            args.profile_root, "profile_fa4_fused_infer_sink_l2", _PROFILE_FREE_FLOOR,
        )
        try:
            csv_free = _dir_free(os.path.dirname(os.path.abspath(args.result_csv)) or ".")
        except OSError:
            csv_free = 0
        if csv_free < _CANN_MIN_FREE:
            args.result_csv = os.path.join(args.profile_root, os.path.basename(args.result_csv))
            print(f"结果 CSV 改到 {args.result_csv}")

    row_bytes = ELEM_BYTES * 2 * args.nkv * args.head_dim
    batch_bytes = row_bytes * args.batch
    l2_mib = 192 * 1024 * 1024
    row_tokens = l2_mib / row_bytes if row_bytes else float("nan")
    batch_tokens = l2_mib / batch_bytes if batch_bytes else float("nan")
    print(
        f"q={fmt_len(q_s)} kv={[fmt_len(n) for n in kv_list]} | "
        f"{args.layout} B={args.batch} N={args.nheads}x{args.nkv} d={args.head_dim} "
        f"{'causal' if args.causal else 'nonmask'} | sink={args.sink}"
    )
    print(
        f"K+V：单行 {row_bytes} B/token，整 batch {batch_bytes} B/token。"
        f"填满 192MiB 约需单行 {row_tokens / K:.1f}K tokens，或整 batch {batch_tokens / K:.1f}K tokens。"
        f"拐点观察点 kv≈{fmt_len(KNEE_KV)}。"
    )
    print(
        f"profiler l2_cache=True | metrics={','.join(metric_passes)} | "
        f"第{TARGET_ITER}次迭代 | e2e warmup={args.e2e_warmup} iters={args.e2e_iters} | device={args.device}"
    )
    print("FA4 与 fused 分开 profile。带宽是 kernel_details 里 aic/aiv 同名读带宽列之和；miss 同样按列求和。")
    for case in cases:
        print(
            f"  {case['name']}  K+V={case['kv_bytes'] / (1024 ** 3):.3f}GiB  flops={case['flops']:.3e}"
        )
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
            rows = run_case(
                case, args.device, args.profile_root, atten_mask, seed,
                args.e2e_warmup, args.e2e_iters, metric_passes,
            )
            results.extend(rows)
            for row in rows:
                hit = row.get("hit_rate")
                if hit is None:
                    hit = row.get("derived_hit_rate")
                print(
                    f"    {row['impl']}: kernel={_fmt(row.get('kernel_ms'), '.3f')}ms "
                    f"e2e={_fmt(row.get('e2e_ms'), '.3f')}ms "
                    f"hit={_fmt(hit, '.4f')} "
                    f"l2_bw={_fmt(row.get('l2_read_bw'), '.2f')} "
                    f"hbm_bw={_fmt(row.get('hbm_read_bw'), '.2f')} "
                    f"read_miss={_fmt(row.get('read_miss'), '.3g')}"
                )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            results.append({**case, "impl": "fa4", "kernel_ms": None})
            results.append({**case, "impl": "fused", "kernel_ms": None})
        finally:
            torch.npu.empty_cache()

    save_csv(results, args.result_csv)
    print_table(results)
    analyze_results(results)


if __name__ == "__main__":
    main()
