"""
FA4 learnable_sink vs torch_npu.npu_fused_infer_attention_score_v2(learnable_sink)

torch_npu 的 npu_fused_infer_attention_score schema 没有 learnable_sink。
能传入 sink 的是 v2，对应 aclnnFusedInferAttentionScoreV4/V5。

场景网格对齐 compare_fa3_fused_infer.py（causal_prefill / causal_decode / nonmask）。
精度：同一输入下对比 FA4 与 fused 的 out（及可选 lse）数值差。
性能：warmup 后第 3 次迭代 profiler 解析 kernel Duration；另采 NPU Event e2e 中位数。

FIA learnable_sink 约束（文档）：仅 TND/NTD_TND，且 value D<=128。
默认 --layout TND 且过滤 head_dim>128；可用 --include-unsupported 保留全量（不支持的用例跳过 fused）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import re
import shutil
import sys
from statistics import mean, median

script_dir = sys.path.pop(0)
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
M = 1024 * 1024
DTYPE = torch.bfloat16
TARGET_ITER = 3
E2E_WARMUP = 5
E2E_ITERS = 10
NUM_CASES = 50
PROFILE_ROOT = "./batchtest/profile_fa4_fused_infer_sink"
RESULT_CSV = "./batchtest/compare_fa4_fused_infer_sink.csv"
# CANN file ageing 在目标盘可用空间低于 20MiB 时直接丢弃 device/host 数据。
_CANN_MIN_FREE = 20 * 1024 * 1024
_PROFILE_FREE_FLOOR = 1024 * 1024 * 1024
ATTN_MASK_SIZE = 2048
SPARSE_MODE_CAUSAL = 3
HBM_RATIO = 0.45
ELEM_BYTES = 2
FUSED_SINK_MAX_HEAD_DIM = 128

AUX_PAT = re.compile(r"Fill|ReduceSum|Cast", re.I)
FA4_PAT = re.compile(r"SplitFuse|FAInfer|FACombine|FlashDecode", re.I)
FUSED_PAT = re.compile(
    r"FusedInferAttentionScore|PromptFlashAttention|IncreFlashAttention|"
    r"IFAKernel|PFAKernel|FusedInfer",
    re.I,
)

B_PREFILL = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 256, 512, 1024]
B_OTHER = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96]
GQA_CAUSAL = [(32, 2), (20, 20), (16, 2), (8, 1), (5, 1), (4, 1)]
GQA_NONMASK = [(40, 40), (24, 24), (6, 6), (5, 5), (4, 4), (8, 1)]
S_EQUAL = [128, 256, 512, 1024, 2048, 4096, 16 * K, 32 * K, 64 * K, 128 * K, 256 * K]
QS_DECODE = [1, 2, 3, 4, 8]
D_CAUSAL = [128, 256]
D_NONMASK = [64, 128]
NONMASK_PAIRS = [
    ("qS1", 2 * K, 16 * K),
    ("qS1", 4 * K, 32 * K),
    ("qS1", 8 * K, 64 * K),
    ("qS1", 16 * K, 128 * K),
    ("qS1", 32 * K, 256 * K),
    ("qS2", 8 * K, 16 * K),
    ("qS2", 16 * K, 32 * K),
    ("qS2", 32 * K, 64 * K),
    ("qS2", 64 * K, 128 * K),
    ("qS2", 128 * K, 256 * K),
    ("qS3", 16 * K, 16 * K),
    ("qS3", 32 * K, 32 * K),
    ("qS3", 64 * K, 64 * K),
    ("qS3", 128 * K, 128 * K),
    ("qS3", 256 * K, 256 * K),
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


def tensor_bytes(batch: int, nheads: int, nkv: int, q_s: int, kv_s: int, head_dim: int) -> int:
    q_elems = batch * q_s * nheads * head_dim
    kv_elems = batch * kv_s * nkv * head_dim
    sink_elems = nheads
    return ELEM_BYTES * (2 * q_elems + 2 * kv_elems + sink_elems)


def flops(batch: int, nheads: int, q_s: int, kv_s: int, head_dim: int, causal: bool) -> int:
    qk_pv = 4 * batch * nheads * q_s * kv_s * head_dim
    if causal and q_s == kv_s:
        return qk_pv // 2
    return qk_pv


def fused_sink_supported(case: dict) -> bool:
    """FIA learnable_sink: TND only, value D <= 128."""
    return case["layout"] == "TND" and case["head_dim"] <= FUSED_SINK_MAX_HEAD_DIM


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
    case = {
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
    case["fused_ok"] = fused_sink_supported(case)
    return case


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
    sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
    if layout == "BSND":
        q = torch.randn(batch, q_s, nheads, head_dim, **kw)
        k = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch, kv_s, nkv, head_dim, **kw)
        cu_q = None
        cu_k = None
        max_q = None
        max_kv = None
        actual_q = [q_s] * batch
        actual_kv = [kv_s] * batch
    else:
        q = torch.randn(batch * q_s, nheads, head_dim, **kw)
        k = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        v = torch.randn(batch * kv_s, nkv, head_dim, **kw)
        cu_q = torch.tensor([0] + prefix_sums(q_s, batch), dtype=torch.int32, device=device)
        cu_k = torch.tensor([0] + prefix_sums(kv_s, batch), dtype=torch.int32, device=device)
        max_q = q_s
        max_kv = kv_s
        actual_q = prefix_sums(q_s, batch)
        actual_kv = prefix_sums(kv_s, batch)
    return {
        "q": q,
        "k": k,
        "v": v,
        "sink": sink,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "max_q": max_q,
        "max_kv": max_kv,
        "actual_q": actual_q,
        "actual_kv": actual_kv,
    }


def run_fa4(inp: dict, case: dict, *, return_lse: bool = False):
    _load_runtime()
    if case["layout"] == "BSND":
        return flash_attn_func(
            inp["q"],
            inp["k"],
            inp["v"],
            softmax_scale=case["scale"],
            causal=case["causal"],
            window_size=(-1, -1),
            learnable_sink=inp["sink"],
            return_lse=True,
        )
    return flash_attn_varlen_func(
        inp["q"],
        inp["k"],
        inp["v"],
        cu_seqlens_q=inp["cu_q"],
        cu_seqlens_k=inp["cu_k"],
        max_seqlen_q=inp["max_q"],
        max_seqlen_k=inp["max_kv"],
        softmax_scale=case["scale"],
        causal=case["causal"],
        window_size=(-1, -1),
        learnable_sink=inp["sink"],
        return_lse=True,
    )


def run_fused(inp: dict, case: dict, atten_mask: torch.Tensor, *, return_lse: bool = False):
    # v2 才暴露 learnable_sink，对应 aclnn FusedInferAttentionScore V4/V5。
    return torch_npu.npu_fused_infer_attention_score_v2(
        inp["q"],
        inp["k"],
        inp["v"],
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
        learnable_sink=inp["sink"],
        return_softmax_lse=True,
    )


def align_fused_lse(lse_fa4: torch.Tensor, lse_fused: torch.Tensor) -> torch.Tensor:
    """把 FIA softmaxLse 转成 FA4 排布再比较。

    FA4 TND 是 (N, T)，FIA TND 是 (T, N, 1)。
    FA4 BSND 是 (B, N, S)，FIA 一般是 (B, N, S, 1)。
    """
    fused = lse_fused
    while fused.dim() > lse_fa4.dim() and fused.shape[-1] == 1:
        fused = fused.squeeze(-1)
    if fused.shape == lse_fa4.shape:
        return fused
    if fused.dim() == 2 and lse_fa4.dim() == 2 and fused.shape == lse_fa4.shape[::-1]:
        return fused.transpose(0, 1).contiguous()
    return fused


def precision_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    if tuple(a.shape) != tuple(b.shape):
        return {
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
            "cosine": None,
            "shape_mismatch": f"{tuple(a.shape)} vs {tuple(b.shape)}",
        }
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rmse = torch.sqrt((diff * diff).mean()).item()
    denom = (af.norm() * bf.norm()).item()
    cosine = (af @ bf).item() / denom if denom > 0 else float("nan")
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "cosine": cosine,
        "shape_mismatch": None,
    }


def compare_precision(inp: dict, case: dict, atten_mask: torch.Tensor) -> dict:
    fa4_out = run_fa4(inp, case, return_lse=True)
    if isinstance(fa4_out, tuple):
        out_fa4, lse_fa4 = fa4_out
    else:
        out_fa4, lse_fa4 = fa4_out, None
    fused_out, lse_fused = run_fused(inp, case, atten_mask, return_lse=True)
    out_m = precision_metrics(out_fa4, fused_out)
    if lse_fa4 is not None and lse_fused is not None and lse_fused.numel() > 0:
        lse_m = precision_metrics(lse_fa4, align_fused_lse(lse_fa4, lse_fused))
    else:
        lse_m = {
            "max_abs": None, "mean_abs": None, "rmse": None, "cosine": None,
            "shape_mismatch": "lse_skip",
        }
    return {
        "out_max_abs": out_m["max_abs"],
        "out_mean_abs": out_m["mean_abs"],
        "out_rmse": out_m["rmse"],
        "out_cosine": out_m["cosine"],
        "out_shape_mismatch": out_m["shape_mismatch"],
        "lse_max_abs": lse_m["max_abs"],
        "lse_mean_abs": lse_m["mean_abs"],
        "lse_rmse": lse_m["rmse"],
        "lse_cosine": lse_m["cosine"],
        "lse_shape_mismatch": lse_m.get("shape_mismatch"),
    }


def _row_ms(row: dict) -> float:
    return float(row["Duration(us)"]) / 1000.0


def parse_iter3_ms(ascend_pt: str) -> tuple[float | None, float | None]:
    csv_paths = glob.glob(os.path.join(ascend_pt, "**/kernel_details.csv"), recursive=True)
    if not csv_paths:
        raise FileNotFoundError(f"kernel_details.csv not found under current profile: {ascend_pt}")
    csv_path = max(csv_paths, key=os.path.getmtime)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fa4_ms = fused_ms = 0.0
    fa4_hit = fused_hit = False
    for row in rows:
        name, typ = row.get("Name", ""), row.get("Type", "")
        if AUX_PAT.search(name) or AUX_PAT.search(typ):
            continue
        blob = f"{name} {typ}"
        if FA4_PAT.search(blob):
            fa4_ms += _row_ms(row)
            fa4_hit = True
        elif FUSED_PAT.search(blob):
            fused_ms += _row_ms(row)
            fused_hit = True
    return (fa4_ms if fa4_hit else None, fused_ms if fused_hit else None)


def e2e_median_ms(fn, warmup: int, iters: int) -> float:
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
    skip_profile: bool = False,
    skip_precision: bool = False,
) -> dict:
    inp = build_inputs(case, device, input_seed=input_seed)
    fused_ok = case["fused_ok"]
    prec = {}
    if not skip_precision and fused_ok:
        prec = compare_precision(inp, case, atten_mask)
        torch.npu.synchronize()

    fa4_e2e_ms = fused_e2e_ms = None
    if e2e_iters > 0:
        fa4_e2e_ms = e2e_median_ms(lambda: run_fa4(inp, case), e2e_warmup, e2e_iters)
        if fused_ok:
            fused_e2e_ms = e2e_median_ms(
                lambda: run_fused(inp, case, atten_mask), e2e_warmup, e2e_iters,
            )
    e2e_delta_ms = (
        fa4_e2e_ms - fused_e2e_ms
        if fa4_e2e_ms is not None and fused_e2e_ms is not None
        else None
    )
    e2e_ratio = fa4_e2e_ms / fused_e2e_ms if fa4_e2e_ms and fused_e2e_ms else None
    e2e_degrade_pct = (e2e_ratio - 1.0) * 100 if e2e_ratio else None

    fa4_ms = fused_ms = None
    if not skip_profile and fused_ok:
        prof_dir = os.path.join(profile_root, case["label"])
        os.makedirs(prof_dir, exist_ok=True)
        profiles_before = profile_dirs(prof_dir)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            data_simplification=False,
            l2_cache=False,
        )
        # 不传 schedule 时默认停在 RECORD，device/host 为空，解析不到 kernel_details.csv。
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
                run_fa4(inp, case)
                run_fused(inp, case, atten_mask)
                # fused 的 launch 是异步的。不在 step 前同步时，device kernel 不会进入 trace。
                torch.npu.synchronize()
                prof.step()
                prof.stop()
                break
            run_fa4(inp, case)
            run_fused(inp, case, atten_mask)
            torch.npu.synchronize()

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
        fa4_ms, fused_ms = parse_iter3_ms(ascend_pt)

    delta_ms = fa4_ms - fused_ms if fa4_ms is not None and fused_ms is not None else None
    ratio = fa4_ms / fused_ms if fa4_ms and fused_ms else None
    degrade_pct = (ratio - 1.0) * 100 if ratio else None
    return {
        **case,
        "fa4_ms": fa4_ms,
        "fused_ms": fused_ms,
        "delta_ms": delta_ms,
        "ratio": ratio,
        "degrade_pct": degrade_pct,
        "fa4_e2e_ms": fa4_e2e_ms,
        "fused_e2e_ms": fused_e2e_ms,
        "e2e_delta_ms": e2e_delta_ms,
        "e2e_ratio": e2e_ratio,
        "e2e_degrade_pct": e2e_degrade_pct,
        **prec,
    }


def analyze_results(results: list[dict]) -> None:
    ok = [r for r in results if r.get("fa4_ms") and r.get("fused_ms")]
    e2e_ok = [r for r in results if r.get("fa4_e2e_ms") and r.get("fused_e2e_ms")]
    prec_ok = [r for r in results if r.get("out_max_abs") is not None]

    def stat(vals):
        return mean(vals), median(vals), min(vals), max(vals)

    if prec_ok:
        print("\n" + "=" * 96)
        print("精度 FA4(learnable_sink) vs fused_infer(learnable_sink)")
        print("=" * 96)
        print(f"有效用例: {len(prec_ok)}/{len(results)}")
        for key, label in (
            ("out_max_abs", "out max|Δ|"),
            ("out_mean_abs", "out mean|Δ|"),
            ("out_rmse", "out rmse"),
            ("out_cosine", "out cosine"),
        ):
            vals = [r[key] for r in prec_ok if r.get(key) is not None and not math.isnan(r[key])]
            if not vals:
                continue
            v_mean, v_med, v_min, v_max = stat(vals)
            print(f"{label}: mean={v_mean:.6g} median={v_med:.6g} min={v_min:.6g} max={v_max:.6g}")
        worst = sorted(prec_ok, key=lambda r: r["out_max_abs"], reverse=True)[:5]
        print("\nout max|Δ| TOP5:")
        for r in worst:
            print(
                f"  {r['label']} {r['name']} max_abs={r['out_max_abs']:.6g} "
                f"rmse={r['out_rmse']:.6g} cos={r['out_cosine']:.6f}"
            )

    if ok:
        ratios = [r["ratio"] for r in ok]
        deltas = [r["delta_ms"] for r in ok]
        r_mean, r_med, r_min, r_max = stat(ratios)
        d_mean, d_med, d_min, d_max = stat(deltas)
        print("\n" + "=" * 96)
        print("FA4 sink 相对 fused_infer sink（第 3 次迭代 kernel Duration）")
        print("=" * 96)
        print(f"有效用例: {len(ok)}/{len(results)}")
        print(f"ratio  (fa4/fused): mean={r_mean:.3f}x  median={r_med:.3f}x  min={r_min:.3f}x  max={r_max:.3f}x")
        print(f"delta  (fa4-fused): mean={d_mean:.3f}ms median={d_med:.3f}ms min={d_min:.3f}ms max={d_max:.3f}ms")
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
            f"e2e ratio  (fa4/fused): mean={er_mean:.3f}x  median={er_med:.3f}x  "
            f"min={er_min:.3f}x  max={er_max:.3f}x"
        )
        print(
            f"e2e delta  (fa4-fused): mean={ed_mean:.3f}ms median={ed_med:.3f}ms "
            f"min={ed_min:.3f}ms max={ed_max:.3f}ms"
        )
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
    if not ok and not e2e_ok and not prec_ok:
        print("无有效结果")
    print("=" * 96)


def save_csv(results: list[dict], path: str) -> None:
    fields = [
        "case_id", "label", "scene", "layout", "name", "causal", "fused_ok",
        "batch", "nheads", "nkv", "q_s", "kv_s", "head_dim", "pair_family",
        "total_q", "total_kv", "bytes", "flops",
        "out_max_abs", "out_mean_abs", "out_rmse", "out_cosine", "out_shape_mismatch",
        "lse_max_abs", "lse_mean_abs", "lse_rmse", "lse_cosine", "lse_shape_mismatch",
        "fa4_ms", "fused_ms", "delta_ms", "ratio", "degrade_pct",
        "fa4_e2e_ms", "fused_e2e_ms", "e2e_delta_ms", "e2e_ratio", "e2e_degrade_pct",
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
        description="FA4 learnable_sink vs npu_fused_infer_attention_score_v2 learnable_sink"
    )
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--scene", default="all", help="all / causal_prefill,causal_decode,nonmask")
    parser.add_argument(
        "--layout",
        default="TND",
        help="默认 TND（FIA sink 仅支持 TND）；all / BSND / TND，或逗号组合",
    )
    parser.add_argument("--num-cases", type=int, default=NUM_CASES, help="均衡抽样条数；--full 时忽略")
    parser.add_argument("--full", action="store_true", help="跑 HBM 过滤后的全部组合")
    parser.add_argument(
        "--name",
        default=None,
        help="只跑指定用例，逗号分隔，匹配 case name 的精确名或子串",
    )
    parser.add_argument("--list-only", action="store_true", help="只打印用例，不跑 NPU")
    parser.add_argument("--e2e-warmup", type=int, default=E2E_WARMUP)
    parser.add_argument("--e2e-iters", type=int, default=E2E_ITERS, help="端到端采样次数；0 跳过")
    parser.add_argument("--skip-profile", action="store_true", help="跳过 profiler kernel 计时")
    parser.add_argument("--skip-precision", action="store_true", help="跳过精度对比")
    parser.add_argument("--profile-root", default=PROFILE_ROOT)
    parser.add_argument("--result-csv", default=RESULT_CSV)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hbm-ratio", type=float, default=HBM_RATIO)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--long", action="store_true", help="只保留长序列：kvS>=16k")
    parser.add_argument("--min-kv-s", default=None)
    parser.add_argument("--max-kv-s", default=None)
    parser.add_argument("--min-q-s", default=None)
    parser.add_argument("--max-q-s", default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument(
        "--max-head-dim",
        type=int,
        default=FUSED_SINK_MAX_HEAD_DIM,
        help="过滤 head_dim 上限；FIA sink 要求 D<=128，默认 128",
    )
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="保留 FIA sink 不支持的用例（BSND 或 D>128）；这些用例只跑 FA4 e2e，跳过 fused/精度",
    )
    args = parser.parse_args()

    scenes = parse_scenes(args.scene)
    layouts = {p.strip().upper() for p in args.layout.split(",") if p.strip()}
    if "ALL" in layouts:
        layouts = {"BSND", "TND"}
    if layouts - {"BSND", "TND"}:
        raise ValueError(f"未知 --layout {args.layout!r}，可选 all,BSND,TND")
    universe = [c for c in iter_all_cases(scenes) if c["layout"] in layouts]
    if not args.include_unsupported:
        before = len(universe)
        universe = [c for c in universe if c["fused_ok"] and c["head_dim"] <= args.max_head_dim]
        print(
            f"FIA sink 过滤: layout=TND & head_dim<={args.max_head_dim} → "
            f"{len(universe)}/{before} 组（可用 --include-unsupported 放宽）"
        )
    else:
        universe = [c for c in universe if c["head_dim"] <= args.max_head_dim or c["layout"] == "BSND"]
        # still apply max-head-dim only when fused path needed; keep BSND regardless of dim if requested
        if args.max_head_dim is not None:
            universe = [
                c for c in universe
                if c["head_dim"] <= args.max_head_dim or not c["fused_ok"]
            ]

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
        raise RuntimeError("过滤后无用例，请放宽 --layout / --max-head-dim / --include-unsupported")

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
        f"dtype=bf16 | sink | 第{TARGET_ITER}次迭代 kernel Duration | "
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
        print(f"HBM 过滤: Q+K+V+O+sink <= {hbm_limit / (1024 ** 3):.2f} GiB ({args.hbm_ratio:.0%} 卡容量)")
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
        tag = "" if case["fused_ok"] else " [FA4-only]"
        print(
            f"  {case['label']} {case['name']}  "
            f"bytes={case['bytes'] / (1024 ** 3):.3f}GiB  flops={case['flops']:.3e}{tag}"
        )

    if args.list_only:
        return

    args.profile_root = place_on_writable_fs(
        args.profile_root, "profile_fa4_fused_infer_sink", _PROFILE_FREE_FLOOR,
    )
    try:
        csv_free = _dir_free(os.path.dirname(os.path.abspath(args.result_csv)) or ".")
    except OSError:
        csv_free = 0
    if csv_free < _CANN_MIN_FREE:
        args.result_csv = os.path.join(args.profile_root, os.path.basename(args.result_csv))
        print(f"结果 CSV 改到 {args.result_csv}")

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
                skip_profile=args.skip_profile,
                skip_precision=args.skip_precision,
            )
            results.append(row)
            if row.get("out_max_abs") is not None:
                print(
                    f"    prec out max|Δ|={row['out_max_abs']:.6g} "
                    f"mean|Δ|={row['out_mean_abs']:.6g} rmse={row['out_rmse']:.6g} "
                    f"cos={row['out_cosine']:.6f}"
                )
                if row.get("lse_max_abs") is not None:
                    print(
                        f"    prec lse max|Δ|={row['lse_max_abs']:.6g} "
                        f"rmse={row['lse_rmse']:.6g} cos={row['lse_cosine']:.6f}"
                    )
                elif row.get("lse_shape_mismatch"):
                    print(f"    prec lse shape mismatch {row['lse_shape_mismatch']}")
            if row.get("fa4_e2e_ms") is not None:
                if row.get("fused_e2e_ms") is not None:
                    print(
                        f"    e2e fa4={row['fa4_e2e_ms']:.3f}ms fused={row['fused_e2e_ms']:.3f}ms "
                        f"delta={row['e2e_delta_ms']:+.3f}ms ratio={row['e2e_ratio']:.3f}x "
                        f"degrade={row['e2e_degrade_pct']:.1f}%"
                    )
                else:
                    print(f"    e2e fa4={row['fa4_e2e_ms']:.3f}ms fused=skip")
            if not args.skip_profile and case["fused_ok"]:
                if row.get("fa4_ms") is None or row.get("fused_ms") is None:
                    print(f"    PARSE MISS fa4={row.get('fa4_ms')} fused={row.get('fused_ms')}")
                else:
                    print(
                        f"    kernel fa4={row['fa4_ms']:.3f}ms fused={row['fused_ms']:.3f}ms "
                        f"delta={row['delta_ms']:+.3f}ms ratio={row['ratio']:.3f}x "
                        f"degrade={row['degrade_pct']:.1f}%"
                    )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            results.append({**case, "fa4_ms": None, "fused_ms": None})
        finally:
            torch.npu.empty_cache()

    save_csv(results, args.result_csv)
    print("\n" + "-" * 168)
    print(
        f"{'id':>3} {'label':>7} {'scene':<16} {'layout':<5} {'name':<40} "
        f"{'max|d|':>10} {'cos':>7} "
        f"{'k_fa4':>8} {'k_fused':>8} {'k_ratio':>7} "
        f"{'e2e_fa4':>8} {'e2e_fus':>8} {'e2e_ratio':>9}"
    )
    print("-" * 168)
    for row in results:
        if not row.get("fa4_ms") and not row.get("fa4_e2e_ms") and row.get("out_max_abs") is None:
            continue
        k_fa4 = f"{row['fa4_ms']:.3f}" if row.get("fa4_ms") else "-"
        k_fused = f"{row['fused_ms']:.3f}" if row.get("fused_ms") else "-"
        k_ratio = f"{row['ratio']:.3f}x" if row.get("ratio") else "-"
        e_fa4 = f"{row['fa4_e2e_ms']:.3f}" if row.get("fa4_e2e_ms") else "-"
        e_fused = f"{row['fused_e2e_ms']:.3f}" if row.get("fused_e2e_ms") else "-"
        e_ratio = f"{row['e2e_ratio']:.3f}x" if row.get("e2e_ratio") else "-"
        max_d = f"{row['out_max_abs']:.3g}" if row.get("out_max_abs") is not None else "-"
        cos = f"{row['out_cosine']:.4f}" if row.get("out_cosine") is not None else "-"
        print(
            f"{row['case_id']:>3} {row['label']:>7} {row['scene']:<16} {row['layout']:<5} "
            f"{row['name']:<40.40} {max_d:>10} {cos:>7} "
            f"{k_fa4:>8} {k_fused:>8} {k_ratio:>7} "
            f"{e_fa4:>8} {e_fused:>8} {e_ratio:>9}"
        )
    analyze_results(results)


if __name__ == "__main__":
    main()
