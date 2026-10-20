"""
FA4 vs torch_npu.npu_fused_infer_attention_score_v2，paged KV。

场景网格对齐 compare_fa4_fused_infer_sink.py：
  causal_prefill / causal_decode 使用 Causal mask（sparse_mode=3）
  nonmask 使用 No mask（sparse_mode=0，不传 atten_mask）
每个 shape 再跑 sink 与 nosink。

Shape 为 TND query + paged KV，--kv-layout 选 BBND / BNBD / both：
  q: (total_q, N, D)
  block_table: (B, ceil(kv_s / block_size))
  BBND 物理页: (num_blocks, block_size, Nkv, D)，FA4 直接读。
    FIA 只认 BNBD，所以另存一份 transpose(1, 2).contiguous()。
  BNBD 物理页: (num_blocks, Nkv, block_size, D)，FIA 直接读。
    FA4 吃 transpose(1, 2) 视图，shape 仍是 (num_blocks, block_size, Nkv, D)，
    stride 满足 BNBD。视图与物理页共享存储，计时区间里不再搬运。
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

flash_attn_varlen_func = None
analyse = None


def _load_runtime():
    global flash_attn_varlen_func
    if flash_attn_varlen_func is not None:
        return
    from flash_attn_npu_4 import flash_attn_varlen_func as _fa4_varlen
    flash_attn_varlen_func = _fa4_varlen


def _load_analyse():
    global analyse
    if analyse is not None:
        return
    from torch_npu.profiler.profiler import analyse as _analyse
    analyse = _analyse


def external_msprof() -> str | None:
    """本进程若由 msprof / msprof op 拉起，返回其命令行，否则 None。"""
    for key in ("PROFILING_MODE", "PROFILER_SAMPLECONFIG"):
        if os.environ.get(key):
            return f"env {key}={os.environ.get(key)}"
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            comm = open(f"/proc/{pid}/comm", encoding="utf-8", errors="replace").read().strip()
            raw = (
                open(f"/proc/{pid}/cmdline", "rb")
                .read()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
                .strip()
            )
            status = open(f"/proc/{pid}/status", encoding="utf-8", errors="replace").read()
            ppid = int(status.split("PPid:", 1)[1].split()[0])
        except (OSError, IndexError, ValueError):
            break
        base = os.path.basename(raw.split()[0]) if raw else comm
        if comm == "msprof" or base == "msprof":
            return raw or comm
        pid = ppid
    return None


K = 1024
M = 1024 * 1024
DTYPE = torch.bfloat16
TARGET_ITER = 3
E2E_WARMUP = 5
E2E_ITERS = 10
NUM_CASES = 50
BLOCK_SIZE = 128
PROFILE_ROOT = "./batchtest/profile_fa4_fused_infer_paged"
RESULT_CSV = "./batchtest/compare_fa4_fused_infer_paged.csv"
_CANN_MIN_FREE = 20 * 1024 * 1024
_PROFILE_FREE_FLOOR = 1024 * 1024 * 1024
ATTN_MASK_SIZE = 2048
SPARSE_MODE_CAUSAL = 3
HBM_RATIO = 0.45
ELEM_BYTES = 2
FUSED_SINK_MAX_HEAD_DIM = 128
FUSED_MAX_HEAD_DIM = 256
MAX_BLOCK_SIZE = 512

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
SCENES = ("causal_prefill", "causal_decode", "nonmask")


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


def pages_per_seq(kv_s: int, block_size: int) -> int:
    return (kv_s + block_size - 1) // block_size


def tensor_bytes(
    batch: int,
    nheads: int,
    nkv: int,
    q_s: int,
    kv_s: int,
    head_dim: int,
    block_size: int,
    use_sink: bool,
    kv_layout: str,
) -> int:
    """Q+O 一份。BBND 另有一份 FIA 用的 BNBD 副本；BNBD 与 FA4 视图共享物理页。"""
    q_elems = batch * q_s * nheads * head_dim
    kv_tokens = batch * pages_per_seq(kv_s, block_size) * block_size
    kv_elems = kv_tokens * nkv * head_dim
    sink_elems = nheads if use_sink else 0
    kv_copies = 4 if kv_layout == "BBND" else 2
    return ELEM_BYTES * (2 * q_elems + kv_copies * kv_elems + sink_elems)


def flops(batch: int, nheads: int, q_s: int, kv_s: int, head_dim: int, causal: bool) -> int:
    qk_pv = 4 * batch * nheads * q_s * kv_s * head_dim
    if causal and q_s == kv_s:
        return qk_pv // 2
    return qk_pv


def fused_supported(head_dim: int, use_sink: bool) -> bool:
    """FIA TND head_dim<=256；learnable_sink 还要求 value D 为 64 或 128。"""
    if head_dim > FUSED_MAX_HEAD_DIM:
        return False
    if use_sink and head_dim not in (64, FUSED_SINK_MAX_HEAD_DIM):
        return False
    return True


def make_case(
    scene: str,
    causal: bool,
    use_sink: bool,
    batch: int,
    nheads: int,
    nkv: int,
    q_s: int,
    kv_s: int,
    head_dim: int,
    block_size: int,
    kv_layout: str,
    family: str = "",
) -> dict:
    scale = head_dim ** -0.5
    pages = pages_per_seq(kv_s, block_size)
    sink_tag = "sink" if use_sink else "nosink"
    mask_tag = "causal" if causal else "nomask"
    name = (
        f"{scene}_TND_{kv_layout}_{mask_tag}_{sink_tag}_B{batch}_N{nheads}x{nkv}_"
        f"q{fmt_len(q_s)}_kv{fmt_len(kv_s)}_d{head_dim}_blk{block_size}"
    )
    case = {
        "scene": scene,
        "layout": "TND",
        "kv_layout": kv_layout,
        "mask": mask_tag,
        "causal": causal,
        "use_sink": use_sink,
        "batch": batch,
        "nheads": nheads,
        "nkv": nkv,
        "q_s": q_s,
        "kv_s": kv_s,
        "head_dim": head_dim,
        "block_size": block_size,
        "pages_per_seq": pages,
        "num_blocks": batch * pages,
        "pair_family": family,
        "scale": scale,
        "name": name,
        "total_q": batch * q_s,
        "total_kv": batch * kv_s,
        "bytes": tensor_bytes(
            batch, nheads, nkv, q_s, kv_s, head_dim, block_size, use_sink, kv_layout,
        ),
        "flops": flops(batch, nheads, q_s, kv_s, head_dim, causal),
    }
    case["fused_ok"] = fused_supported(head_dim, use_sink)
    return case


def iter_all_cases(
    scenes: set[str],
    block_size: int,
    sink_modes: tuple[bool, ...],
    kv_layouts: tuple[str, ...],
) -> list[dict]:
    cases = []

    def emit(scene, causal, batch, nheads, nkv, q_s, kv_s, head_dim, family=""):
        for kv_layout in kv_layouts:
            for use_sink in sink_modes:
                cases.append(
                    make_case(
                        scene, causal, use_sink,
                        batch, nheads, nkv, q_s, kv_s, head_dim, block_size, kv_layout, family,
                    )
                )

    if "causal_prefill" in scenes:
        for batch in B_PREFILL:
            for nheads, nkv in GQA_CAUSAL:
                for seq in S_EQUAL:
                    for head_dim in D_CAUSAL:
                        emit("causal_prefill", True, batch, nheads, nkv, seq, seq, head_dim)
    if "causal_decode" in scenes:
        for batch in B_OTHER:
            for nheads, nkv in GQA_CAUSAL:
                for q_s in QS_DECODE:
                    for kv_s in S_EQUAL:
                        for head_dim in D_CAUSAL:
                            emit("causal_decode", True, batch, nheads, nkv, q_s, kv_s, head_dim)
    if "nonmask" in scenes:
        for batch in B_OTHER:
            for nheads, nkv in GQA_NONMASK:
                for family, q_s, kv_s in NONMASK_PAIRS:
                    for head_dim in D_NONMASK:
                        emit("nonmask", False, batch, nheads, nkv, q_s, kv_s, head_dim, family)
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
    by_bucket: dict[tuple[str, bool, str], list[dict]] = {}
    for case in cases:
        by_bucket.setdefault((case["scene"], case["use_sink"], case["kv_layout"]), []).append(case)
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
    block_size = case["block_size"]
    num_blocks = case["num_blocks"]
    pages = case["pages_per_seq"]
    kw = dict(dtype=DTYPE, device=device, generator=gen)
    q = torch.randn(batch * q_s, nheads, head_dim, **kw)
    if case["kv_layout"] == "BBND":
        k = torch.randn(num_blocks, block_size, nkv, head_dim, **kw)
        v = torch.randn(num_blocks, block_size, nkv, head_dim, **kw)
        # FIA 4D paged 按 (num_blocks, Nkv, block_size, D) 解读。
        k_fia = k.transpose(1, 2).contiguous()
        v_fia = v.transpose(1, 2).contiguous()
    elif case["kv_layout"] == "BNBD":
        k_fia = torch.randn(num_blocks, nkv, block_size, head_dim, **kw)
        v_fia = torch.randn(num_blocks, nkv, block_size, head_dim, **kw)
        # FA4 逻辑 shape 仍是 (num_blocks, block_size, Nkv, D)，stride 走 BNBD。
        k = k_fia.transpose(1, 2)
        v = v_fia.transpose(1, 2)
    else:
        raise ValueError(f"未知 kv_layout {case['kv_layout']}")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(batch, pages)
    cu_q = torch.tensor([0] + prefix_sums(q_s, batch), dtype=torch.int32, device=device)
    seqused_k = torch.full((batch,), kv_s, dtype=torch.int32, device=device)
    sink = None
    if case["use_sink"]:
        sink = torch.randn(nheads, dtype=torch.bfloat16, device=device, generator=gen)
    return {
        "q": q,
        "k": k,
        "v": v,
        "k_fia": k_fia,
        "v_fia": v_fia,
        "sink": sink,
        "block_table": block_table,
        "cu_q": cu_q,
        "seqused_k": seqused_k,
        "max_q": q_s,
        "max_kv": kv_s,
        # TND query 的 actual_seq 是前缀和；paged KV 的 actual_seq 是每条序列的长度。
        "actual_q": prefix_sums(q_s, batch),
        "actual_kv": [kv_s] * batch,
    }


def run_fa4(inp: dict, case: dict, *, return_lse: bool = False):
    _load_runtime()
    kwargs = dict(
        cu_seqlens_q=inp["cu_q"],
        max_seqlen_q=inp["max_q"],
        max_seqlen_k=inp["max_kv"],
        seqused_k=inp["seqused_k"],
        page_table=inp["block_table"],
        softmax_scale=case["scale"],
        causal=case["causal"],
        window_size=(-1, -1),
        return_lse=True,
    )
    if inp["sink"] is not None:
        kwargs["learnable_sink"] = inp["sink"]
    return flash_attn_varlen_func(inp["q"], inp["k"], inp["v"], **kwargs)


def run_fused(inp: dict, case: dict, atten_mask: torch.Tensor, *, return_lse: bool = False):
    kwargs = dict(
        num_query_heads=case["nheads"],
        num_key_value_heads=case["nkv"],
        input_layout="TND",
        softmax_scale=case["scale"],
        pre_tokens=2147483647,
        next_tokens=0 if case["causal"] else 2147483647,
        sparse_mode=SPARSE_MODE_CAUSAL if case["causal"] else 0,
        inner_precise=0,
        atten_mask=select_fused_mask(case, atten_mask),
        actual_seq_qlen=inp["actual_q"],
        actual_seq_kvlen=inp["actual_kv"],
        block_table=inp["block_table"],
        block_size=case["block_size"],
        return_softmax_lse=True,
    )
    if inp["sink"] is not None:
        kwargs["learnable_sink"] = inp["sink"]
    return torch_npu.npu_fused_infer_attention_score_v2(
        inp["q"], inp["k_fia"], inp["v_fia"], **kwargs,
    )


def align_fused_lse(lse_fa4: torch.Tensor, lse_fused: torch.Tensor) -> torch.Tensor:
    """FA4 TND lse 是 (N, T)，FIA TND lse 是 (T, N, 1)。"""
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
        _load_analyse()
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


def _stat_line(rows: list[dict], ratio_key: str, delta_key: str) -> str:
    ratios = [r[ratio_key] for r in rows]
    deltas = [r[delta_key] for r in rows]
    return (
        f"n={len(rows)} ratio mean={mean(ratios):.3f}x median={median(ratios):.3f}x | "
        f"delta mean={mean(deltas):.3f}ms median={median(deltas):.3f}ms"
    )


def analyze_results(results: list[dict]) -> None:
    ok = [r for r in results if r.get("fa4_ms") and r.get("fused_ms")]
    e2e_ok = [r for r in results if r.get("fa4_e2e_ms") and r.get("fused_e2e_ms")]
    prec_ok = [r for r in results if r.get("out_max_abs") is not None]

    def stat(vals):
        return mean(vals), median(vals), min(vals), max(vals)

    if prec_ok:
        print("\n" + "=" * 96)
        print("精度 FA4 vs fused_infer（TND + BBND/BNBD paged KV，sink/nosink）")
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
        print("FA4 相对 fused_infer（第 3 次迭代 kernel Duration）")
        print("=" * 96)
        print(f"有效用例: {len(ok)}/{len(results)}")
        print(f"ratio  (fa4/fused): mean={r_mean:.3f}x  median={r_med:.3f}x  min={r_min:.3f}x  max={r_max:.3f}x")
        print(f"delta  (fa4-fused): mean={d_mean:.3f}ms median={d_med:.3f}ms min={d_min:.3f}ms max={d_max:.3f}ms")
        print(f"劣化百分比 (ratio-1): mean={(r_mean - 1) * 100:.1f}%  median={(r_med - 1) * 100:.1f}%")
        for scene in SCENES:
            for kv_layout in ("BBND", "BNBD"):
                for use_sink in (True, False):
                    sub = [
                        r for r in ok
                        if r["scene"] == scene and r["use_sink"] is use_sink and r["kv_layout"] == kv_layout
                    ]
                    if not sub:
                        continue
                    tag = "sink" if use_sink else "nosink"
                    print(f"\n[{scene}/{kv_layout}/{tag}] {_stat_line(sub, 'ratio', 'delta_ms')}")

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
        for scene in SCENES:
            for kv_layout in ("BBND", "BNBD"):
                for use_sink in (True, False):
                    sub = [
                        r for r in e2e_ok
                        if r["scene"] == scene and r["use_sink"] is use_sink and r["kv_layout"] == kv_layout
                    ]
                    if not sub:
                        continue
                    tag = "sink" if use_sink else "nosink"
                    print(f"\n[{scene}/{kv_layout}/{tag}] {_stat_line(sub, 'e2e_ratio', 'e2e_delta_ms')}")
    if not ok and not e2e_ok and not prec_ok:
        print("无有效结果")
    print("=" * 96)


def save_csv(results: list[dict], path: str) -> None:
    fields = [
        "case_id", "label", "scene", "layout", "kv_layout", "mask", "name",
        "causal", "use_sink", "fused_ok",
        "batch", "nheads", "nkv", "q_s", "kv_s", "head_dim", "block_size",
        "pages_per_seq", "num_blocks", "pair_family",
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
        "causal": {"causal_prefill", "causal_decode"},
        "causal_prefill": {"causal_prefill"},
        "causal_decode": {"causal_decode"},
        "nonmask": {"nonmask"},
        "nomask": {"nonmask"},
        "none": {"nonmask"},
    }
    out: set[str] = set()
    for part in raw.split(","):
        key = part.strip().lower()
        if key not in alias:
            raise ValueError(
                f"未知 --scene {part!r}，可选 all,causal,causal_prefill,causal_decode,nonmask"
            )
        out |= alias[key]
    return out


def parse_kv_layouts(raw: str) -> tuple[str, ...]:
    order: list[str] = []
    for part in raw.split(","):
        key = part.strip().upper()
        if key in ("BOTH", "ALL"):
            picked = ("BBND", "BNBD")
        elif key in ("BBND", "BNBD"):
            picked = (key,)
        else:
            raise ValueError(f"未知 --kv-layout {part!r}，可选 BBND,BNBD,both")
        for item in picked:
            if item not in order:
                order.append(item)
    return tuple(order)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FA4 vs fused_infer，TND query + BBND/BNBD paged KV，sink/nosink，causal/no mask"
    )
    parser.add_argument("--device", default="npu:5")
    parser.add_argument(
        "--scene",
        default="all",
        help="all / causal / causal_prefill,causal_decode,nonmask。causal 含 prefill+decode",
    )
    parser.add_argument(
        "--sink",
        default="both",
        choices=["both", "sink", "none"],
        help="both=sink 与 nosink；sink=只跑 learnable_sink；none=不传 sink",
    )
    parser.add_argument(
        "--kv-layout",
        default="both",
        help="BBND / BNBD / both。BBND 物理页给 FA4，BNBD 物理页给 FIA",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help="paged KV block size，须为 16 的倍数且不超过 512",
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
    parser.add_argument(
        "--disable-msprof",
        action="store_true",
        help="关闭脚本内 torch_npu profiler。外层 msprof op 采集时必须关掉，否则两层 profiler 抢同一条通道",
    )
    parser.add_argument(
        "--enable-msprof",
        action="store_true",
        help="检测到外层 msprof 时仍开启脚本内 profiler",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="同 --disable-msprof，跳过脚本内 profiler kernel 计时",
    )
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
        default=FUSED_MAX_HEAD_DIM,
        help="过滤 head_dim 上限；默认 256。sink 仍只保留 D=64/128",
    )
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="保留 FIA 不支持的用例（sink 且 D 不是 64/128）；这些用例只跑 FA4 e2e",
    )
    args = parser.parse_args()
    if args.enable_msprof and (args.disable_msprof or args.skip_profile):
        raise ValueError("--enable-msprof 与 --disable-msprof / --skip-profile 不能同时使用")
    external = external_msprof()
    if args.disable_msprof or args.skip_profile:
        args.skip_profile = True
        print("脚本内 profiler 已关闭（--disable-msprof / --skip-profile），可配合外层 msprof op。")
    elif external and not args.enable_msprof:
        args.skip_profile = True
        print(
            f"检测到外层 msprof（{external}），已关闭脚本内 profiler，"
            "避免双采集抢通道。需要脚本内 kernel 计时时去掉外层 msprof，或加 --enable-msprof。"
        )
    elif external and args.enable_msprof:
        print(
            f"检测到外层 msprof（{external}）；已按 --enable-msprof 保留脚本内 profiler，"
            "两层采集可能互相干扰。"
        )

    if args.block_size <= 0 or args.block_size > MAX_BLOCK_SIZE or args.block_size % 16 != 0:
        raise ValueError(f"--block-size 必须是 (0, {MAX_BLOCK_SIZE}] 内的 16 倍数，收到 {args.block_size}")

    scenes = parse_scenes(args.scene)
    sink_modes = {"both": (True, False), "sink": (True,), "none": (False,)}[args.sink]
    kv_layouts = parse_kv_layouts(args.kv_layout)
    universe = iter_all_cases(scenes, args.block_size, sink_modes, kv_layouts)
    universe = [c for c in universe if c["head_dim"] <= args.max_head_dim]
    if not args.include_unsupported:
        before = len(universe)
        universe = [c for c in universe if c["fused_ok"]]
        print(
            f"FIA 过滤: TND+paged，sink 仅 D=64/128，nosink D<={args.max_head_dim} → "
            f"{len(universe)}/{before} 组（可用 --include-unsupported 放宽）"
        )

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
        raise RuntimeError("过滤后无用例，请放宽 --max-head-dim / --sink / --include-unsupported")

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
        f"场景 {sorted(scenes)} sink={args.sink} kv={list(kv_layouts)} block={args.block_size} "
        f"候选 {len(universe)} 组 | dtype=bf16 | TND+paged | "
        + (
            "脚本内 profiler 关闭 | "
            if args.skip_profile
            else f"第{TARGET_ITER}次迭代 kernel Duration | "
        )
        + f"e2e median warmup={args.e2e_warmup} iters={args.e2e_iters}"
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
        print(
            f"HBM 过滤: Q+O+paged K/V+sink <= "
            f"{hbm_limit / (1024 ** 3):.2f} GiB ({args.hbm_ratio:.0%} 卡容量)"
        )
        feasible, skipped = filter_hbm(universe, hbm_limit)
        print(f"可装下 {len(feasible)} 组，跳过 {skipped} 组（估算会 OOM）")

    if not feasible:
        raise RuntimeError("HBM 过滤后无可用用例，请增大 --hbm-ratio 或缩小场景")

    cases = feasible if (args.full or args.name) else sample_cases(feasible, args.num_cases)
    prefix = {
        ("causal_prefill", True): "cps",
        ("causal_prefill", False): "cpn",
        ("causal_decode", True): "cds",
        ("causal_decode", False): "cdn",
        ("nonmask", True): "nms",
        ("nonmask", False): "nmn",
    }
    for i, case in enumerate(cases, 1):
        case["case_id"] = i
        layout_tag = "bb" if case["kv_layout"] == "BBND" else "bn"
        case["label"] = f"{layout_tag}{prefix[(case['scene'], case['use_sink'])]}_{i:03d}"

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
        args.profile_root, "profile_fa4_fused_infer_paged", _PROFILE_FREE_FLOOR,
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
    print("\n" + "-" * 188)
    print(
        f"{'id':>3} {'label':>10} {'kv':<5} {'scene':<16} {'mask':<7} {'sink':<7} {'name':<42} "
        f"{'max|d|':>10} {'cos':>7} "
        f"{'k_fa4':>8} {'k_fused':>8} {'k_ratio':>7} "
        f"{'e2e_fa4':>8} {'e2e_fus':>8} {'e2e_ratio':>9}"
    )
    print("-" * 188)
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
        sink_tag = "sink" if row["use_sink"] else "nosink"
        print(
            f"{row['case_id']:>3} {row['label']:>10} {row['kv_layout']:<5} {row['scene']:<16} {row['mask']:<7} {sink_tag:<7} "
            f"{row['name']:<42.42} {max_d:>10} {cos:>7} "
            f"{k_fa4:>8} {k_fused:>8} {k_ratio:>7} "
            f"{e_fa4:>8} {e_fused:>8} {e_ratio:>9}"
        )
    analyze_results(results)


if __name__ == "__main__":
    main()
