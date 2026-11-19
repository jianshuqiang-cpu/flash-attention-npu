"""Check and profile Ascend950 FA3 on the CATLASS ASWT attention analogues."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from statistics import median


DOC_MNK = (
    (1024, 1024, 1024),
    (2048, 2048, 256),
    (2208, 2048, 512),
    (2208, 2048, 1024),
    (1024, 2368, 512),
    (1024, 2368, 1024),
    (1024, 2368, 2048),
)
CASES = {"decode-d192": (1, 65535, 64, 4, 192)}
for queries, keys, reduction in DOC_MNK:
    width = min(512, reduction)
    heads = reduction // width
    CASES[f"{queries}x{keys}x{reduction}"] = (queries, keys, heads, heads, width)


def summarize(directory):
    log = (directory / "msprof.log").read_text()
    if (
        "Completed 10 attention samples" not in log
        or "Traceback" in log
        or re.search(r"signal\s+\d+", log, re.I)
    ):
        raise ValueError(f"Incomplete or abnormal application in {directory}")
    (identity,) = [
        json.loads(line) for line in log.splitlines() if line.startswith("{") and '"binary"' in line
    ]
    processes = (directory / "processes.log").read_text()
    observed = set(map(int, re.findall(r"Process id:\s*(\d+)", processes)))
    if "query failed" in processes or observed != {identity["pid"]}:
        raise ValueError(f"Unverified or contended device in {directory}: {observed}")
    (trace,) = directory.rglob("op_summary_*.csv")
    with trace.open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: float(row["Task Start Time(us)"]))
    markers = [index for index, row in enumerate(rows) if "ArgMax" in row["Op Name"]]
    if len(markers) != 11:
        raise ValueError(f"Expected 11 ArgMax markers, got {len(markers)}")
    samples = []
    for start, end in zip(markers, markers[1:]):
        (main,) = [row for row in rows[start + 1 : end] if "FAInfer" in row["Op Name"]]
        duration = float(main["Task Duration(us)"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Bad main-kernel duration: {duration}")
        samples.append(duration)
    return dict(
        binary=identity["binary"],
        sha256=identity.get("sha256", identity.get("binary_sha256")),
        samples_us=samples,
        median_us=median(samples),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", type=Path, help="Read one completed, monitored msprof round")
    parser.add_argument("--compare", type=Path, help="Compare all A/B/B/A round summaries")
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.summary:
        print(json.dumps(summarize(args.summary)))
        return
    if args.compare:
        writer = csv.writer(sys.stdout)
        writer.writerow(("case", "baseline_us", "candidate_us", "speedup", "samples_per_variant"))
        for case in CASES:
            rounds = [
                json.loads((args.compare / f"{case}-{label}" / "summary.json").read_text())
                for label in ("A1", "B1", "B2", "A2")
            ]
            first, second, third, fourth = [record["sha256"] for record in rounds]
            if not first or not second or first != fourth or second != third or first == second:
                raise ValueError(f"Invalid A/B binary identities for {case}")
            baseline = median(rounds[0]["samples_us"] + rounds[3]["samples_us"])
            candidate = median(rounds[1]["samples_us"] + rounds[2]["samples_us"])
            writer.writerow((case, baseline, candidate, baseline / candidate, 20))
        return
    if args.list:
        for case, shape in CASES.items():
            print(f"{case}: Sq,Sk,Hq,Hkv,D={shape}")
        return
    if not args.case:
        parser.error("--case is required unless --list is given")

    import torch
    import torch_npu

    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name(device)
    if "Ascend950" not in device_name:
        raise RuntimeError(f"Expected Ascend950, found {device_name}")
    api = import_module("flash_attn_npu_3.flash_attn_npu_interface_950")
    binary = Path(import_module("flash_attn_npu_3_950").__file__).resolve()
    if binary.parent != checkout or not Path(api.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError(f"Unexpected FA3 checkout: {api.__file__}, {binary}")
    from tests.common.attention_ref import ref_flash_attention_pair
    from tests.common.compare import assert_fa_close

    q_len, kv_len, heads_q, heads_kv, width = CASES[args.case]
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(
        json.dumps(
            dict(
                case=args.case,
                shape=CASES[args.case],
                pid=os.getpid(),
                binary=str(binary),
                sha256=digest,
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
                ).strip(),
                device=device_name,
                logical_device=args.device,
                visible=os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                splits=1,
            )
        ),
        flush=True,
    )
    generator = torch.Generator().manual_seed(0)
    dtype = torch.float16
    inputs = [
        torch.randn((1, length, heads, width), generator=generator, dtype=dtype)
        for length, heads in ((q_len, heads_q), (kv_len, heads_kv), (kv_len, heads_kv))
    ]
    with torch.inference_mode():
        tensors = [tensor.to(device) for tensor in inputs]
        out, lse = api.flash_attn_func(*tensors, num_splits=1, return_attn_probs=True)
        out_ref, lse_ref, out_pt, lse_pt = ref_flash_attention_pair(
            *inputs, width**-0.5, None, dtype
        )
        for actual, ref, low, name in (
            (out, out_ref, out_pt, "out"),
            (lse, lse_ref, lse_pt, "lse"),
        ):
            if actual.shape != ref.shape:
                raise ValueError(f"{name}: {actual.shape} != {ref.shape}")
            assert_fa_close(actual.cpu(), ref, low, name=name)
        print("O and LSE passed", flush=True)
        if args.check_only:
            return
        marker = torch.empty((128, 1024, 1024), dtype=dtype, device=device)
        flush_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=device)
        for _ in range(10):
            torch.argmax(marker)
            torch.npu.synchronize()
            flush_buffer.zero_()
            torch.npu.synchronize()
            api.flash_attn_func(*tensors, num_splits=1)
            torch.npu.synchronize()
        torch.argmax(marker)
        torch.npu.synchronize()
        print("Completed 10 attention samples", flush=True)


if __name__ == "__main__":
    main()
