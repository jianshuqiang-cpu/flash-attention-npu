"""Fixed-shape Ascend950 attention runner for msprof application profiling."""

import argparse
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch_npu


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--fa", type=int, choices=(3, 4))
    backend.add_argument("--fia", action="store_true", help="Use torch_npu.npu_fused_infer_attention_score")
    parser.add_argument("--num-splits", type=int, choices=(0, 1), default=1)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True,
                        help="Sweep a separate 256 MiB NPU buffer before each measured call")
    parser.add_argument("--device", type=int, default=0, help="Logical NPU index")
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    if args.fia:
        binary = Path(torch_npu._C.__file__).resolve()
        commit = None
    else:
        sys.path.insert(0, str(checkout))
        module = f"flash_attn_npu_{args.fa}"
        api = import_module(f"{module}.flash_attn_npu_interface_950")
        binary = Path(import_module(f"{module}_950").__file__).resolve()
        if not Path(api.__file__).resolve().is_relative_to(checkout) or binary.parent != checkout:
            raise RuntimeError(f"Expected an in-place 950 build in {checkout}, got {api.__file__}, {binary}")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name(device)
    if "Ascend950" not in device_name:
        raise RuntimeError(f"Expected Ascend950, got {device_name}")
    torch.npu.manual_seed(0)
    q = torch.randn((1, 1, 64, 192), dtype=torch.float16, device=device)
    k = torch.randn((1, 65535, 4, 192), dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    warmup = torch.empty((128, 1024, 1024), dtype=torch.float16, device=device)
    flush_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=device) if args.flush_l2 else None
    with binary.open("rb") as binary_file:
        binary_sha256 = hashlib.file_digest(binary_file, "sha256").hexdigest()
    print(json.dumps({
        "pid": os.getpid(),
        "commit": commit,
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "device": device_name,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "api": "npu_fused_infer_attention_score" if args.fia else "flash_attn_func",
        "fa": args.fa,
        "num_splits": None if args.fia else args.num_splits,
        "flush_l2_bytes": 256 * 1024 * 1024 if args.flush_l2 else 0,
        "q": list(q.shape),
        "k": list(k.shape),
        "v": list(v.shape),
        "dtype": str(q.dtype),
    }), flush=True)

    def forward():
        if args.fia:
            return torch_npu.npu_fused_infer_attention_score(
                q, k, v,
                input_layout="BSND",
                num_heads=q.shape[2],
                num_key_value_heads=k.shape[2],
                scale=q.shape[-1] ** -0.5,
            )
        return api.flash_attn_func(q, k, v, num_splits=args.num_splits)

    result = forward()
    out = result[0] if isinstance(result, tuple) else result
    if out.shape != q.shape or not torch.isfinite(out).all().item():
        raise RuntimeError("Attention produced an invalid output")
    torch.npu.synchronize()

    print(f"Sample window begin {time.time_ns()}", flush=True)
    for _ in range(10):
        torch.argmax(warmup)
        torch.npu.synchronize()
        if flush_buffer is not None:
            flush_buffer.zero_()
            torch.npu.synchronize()
        forward()
        torch.npu.synchronize()
    torch.argmax(warmup)  # Delimit the final measured call in the device trace.
    torch.npu.synchronize()
    print(f"Sample window end {time.time_ns()}", flush=True)
    print("Completed 10 attention samples", flush=True)


if __name__ == "__main__":
    main()
