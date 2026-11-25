#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np


def set_blas_threads(thread_num):
    os.environ["OMP_NUM_THREADS"] = str(thread_num)
    os.environ["MKL_NUM_THREADS"] = str(thread_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_num)


def get_data_dir(save_path):
    return os.path.join(save_path, "data")


def get_golden_dir(save_path):
    return os.path.join(save_path, "golden")


def gen_data(m, n, k, seed, blas_threads, save_path):
    set_blas_threads(blas_threads)
    data_dir = get_data_dir(save_path)
    golden_dir = get_golden_dir(save_path)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    a_real = rng.uniform(-5.0, 5.0, size=(m, k)).astype(np.float16)
    a_imag = rng.uniform(-5.0, 5.0, size=(m, k)).astype(np.float16)
    b_real = rng.uniform(-5.0, 5.0, size=(k, n)).astype(np.float16)
    b_imag = rng.uniform(-5.0, 5.0, size=(k, n)).astype(np.float16)

    a_real.tofile(os.path.join(data_dir, "inputA_real.dat"))
    a_imag.tofile(os.path.join(data_dir, "inputA_imag.dat"))
    b_real.T.copy().tofile(os.path.join(data_dir, "inputB_real.dat"))
    b_imag.T.copy().tofile(os.path.join(data_dir, "inputB_imag.dat"))

    a_r = a_real.astype(np.float32)
    a_i = a_imag.astype(np.float32)
    b_r = b_real.astype(np.float32)
    b_i = b_imag.astype(np.float32)
    c_real = np.matmul(a_r, b_r) - np.matmul(a_i, b_i)
    c_imag = np.matmul(a_r, b_i) + np.matmul(a_i, b_r)

    c_real.tofile(os.path.join(golden_dir, "goldenC_real.dat"))
    c_imag.tofile(os.path.join(golden_dir, "goldenC_imag.dat"))
    print(f"Data generated: M={m}, N={n}, K={k}, BLAS threads={blas_threads}")


def get_default_op_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    catlass_home_dir = os.path.dirname(os.path.dirname(current_dir))
    return os.path.join(catlass_home_dir, "output", "bin", "77_planar_complex_matmul")


def run_op(op_path, m, n, k, device_id, save_path):
    data_dir = os.path.abspath(get_data_dir(save_path))
    print("------计算npu------")
    result = subprocess.run(
        [op_path, str(m), str(n), str(k), str(device_id), "--datapath", data_dir],
        capture_output=True,
        text=True,
    )
    print(f"npu op run log = {result.stdout}")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        print(
            f"Kernel exited with non-zero returncode {result.returncode}",
            file=sys.stderr,
        )
        return False
    if "KERNEL FAILED" in result.stdout or "Kernel failed" in result.stderr:
        print("Kernel failure marker detected in output", file=sys.stderr)
        return False
    return True


def compute_errors(result, golden):
    diff = np.abs(result - golden)
    relative_errors = diff / np.maximum(1.0, np.abs(golden))
    return (
        float(np.max(relative_errors)),
        float(np.mean(relative_errors)),
        float(np.max(diff)),
    )


def compare(max_rel_err, save_path):
    data_dir = get_data_dir(save_path)
    golden_dir = get_golden_dir(save_path)
    result_real_path = os.path.join(data_dir, "outputC_real.dat")
    result_imag_path = os.path.join(data_dir, "outputC_imag.dat")
    if not os.path.exists(result_real_path) or not os.path.exists(result_imag_path):
        print("NPU output files not found", file=sys.stderr)
        return False

    result_real = np.fromfile(result_real_path, dtype=np.float32)
    result_imag = np.fromfile(result_imag_path, dtype=np.float32)
    golden_real = np.fromfile(
        os.path.join(golden_dir, "goldenC_real.dat"), dtype=np.float32
    )
    golden_imag = np.fromfile(
        os.path.join(golden_dir, "goldenC_imag.dat"), dtype=np.float32
    )

    if result_real.shape != golden_real.shape or result_imag.shape != golden_imag.shape:
        print(
            f"Shape mismatch: real {result_real.shape} vs {golden_real.shape}, imag {result_imag.shape} vs {golden_imag.shape}"
        )
        return False

    real_max_rel, real_mean_rel, real_max_abs = compute_errors(result_real, golden_real)
    imag_max_rel, imag_mean_rel, imag_max_abs = compute_errors(result_imag, golden_imag)
    precision_metric = max(real_max_rel, imag_max_rel)

    print("------ 计算相对误差 -----")
    print(
        f"real max_rel_error = {real_max_rel:.8f}, mean_rel_error = {real_mean_rel:.8f}, max_abs_error = {real_max_abs:.8f}"
    )
    print(
        f"imag max_rel_error = {imag_max_rel:.8f}, mean_rel_error = {imag_mean_rel:.8f}, max_abs_error = {imag_max_abs:.8f}"
    )
    print(f"Precision metric: {precision_metric:.8f}")
    print("------ 开始比较 ------")
    return np.isfinite(precision_metric) and precision_metric <= max_rel_err


def main():
    parser = argparse.ArgumentParser(
        description="Generate planar complex GEMM data, run NPU operator, and compare result"
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("device_id", type=int, nargs="?", default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blas_threads", type=int, default=8)
    parser.add_argument("--max_rel_err", type=float, default=5e-2)
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--clean", choices=["true", "false"], default="true")
    args = parser.parse_args()
    save_path = os.path.abspath(args.save_path)

    print(f"m={args.m}")
    print(f"n={args.n}")
    print(f"k={args.k}")
    print(f"device_id={args.device_id}")
    print(f"seed={args.seed}")
    print(f"blas_threads={args.blas_threads}")
    print(f"max_rel_err={args.max_rel_err}")
    print(f"save_path={save_path}")
    print(f"clean={args.clean}")

    try:
        print("------计算golden------")
        gen_data(args.m, args.n, args.k, args.seed, args.blas_threads, save_path)

        op_path = get_default_op_path()
        if not os.path.exists(op_path):
            print(f"operator binary not found: {op_path}", file=sys.stderr)
            sys.exit(1)

        run_ok = run_op(op_path, args.m, args.n, args.k, args.device_id, save_path)
        compare_ok = run_ok and compare(args.max_rel_err, save_path)
        res = "Compare success" if compare_ok else "Compare false"
        print(f"比较结果：{res}")
        sys.exit(0 if compare_ok else 1)
    finally:
        if args.clean == "true":
            shutil.rmtree(get_data_dir(save_path), ignore_errors=True)
            shutil.rmtree(get_golden_dir(save_path), ignore_errors=True)


if __name__ == "__main__":
    main()
