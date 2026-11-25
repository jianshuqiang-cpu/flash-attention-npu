# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

_DSL_BASE_PATH = str((Path(__file__).resolve().parent / "../../../").resolve())

_DSL_PATH_ADDED = _DSL_BASE_PATH not in sys.path
if _DSL_PATH_ADDED:
    sys.path.insert(0, _DSL_BASE_PATH)

import argparse

import catlass.tla as tla

M_DIM = 32
N_DIM = 32
K_DIM = 32
UB_A_TILE_BYTES = M_DIM // 2 * K_DIM * 4
L1_STAGE_BYTES = 32 * 32 * 4
L0A_BYTES = 32 * 32 * 4
L0B_BYTES = 32 * 32 * 4
L0C_BYTES = 32 * 32 * 4

DESCRIPTION = "Basic Mixed UB→L1 + cross_core sync; f32 only."

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_mixed_ub2l1(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
) -> None:
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    ub_load_ready = tla.flag("ub_load_ready", tla.arch.MTE3, tla.arch.MTE2)
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.MTE3)

    ub2l1_ready = tla.cross_flag("ub2l1_ready")
    ub2l1_done = tla.cross_flag("ub2l1_done")

    l1a_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 512)

    ub_a_ptr = tla.allocate(UB_A_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        tla.cross_core_set_flag(ub2l1_ready, tla.arch.MTE1)

        gm_a = tla.tile_view(lhs, tla.make_shape(M_DIM, K_DIM), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(K_DIM, N_DIM), tla.make_coord(0, 0))
        gm_c = tla.tile_view(out, tla.make_shape(M_DIM, N_DIM), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)

        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        tla.cross_core_wait_flag(ub2l1_done, tla.arch.MTE1)

        l1_a_l0 = tla.tile_view(
            l1_a, tla.make_shape(M_DIM, K_DIM), tla.make_coord(0, 0)
        )
        l1_b_l0 = tla.tile_view(
            l1_b, tla.make_shape(K_DIM, N_DIM), tla.make_coord(0, 0)
        )
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        tla.copy(gm_c, l0_c)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_a_full = tla.tile_view(
            lhs, tla.make_shape(M_DIM, K_DIM), tla.make_coord(0, 0)
        )
        gm_a = tla.tile_view(
            lhs, tla.make_shape(M_DIM // 2, K_DIM), tla.make_coord(vec_idx, 0)
        )

        ub_a = tla.make_tensor_like(ub_a_ptr, gm_a, tla.arch.RowMajor)

        tla.set_flag(ub_load_ready)
        tla.wait_flag(ub_load_ready)

        tla.copy(ub_a, gm_a)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        l1_a_full = tla.make_tensor_like(l1a_ptr, gm_a_full, tla.arch.zN)
        l1_a = tla.tile_view(
            l1_a_full, tla.make_shape(M_DIM // 2, K_DIM), tla.make_coord(vec_idx, 0)
        )
        tla.cross_core_wait_flag(ub2l1_ready, tla.arch.MTE3)
        tla.copy(l1_a, ub_a)
        tla.cross_core_set_flag(ub2l1_done, tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def golden(lhs, rhs):
    import torch

    return lhs.to(torch.float32) @ rhs.to(torch.float32)


def prepare_npu(buf, layout: str):
    storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
    return storage.npu()


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    from examples.end_to_end.common import (
        get_block_num,
        create_tla_tensor,
        compare,
    )

    mi, ni, ki = int(args.m), int(args.n), int(args.k)

    torch.npu.set_device(args.device)
    print(f"--- mnk=({mi},{ni},{ki}) ---")
    lhs = torch.rand(mi, ki, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    rhs = torch.rand(ki, ni, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    out = torch.full((mi, ni), args.sentinel, dtype=torch.float32, device="cpu")
    ref = golden(lhs, rhs)

    lhs = prepare_npu(lhs, args.layout_a)
    rhs = prepare_npu(rhs, args.layout_b)
    out = prepare_npu(out, "row")
    a_tensor = create_tla_tensor(lhs, args.layout_a)
    b_tensor = create_tla_tensor(rhs, args.layout_b)
    c_tensor = create_tla_tensor(out, "row")

    artifact = tla.compile(
        basic_mixed_ub2l1, a_tensor, b_tensor, c_tensor, options="--npu-arch 3510"
    )
    block_num = get_block_num(args.block_num, args.device, kind="mix")
    artifact(a_tensor, b_tensor, c_tensor, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(out.detach().cpu(), ref, ki)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=M_DIM)
    parser.add_argument("--n", type=int, default=N_DIM)
    parser.add_argument("--k", type=int, default=K_DIM)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--block-num", type=int, default=-1)
    parser.add_argument("--sentinel", type=float, default=-9.0)
    try:
        return run(parser.parse_args())
    finally:
        if _DSL_PATH_ADDED:
            sys.path.remove(_DSL_BASE_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
