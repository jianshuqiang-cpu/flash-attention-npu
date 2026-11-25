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

import argparse

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.params import UnalignStoreParams, NormalLoadParams, LoadDist
from catlass.tla.runtime import from_dlpack

from fa_tiling import (
    QK_L1_TILE_M,
    QK_L1_TILE_N,
    QK_L1_TILE_K_LEFT,
    QK_L1_TILE_K_RIGHT,
    PV_L1_TILE_M,
    PV_L1_TILE_N,
    PV_L1_TILE_K_LEFT,
    PV_L1_TILE_K_RIGHT,
    Q_L1_BUF_NUM as Q_L1_BUF,
    K_L1_BUF_NUM as K_L1_BUF,
    V_L1_BUF_NUM as V_L1_BUF,
    P_L1_BUF_NUM as P_L1_BUF,
)
from fa_tiling import compute_tiling, pack_tiling_int, make_actual_seqlen

# tile 大小与硬件参数 (编译期常量)
HEAD_DIM = 128

Q_BLOCK = 128
KV_BLOCK = 128
Q_BLOCK_SUB = Q_BLOCK // 2

# 编译期形状参数（默认值，可由命令行 --batch/--headnum/--kvheadnum/--qseqlen/--kvseqlen 覆盖；
# run() 会按命令行参数重写这些模块全局并重算派生量，tla.compile 在 trace 时读取最新值）
BATCH = 1
HEAD_NUM = 8
KV_HEAD_NUM = 1
Q_SEQ = 117
KV_SEQ = 512
GROUP_SIZE = HEAD_NUM // KV_HEAD_NUM
Q_BLOCK_COUNT = (Q_SEQ + Q_BLOCK - 1) // Q_BLOCK
KV_BLOCK_COUNT = (KV_SEQ + KV_BLOCK - 1) // KV_BLOCK
TOTAL_TASKS = BATCH * HEAD_NUM * Q_BLOCK_COUNT

ALIGNMENT = 512
L0_HALF_SIZE = 32768
VL_FLOAT_ELE = 64
QK_SCALE = 1.0 / (HEAD_DIM**0.5)
MIN_VALUE = -3e38

# Prelaunch
PRE_LAUNCH = 2
MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1
UB_S_OTMP_BUF_STAGES = 2
UB_S_BUF_STAGES = 2

L0_TILE_M = 128
L0_TILE_N = 128
L0_TILE_K = 128
L0_STAGES = 2

mm1L0ATotalStages_ = (Q_BLOCK // L0_TILE_M) * (HEAD_DIM // L0_TILE_K)
mm1L0BTotalStages_ = (KV_BLOCK // L0_TILE_N) * (HEAD_DIM // L0_TILE_K)
mm2L0ATotalStages_ = (Q_BLOCK // L0_TILE_M) * (KV_BLOCK // L0_TILE_K)
mm2L0BTotalStages_ = (KV_BLOCK // L0_TILE_K) * (HEAD_DIM // L0_TILE_N)

QK_ML0_LOOP_NUM = (Q_BLOCK + L0_TILE_M - 1) // L0_TILE_M
QK_NL0_LOOP_NUM = (KV_BLOCK + L0_TILE_N - 1) // L0_TILE_N
QK_KL0_LOOP_NUM = (HEAD_DIM + L0_TILE_K - 1) // L0_TILE_K

PV_ML0_LOOP_NUM = (Q_BLOCK + L0_TILE_M - 1) // L0_TILE_M
PV_NL0_LOOP_NUM = (HEAD_DIM + L0_TILE_N - 1) // L0_TILE_N
PV_KL0_LOOP_NUM = (KV_BLOCK + L0_TILE_K - 1) // L0_TILE_K

UNCHANGED_THRESHOLD = 0.1


@tla.kernel
def flash_attention_infer_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_v: tla.Tensor,
    mem_o: tla.Tensor,
    mem_mask: tla.Tensor,  # 暂不支持，只能为全0
    tiling_data: tla.Tensor,
    actual_q_seqlen: tla.Tensor,
    actual_kv_seqlen: tla.Tensor,
):
    # QK->softmax (qkReadyFlag, id=0/1, AIC FIX set/wait <-> AIV V wait/set)
    qk_ready_0 = tla.cross_flag("qk_ready_0")
    qk_ready_1 = tla.cross_flag("qk_ready_1")
    # softmax->PV (softmaxReadyFlag, id=2/3/4, AIV MTE3 set <-> AIC MTE1 wait)
    sm_ready_mm2_0 = tla.cross_flag("sm_ready_mm2_0")
    sm_ready_mm2_1 = tla.cross_flag("sm_ready_mm2_1")
    sm_ready_mm2_2 = tla.cross_flag("sm_ready_mm2_2")
    # PV->rescale (pvReadyFlag, id=5/6, AIC FIX set <-> AIV V wait) — PV fixpipe L0C->UB 后通知 rescale
    pv_ready_0 = tla.cross_flag("pv_ready_0")
    pv_ready_1 = tla.cross_flag("pv_ready_1")

    q_l0a_ready_l1 = tla.flag("q_l0a_ready_l1", tla.arch.MTE1, tla.arch.MTE2)
    q_l1_ready_l0 = tla.flag("q_l1_ready_l0", tla.arch.MTE2, tla.arch.MTE1)

    k_l0b_ready_l1_0 = tla.flag("k_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_1 = tla.flag("k_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    k_l1_ready_l0_0 = tla.flag("k_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_1 = tla.flag("k_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    mmad_ready_l0a_0 = tla.flag("mmad_ready_l0a_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0a_1 = tla.flag("mmad_ready_l0a_1", tla.arch.CUBE, tla.arch.MTE1)
    l0a_ready_mmad_0 = tla.flag("l0a_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0a_ready_mmad_1 = tla.flag("l0a_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    mmad_ready_l0b_0 = tla.flag("mmad_ready_l0b_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_1 = tla.flag("mmad_ready_l0b_1", tla.arch.CUBE, tla.arch.MTE1)
    l0b_ready_mmad_0 = tla.flag("l0b_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_1 = tla.flag("l0b_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    fix_ready_mmad_qk_0 = tla.flag("fix_ready_mmad_qk_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_qk_1 = tla.flag("fix_ready_mmad_qk_1", tla.arch.FIX, tla.arch.CUBE)
    mmad_ready_fix_qk_0 = tla.flag("mmad_ready_fix_qk_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_qk_1 = tla.flag("mmad_ready_fix_qk_1", tla.arch.CUBE, tla.arch.FIX)

    fix_ready_mmad_pv_0 = tla.flag("fix_ready_mmad_pv_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_pv_1 = tla.flag("fix_ready_mmad_pv_1", tla.arch.FIX, tla.arch.CUBE)
    mmad_ready_fix_pv_0 = tla.flag("mmad_ready_fix_pv_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_pv_1 = tla.flag("mmad_ready_fix_pv_1", tla.arch.CUBE, tla.arch.FIX)

    v_l0b_ready_l1_0 = tla.flag("v_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_1 = tla.flag("v_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    v_l1_ready_l0_0 = tla.flag("v_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_1 = tla.flag("v_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    #   V_MTE3(ubSBufId)         EVENT_ID0/1 — softmax V 完 -> 触发 P->L1 MTE3
    #   MTE3_V(ubSBufId + 2)     EVENT_ID2/3 — 上一轮 P->L1 MTE3 完成 -> 本轮 softmax 可动
    mte3_ready_softmax_0 = tla.flag(
        "mte3_ready_softmax_0", tla.arch.MTE3, tla.arch.VECTOR
    )
    mte3_ready_softmax_1 = tla.flag(
        "mte3_ready_softmax_1", tla.arch.MTE3, tla.arch.VECTOR
    )
    v_mte3_0 = tla.flag("v_mte3_0", tla.arch.VECTOR, tla.arch.MTE3)
    v_mte3_1 = tla.flag("v_mte3_1", tla.arch.VECTOR, tla.arch.MTE3)
    # mask copy (MTE2->V)
    mask_copy_end = tla.flag("mask_copy_end", tla.arch.MTE2, tla.arch.VECTOR)
    # rescale MTE3→V : 上一轮 O->GM MTE3 完成后本轮 rescale 可动
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)
    # rescale V→MTE3 : cast 写完 ub_out_f16 后才 O->GM MTE3
    rescale_v_mte3 = tla.flag("rescale_v_mte3", tla.arch.VECTOR, tla.arch.MTE3)

    c0 = 0
    c1 = 1

    # Q/K/V/O 均为 b16（f16/bf16），中间 S/PV/acc 为 f32
    io_dtype = mem_q.ptr.dtype

    l1_q_ptr = tla.allocate(
        Q_BLOCK * HEAD_DIM, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )
    l1_k_ping_ptr = tla.allocate(
        HEAD_DIM * KV_BLOCK, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )
    l1_k_pong_ptr = tla.allocate(
        HEAD_DIM * KV_BLOCK, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )
    l1_v_ping_ptr = tla.allocate(
        KV_BLOCK * HEAD_DIM, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )
    l1_v_pong_ptr = tla.allocate(
        KV_BLOCK * HEAD_DIM, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )
    l1_p_0_ptr = tla.allocate(
        Q_BLOCK * KV_BLOCK, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )
    l1_p_1_ptr = tla.allocate(
        Q_BLOCK * KV_BLOCK, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )
    l1_p_2_ptr = tla.allocate(
        Q_BLOCK * KV_BLOCK, io_dtype, tla.AddressSpace.l1, ALIGNMENT
    )

    l0a_ping_ptr = tla.allocate(
        L0_HALF_SIZE // 2, io_dtype, tla.AddressSpace.l0a, ALIGNMENT
    )
    l0a_pong_ptr = tla.allocate(
        L0_HALF_SIZE // 2, io_dtype, tla.AddressSpace.l0a, ALIGNMENT
    )
    l0b_ping_ptr = tla.allocate(
        L0_HALF_SIZE // 2, io_dtype, tla.AddressSpace.l0b, ALIGNMENT
    )
    l0b_pong_ptr = tla.allocate(
        L0_HALF_SIZE // 2, io_dtype, tla.AddressSpace.l0b, ALIGNMENT
    )

    l0_s_ping_ptr = tla.allocate(
        Q_BLOCK * KV_BLOCK, tla.Float32, tla.AddressSpace.l0c, ALIGNMENT
    )
    l0_s_pong_ptr = tla.allocate(
        Q_BLOCK * KV_BLOCK, tla.Float32, tla.AddressSpace.l0c, ALIGNMENT
    )
    l0_pv_ping_ptr = tla.allocate(
        Q_BLOCK * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, ALIGNMENT
    )
    l0_pv_pong_ptr = tla.allocate(
        Q_BLOCK * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, ALIGNMENT
    )

    ub_s_ping_ptr = tla.allocate(
        Q_BLOCK_SUB * KV_BLOCK, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_s_pong_ptr = tla.allocate(
        Q_BLOCK_SUB * KV_BLOCK, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_p_f16_ping_ptr = tla.allocate(
        (Q_BLOCK_SUB + 1) * KV_BLOCK, io_dtype, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_p_f16_pong_ptr = tla.allocate(
        (Q_BLOCK_SUB + 1) * KV_BLOCK, io_dtype, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_pv_ping_ptr = tla.allocate(
        Q_BLOCK_SUB * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_pv_pong_ptr = tla.allocate(
        Q_BLOCK_SUB * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )

    ub_acc_ptr = tla.allocate(
        Q_BLOCK_SUB * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_out_f16_ptr = tla.recast_ptr(ub_acc_ptr, dtype=io_dtype)

    ub_now_max_ptr = tla.allocate(
        Q_BLOCK_SUB, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_last_max_ptr = tla.allocate(
        Q_BLOCK_SUB, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_sum_ptr = tla.allocate(Q_BLOCK_SUB, tla.Float32, tla.AddressSpace.ub, ALIGNMENT)
    ub_tmp_ptr = tla.allocate(
        2 * VL_FLOAT_ELE, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_mask_ptr = tla.allocate(
        Q_BLOCK_SUB * KV_BLOCK, tla.Int8, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_exp_sum_ptr = tla.allocate(
        Q_BLOCK_SUB, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_exp_max_0_ptr = tla.allocate(
        Q_BLOCK_SUB, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_exp_max_1_ptr = tla.allocate(
        Q_BLOCK_SUB, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )
    ub_exp_max_2_ptr = tla.allocate(
        Q_BLOCK_SUB, tla.Float32, tla.AddressSpace.ub, ALIGNMENT
    )

    cast_trait_zero = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast_trait_one = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )

    # tilingdata
    q_heads = HEAD_NUM
    kv_heads = KV_HEAD_NUM
    group_size = GROUP_SIZE
    max_q_seqlen = Q_SEQ
    max_kv_seqlen = KV_SEQ
    first_batch_task_num = HEAD_NUM * Q_BLOCK_COUNT
    total_task_num = TOTAL_TASKS

    with tla.cube():
        tla.set_flag(q_l0a_ready_l1)
        tla.set_flag(k_l0b_ready_l1_0)
        tla.set_flag(k_l0b_ready_l1_1)
        tla.set_flag(v_l0b_ready_l1_0)
        tla.set_flag(v_l0b_ready_l1_1)
        tla.set_flag(mmad_ready_l0a_0)
        tla.set_flag(mmad_ready_l0a_1)
        tla.set_flag(mmad_ready_l0b_0)
        tla.set_flag(mmad_ready_l0b_1)
        tla.set_flag(fix_ready_mmad_qk_0)
        tla.set_flag(fix_ready_mmad_qk_1)
        tla.set_flag(fix_ready_mmad_pv_0)
        tla.set_flag(fix_ready_mmad_pv_1)

        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)

    with tla.vector():
        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR)

        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR)

        tla.set_flag(mte3_ready_softmax_0)
        tla.set_flag(mte3_ready_softmax_1)

        tla.set_flag(mte3_ready_rescale)

    task_range = tla.range(
        tla.arch.block_idx(),
        TOTAL_TASKS,
        tla.arch.block_num(),
    )
    cur_batch = tla.as_numeric(c0)
    pre_total_task_num = tla.as_numeric(c0)
    cur_total_task_num = tla.as_numeric(first_batch_task_num)
    q_b_offset = tla.as_numeric(c0)
    o_b_offset = tla.as_numeric(c0)
    q_seqlen_cur = tla.as_numeric(Q_SEQ)
    kv_block_count_cur = KV_BLOCK_COUNT
    for task in task_range:
        if task >= cur_total_task_num:
            cur_batch = cur_batch + c1
            pre_total_task_num = cur_total_task_num
            q_b_offset = q_b_offset + q_seqlen_cur * q_heads * HEAD_DIM
            o_b_offset = o_b_offset + q_seqlen_cur * q_heads * HEAD_DIM
            q_seqlen_cur = Q_SEQ
            cur_total_task_num = (
                cur_total_task_num
                + ((q_seqlen_cur + Q_BLOCK - c1) // Q_BLOCK) * q_heads
            )

        task_idx_cur_batch = task - pre_total_task_num
        q_block_idx = task_idx_cur_batch // q_heads
        head_idx = task_idx_cur_batch - q_block_idx * q_heads
        kv_head_idx = head_idx // group_size

        q_tile_size = (
            (Q_SEQ - (Q_BLOCK_COUNT - 1) * Q_BLOCK)
            if q_block_idx == Q_BLOCK_COUNT - 1
            else Q_BLOCK
        )

        qo_stride = q_heads * HEAD_DIM
        kv_stride = kv_heads * HEAD_DIM
        kv_b_offset = cur_batch * max_kv_seqlen * kv_heads * HEAD_DIM

        mem_q_block = tla.make_tensor(
            mem_q.ptr + q_b_offset + head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(max_q_seqlen, HEAD_DIM),
                tla.make_stride(qo_stride, 1),
            ),
        )
        mem_k_block = tla.make_tensor(
            mem_k.ptr + kv_b_offset + kv_head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(HEAD_DIM, max_kv_seqlen),
                tla.make_stride(1, kv_stride),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        mem_v_block = tla.make_tensor(
            mem_v.ptr + kv_b_offset + kv_head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(max_kv_seqlen, HEAD_DIM), tla.make_stride(kv_stride, 1)
            ),
        )
        mem_o_block = tla.make_tensor(
            mem_o.ptr + o_b_offset + head_idx * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(max_q_seqlen, HEAD_DIM),
                tla.make_stride(qo_stride, 1),
            ),
        )

        with tla.cube():
            q_gm = tla.tile_view(
                mem_q_block,
                tla.make_shape(Q_BLOCK, HEAD_DIM),
                tla.make_coord(q_block_idx, c0),
            )
            l1_q = tla.make_tensor_like(l1_q_ptr, q_gm)
            tla.wait_flag(q_l0a_ready_l1)
            tla.copy(l1_q, q_gm)
            tla.set_flag(q_l1_ready_l0)
            tla.wait_flag(q_l1_ready_l0)

        for kv_iter in tla.range(c0, kv_block_count_cur + PRE_LAUNCH, c1):
            if kv_iter < kv_block_count_cur:
                ubSBufId = kv_iter % UB_S_OTMP_BUF_STAGES
                l1PBufId_qk = kv_iter % P_L1_BUF
                l1BBufId = kv_iter % K_L1_BUF
                l0CBufId = kv_iter % L0_STAGES

                kv_tile_size = (
                    (KV_SEQ - (KV_BLOCK_COUNT - 1) * KV_BLOCK)
                    if kv_iter == KV_BLOCK_COUNT - 1
                    else KV_BLOCK
                )

                # L0A/L0B ping/pong flag
                prefixSumL0AStages = (
                    (kv_iter * mm1L0ATotalStages_)
                    if (kv_iter <= PRE_LAUNCH)
                    else (
                        kv_iter * mm1L0ATotalStages_
                        + (kv_iter - PRE_LAUNCH) * mm2L0ATotalStages_
                    )
                )
                prefixSumL0BStages = (
                    (kv_iter * mm1L0BTotalStages_)
                    if (kv_iter <= PRE_LAUNCH)
                    else (
                        kv_iter * mm1L0BTotalStages_
                        + (kv_iter - PRE_LAUNCH) * mm2L0BTotalStages_
                    )
                )

                with tla.cube():
                    l1_k_ptr = l1_k_ping_ptr if l1BBufId == c0 else l1_k_pong_ptr
                    l0_s_ptr = l0_s_ping_ptr if l0CBufId == c0 else l0_s_pong_ptr
                    ub_s_ptr = ub_s_ping_ptr if ubSBufId == c0 else ub_s_pong_ptr

                    k_gm = tla.tile_view(
                        mem_k_block,
                        tla.make_shape(HEAD_DIM, KV_BLOCK),
                        tla.make_coord(c0, kv_iter),
                    )
                    q_gm = tla.tile_view(
                        mem_q_block,
                        tla.make_shape(Q_BLOCK, HEAD_DIM),
                        tla.make_coord(q_block_idx, c0),
                    )
                    l1_q = tla.make_tensor_like(l1_q_ptr, q_gm)
                    l1_k = tla.make_tensor_like(l1_k_ptr, k_gm, layoutTag=tla.arch.nZ)

                    # QK L0: 扁平化（QK_NL0_LOOP_NUM=QK_ML0_LOOP_NUM=QK_KL0_LOOP_NUM=1）
                    l0c_s = tla.make_tensor_like(l0_s_ptr, k_gm, tla.arch.L0Clayout)
                    s_ub = tla.make_tensor(
                        ub_s_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, kv_tile_size),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    ub_s_fix = tla.tile_view(
                        s_ub,
                        tla.make_shape(Q_BLOCK, KV_BLOCK),
                        tla.make_coord(c0, c0),
                    )

                    if l1BBufId == c0:
                        tla.wait_flag(k_l0b_ready_l1_0)
                    else:
                        tla.wait_flag(k_l0b_ready_l1_1)
                    tla.copy(l1_k, k_gm)
                    if l1BBufId == c0:
                        tla.set_flag(k_l1_ready_l0_0)
                        tla.wait_flag(k_l1_ready_l0_0)
                    else:
                        tla.set_flag(k_l1_ready_l0_1)
                        tla.wait_flag(k_l1_ready_l0_1)

                    l0ABufId = prefixSumL0AStages % L0_STAGES
                    l0BBufId_l0 = prefixSumL0BStages % L0_STAGES
                    l0_q_ptr = l0a_ping_ptr if l0ABufId == c0 else l0a_pong_ptr
                    l0_k_ptr = l0b_ping_ptr if l0BBufId_l0 == c0 else l0b_pong_ptr
                    l0_q = tla.make_tensor_like(l0_q_ptr, l1_q)
                    l0_k = tla.make_tensor_like(l0_k_ptr, l1_k)

                    if l0BBufId_l0 == c0:
                        tla.wait_flag(mmad_ready_l0b_0)
                    else:
                        tla.wait_flag(mmad_ready_l0b_1)
                    tla.copy(l0_k, l1_k)
                    if l0BBufId_l0 == c0:
                        tla.set_flag(l0b_ready_mmad_0)
                    else:
                        tla.set_flag(l0b_ready_mmad_1)
                    if l1BBufId == c0:
                        tla.set_flag(k_l0b_ready_l1_0)
                    else:
                        tla.set_flag(k_l0b_ready_l1_1)

                    if l0ABufId == c0:
                        tla.wait_flag(mmad_ready_l0a_0)
                    else:
                        tla.wait_flag(mmad_ready_l0a_1)
                    tla.copy(l0_q, l1_q)
                    if l0ABufId == c0:
                        tla.set_flag(l0a_ready_mmad_0)
                    else:
                        tla.set_flag(l0a_ready_mmad_1)

                    if l0ABufId == c0:
                        tla.wait_flag(l0a_ready_mmad_0)
                    else:
                        tla.wait_flag(l0a_ready_mmad_1)
                    if l0BBufId_l0 == c0:
                        tla.wait_flag(l0b_ready_mmad_0)
                    else:
                        tla.wait_flag(l0b_ready_mmad_1)
                    if l0CBufId == c0:
                        tla.wait_flag(fix_ready_mmad_qk_0)
                    else:
                        tla.wait_flag(fix_ready_mmad_qk_1)
                    tla.mmad(l0c_s, l0_q, l0_k, init_c=True)
                    if l0ABufId == c0:
                        tla.set_flag(mmad_ready_l0a_0)
                    else:
                        tla.set_flag(mmad_ready_l0a_1)
                    if l0BBufId_l0 == c0:
                        tla.set_flag(mmad_ready_l0b_0)
                    else:
                        tla.set_flag(mmad_ready_l0b_1)

                    if ubSBufId == c0:
                        tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX)
                    if l0CBufId == c0:
                        tla.set_flag(mmad_ready_fix_qk_0)
                        tla.wait_flag(mmad_ready_fix_qk_0)
                    else:
                        tla.set_flag(mmad_ready_fix_qk_1)
                        tla.wait_flag(mmad_ready_fix_qk_1)
                    tla.copy(
                        ub_s_fix,
                        l0c_s,
                        tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                    if l0CBufId == c0:
                        tla.set_flag(fix_ready_mmad_qk_0)
                    else:
                        tla.set_flag(fix_ready_mmad_qk_1)
                    if ubSBufId == c0:
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.FIX)
                    if kv_iter == (kv_block_count_cur - c1):
                        tla.set_flag(q_l0a_ready_l1)

                with tla.vector():
                    if ubSBufId == c0:
                        tla.cross_core_wait_flag(qk_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_wait_flag(qk_ready_1, tla.arch.VECTOR)
                    if ubSBufId == c0:
                        tla.wait_flag(mte3_ready_softmax_0)
                    else:
                        tla.wait_flag(mte3_ready_softmax_1)

                    vec_idx = tla.arch.sub_block_idx()
                    update = kv_iter != c0

                    q_tile_size_half = (q_tile_size + 1) // 2
                    q_tile_size_sub = (
                        q_tile_size_half
                        if vec_idx == c0
                        else (q_tile_size - q_tile_size_half)
                    )

                    ub_s_ptr = ub_s_ping_ptr if ubSBufId == c0 else ub_s_pong_ptr
                    ub_p_f16_ptr = (
                        ub_p_f16_ping_ptr if ubSBufId == c0 else ub_p_f16_pong_ptr
                    )
                    l1_p_ptr_qk = l1_p_0_ptr
                    if l1PBufId_qk == c0:
                        l1_p_ptr_qk = l1_p_0_ptr
                    elif l1PBufId_qk == c1:
                        l1_p_ptr_qk = l1_p_1_ptr
                    else:
                        l1_p_ptr_qk = l1_p_2_ptr
                    ub_exp_max_ptr_qk = ub_exp_max_0_ptr
                    if l1PBufId_qk == c0:
                        ub_exp_max_ptr_qk = ub_exp_max_0_ptr
                    elif l1PBufId_qk == c1:
                        ub_exp_max_ptr_qk = ub_exp_max_1_ptr
                    else:
                        ub_exp_max_ptr_qk = ub_exp_max_2_ptr

                    ub_s = tla.make_tensor(
                        ub_s_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub, KV_BLOCK),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )

                    ub_now_max = tla.make_tensor(
                        ub_now_max_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_last_max = tla.make_tensor(
                        ub_last_max_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_sum = tla.make_tensor(
                        ub_sum_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_exp_sum = tla.make_tensor(
                        ub_exp_sum_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_exp_max_qk = tla.make_tensor(
                        ub_exp_max_ptr_qk,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    # P UB layout: RowMajor (65, 128), zN
                    ub_p_f16 = tla.make_tensor(
                        ub_p_f16_ptr,
                        tla.make_layout(
                            tla.make_shape(Q_BLOCK_SUB + 1, KV_BLOCK),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    ub_p_zN_full = tla.make_tensor_like(
                        ub_p_f16_ptr, ub_p_f16, tla.arch.zNUnAlign
                    )
                    ub_p_zN = tla.tile_view(
                        ub_p_zN_full,
                        tla.make_shape(q_tile_size_sub, kv_tile_size),
                        tla.make_coord(c0, c0),
                    )
                    ub_tmp = tla.make_tensor(
                        ub_tmp_ptr,
                        tla.make_layout(
                            tla.make_shape(2 * VL_FLOAT_ELE), tla.make_stride(1)
                        ),
                    )

                    tailN = (kv_tile_size - 1) % VL_FLOAT_ELE + 1
                    if kv_iter == c0:
                        if kv_tile_size > VL_FLOAT_ELE:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_i1_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i0, c1),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg1_p1 = ub_s_i1_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, QK_SCALE, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.mul(
                                        ub_s_reg1_p1, QK_SCALE, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.where(
                                        mask_tail_n, ub_s_reg1_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    ub_s_i1_p1.store(ub_s_reg1_p1, mask=pregFull)
                                    tmp_reg_p1 = tla.max(
                                        ub_s_reg0_p1, ub_s_reg1_p1, mask=pregFull
                                    )
                                    max_reg_p1 = tmp_reg_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=NormalLoadParams(
                                            load_dist=LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_row_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    s_odd_p2, s_even_p2 = ub_s_row_p2.load(
                                        params=NormalLoadParams(
                                            load_dist=LoadDist.DIST_DINTLV_B32
                                        )
                                    )
                                    exp_odd_p2 = tla.exp(
                                        tla.sub(s_odd_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    exp_even_p2 = tla.exp(
                                        tla.sub(s_even_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    tmp_reg_p2 = tla.add(
                                        exp_odd_p2, exp_even_p2, mask=pregFull
                                    )
                                    block_sum_p2 = tmp_reg_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_sum_i_p2 = tla.tile_view(
                                        ub_sum, tla.make_shape(1), tla.make_coord(i2)
                                    )
                                    ub_sum_i_p2.store(
                                        block_sum_p2,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_even_p2 = exp_even_p2.to(
                                        io_dtype, cast_trait_one, mask=pregFull
                                    )
                                    p_odd_p2 = exp_odd_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    p_zn_p2 = tla.bitwise_or(
                                        p_even_p2, p_odd_p2, mask=preg_all_b16
                                    )
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        p_zn_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=Q_BLOCK_SUB + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                        else:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, QK_SCALE, mask=pregFull
                                    )
                                    ub_s_reg0_p1 = tla.where(
                                        mask_tail_n, ub_s_reg0_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    max_reg_p1 = ub_s_reg0_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=NormalLoadParams(
                                            load_dist=LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_i0_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_s_reg0_p2 = ub_s_i0_p2.load()
                                    exp_reg0_p2 = tla.exp(
                                        tla.sub(
                                            ub_s_reg0_p2, max_reg_p2, mask=pregFull
                                        ),
                                        mask=pregFull,
                                    )
                                    block_sum_p2 = exp_reg0_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_sum_i_p2 = tla.tile_view(
                                        ub_sum, tla.make_shape(1), tla.make_coord(i2)
                                    )
                                    ub_sum_i_p2.store(
                                        block_sum_p2,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_reg0_p2 = exp_reg0_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    r0_qk_p2, _ = tla.deinterleave(p_reg0_p2, p_reg0_p2)
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        r0_qk_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=Q_BLOCK_SUB + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                    else:
                        if kv_tile_size > VL_FLOAT_ELE:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_i1_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i0, c1),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg1_p1 = ub_s_i1_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, QK_SCALE, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.mul(
                                        ub_s_reg1_p1, QK_SCALE, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.where(
                                        mask_tail_n, ub_s_reg1_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    ub_s_i1_p1.store(ub_s_reg1_p1, mask=pregFull)
                                    tmp_reg_p1 = tla.max(
                                        ub_s_reg0_p1, ub_s_reg1_p1, mask=pregFull
                                    )
                                    max_reg_p1 = tmp_reg_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_last_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_now_max_vu = tla.tile_view(
                                    ub_now_max,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_last_max_vu = tla.tile_view(
                                    ub_last_max,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_sum_vu = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_exp_max_vu = tla.tile_view(
                                    ub_exp_max_qk,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                now_max_vu = (
                                    ub_now_max_vu.load()
                                )  # 旧 global max（上一轮存入）
                                blk_max_vu = (
                                    ub_last_max_vu.load()
                                )  # 本块 max（i0 存入）
                                global_max_vu = tla.max(
                                    now_max_vu, blk_max_vu, mask=pregFull
                                )
                                dm_vu = tla.exp(
                                    tla.sub(now_max_vu, global_max_vu, mask=pregFull),
                                    mask=pregFull,
                                )  # exp(旧-新)
                                sum_vu = ub_sum_vu.load()  # 旧 global sum
                                update_vu = tla.mul(sum_vu, dm_vu, mask=pregFull)
                                ub_now_max_vu.store(global_max_vu, mask=pregFull)
                                ub_exp_max_vu.store(dm_vu, mask=pregFull)
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=NormalLoadParams(
                                            load_dist=LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_row_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    s_odd_p2, s_even_p2 = ub_s_row_p2.load(
                                        params=NormalLoadParams(
                                            load_dist=LoadDist.DIST_DINTLV_B32
                                        )
                                    )
                                    exp_odd_p2 = tla.exp(
                                        tla.sub(s_odd_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    exp_even_p2 = tla.exp(
                                        tla.sub(s_even_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    tmp_reg_p2 = tla.add(
                                        exp_odd_p2, exp_even_p2, mask=pregFull
                                    )
                                    block_sum_p2 = tmp_reg_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_exp_sum_i_p2 = tla.tile_view(
                                        ub_exp_sum,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    ub_exp_sum_i_p2.store(
                                        block_sum_p2,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_even_p2 = exp_even_p2.to(
                                        io_dtype, cast_trait_one, mask=pregFull
                                    )
                                    p_odd_p2 = exp_odd_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    p_zn_p2 = tla.bitwise_or(
                                        p_even_p2, p_odd_p2, mask=preg_all_b16
                                    )
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        p_zn_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=Q_BLOCK_SUB + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                                # 向量化 final add（替代 i3 逐行 sum-merge；update_vu + 本块 sum → 新 global sum）
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_exp_sum_vf = tla.tile_view(
                                    ub_exp_sum,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                blk_sum_vf = ub_exp_sum_vf.load()
                                new_sum_vf = tla.add(
                                    update_vu, blk_sum_vf, mask=pregFull
                                )
                                ub_sum_vf = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_sum_vf.store(new_sum_vf, mask=pregFull)
                        else:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(MIN_VALUE, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, QK_SCALE, mask=pregFull
                                    )
                                    ub_s_reg0_p1 = tla.where(
                                        mask_tail_n, ub_s_reg0_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    max_reg_p1 = ub_s_reg0_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_last_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_now_max_vu = tla.tile_view(
                                    ub_now_max,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_last_max_vu = tla.tile_view(
                                    ub_last_max,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_sum_vu = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_exp_max_vu = tla.tile_view(
                                    ub_exp_max_qk,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                now_max_vu = (
                                    ub_now_max_vu.load()
                                )  # 旧 global max（上一轮存入）
                                blk_max_vu = (
                                    ub_last_max_vu.load()
                                )  # 本块 max（i0 存入）
                                global_max_vu = tla.max(
                                    now_max_vu, blk_max_vu, mask=pregFull
                                )
                                dm_vu = tla.exp(
                                    tla.sub(now_max_vu, global_max_vu, mask=pregFull),
                                    mask=pregFull,
                                )  # exp(旧-新)
                                sum_vu = ub_sum_vu.load()  # 旧 global sum
                                update_vu = tla.mul(sum_vu, dm_vu, mask=pregFull)
                                ub_now_max_vu.store(global_max_vu, mask=pregFull)
                                ub_exp_max_vu.store(dm_vu, mask=pregFull)
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=NormalLoadParams(
                                            load_dist=LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_i0_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_s_reg0_p2 = ub_s_i0_p2.load()
                                    exp_reg0_p2 = tla.exp(
                                        tla.sub(
                                            ub_s_reg0_p2, max_reg_p2, mask=pregFull
                                        ),
                                        mask=pregFull,
                                    )
                                    block_sum_p2 = exp_reg0_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_exp_sum_i_p2 = tla.tile_view(
                                        ub_exp_sum,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    ub_exp_sum_i_p2.store(
                                        block_sum_p2,
                                        params=UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_reg0_p2 = exp_reg0_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    r0_qk_p2, _ = tla.deinterleave(p_reg0_p2, p_reg0_p2)
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        r0_qk_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=Q_BLOCK_SUB + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                                # 向量化 final add（替代 i3 逐行 sum-merge；update_vu + 本块 sum → 新 global sum）
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_exp_sum_vf = tla.tile_view(
                                    ub_exp_sum,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                blk_sum_vf = ub_exp_sum_vf.load()
                                new_sum_vf = tla.add(
                                    update_vu, blk_sum_vf, mask=pregFull
                                )
                                ub_sum_vf = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(VL_FLOAT_ELE),
                                    tla.make_coord(0),
                                )
                                ub_sum_vf.store(new_sum_vf, mask=pregFull)

                    if ubSBufId == c0:
                        tla.set_flag(v_mte3_0)
                        tla.wait_flag(v_mte3_0)
                    else:
                        tla.set_flag(v_mte3_1)
                        tla.wait_flag(v_mte3_1)
                    if ubSBufId == c0:
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR)

                    # softmaxReadyFlag = l1PBufId + 2; 等 PV 用完上一轮 P buffer
                    if l1PBufId_qk == c0:
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                    elif l1PBufId_qk == c1:
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                    else:
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
                    ub_p_zN_copy = tla.tile_view(
                        ub_p_zN_full,
                        tla.make_shape(q_tile_size_sub, kv_tile_size),
                        tla.make_coord(c0, c0),
                    )
                    l1_p_ref = tla.make_tensor(
                        l1_p_ptr_qk,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, kv_tile_size),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    l1_p_qk = tla.make_tensor_like(l1_p_ptr_qk, l1_p_ref, tla.arch.zN)
                    l1_p_sub = tla.tile_view(
                        l1_p_qk,
                        tla.make_shape(q_tile_size_half, kv_tile_size),
                        tla.make_coord(vec_idx, c0),
                    )
                    tla.copy(l1_p_sub, ub_p_zN_copy)
                    # 释放 ub_s 给下一轮 softmax
                    if ubSBufId == c0:
                        tla.set_flag(mte3_ready_softmax_0)
                    else:
                        tla.set_flag(mte3_ready_softmax_1)

                    if l1PBufId_qk == c0:
                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                    elif l1PBufId_qk == c1:
                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                    else:
                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)

            if kv_iter >= PRE_LAUNCH:
                kv_pv = kv_iter - PRE_LAUNCH

                kv_tile_size_pv = (
                    (KV_SEQ - (KV_BLOCK_COUNT - 1) * KV_BLOCK)
                    if kv_pv == KV_BLOCK_COUNT - 1
                    else KV_BLOCK
                )
                ubOTmpBufId = kv_pv % UB_S_OTMP_BUF_STAGES
                l1PBufId_pv = kv_pv % P_L1_BUF
                l1BBufId_pv = kv_pv % V_L1_BUF
                l0CBufId_pv = kv_pv % L0_STAGES

                prefixSumL0AStages_pv = (
                    (
                        (kv_pv + c1 + PRE_LAUNCH) * mm1L0ATotalStages_
                        + kv_pv * mm2L0ATotalStages_
                    )
                    if (kv_pv < (KV_BLOCK_COUNT - PRE_LAUNCH))
                    else (
                        KV_BLOCK_COUNT * mm1L0ATotalStages_ + kv_pv * mm2L0ATotalStages_
                    )
                )
                prefixSumL0BStages_pv = (
                    (
                        (kv_pv + c1 + PRE_LAUNCH) * mm1L0BTotalStages_
                        + kv_pv * mm2L0BTotalStages_
                    )
                    if (kv_pv < (KV_BLOCK_COUNT - PRE_LAUNCH))
                    else (
                        KV_BLOCK_COUNT * mm1L0BTotalStages_ + kv_pv * mm2L0BTotalStages_
                    )
                )

                with tla.cube():
                    l1_v_ptr = l1_v_ping_ptr if l1BBufId_pv == c0 else l1_v_pong_ptr
                    l0_pv_ptr = l0_pv_ping_ptr if l0CBufId_pv == c0 else l0_pv_pong_ptr

                    v_gm = tla.tile_view(
                        mem_v_block,
                        tla.make_shape(KV_BLOCK, HEAD_DIM),
                        tla.make_coord(kv_pv, c0),
                    )
                    l1_v = tla.make_tensor_like(l1_v_ptr, v_gm)

                    if l1BBufId_pv == c0:
                        tla.wait_flag(v_l0b_ready_l1_0)
                    else:
                        tla.wait_flag(v_l0b_ready_l1_1)
                    tla.copy(l1_v, v_gm)
                    if l1BBufId_pv == c0:
                        tla.set_flag(v_l1_ready_l0_0)
                        tla.wait_flag(v_l1_ready_l0_0)
                    else:
                        tla.set_flag(v_l1_ready_l0_1)
                        tla.wait_flag(v_l1_ready_l0_1)

                    if l1PBufId_pv == c0:
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE1)
                    elif l1PBufId_pv == c1:
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE1)
                    else:
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE1)

                    l1_p_ptr_pv = l1_p_0_ptr
                    if l1PBufId_pv == c0:
                        l1_p_ptr_pv = l1_p_0_ptr
                    elif l1PBufId_pv == c1:
                        l1_p_ptr_pv = l1_p_1_ptr
                    else:
                        l1_p_ptr_pv = l1_p_2_ptr
                    ub_p_full_pv = tla.make_tensor(
                        l1_p_ptr_pv,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, kv_tile_size_pv),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    l1_p_pv = tla.make_tensor_like(
                        l1_p_ptr_pv, ub_p_full_pv, tla.arch.zN
                    )

                    # PV L0: 扁平化（PV_NL0_LOOP_NUM=PV_ML0_LOOP_NUM=PV_KL0_LOOP_NUM=1）
                    l0c_pv_ref = tla.make_tensor(
                        l0_pv_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                        ),
                    )
                    l0c_pv = tla.make_tensor_like(
                        l0_pv_ptr, l0c_pv_ref, tla.arch.L0Clayout
                    )

                    l0ABufId_pv = prefixSumL0AStages_pv % L0_STAGES
                    l0BBufId_pv = prefixSumL0BStages_pv % L0_STAGES
                    l0_p_ptr = l0a_ping_ptr if l0ABufId_pv == c0 else l0a_pong_ptr
                    l0_v_ptr = l0b_ping_ptr if l0BBufId_pv == c0 else l0b_pong_ptr
                    l0_p = tla.make_tensor_like(l0_p_ptr, l1_p_pv)
                    l0_v = tla.make_tensor_like(l0_v_ptr, l1_v)

                    if l0BBufId_pv == c0:
                        tla.wait_flag(mmad_ready_l0b_0)
                    else:
                        tla.wait_flag(mmad_ready_l0b_1)
                    tla.copy(l0_v, l1_v)
                    if l0BBufId_pv == c0:
                        tla.set_flag(l0b_ready_mmad_0)
                    else:
                        tla.set_flag(l0b_ready_mmad_1)
                    if l1BBufId_pv == c0:
                        tla.set_flag(v_l0b_ready_l1_0)
                    else:
                        tla.set_flag(v_l0b_ready_l1_1)

                    if l0ABufId_pv == c0:
                        tla.wait_flag(mmad_ready_l0a_0)
                    else:
                        tla.wait_flag(mmad_ready_l0a_1)
                    tla.copy(l0_p, l1_p_pv)
                    if l0ABufId_pv == c0:
                        tla.set_flag(l0a_ready_mmad_0)
                    else:
                        tla.set_flag(l0a_ready_mmad_1)
                    if l1PBufId_pv == c0:
                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
                    elif l1PBufId_pv == c1:
                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
                    else:
                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)

                    if l0ABufId_pv == c0:
                        tla.wait_flag(l0a_ready_mmad_0)
                    else:
                        tla.wait_flag(l0a_ready_mmad_1)
                    if l0BBufId_pv == c0:
                        tla.wait_flag(l0b_ready_mmad_0)
                    else:
                        tla.wait_flag(l0b_ready_mmad_1)
                    if l0CBufId_pv == c0:
                        tla.wait_flag(fix_ready_mmad_pv_0)
                    else:
                        tla.wait_flag(fix_ready_mmad_pv_1)
                    tla.mmad(l0c_pv, l0_p, l0_v, init_c=True)
                    if l0ABufId_pv == c0:
                        tla.set_flag(mmad_ready_l0a_0)
                    else:
                        tla.set_flag(mmad_ready_l0a_1)
                    if l0BBufId_pv == c0:
                        tla.set_flag(mmad_ready_l0b_0)
                    else:
                        tla.set_flag(mmad_ready_l0b_1)

                    if ubOTmpBufId == c0:
                        tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX)
                    if l0CBufId_pv == c0:
                        tla.set_flag(mmad_ready_fix_pv_0)
                        tla.wait_flag(mmad_ready_fix_pv_0)
                    else:
                        tla.set_flag(mmad_ready_fix_pv_1)
                        tla.wait_flag(mmad_ready_fix_pv_1)
                    ub_pv_ptr_pv = (
                        ub_pv_ping_ptr if ubOTmpBufId == c0 else ub_pv_pong_ptr
                    )
                    ub_pv_fix = tla.make_tensor(
                        ub_pv_ptr_pv,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    tla.copy(
                        ub_pv_fix,
                        l0c_pv,
                        tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                    if l0CBufId_pv == c0:
                        tla.set_flag(fix_ready_mmad_pv_0)
                    else:
                        tla.set_flag(fix_ready_mmad_pv_1)
                    if ubOTmpBufId == c0:
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.FIX)

                with tla.vector():
                    if ubOTmpBufId == c0:
                        tla.cross_core_wait_flag(pv_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_wait_flag(pv_ready_1, tla.arch.VECTOR)
                    tla.wait_flag(mte3_ready_rescale)

                    ub_pv_ptr_re = (
                        ub_pv_ping_ptr if ubOTmpBufId == c0 else ub_pv_pong_ptr
                    )
                    ub_exp_max_ptr_re = ub_exp_max_0_ptr
                    if l1PBufId_pv == c0:
                        ub_exp_max_ptr_re = ub_exp_max_0_ptr
                    elif l1PBufId_pv == c1:
                        ub_exp_max_ptr_re = ub_exp_max_1_ptr
                    else:
                        ub_exp_max_ptr_re = ub_exp_max_2_ptr

                    vec_idx_re = tla.arch.sub_block_idx()
                    is_first_pv = kv_pv == c0
                    is_last_pv = kv_pv == (kv_block_count_cur - c1)

                    q_tile_size_half_re = (q_tile_size + 1) // 2
                    q_tile_size_sub_re = (
                        q_tile_size_half_re
                        if vec_idx_re == c0
                        else (q_tile_size - q_tile_size_half_re)
                    )

                    ub_pv_re = tla.make_tensor(
                        ub_pv_ptr_re,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                        ),
                    )
                    ub_acc_re = tla.make_tensor(
                        ub_acc_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                        ),
                    )
                    ub_out_f16_re = tla.make_tensor(
                        ub_out_f16_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                        ),
                    )
                    ub_sum_re = tla.make_tensor(
                        ub_sum_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re), tla.make_stride(1)
                        ),
                    )
                    ub_exp_max_re = tla.make_tensor(
                        ub_exp_max_ptr_re,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re), tla.make_stride(1)
                        ),
                    )

                    if is_first_pv and is_last_pv:
                        # Single block: O = PV / sum → cast → GM (1 个 vec.func + local_mem_bar)
                        with tla.vec.func(mode="simd"):
                            pregFull_re0 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            preg_all_b16_re0 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=io_dtype
                            )
                            for i_re0 in tla.range(q_tile_size_sub_re):
                                ub_sum_i_re0 = tla.tile_view(
                                    ub_sum_re, tla.make_shape(1), tla.make_coord(i_re0)
                                )
                                sum_reg_re0 = ub_sum_i_re0.load(
                                    params=NormalLoadParams(
                                        load_dist=LoadDist.DIST_BRC_B32
                                    )
                                )
                                for j_re0 in tla.range(
                                    c0, HEAD_DIM // VL_FLOAT_ELE, c1
                                ):
                                    cur_tile_re0 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re0, j_re0),
                                    )
                                    cur_reg_re0 = cur_tile_re0.load()
                                    div_reg_re0 = tla.div(
                                        cur_reg_re0, sum_reg_re0, mask=pregFull_re0
                                    )
                                    acc_tile_re0 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re0, j_re0),
                                    )
                                    acc_tile_re0.store(div_reg_re0, mask=pregFull_re0)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                            for i_re1 in tla.range(q_tile_size_sub_re):
                                for j_re1 in tla.range(
                                    c0, HEAD_DIM // (2 * VL_FLOAT_ELE), c1
                                ):
                                    acc_tile_re1_0 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re1, 2 * j_re1),
                                    )
                                    acc_tile_re1_1 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re1, 2 * j_re1 + c1),
                                    )
                                    out_tile_re1 = tla.tile_view(
                                        ub_out_f16_re,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i_re1, j_re1),
                                    )
                                    acc_reg_re1_0 = acc_tile_re1_0.load()
                                    acc_reg_re1_1 = acc_tile_re1_1.load()
                                    out_reg_re1_0 = acc_reg_re1_0.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re0
                                    )
                                    out_reg_re1_1 = acc_reg_re1_1.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re0
                                    )
                                    r0_re1, _ = tla.deinterleave(
                                        out_reg_re1_0, out_reg_re1_1
                                    )
                                    out_tile_re1.store(r0_re1, mask=preg_all_b16_re0)
                        tla.set_flag(rescale_v_mte3)
                        tla.wait_flag(rescale_v_mte3)
                        o_gm_re0 = tla.tile_view(
                            mem_o_block,
                            tla.make_shape(Q_BLOCK, HEAD_DIM),
                            tla.make_coord(q_block_idx, c0),
                        )
                        o_sub_gm_re0 = tla.tile_view(
                            o_gm_re0,
                            tla.make_shape(q_tile_size_half_re, HEAD_DIM),
                            tla.make_coord(vec_idx_re, c0),
                        )
                        tla.copy(o_sub_gm_re0, ub_out_f16_re)

                    elif is_first_pv and (not is_last_pv):
                        # First block: copy PV to acc
                        with tla.vec.func(mode="simd"):
                            pregFull_re2 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            for i_re2 in tla.range(q_tile_size_sub_re):
                                for j_re2 in tla.range(
                                    c0, HEAD_DIM // VL_FLOAT_ELE, c1
                                ):
                                    cur_tile_re2 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re2, j_re2),
                                    )
                                    acc_tile_re2 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re2, j_re2),
                                    )
                                    cur_reg_re2 = cur_tile_re2.load()
                                    acc_tile_re2.store(cur_reg_re2, mask=pregFull_re2)

                    elif (not is_first_pv) and (not is_last_pv):
                        with tla.vec.func(mode="simd"):
                            pregFull_re3 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            for i_re3 in tla.range(q_tile_size_sub_re):
                                ub_exp_max_i_re3 = tla.tile_view(
                                    ub_exp_max_re,
                                    tla.make_shape(1),
                                    tla.make_coord(i_re3),
                                )
                                exp_max_reg_re3 = ub_exp_max_i_re3.load(
                                    params=NormalLoadParams(
                                        load_dist=LoadDist.DIST_BRC_B32
                                    )
                                )
                                for j_re3 in tla.range(
                                    c0, HEAD_DIM // VL_FLOAT_ELE, c1
                                ):
                                    pre_tile_re3 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re3, j_re3),
                                    )
                                    cur_tile_re3 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re3, j_re3),
                                    )
                                    pre_reg_re3 = pre_tile_re3.load()
                                    cur_reg_re3 = cur_tile_re3.load()
                                    mul_reg_re3 = tla.mul(
                                        exp_max_reg_re3, pre_reg_re3, mask=pregFull_re3
                                    )
                                    add_reg_re3 = tla.add(
                                        mul_reg_re3, cur_reg_re3, mask=pregFull_re3
                                    )
                                    pre_tile_re3.store(add_reg_re3, mask=pregFull_re3)

                    else:
                        # Last block: O = (O*expMax + PV) / sum → cast → GM (1 个 vec.func + local_mem_bar)
                        with tla.vec.func(mode="simd"):
                            pregFull_re4 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            preg_all_b16_re4 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=io_dtype
                            )
                            for i_re4 in tla.range(q_tile_size_sub_re):
                                ub_exp_max_i_re4 = tla.tile_view(
                                    ub_exp_max_re,
                                    tla.make_shape(1),
                                    tla.make_coord(i_re4),
                                )
                                exp_max_reg_re4 = ub_exp_max_i_re4.load(
                                    params=NormalLoadParams(
                                        load_dist=LoadDist.DIST_BRC_B32
                                    )
                                )
                                ub_sum_i_re4 = tla.tile_view(
                                    ub_sum_re, tla.make_shape(1), tla.make_coord(i_re4)
                                )
                                sum_reg_re4 = ub_sum_i_re4.load(
                                    params=NormalLoadParams(
                                        load_dist=LoadDist.DIST_BRC_B32
                                    )
                                )
                                for j_re4 in tla.range(
                                    c0, HEAD_DIM // VL_FLOAT_ELE, c1
                                ):
                                    pre_tile_re4 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re4, j_re4),
                                    )
                                    cur_tile_re4 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re4, j_re4),
                                    )
                                    pre_reg_re4 = pre_tile_re4.load()
                                    cur_reg_re4 = cur_tile_re4.load()
                                    mul_reg_re4 = tla.mul(
                                        exp_max_reg_re4, pre_reg_re4, mask=pregFull_re4
                                    )
                                    add_reg_re4 = tla.add(
                                        mul_reg_re4, cur_reg_re4, mask=pregFull_re4
                                    )
                                    div_reg_re4 = tla.div(
                                        add_reg_re4, sum_reg_re4, mask=pregFull_re4
                                    )
                                    pre_tile_re4.store(div_reg_re4, mask=pregFull_re4)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                            for i_re5 in tla.range(q_tile_size_sub_re):
                                for j_re5 in tla.range(
                                    c0, HEAD_DIM // (2 * VL_FLOAT_ELE), c1
                                ):
                                    acc_tile_re5_0 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re5, 2 * j_re5),
                                    )
                                    acc_tile_re5_1 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, VL_FLOAT_ELE),
                                        tla.make_coord(i_re5, 2 * j_re5 + c1),
                                    )
                                    out_tile_re5 = tla.tile_view(
                                        ub_out_f16_re,
                                        tla.make_shape(1, 2 * VL_FLOAT_ELE),
                                        tla.make_coord(i_re5, j_re5),
                                    )
                                    acc_reg_re5_0 = acc_tile_re5_0.load()
                                    acc_reg_re5_1 = acc_tile_re5_1.load()
                                    out_reg_re5_0 = acc_reg_re5_0.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re4
                                    )
                                    out_reg_re5_1 = acc_reg_re5_1.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re4
                                    )
                                    r0_re5, _ = tla.deinterleave(
                                        out_reg_re5_0, out_reg_re5_1
                                    )
                                    out_tile_re5.store(r0_re5, mask=preg_all_b16_re4)
                        tla.set_flag(rescale_v_mte3)
                        tla.wait_flag(rescale_v_mte3)
                        o_gm_re5 = tla.tile_view(
                            mem_o_block,
                            tla.make_shape(Q_BLOCK, HEAD_DIM),
                            tla.make_coord(q_block_idx, c0),
                        )
                        o_sub_gm_re5 = tla.tile_view(
                            o_gm_re5,
                            tla.make_shape(q_tile_size_half_re, HEAD_DIM),
                            tla.make_coord(vec_idx_re, c0),
                        )
                        tla.copy(o_sub_gm_re5, ub_out_f16_re)

                    if ubOTmpBufId == c0:
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR)
                    # O->GM copy 完成, 释放给下一轮 rescale
                    tla.set_flag(mte3_ready_rescale)

    with tla.cube():
        tla.wait_flag(q_l0a_ready_l1)
        tla.wait_flag(k_l0b_ready_l1_0)
        tla.wait_flag(k_l0b_ready_l1_1)
        tla.wait_flag(v_l0b_ready_l1_0)
        tla.wait_flag(v_l0b_ready_l1_1)
        tla.wait_flag(mmad_ready_l0a_0)
        tla.wait_flag(mmad_ready_l0a_1)
        tla.wait_flag(mmad_ready_l0b_0)
        tla.wait_flag(mmad_ready_l0b_1)
        tla.wait_flag(fix_ready_mmad_qk_0)
        tla.wait_flag(fix_ready_mmad_qk_1)
        tla.wait_flag(fix_ready_mmad_pv_0)
        tla.wait_flag(fix_ready_mmad_pv_1)

        tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX)
        tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX)

        tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX)
        tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX)

    with tla.vector():
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
        tla.wait_flag(mte3_ready_softmax_0)
        tla.wait_flag(mte3_ready_softmax_1)
        tla.wait_flag(mte3_ready_rescale)


# Host侧：构造输入、编译、调用 kernel、精度校验
def get_block_num(block_num: int, device: int = 0, *, kind: str = "vector") -> int:
    """Get launch ``block_num``.

    Non-``-1`` uses the host argument. ``-1`` means full-device launch:
    pure vector → ``vector_core_num`` (AIV); cube/mix → ``cube_core_num`` (AIC).
    """
    if int(block_num) != -1:
        return max(1, int(block_num))
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")


def create_tla_tensor(buf, layout_tag=tla.arch.RowMajor):
    return from_dlpack(buf.contiguous(), layout_tag=layout_tag)


def reshape_qk_to_2d(tensor):
    return tensor.reshape(-1, HEAD_DIM).contiguous()


def apply_shape_args(args: argparse.Namespace) -> None:
    """按命令行参数覆盖编译期形状全局，并重算派生量。

    kernel 体在 tla.compile(...) 时被 trace，通过 Python LOAD_GLOBAL 读取这些
    模块全局；在 compile 之前覆盖即可让新形状生效，缓存 key 也会按新 MLIR 区分。
    """
    global BATCH, HEAD_NUM, KV_HEAD_NUM, Q_SEQ, KV_SEQ
    global GROUP_SIZE, Q_BLOCK_COUNT, KV_BLOCK_COUNT, TOTAL_TASKS

    if args.headnum % args.kvheadnum != 0:
        raise SystemExit(
            f"--headnum ({args.headnum}) must be a multiple of "
            f"--kvheadnum ({args.kvheadnum}) for GQA"
        )

    BATCH = args.batch
    HEAD_NUM = args.headnum
    KV_HEAD_NUM = args.kvheadnum
    Q_SEQ = args.qseqlen
    KV_SEQ = args.kvseqlen
    GROUP_SIZE = HEAD_NUM // KV_HEAD_NUM
    Q_BLOCK_COUNT = (Q_SEQ + Q_BLOCK - 1) // Q_BLOCK
    KV_BLOCK_COUNT = (KV_SEQ + KV_BLOCK - 1) // KV_BLOCK
    TOTAL_TASKS = BATCH * HEAD_NUM * Q_BLOCK_COUNT


def run(args: argparse.Namespace) -> int:
    torch.npu.set_device(args.device)
    apply_shape_args(args)
    dtypes = {"f16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtypes[args.dtype]
    print(
        f"--- BATCH=({BATCH},{Q_SEQ},{KV_SEQ}) "
        f"HEAD=({HEAD_NUM},{KV_HEAD_NUM}) "
        f"HEAD_DIM={HEAD_DIM} "
        f"dtype={args.dtype} "
        f"sentinel={args.sentinel} ---"
    )

    q_shape = (BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM)
    k_shape = (BATCH, KV_SEQ, KV_HEAD_NUM, HEAD_DIM)
    v_shape = (BATCH, KV_SEQ, KV_HEAD_NUM, HEAD_DIM)
    o_shape = (BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM)

    torch.manual_seed(42)
    torch_q = torch.rand(q_shape, dtype=dtype).npu()
    torch_k = torch.rand(k_shape, dtype=dtype).npu()
    torch_v = torch.rand(v_shape, dtype=dtype).npu()
    torch_o = torch.full(o_shape, args.sentinel, dtype=dtype).npu()

    tla_q = create_tla_tensor(reshape_qk_to_2d(torch_q), tla.arch.RowMajor)
    tla_k = create_tla_tensor(reshape_qk_to_2d(torch_k), tla.arch.ColumnMajor)
    tla_v = create_tla_tensor(reshape_qk_to_2d(torch_v), tla.arch.RowMajor)
    tla_o = create_tla_tensor(reshape_qk_to_2d(torch_o), tla.arch.RowMajor)
    mask_2d = torch.zeros((Q_SEQ, KV_SEQ), dtype=torch.int8).npu()
    tla_mask = create_tla_tensor(mask_2d, tla.arch.RowMajor)

    q_seqlen_list = [Q_SEQ] * BATCH
    kv_seqlen_list = [KV_SEQ] * BATCH
    td = compute_tiling(
        batch=BATCH,
        num_heads=HEAD_NUM,
        kv_heads=KV_HEAD_NUM,
        q_seqlen_list=q_seqlen_list,
        kv_seqlen_list=kv_seqlen_list,
        head_dim=HEAD_DIM,
        q_base_tile=Q_BLOCK,
        kv_base_tile=KV_BLOCK,
    )
    tiling_int_list = pack_tiling_int(td)
    tiling_tensor = torch.tensor(tiling_int_list, dtype=torch.int32).npu()
    tla_tiling = create_tla_tensor(tiling_tensor, tla.arch.RowMajor)

    actual_q_list = make_actual_seqlen(q_seqlen_list)
    actual_kv_list = make_actual_seqlen(kv_seqlen_list)
    actual_q = torch.tensor(actual_q_list, dtype=torch.int32).npu()
    actual_kv = torch.tensor(actual_kv_list, dtype=torch.int32).npu()
    tla_actual_q = create_tla_tensor(actual_q, tla.arch.RowMajor)
    tla_actual_kv = create_tla_tensor(actual_kv, tla.arch.RowMajor)

    artifact = tla.compile(
        flash_attention_infer_kernel,
        tla_q,
        tla_k,
        tla_v,
        tla_o,
        tla_mask,
        tla_tiling,
        tla_actual_q,
        tla_actual_kv,
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="mix")
    artifact(
        tla_q,
        tla_k,
        tla_v,
        tla_o,
        tla_mask,
        tla_tiling,
        tla_actual_q,
        tla_actual_kv,
        block_num=block_num,
    )
    torch.npu.synchronize()

    # ==================== 精度校验 ====================
    scale = 1.0 / (HEAD_DIM**0.5)
    q_cpu = torch_q.cpu()
    k_cpu = torch_k.cpu()
    v_cpu = torch_v.cpu()

    # --- 真值: 全量 f32 attention ---
    golden_truth = torch.empty(BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM, dtype=torch.float32)
    for b in range(BATCH):
        q_b = q_cpu[b].permute(1, 0, 2).float()  # [H, Sq, D] f32
        k_b = k_cpu[b].permute(1, 0, 2).float()  # [Hkv, Svk, D] f32
        v_b = v_cpu[b].permute(1, 0, 2).float()
        if KV_HEAD_NUM != HEAD_NUM:
            k_b = k_b.repeat_interleave(GROUP_SIZE, dim=0)
            v_b = v_b.repeat_interleave(GROUP_SIZE, dim=0)
        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_b)
        golden_truth[b] = out.permute(1, 0, 2)

    # --- 标杆: ref_flash_attention (分块 online softmax, 对齐 kernel 实现) ---
    def _group_matmul(head, kv_head, left, right):
        gn = head // kv_head
        score = None
        for i in range(kv_head):
            gs = torch.matmul(
                left[i * gn : (i + 1) * gn, :, :].half(),
                right[i : (i + 1), :, :].half(),
            ).float()
            score = gs if score is None else torch.cat((score, gs), 0)
        return score

    def _qkMM1(query, key):
        result = None
        qk_k = key.shape[1]
        for s in range(0, qk_k, 128):
            sub = min(128, qk_k - s)
            pq = query[:, :, s : s + sub]
            pk = key[:, s : s + sub, :]
            split = _group_matmul(pq.shape[0], pk.shape[0], pq, pk)
            result = split if result is None else result + split
        return result

    def _pvMM2(p, value):
        result = None
        pv_k = value.shape[1]
        for s in range(0, pv_k, 128):
            sub = min(128, pv_k - s)
            pp = p[:, :, s : s + sub]
            pv = value[:, s : s + sub, :]
            split = _group_matmul(pp.shape[0], pv.shape[0], pp, pv)
            result = split if result is None else result + split
        return result

    dt = torch.float16 if args.dtype == "f16" else torch.bfloat16
    golden_bm = torch.empty(BATCH, Q_SEQ, HEAD_NUM, HEAD_DIM, dtype=dt)
    for b in range(BATCH):
        query = q_cpu[b].permute(1, 0, 2)  # [H, Sq, D]
        key = k_cpu[b].permute(1, 2, 0)  # [Hkv, D, Svk]
        value = v_cpu[b].permute(1, 0, 2)  # [Hkv, Svk, D]
        context_len = key.shape[2]
        gl = None
        gm = None
        go = None
        for kv_start in range(0, context_len, KV_BLOCK):
            sub_len = min(KV_BLOCK, context_len - kv_start)
            sub_key = key[:, :, kv_start : kv_start + sub_len]
            sub_value = value[:, kv_start : kv_start + sub_len]
            qk_result = _qkMM1(query, sub_key).float()
            qk_result = qk_result * scale
            # online softmax
            lm = torch.max(qk_result, dim=-1, keepdims=True)[0]
            if kv_start == 0:
                hm = lm
                dm = 0
            else:
                hm = torch.maximum(gm, lm)
                dm = gm - hm
            gm = hm
            sim_sub = torch.exp(qk_result - hm)
            row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
            p_result = sim_sub.to(dt)
            lo = _pvMM2(p_result, sub_value).float()
            if kv_start == 0:
                gl = row_sum
                go = lo
            else:
                dm = torch.exp(dm)
                gl = gl * dm + row_sum
                go = go * dm + lo
        go = go / gl
        go = torch.nan_to_num(go, nan=0.0)
        golden_bm[b] = go.permute(1, 0, 2)

    # --- 比值计算 ---
    result = torch_o.cpu().float()
    sentinel_t = torch.full_like(result, args.sentinel)
    unchanged = torch.isclose(result, sentinel_t, rtol=0.0, atol=UNCHANGED_THRESHOLD)

    epsilon = 1e-7

    def _metrics(actual, golden):
        a = actual.float()
        g = golden.float()
        diff = (a - g).abs()
        rel = diff / (g.abs() + epsilon)
        mare = rel.max().item()
        mere = rel.mean().item()
        rmse = torch.sqrt((diff**2).mean()).item()
        return mare, mere, rmse

    mare_n, mere_n, rmse_n = _metrics(result, golden_truth)
    mare_d, mere_d, rmse_d = _metrics(golden_bm, golden_truth)
    eps = (2.0**-7) if args.dtype == "f16" else (2.0**-6)
    mare_ratio = mare_n / max(mare_d, eps)
    mere_ratio = mere_n / max(mere_d, eps)
    rmse_ratio = rmse_n / max(rmse_d, eps)
    passed = mare_ratio <= 2.0 and mere_ratio <= 1.2 and rmse_ratio <= 1.2

    print(
        f"host=torch_npu BATCH={BATCH} Q_SEQ={Q_SEQ} KV_SEQ={KV_SEQ} "
        f"HEAD_NUM={HEAD_NUM} KV_HEAD_NUM={KV_HEAD_NUM} HEAD_DIM={HEAD_DIM} "
        f"Q_BLOCK_COUNT={Q_BLOCK_COUNT} KV_BLOCK_COUNT={KV_BLOCK_COUNT}"
    )
    print(
        f"O unchanged (sentinel)? {bool(unchanged.all())} "
        f"changed_count={int((~unchanged).sum().item())} / {result.numel()}"
    )
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="flash_attention_infer: full FA kernel (QK + softmax + PV + rescale)."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--headnum", type=int, default=HEAD_NUM)
    parser.add_argument("--kvheadnum", type=int, default=KV_HEAD_NUM)
    parser.add_argument("--qseqlen", type=int, default=Q_SEQ)
    parser.add_argument("--kvseqlen", type=int, default=KV_SEQ)
    parser.add_argument("--block-num", type=int, default=-1)
    parser.add_argument("--sentinel", type=float, default=-7.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
