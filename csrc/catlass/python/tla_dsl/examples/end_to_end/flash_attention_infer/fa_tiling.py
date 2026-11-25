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
from dataclasses import dataclass
from typing import List

Q_BASE_TILE = 128
KV_BASE_TILE = 128

QK_L1_TILE_M = 128
QK_L1_TILE_N = 256
QK_L1_TILE_K_LEFT = 192
QK_L1_TILE_K_RIGHT = 192

PV_L1_TILE_M = 128
PV_L1_TILE_N = 128
PV_L1_TILE_K_LEFT = 256
PV_L1_TILE_K_RIGHT = 128

Q_L1_BUF_NUM = 1
K_L1_BUF_NUM = 2
V_L1_BUF_NUM = 2
P_L1_BUF_NUM = 3


@dataclass
class FATilingData:
    batch: int = 0
    num_heads: int = 0
    kv_heads: int = 0
    max_q_seqlen: int = 0
    max_kv_seqlen: int = 0
    first_batch_task_num: int = 0
    total_task_num: int = 0
    q_base_tile: int = Q_BASE_TILE
    kv_base_tile: int = KV_BASE_TILE
    head_dim: int = 128
    group_size: int = 0
    qk_l1_tile_m: int = QK_L1_TILE_M
    qk_l1_tile_n: int = QK_L1_TILE_N
    qk_l1_tile_k_left: int = QK_L1_TILE_K_LEFT
    qk_l1_tile_k_right: int = QK_L1_TILE_K_RIGHT
    pv_l1_tile_m: int = PV_L1_TILE_M
    pv_l1_tile_n: int = PV_L1_TILE_N
    pv_l1_tile_k_left: int = PV_L1_TILE_K_LEFT
    pv_l1_tile_k_right: int = PV_L1_TILE_K_RIGHT
    q_l1_buf_num: int = Q_L1_BUF_NUM
    k_l1_buf_num: int = K_L1_BUF_NUM
    v_l1_buf_num: int = V_L1_BUF_NUM
    p_l1_buf_num: int = P_L1_BUF_NUM


def compute_tiling(
    batch: int,
    num_heads: int,
    kv_heads: int,
    q_seqlen_list: List[int],
    kv_seqlen_list: List[int],
    head_dim: int = 128,
    q_base_tile: int = Q_BASE_TILE,
    kv_base_tile: int = KV_BASE_TILE,
) -> FATilingData:
    td = FATilingData()
    td.batch = batch
    td.num_heads = num_heads
    td.kv_heads = kv_heads
    td.head_dim = head_dim
    td.q_base_tile = q_base_tile
    td.kv_base_tile = kv_base_tile
    td.group_size = num_heads // kv_heads
    td.max_q_seqlen = max(q_seqlen_list)
    td.max_kv_seqlen = max(kv_seqlen_list)

    td.qk_l1_tile_m = QK_L1_TILE_M
    td.qk_l1_tile_n = QK_L1_TILE_N
    td.qk_l1_tile_k_left = QK_L1_TILE_K_LEFT
    td.qk_l1_tile_k_right = QK_L1_TILE_K_RIGHT
    td.pv_l1_tile_m = PV_L1_TILE_M
    td.pv_l1_tile_n = PV_L1_TILE_N
    td.pv_l1_tile_k_left = PV_L1_TILE_K_LEFT
    td.pv_l1_tile_k_right = PV_L1_TILE_K_RIGHT
    td.q_l1_buf_num = Q_L1_BUF_NUM
    td.k_l1_buf_num = K_L1_BUF_NUM
    td.v_l1_buf_num = V_L1_BUF_NUM
    td.p_l1_buf_num = P_L1_BUF_NUM

    total_task_num = 0
    first_batch_task_num = 0
    for b_idx in range(batch):
        q_s = q_seqlen_list[b_idx]
        q_block_count_cur = (q_s + q_base_tile - 1) // q_base_tile
        cur_task_num = num_heads * q_block_count_cur
        if b_idx == 0:
            first_batch_task_num = cur_task_num
        total_task_num += cur_task_num

    td.first_batch_task_num = first_batch_task_num
    td.total_task_num = total_task_num
    return td


def pack_tiling_int(td: FATilingData) -> List[int]:
    return [
        td.batch,
        td.num_heads,
        td.kv_heads,
        td.max_q_seqlen,
        td.max_kv_seqlen,
        td.first_batch_task_num,
        td.total_task_num,
        td.q_base_tile,
        td.kv_base_tile,
        td.qk_l1_tile_m,
        td.qk_l1_tile_n,
        td.qk_l1_tile_k_left,
        td.qk_l1_tile_k_right,
        td.pv_l1_tile_m,
        td.pv_l1_tile_n,
        td.pv_l1_tile_k_left,
        td.pv_l1_tile_k_right,
        td.q_l1_buf_num,
        td.k_l1_buf_num,
        td.v_l1_buf_num,
        td.p_l1_buf_num,
    ]


def make_actual_seqlen(seqlen_list: List[int]):
    import itertools

    return [0] + list(itertools.accumulate(seqlen_list))
