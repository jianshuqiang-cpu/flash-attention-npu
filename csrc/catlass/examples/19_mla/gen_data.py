#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
from ml_dtypes import bfloat16
from dataclasses import dataclass

np.random.seed(1)


WORKSPACE = os.path.dirname(os.path.abspath(__file__))


class TestPagedMLAttention:
    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        mask_type: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        head_size_rope: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any

    @classmethod
    def check_attr(
        cls,
        batch: int,
        q_seqlen_list: list,
        k_seqlen_list: list,
        num_blocks: int,
        block_size: int,
    ):
        # 检查列表长度是否与 batch size 匹配
        if len(q_seqlen_list) != batch or len(k_seqlen_list) != batch:
            logging(
                "[ERROR] The length of q_seqlen_list and k_seqlen_list must be equal to batch size."
            )
            sys.exit()

        # 检查缓存大小是否足够
        if sum(k_seqlen_list) > num_blocks * block_size:
            logging(
                "[ERROR] the number of K and V tokens is too big to fit in the paged cache."
            )
            sys.exit()

        if block_size != 128:
            logging("[ERROR] blockSize != 128 is not supported.")
            sys.exit()

        # 检查每个 q_seqlen 是否在有效范围内 [1, 4]
        for q_seqlen in q_seqlen_list:
            if q_seqlen > 4 or q_seqlen < 1:
                logging(
                    f"[ERROR] q_seqlen value {q_seqlen} is not in the valid range of [1, 4]."
                )
                sys.exit()

        # 检查每个 kv_seqlen 是否在有效范围内 [128, 16384]
        for kv_seqlen in k_seqlen_list:
            if kv_seqlen > 16384 or kv_seqlen < 128:
                logging(
                    f"[ERROR] kv_seqlen value {kv_seqlen} is not in the valid range of [128, 16384]."
                )
                sys.exit()

    @classmethod
    def group_matmul(cls, head, kv_head, left, right):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(
                left[i * group_num : (i + 1) * group_num, :, :].astype(np.float32),
                right[i : (i + 1), :, :].astype(np.float32),
            )
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def ref_masked_attention(
        self,
        query,  # (q_seqlen, num_heads, head_size)
        key,  # (k_seqlen, kv_heads, head_size)
        value,
        scale: float,
        mask,  # (q_seqlen, k_seqlen)
    ):
        # Q * K.T
        query = query
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(
            query.shape[0], key.shape[0], query, key
        )  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (
                mask[: sim_high.shape[-2], : sim_high.shape[-1]] * self.post_mask_factor
            ).astype(np.float32)

        # softmax
        p_high = self.softmax_numpy(sim_high)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_high = np.transpose(out_high, (1, 0, 2))
        out = np.transpose(out, (1, 0, 2))
        out = out.astype(query.dtype)
        return out, out_high

    def ref_single_query_cached_kv_attention(
        self, attention_inputs: AttentionInputs, output, true_out
    ) -> None:
        num_heads = attention_inputs.query.shape[1]
        kv_heads = attention_inputs.value_cache.shape[2]
        head_size_qk = attention_inputs.key_cache.shape[3]
        head_size_vo = attention_inputs.value_cache.shape[3]
        block_size = attention_inputs.value_cache.shape[1]

        batch = len(attention_inputs.q_seqlen_list)
        cu_seqlen = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = attention_inputs.query[cu_seqlen : (cu_seqlen + q_seqlen), :, :]
            block_table = attention_inputs.block_tables[i]
            keys = []
            values = []
            for j in range(k_seqlen):  # j 每个k token拼接
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = attention_inputs.key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = attention_inputs.value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)
            keys = np.stack(keys, axis=0)
            values = np.stack(values, axis=0)
            scale = 1.0 / (head_size_qk**0.5)
            if attention_inputs.mask_type == 1:
                mask = attention_inputs.global_mask[
                    cu_seqlen : (cu_seqlen + q_seqlen), :
                ]
            else:
                mask = None
            out, out_high = self.ref_masked_attention(q, keys, values, scale, mask)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out_high
            cu_seqlen += attention_inputs.q_seqlen_list[i]

    def calc_data(self, gen_data_params: GenDataParams):
        head_size_qk = gen_data_params.head_size + gen_data_params.head_size_rope
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        batch_size = len(gen_data_params.q_seqlen_list)
        query = np.random.uniform(
            q_min_range,
            q_max_range,
            size=(num_tokens, gen_data_params.num_heads, head_size_qk),
        ).astype(gen_data_params.dtype)
        query_nope = query[:, :, :head_size_vo]
        query_rope = query[:, :, -gen_data_params.head_size_rope :]

        key_cache = np.random.uniform(
            kv_min_range,
            kv_max_range,
            size=(
                gen_data_params.num_blocks,
                gen_data_params.block_size,
                gen_data_params.kv_heads,
                head_size_qk,
            ),
        ).astype(gen_data_params.dtype)
        kv_nope_cache = key_cache[:, :, :, :head_size_vo]
        kv_rope_cache = key_cache[:, :, :, -gen_data_params.head_size_rope :]
        value_cache = kv_nope_cache

        max_k_seqlen = max(gen_data_params.k_seqlen_list)
        max_num_blocks_per_seq = (
            max_k_seqlen + gen_data_params.block_size - 1
        ) // gen_data_params.block_size
        block_tables = []  # (num_tokens, max_num_blocks_per_seq）
        for i in range(batch_size):
            block_table = [
                max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        pre_mask_factor = -10000.0
        if gen_data_params.mask_type == 1:
            mask = np.zeros(shape=(num_tokens, max_k_seqlen)).astype(
                gen_data_params.dtype
            )
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = gen_data_params.q_seqlen_list[i]
                kseqlen = gen_data_params.k_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= pre_mask_factor
                mask[
                    pre_qseqlen : (pre_qseqlen + qseqlen), kseqlen - qseqlen : kseqlen
                ] = tri
                pre_qseqlen += qseqlen
            mask = mask.astype(gen_data_params.dtype)
        elif gen_data_params.mask_type == 0:
            mask = None

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        ref_output = np.zeros(shape_out, dtype=gen_data_params.dtype)
        true_out = np.zeros(shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(
            query,
            key_cache,
            value_cache,
            block_tables,
            gen_data_params.q_seqlen_list,
            gen_data_params.k_seqlen_list,
            mask,
            gen_data_params.mask_type,
        )
        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            ref_output,
            true_out,
        )

        num_tokens.astype(np.int32).tofile(
            os.path.join(WORKSPACE, "data", "q_ntokens.bin")
        )
        query_nope.tofile(os.path.join(WORKSPACE, "data", "q.bin"))
        query_rope.tofile(os.path.join(WORKSPACE, "data", "q_rope.bin"))
        kv_nope_cache.tofile(os.path.join(WORKSPACE, "data", "k.bin"))
        kv_rope_cache.tofile(os.path.join(WORKSPACE, "data", "k_rope.bin"))
        np.array(block_tables).astype(np.int32).tofile(
            os.path.join(WORKSPACE, "data", "block_table.bin")
        )
        np.array(gen_data_params.q_seqlen_list).astype(np.int32).tofile(
            os.path.join(WORKSPACE, "data", "q_seqlen.bin")
        )
        np.array(gen_data_params.k_seqlen_list).astype(np.int32).tofile(
            os.path.join(WORKSPACE, "data", "kv_seqlen.bin")
        )
        if mask:
            mask.tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
        ref_output.astype(gen_data_params.dtype).tofile(
            os.path.join(WORKSPACE, "data", "cpu_low.bin")
        )
        true_out.astype(np.float32).tofile(
            os.path.join(WORKSPACE, "data", "golden.bin")
        )


if __name__ == "__main__":
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    # 修改命令行参数解析逻辑，以接受列表形式的输入
    if len(sys.argv) != 8:
        print(
            'Usage: python gen_data.py <batchSize> "<qSeqlen_list>" "<kvSeqlen_list>" <qheadNum> <numBlock> <blockSize> <dtype>'
        )
        print(
            'Example: python gen_data.py 4 "1,2,3,4" "128,256,512,1024" 16 16 128 half'
        )
        sys.exit(1)

    batch = int(sys.argv[1])

    # 将逗号分隔的字符串解析为整数列表
    q_seqlen_list_str = sys.argv[2]
    kv_seqlen_list_str = sys.argv[3]
    q_seqlen_list = [int(x.strip()) for x in q_seqlen_list_str.split(",")]
    kv_seqlen_list = [int(x.strip()) for x in kv_seqlen_list_str.split(",")]

    num_head = int(sys.argv[4])
    num_blocks = int(sys.argv[5])
    block_size = int(sys.argv[6])
    str_dtype = str(sys.argv[7])
    max_kv_seqlen = max(kv_seqlen_list) if kv_seqlen_list else 0

    mask_type = 0
    kv_heads = 1
    embedding_size = 512
    embedding_size_rope = 64
    if str_dtype == "half":
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        logging("[ERROR] dtype must be half or bf16")
        sys.exit()

    testObj = TestPagedMLAttention()

    testObj.check_attr(batch, q_seqlen_list, kv_seqlen_list, num_blocks, block_size)
    gen_data_params = testObj.GenDataParams(
        q_seqlen_list,
        kv_seqlen_list,
        num_head,
        kv_heads,
        embedding_size,
        embedding_size_rope,
        num_blocks,
        block_size,
        mask_type,
        dtype,
    )

    testObj.calc_data(gen_data_params)
