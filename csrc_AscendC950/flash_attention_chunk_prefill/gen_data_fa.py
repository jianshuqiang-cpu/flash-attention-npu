#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import random
from ml_dtypes import bfloat16
from dataclasses import dataclass
# import flash_decoding_tiling
np.random.seed(1)
random.seed(1)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for i in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            kv_seq = random.randint(q_seq, max_kv_seqlen)
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    print(f"q_seqlen_list:{q_seqlen_list}")
    print(f"kv_seqlen_list:{kv_seqlen_list}")
    return q_seqlen_list, kv_seqlen_list

class TestFlashAttentionInfer():

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
        inner_prec: int
        shape_param: any
        sink_vector: any
        pretoken: int
        nexttoken: int

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        qk_head_size: int
        v_head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        layout_dtype: int
        inner_prec: int
        max_q_seqlen: int
        max_kv_seqlen: int
        lse_flag: int
        sink_flag: int
        pretoken: int
        nexttoken: int
        cache_layout: any
        isBnNBsD: int
        global_window_size: int
        local_window_size: int
        core_num: int

    @classmethod
    def check_attr(cls, batch: int, q_seqlen: int, kv_seqlen: int, num_blocks: int, block_size: int):
        if q_seqlen > kv_seqlen:
            logging("[ERROR] q_seqlen cannot exceed kv_seqlen.")
            sys.exit()

    @classmethod
    def group_matmul(cls, head, kv_head, left, right):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                                    right[i:(i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim, sink_matrix):
        row_max = np.max(sim, axis=-1, keepdims=True)
        if sink_matrix is not None:
            assert sink_matrix.shape == row_max.shape, \
                f"sink_matrix 形状 {sink_matrix.shape} 与 row_max 形状 {row_max.shape} 不一致！"
            # 更新含sink的rowmax
            row_max = np.maximum(row_max, sink_matrix)

        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)

        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        # add sink rowsum
        if sink_matrix is not None:
            # 更新含sink的rowmax
            sink_exp = np.exp(sink_matrix - row_max)
            row_sum = row_sum + sink_exp

        soft_res = sim_sub / row_sum
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

        return soft_res, lse, row_max

    def softmax1(
        self,
        qk_result,
        is_first,
        gm,
        is_kvs_last_loop,
 	    sink_matrix,
        data_type = np.float16
    ):
        sim = qk_result
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm

        if sink_matrix is not None and is_first:
            assert sink_matrix.shape == hm.shape, \
            f"sink_matrix 形状 {sink_matrix.shape} 与 hm 形状 {hm.shape} 不一致！"
            hm = np.maximum(hm, sink_matrix)
            dm = gm - hm if not is_first  else 0

        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(np.float32))
        # sim_sub = sim_sub.astype(np.float16)

        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)

        sink_exp = None
        if sink_matrix is not None and is_kvs_last_loop:
            sink_exp = np.exp(sink_matrix - hm)

        return sim_sub, row_sum, dm, gm, sink_exp


    def qkMM1(
        self,
        query,
        key
    ):
        result = None
        qk_k = key.shape[1]
        for qk_k_split in range(0, qk_k, 128):
            sub_k = 128
            if qk_k_split == 512:
                sub_k = 64
            query_k = query[:, :, qk_k_split: qk_k_split + sub_k]
            key_k = key[:, qk_k_split: qk_k_split + sub_k, :]
            result_split = self.group_matmul(query_k.shape[0], key_k.shape[0], query_k, key_k)
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def ref_flash_attention(
        self,
        query,
        key,
        value,
        scale,
        mask,
        attention_inputs: AttentionInputs,
        sink_matrix
    ):
        data_type = attention_inputs.shape_param.dtype
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        value = np.transpose(value, (1, 0, 2))
        context_len = key.shape[2]
        context_size = 128
        group_num = query.shape[0] // key.shape[0]
        gl = None
        gl_high = None
        go = None
        go_high = None

        sink_matrix_high = None
        if sink_matrix is not None:
            sink_matrix_high= sink_matrix.astype(np.float32)

        is_kvs_last_loop = False
        FIRST_CONTEXT_SIZE = 128
        # ============= 第一次循环：自定义长度 =============
        kv_start = 0
        if kv_start < context_len:
            # 第一次取你设置的长度，不超过剩余长度
            sub_len = min(FIRST_CONTEXT_SIZE, context_len - kv_start)
            is_kvs_last_loop = (kv_start + sub_len >= context_len)
            
            sub_key = key[:, :, kv_start: kv_start + sub_len]
            sub_mask = None
            if mask is not None:
                sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len]
            sub_value = value[:, kv_start: kv_start + sub_len, :]
            qk_result = self.qkMM1(query, sub_key).astype(np.float32)
            qk_result_high = self.qkMM1(query.astype(np.float32), sub_key.astype(np.float32))

            qk_result = qk_result * scale
            qk_result_high = qk_result_high * scale

            if mask is not None:
                qk_result += sub_mask
                qk_result_high += sub_mask.astype(np.float32)
            
            if kv_start == 0:
                gm = None
            p_result, row_sum, dm, gm, sink_exp = self.softmax1(qk_result, kv_start == 0, gm, is_kvs_last_loop, sink_matrix)
            p_result = p_result.astype(data_type)
            if kv_start == 0:
                gm_high = None
            p_result_high, row_sum_high, dm_high, gm_high, sink_exp_high = self.softmax1(qk_result_high, kv_start == 0, gm_high, is_kvs_last_loop, sink_matrix_high)
            lo = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result, sub_value)
            lo_high = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result.astype(np.float32), sub_value.astype(np.float32))

            gl = row_sum
            gl_high = row_sum_high
            go = lo
            go_high = lo_high

            if is_kvs_last_loop and sink_exp is not None:
                assert gl.shape == sink_exp.shape, \
                f"sink_matrix 形状 {gl.shape} 与 hm 形状 {sink_exp.shape} 不一致！"
                gl = gl + sink_exp
                gl_high = gl_high + sink_exp_high

            # 更新起始位置
            kv_start += sub_len

        # ============= 后续循环：固定 128 长度 =============
        for kv_start in range(kv_start, context_len, context_size):
            sub_len = context_size
            if kv_start + context_size > context_len:
                sub_len = context_len - kv_start

            is_kvs_last_loop = (kv_start + context_size >= context_len)
            sub_key = key[:, :, kv_start: kv_start + sub_len]
            sub_mask = None
            if mask is not None:
                sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len]
            sub_value = value[:, kv_start: kv_start + sub_len, :]
            qk_result = self.qkMM1(query, sub_key).astype(np.float32)
            qk_result_high = self.qkMM1(query.astype(np.float32), sub_key.astype(np.float32))

            

            qk_result = qk_result * scale
            qk_result_high = qk_result_high * scale

            if mask is not None:
                qk_result += sub_mask
                qk_result_high += sub_mask.astype(np.float32)
            
            p_result, row_sum, dm, gm, sink_exp = self.softmax1(qk_result, False, gm, is_kvs_last_loop, sink_matrix)
            p_result = p_result.astype(data_type)
            p_result_high, row_sum_high, dm_high, gm_high, sink_exp_high = self.softmax1(qk_result_high, False, gm_high, is_kvs_last_loop, sink_matrix_high)
            lo = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result, sub_value)
            lo_high = self.group_matmul(p_result.shape[0], sub_value.shape[0], p_result.astype(np.float32), sub_value.astype(np.float32))

            dm = np.exp(dm)
            dm_high = np.exp(dm_high)
            gl = gl * dm
            gl = gl + row_sum
            go = go * dm
            go = go + lo
            gl_high = gl_high * dm_high
            gl_high = gl_high + row_sum_high
            go_high = go_high * dm_high
            go_high = go_high + lo_high

            if is_kvs_last_loop and sink_exp is not None:
                assert gl.shape == sink_exp.shape, \
                f"sink_matrix 形状 {gl.shape} 与 hm 形状 {sink_exp.shape} 不一致！"
                gl = gl + sink_exp
                gl_high = gl_high + sink_exp_high

        go = go / gl
        go_high = go_high / gl_high
        go = np.transpose(go, (1, 0, 2))
        go_high = np.transpose(go_high, (1, 0, 2))
        lse = np.squeeze((np.log(gl) + gm), axis=-1)
        lse_high = np.squeeze((np.log(gl_high) + gm_high), axis=-1)
        return go.astype(data_type), go_high, lse, lse_high

    def generate_swa_full_mask(self, maxkvseqlen, local_window_size, global_window_size):
        N = maxkvseqlen
        one_tensor = np.ones((N, N), dtype=np.float32)
        inner_size = N - local_window_size - global_window_size
        if inner_size > 0:
            inner_mask = np.triu(np.ones((inner_size, inner_size), dtype=np.float32), k=1)
            row_start = local_window_size + global_window_size
            col_start = global_window_size
            one_tensor[row_start:row_start+inner_size, col_start:col_start+inner_size] = inner_mask
        result = np.tril(one_tensor)
        # ======================
        # 关键：掩码全部取反 0  1
        # ======================
        result = 1 - result
        return result.astype(np.float16)

    #chunkprefill, 全量, (batch * maxqseqlen, maxkvseqlen)
    #swa 全量 (maxkvseqlen, maxkvseqlen)
    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask,    # (q_seqlen, k_seqlen)
            sink_matrix
    ):
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        # print(sim_high)
        sim_low_prec = sim_high.astype(np.float16) * np.float16(scale)
        sim_high = sim_high * scale
        # sink
        sink_matrix_low_prec = None
        if sink_matrix is not None:
            sink_matrix_low_prec = sink_matrix.astype(np.float32)
        if mask is not None:
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32)
            sim_low_prec = sim_low_prec + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float16)

        sink_matrix_low_prec = None
        if sink_matrix is not None:
            sink_matrix_low_prec = sink_matrix.astype(np.float32)

        p_high, lse_high, gm = self.softmax_numpy(sim_high, sink_matrix)
        p_low_prec, lse_low_prec, gm_low_prec = self.softmax_numpy(sim_low_prec, sink_matrix_low_prec)
        lse = lse_high.astype(query.dtype)
        lse_high = lse_high.astype(np.float32)
        lse_high = lse_high.astype(np.float32)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        
        out_low_prec = self.group_matmul(query.shape[0], key.shape[0], p_low_prec, value)
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_low_prec = np.transpose(out_low_prec, (1, 0, 2))
        out_high = np.transpose(out_high, (1, 0, 2))
        out = np.transpose(out, (1, 0, 2))
        out_low_prec = out_low_prec.astype(np.float16)
        out = out.astype(query.dtype)
        return out, out_high, out_low_prec, lse, lse_high, gm

    def get_batch_inputs(self, attention_inputs, batch_idx, cu_seqlen, kv_seqlen_now):
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.qk_head_size
        head_size_vo = attention_inputs.shape_param.v_head_size
        block_size = attention_inputs.shape_param.block_size
        max_q_seqlen = attention_inputs.shape_param.max_q_seqlen
        
        q_seqlen = int(attention_inputs.q_seqlen_list[batch_idx])
        k_seqlen = int(attention_inputs.k_seqlen_list[batch_idx])
        
        q = None
        if attention_inputs.shape_param.layout_dtype == 1:
            q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
        else:
            q = attention_inputs.query[batch_idx * max_q_seqlen:(batch_idx * max_q_seqlen + q_seqlen), :, :]
            
        keys = None
        values = None
        if attention_inputs.shape_param.kv_dtype == 1:
            keys = []
            values = []
            block_table = attention_inputs.block_tables[batch_idx]
            for j in range(k_seqlen):
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
        elif attention_inputs.shape_param.kv_dtype == 0:
            if attention_inputs.shape_param.layout_dtype == 1:
                keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
            else:
                keys = attention_inputs.key_cache[batch_idx, :, :, :]
                values = attention_inputs.value_cache[batch_idx, :, :, :]
                
        scale = 1.0 / (head_size_qk ** 0.5)
        
        mask = None
        if attention_inputs.mask_type == 1 or attention_inputs.mask_type == 4:
            mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
        elif attention_inputs.mask_type == 2:
            mask = attention_inputs.global_mask

        return q, keys, values, mask, scale

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.qk_head_size
        head_size_vo = attention_inputs.shape_param.v_head_size
        block_size = attention_inputs.shape_param.block_size
        max_q_seqlen = attention_inputs.shape_param.max_q_seqlen
        inner_prec = attention_inputs.inner_prec
        sink_vector = attention_inputs.sink_vector
        if sink_vector is not None:
                sink_vector = sink_vector.astype(np.float32)

        batch = len(attention_inputs.shape_param.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            print(f"batch:{i}")
            print(f"q_seqlen:{q_seqlen}")
            print(f"k_seqlen:{k_seqlen}")
            q = None
            if attention_inputs.shape_param.layout_dtype == 1:
                q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            else:
                q = attention_inputs.query[i * max_q_seqlen:(i * max_q_seqlen + q_seqlen), :, :]
            keys = None
            values = None
            if attention_inputs.shape_param.kv_dtype == 1:
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                if attention_inputs.shape_param.isBnNBsD == 0:
                    for j in range(k_seqlen):
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
                else:
                    for j in range(k_seqlen):
                        block_number = int(block_table[j // block_size])
                        block_offset = j % block_size

                        k = attention_inputs.key_cache[block_number, :, block_offset, :]
                        k = k.reshape(kv_heads, head_size_qk)
                        keys.append(k)

                        v = attention_inputs.value_cache[block_number, :, block_offset, :]
                        v = v.reshape(kv_heads, head_size_vo)
                        values.append(v)
                    keys = np.stack(keys, axis=0)
                    values = np.stack(values, axis=0)
            elif attention_inputs.shape_param.kv_dtype == 0:
                if attention_inputs.shape_param.layout_dtype == 1:
                    keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                    values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                else:
                    keys = attention_inputs.key_cache[i, :, :, :]
                    values = attention_inputs.value_cache[i, :, :, :]
            scale = 1.0 / (head_size_qk ** 0.5)
            if attention_inputs.mask_type == 1 or attention_inputs.mask_type == 4:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            elif attention_inputs.mask_type == 2 or attention_inputs.mask_type == 3:
                mask = attention_inputs.global_mask
            else:
                mask = None
            # sink: [num_heads, 1]->[num_heads, q_seqlen, 1]
            sink_matrix = None
            if sink_vector is not None:
                # [num_heads, 1] → [num_heads, 1, 1]
                sink_expanded = np.expand_dims(sink_vector, axis=1)
                # [num_heads, 1, 1] → [num_heads, q_seqlen, 1]
                sink_matrix = np.broadcast_to(sink_expanded, shape=(sink_vector.shape[0], q_seqlen, 1))
                sink_matrix = sink_matrix.astype(np.float32)
            out_normal, _,  out_low_prec, lse, _, gm = self.ref_masked_attention(q, keys, values, scale, mask, sink_matrix)
            out_high, _, lse_high, _ = self.ref_flash_attention(q, keys, values, scale, mask, attention_inputs, sink_matrix)
            out = None
            if inner_prec == 0:
                out = out_normal.reshape(-1, num_heads, head_size_vo)
            else:
                out = out_low_prec.reshape(-1, num_heads, head_size_vo)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            preTokens = attention_inputs.pretoken
            nextTokens = attention_inputs.nexttoken
            preTokensChange = preTokens - k_seqlen + q_seqlen
            nextTokensChange = nextTokens + k_seqlen - q_seqlen
            nextTokensError = -nextTokensChange if nextTokensChange < 0 else 0
            preTokensError = (q_seqlen - k_seqlen - preTokensChange) if q_seqlen > k_seqlen + preTokensChange else 0
            actualSeq = q_seqlen
            print(f"{i},:{preTokens},{nextTokens},{preTokensChange},{nextTokensChange},{preTokensError},{nextTokensError}")
            actualSeq -= nextTokensError
            actualSeq -= preTokensError
            if actualSeq != q_seqlen and attention_inputs.mask_type == 3:
                if nextTokensError != 0:
                    # 前n行置0
                    actualSeq = q_seqlen - actualSeq
                elif preTokensError != 0:
                    # 后n行置0
                    actualSeq = actualSeq
            if attention_inputs.shape_param.layout_dtype == 1:
                output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high

                golden_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse_high
                if actualSeq != q_seqlen :
                    if nextTokensError != 0:
                        output[cu_seqlen : cu_seqlen  + actualSeq, :, :] = 0  # 前n行置0
                        golden_gpu_output[cu_seqlen: cu_seqlen + actualSeq, :, :] = 0
                        golden_lse_output[:, cu_seqlen: cu_seqlen + actualSeq] = np.inf
                        golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + actualSeq] = np.inf
                    elif preTokensError != 0:
                        output[cu_seqlen + actualSeq: cu_seqlen  + q_seqlen, :, :] = 0  # 后n行置0
                        golden_gpu_output[cu_seqlen + actualSeq: cu_seqlen + q_seqlen, :, :] = 0
                        golden_lse_output[:, cu_seqlen + actualSeq: cu_seqlen  + q_seqlen] =  np.inf
                        golden_gpu_lse_output[:, cu_seqlen + actualSeq: cu_seqlen + q_seqlen] =  np.inf
            else:
                output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out_high

                golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse_high
                if actualSeq != q_seqlen :
                    if nextTokensError != 0:
                        output[i * max_q_seqlen: i * max_q_seqlen + actualSeq, :, :] = 0
                        golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + actualSeq, :, :] = 0

                        golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + actualSeq] = np.inf
                        golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + actualSeq] = np.inf
                    elif preTokensError != 0:
                        output[i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen, :, :] = 0
                        golden_gpu_output[i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen, :, :] = 0

                        golden_lse_output[:, i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen] = np.inf
                        golden_gpu_lse_output[:, i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen] = np.inf
            
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def create_binary_matrix(self, qSeqlen, kvSeqlen, preToken, nextToken):
        preToken = kvSeqlen - qSeqlen - preToken
        nextToken = kvSeqlen - qSeqlen + nextToken
        matrix = [[0 for _ in range(kvSeqlen)] for _ in range(qSeqlen)]
        for i in range(qSeqlen):
            for j in range(kvSeqlen):
                is_below_pretoken_line = (-i + j) < preToken
                is_above_nexttoken_line = (-i + j) > nextToken
                if is_below_pretoken_line or is_above_nexttoken_line:
                    matrix[i][j] = 1
        
        return np.array(matrix)

    def calc_data(self, gen_data_params: GenDataParams):
        head_size_qk = gen_data_params.qk_head_size
        head_size_vo = gen_data_params.v_head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        num_kv_tokens = np.array(gen_data_params.k_seqlen_list).sum()
        batch_size = len(gen_data_params.q_seqlen_list)
        query = np.random.uniform(q_min_range, q_max_range,
            size=(num_tokens, gen_data_params.num_heads, head_size_qk)).astype(gen_data_params.dtype)
        max_k_seqlen = gen_data_params.max_kv_seqlen
        max_q_seqlen = max(gen_data_params.q_seqlen_list)
        block_tables = []
        key_cache = None
        value_cache = None
        if gen_data_params.kv_dtype == 1:
            print(f"isBnNBsD : {isBnNBsD}")
            if gen_data_params.isBnNBsD == 0:
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(gen_data_params.num_blocks, gen_data_params.block_size,
                    gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)

                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(gen_data_params.num_blocks, gen_data_params.block_size,
                    gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
                max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
                for i in range(batch_size):
                    block_table = [
                        max_num_blocks_per_seq * i + j
                        for j in range(max_num_blocks_per_seq)
                    ]
                    block_tables.append(block_table)
            else:
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(gen_data_params.num_blocks, gen_data_params.kv_heads,
                    gen_data_params.block_size, head_size_qk)).astype(gen_data_params.dtype)

                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(gen_data_params.num_blocks, gen_data_params.kv_heads,
                    gen_data_params.block_size, head_size_vo)).astype(gen_data_params.dtype)
                max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
                for i in range(batch_size):
                    block_table = [
                        max_num_blocks_per_seq * i + j
                        for j in range(max_num_blocks_per_seq)
                    ]
                    block_tables.append(block_table)
        elif gen_data_params.kv_dtype == 0:
            if gen_data_params.layout_dtype == 1:
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
            elif gen_data_params.layout_dtype == 0:
                key_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_qk)).astype(gen_data_params.dtype)
                value_cache = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_vo)).astype(gen_data_params.dtype)
        # pre_mask_factor = -10000
        pre_mask_factor = -65500
        if gen_data_params.mask_type == 1:
            mask = np.zeros(shape=(num_tokens, max_k_seqlen)).astype(np.float16)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = gen_data_params.q_seqlen_list[i]
                kseqlen = gen_data_params.k_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= pre_mask_factor
                mask[pre_qseqlen : (pre_qseqlen + qseqlen), kseqlen - qseqlen : kseqlen] = tri
                pre_qseqlen += qseqlen
            mask = mask.astype(gen_data_params.dtype)
        elif gen_data_params.mask_type == 2:
            mask = np.ones(shape=(max_q_seqlen, max_k_seqlen)).astype(np.float16)
            mask = np.triu(mask, 1)
            mask *= pre_mask_factor
        elif gen_data_params.mask_type == 0:
            mask = None
        elif gen_data_params.mask_type == 3:
            mask = np.zeros(shape=(num_tokens, max_k_seqlen)).astype(np.float16)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = gen_data_params.q_seqlen_list[i]
                kseqlen = gen_data_params.k_seqlen_list[i]
                tri = self.create_binary_matrix(qseqlen, kseqlen, pretoken, nexttoken)
                tri = tri.astype(np.float16)
                tri *= pre_mask_factor
                mask[pre_qseqlen : (pre_qseqlen + qseqlen), :] = tri
                pre_qseqlen += qseqlen
            mask = mask.astype(gen_data_params.dtype)
        elif gen_data_params.mask_type == 4:
            swa_mask_full = self.generate_swa_full_mask(
                max_k_seqlen,
                gen_data_params.local_window_size,
                gen_data_params.global_window_size
            )
            np.set_printoptions(
                threshold=100,
                edgeitems=32,   # 首尾各显示5个
                linewidth=200
            )
            # print("mask:", swa_mask_full.shape)
            # print("mask:", swa_mask_full)
            mask = np.zeros((num_tokens, max_k_seqlen), dtype=np.float16)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = gen_data_params.q_seqlen_list[i]
                kseqlen = gen_data_params.k_seqlen_list[i]
                diff = max_k_seqlen - kseqlen
                # print(f"lch########### {diff}")
                # print(f"lch########### {qseqlen}")
                if diff > 0:
                    sub_mask = swa_mask_full[-qseqlen-diff:-diff, :kseqlen]
                else:
                    sub_mask = swa_mask_full[-qseqlen:, :kseqlen]
                mask[pre_qseqlen : pre_qseqlen + qseqlen, :kseqlen] = sub_mask
                pre_qseqlen += qseqlen
            mask = mask.astype(gen_data_params.dtype)
            mask *= pre_mask_factor
            # print("sub mask:", sub_mask)
        # generate sink_vector
        sink_vector = self.gen_sink(gen_data_params.sink_flag, gen_data_params.num_heads, gen_data_params.dtype)
        # sink_bin
        sink_to_write = None
        sink_to_write_fp32 = None
        if sink_vector is None:
            sink_to_write = np.array([], dtype=bfloat16)
            sink_to_write_fp32 = np.array([], dtype=np.float32)
        else:
            print(f"sink_fp32: {sink_vector} \n")
            sink_to_write = sink_vector.astype(bfloat16)
            sink_to_write_fp32 = sink_vector.astype(np.float32)
            print(f"sink_fp32: {sink_to_write_fp32} \n")

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        golden_output = np.zeros(shape_out, dtype=gen_data_params.dtype)
        golden_gpu_output = np.zeros(shape_out, dtype=np.float32)

        lse_shape_out = (gen_data_params.num_heads, num_tokens)
        golden_lse_output = np.zeros(lse_shape_out, dtype=gen_data_params.dtype)
        golden_gpu_lse_output = np.zeros(lse_shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(query, key_cache, value_cache, block_tables,
            gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, mask, gen_data_params.mask_type, 
            gen_data_params.inner_prec, gen_data_params, sink_vector,pretoken, nexttoken)
        
        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            golden_output,
            golden_gpu_output,
            golden_lse_output,
            golden_gpu_lse_output
        )

        # check flash decoding condition
        max_q_seqlen_actual = max(gen_data_params.q_seqlen_list)
        min_q_seqlen_actual = min(gen_data_params.q_seqlen_list)
        min_kv_seqlen_actual = min(gen_data_params.k_seqlen_list)
        batch_size = len(gen_data_params.q_seqlen_list)
        is_paged_attention = (gen_data_params.kv_dtype == 1)

        num_tasks = batch_size * gen_data_params.kv_heads
        core_num = gen_data_params.core_num
        is_long_seq = (num_tasks <= 0.8 * core_num) and (min_kv_seqlen_actual >= core_num * 512)
        is_short_seq = (num_tasks <= 0.4 * core_num) and (min_kv_seqlen_actual >= 1024)

        golden_lse_output = np.transpose(golden_lse_output, (1, 0))
        golden_lse_output = np.expand_dims(golden_lse_output, axis=2)

        num_tokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_ntokens.bin"))
        num_kv_tokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_ntokens.bin"))
        query.tofile(os.path.join(WORKSPACE, "data", "q.bin"))
        print("kv cache : ", key_cache.shape)
        if (gen_data_params.cache_layout == "nd"):
            key_cache.tofile(os.path.join(WORKSPACE, "data", "k.bin"))
            value_cache.tofile(os.path.join(WORKSPACE, "data", "v.bin"))
        elif (gen_data_params.cache_layout == "nz"):
            if (gen_data_params.kv_dtype == 1):
                if (gen_data_params.isBnNBsD == 1):
                    key_cache_nz = key_cache.reshape(gen_data_params.num_blocks, gen_data_params.kv_heads, 
                                                gen_data_params.block_size, head_size_qk // 16, 16)
                    key_cache_nz = np.transpose(key_cache_nz, (0, 1, 3, 2, 4))
                    key_cache_nz.tofile(os.path.join(WORKSPACE, "data", "k_nz.bin"))
                    print("kv cache_nz : ", key_cache_nz.shape)
                    value_cache_nz = value_cache.reshape(gen_data_params.num_blocks, gen_data_params.kv_heads, 
                                                gen_data_params.block_size, head_size_vo // 16, 16)
                    value_cache_nz = np.transpose(value_cache_nz, (0, 1, 3, 2, 4))
                    value_cache_nz.tofile(os.path.join(WORKSPACE, "data", "v_nz.bin"))
                else:
                    assert False, "nz only support PAGE_BnNBsD or TND!!!"
            else:
                if (gen_data_params.layout_dtype == 1):
                    key_cache_nz = key_cache.reshape(num_kv_tokens, gen_data_params.kv_heads, head_size_qk // 16, 16)
                    key_cache_nz = np.transpose(key_cache_nz, (1, 2, 0, 3))
                    key_cache_nz.tofile(os.path.join(WORKSPACE, "data", "k_nz.bin"))
                    value_cache_nz = value_cache.reshape(num_kv_tokens, gen_data_params.kv_heads, head_size_vo // 16, 16)
                    value_cache_nz = np.transpose(value_cache_nz, (1, 2, 0, 3))
                    value_cache_nz.tofile(os.path.join(WORKSPACE, "data", "v_nz.bin"))
                else:
                    assert False, "nz only support PAGE_BnNBsD or TND!!!"
        np.array(block_tables).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "block_table.bin"))
        np.array([num_tokens, num_kv_tokens]).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "dataext.bin"))
        if gen_data_params.layout_dtype == 1:
            new_q_seqlen_list = [0]
            pre_seq_sum = 0
            for i in range(batch):
                pre_seq_sum += gen_data_params.q_seqlen_list[i]
                new_q_seqlen_list.append(pre_seq_sum)
            gen_data_params.q_seqlen_list = new_q_seqlen_list
            if gen_data_params.kv_dtype == 0:
                new_kv_seqlen_list = [0]
                pre_seq_sum = 0
                for i in range(batch):
                    pre_seq_sum += gen_data_params.k_seqlen_list[i]
                    new_kv_seqlen_list.append(pre_seq_sum)
                gen_data_params.k_seqlen_list = new_kv_seqlen_list
        np.array(gen_data_params.q_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "q_seqlen.bin"))
        np.array(gen_data_params.k_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "kv_seqlen.bin"))
        if gen_data_params.mask_type == 1 or  gen_data_params.mask_type == 3: 
            actual_input_mask_triu = np.triu(np.ones((2048, 2048)), 1).astype(np.int8)
            actual_input_mask_triu.tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
        elif gen_data_params.mask_type == 2:
            actual_input_mask_triu = np.triu(np.ones((128, 128)), 1).astype(np.int8)
            actual_input_mask_triu.tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
        elif gen_data_params.mask_type == 4:
            actual_input_mask_triu = self.generate_swa_full_mask(
                2 * global_window_size + local_window_size + 128 + 128,
                gen_data_params.local_window_size,
                gen_data_params.global_window_size
            )
            actual_input_mask_triu = actual_input_mask_triu.astype(np.int8)
            print(actual_input_mask_triu.shape)
            print(actual_input_mask_triu[1028][:256])
            actual_input_mask_triu.tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
        print("query_seqlen:", gen_data_params.q_seqlen_list)
        print("kv_seqlen:", gen_data_params.k_seqlen_list)
        golden_output.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden.bin"))
        golden_gpu_output.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu.bin"))
        golden_lse_output.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_lse.bin"))
        golden_gpu_lse_output.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden_gpu_lse.bin"))
        print("query",query.shape)
        # print("key_cache",key_cache[0][0][0][0])
        print("key_cache",key_cache.shape)
        print("value_cache",value_cache.shape)
        print("block_tables",block_tables)
        # import pdb;pdb.set_trace()
        # if sink_to_write is not None:
        sink_to_write.tofile(os.path.join(WORKSPACE, "data", "sink.bin"))
        sink_to_write_fp32.tofile(os.path.join(WORKSPACE, "data", "sink32.bin"))
 	     
    def gen_sink(self, sink_flag: int, num_heads: int, dtype: any):
        sink_min_range = 30
        sink_max_range = 50

        if sink_flag == 0:
            return None

        sink_fp32 = np.random.uniform(
            low=sink_min_range,
            high=sink_max_range,
            size=(num_heads, 1)
        ).astype(np.float32)
        sink_vector = sink_fp32.astype(bfloat16)

        return sink_vector


if __name__ == "__main__":
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    print("参数总数：", len(sys.argv) - 1)  # 减去脚本名本身
    print("所有参数：", sys.argv)

    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_head = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    qk_head_size = int(sys.argv[6])
    v_head_size = int(sys.argv[7])
    is_varied_len = int(sys.argv[8])
    mask_type = int(sys.argv[9])

    str_dtype = str(sys.argv[10])
    if str_dtype == "half":
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        logging("[ERROR] dtype must be half or bf16")
        sys.exit()

    kv_dtype = int(sys.argv[11])
    layout_dtype = int(sys.argv[12])
    num_blocks = int(sys.argv[13])
    inner_prec = int(sys.argv[14])
    lse_flag = int(sys.argv[15])
    block_size = int(sys.argv[16])
    sink_flag = int(sys.argv[17])
    pretoken = int(sys.argv[18])
    nexttoken = int(sys.argv[19])
    cache_layout = str(sys.argv[20])
    isBnNBsD = int(sys.argv[21])
    global_window_size = int(sys.argv[22])
    local_window_size  = int(sys.argv[23])
    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, batch)
    
    max_kv_seqlen = max(kv_seqlen_list)
    
    testObj = TestFlashAttentionInfer()
    gen_data_params = testObj.GenDataParams(q_seqlen_list, kv_seqlen_list, num_head,
                                            kv_heads, qk_head_size, v_head_size,
                                            num_blocks, block_size, mask_type, dtype, kv_dtype, layout_dtype, inner_prec, 
                                            q_seqlen, kv_seqlen, lse_flag, sink_flag, pretoken, nexttoken, cache_layout, isBnNBsD,
                                            global_window_size, local_window_size,  core_num=24)
    testObj.calc_data(gen_data_params)
