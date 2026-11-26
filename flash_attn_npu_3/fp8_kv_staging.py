"""Explicit compact BF16 staging for paged FP8 causal prefill on Ascend 950."""

from dataclasses import dataclass, field
from typing import Optional

import torch

from .flash_attn_npu_interface_950 import (
    flash_attn_npu_3_950,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)


_PAGE_SIZE = 128
_HEAD_DIM = 192
# The packed FA3 decode dispatcher supports up to 16 query rows per request.
_MAX_DECODE_Q = 16


@dataclass(frozen=True, eq=False)
class FP8KVStagingPlan:
    """Reuse a fixed query layout, KV lengths and page table across calls.

    Prepare the plan outside graph capture. On another stream, wait for
    preparation and keep the plan alive until the consuming stream finishes.
    K/V contents and descales can change on every call.
    Prefill allocates compact BF16 scratch by default. Reuse an explicitly
    allocated workspace on one stream, or order its uses across streams.
    Decode runs the original mixed FP8 path with zero staging workspace.
    """

    _query_shape: tuple[int, ...] = field(repr=False)
    _cache_shape: tuple[int, ...] = field(repr=False)
    _max_seqlen_q: int
    _window_size: tuple[int, int]
    _softcap: float
    _source_pages: Optional[torch.Tensor] = field(repr=False)
    _page_table: torch.Tensor = field(repr=False)
    _cache_seqlens: torch.Tensor = field(repr=False)
    _cu_seqlens_q: torch.Tensor = field(repr=False)
    _scheduler_metadata: torch.Tensor = field(repr=False)

    @property
    def num_staged_pages(self) -> int:
        """Number of distinct physical page pairs to convert."""
        return 0 if self._source_pages is None else self._source_pages.numel()

    @property
    def workspace_bytes(self) -> int:
        """Bytes of BF16 K and V scratch storage used by each forward."""
        if self._source_pages is None:
            return 0
        return 2 * max(self.num_staged_pages, 1) * _PAGE_SIZE * self._cache_shape[2] * _HEAD_DIM * 2

    def allocate_workspace(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate reusable compact BF16 scratch for prefill. Decode returns None."""
        if self._source_pages is None:
            return None
        # The attention ABI requires one physical page even if all KV lengths are zero.
        shape = (max(self.num_staged_pages, 1), self._cache_shape[2], _PAGE_SIZE, _HEAD_DIM)
        k_bf16 = torch.empty(shape, dtype=torch.bfloat16, device=self._source_pages.device).transpose(1, 2)
        return k_bf16, torch.empty_like(k_bf16)

    def _stage(self, k_cache, v_cache, k_descale, v_descale, workspace=None):
        k_bf16, v_bf16 = self.allocate_workspace() if workspace is None else workspace
        flash_attn_npu_3_950._stage_fp8_kv(
            k_cache, v_cache, k_descale, v_descale, self._source_pages, k_bf16, v_bf16,
        )
        return k_bf16, v_bf16

    @torch.no_grad()
    def forward(
        self, q, k_cache, v_cache, *, k_descale, v_descale,
        softmax_scale=None, return_softmax_lse=False, workspace=None,
    ):
        """Run BF16 prefill with staged K/V or dispatch mixed FP8 decode."""
        if q.shape != self._query_shape or q.dtype != torch.bfloat16 or q.device != self._page_table.device:
            raise ValueError("Q must match the BF16 TND shape and NPU device of the staging plan")
        if k_cache.shape != self._cache_shape:
            raise ValueError("K/V cache shape must match the staging plan")
        with torch.npu.device(q.device):
            if self._source_pages is not None:
                k_cache, v_cache = self._stage(k_cache, v_cache, k_descale, v_descale, workspace)
                k_descale = v_descale = None
            return flash_attn_with_kvcache(
                q, k_cache, v_cache, k_descale=k_descale, v_descale=v_descale,
                cache_seqlens=self._cache_seqlens,
                page_table=self._page_table, cu_seqlens_q=self._cu_seqlens_q,
                max_seqlen_q=self._max_seqlen_q, scheduler_metadata=self._scheduler_metadata,
                causal=True, window_size=self._window_size, softcap=self._softcap,
                softmax_scale=softmax_scale, return_softmax_lse=return_softmax_lse,
            )


def _compact_page_table(page_table, q_lengths, kv_lengths, left, num_blocks):
    # Keep logical indices and omit columns that follow every request's KV end.
    live_width = max(1, (max(kv_lengths) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    table = page_table[:, :live_width].cpu()
    remapped = torch.zeros((len(q_lengths), live_width), dtype=torch.int32, device="cpu")
    physical_to_slot: dict[int, int] = {}
    for seq, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        if not q_len or not kv_len:
            continue
        # Bottom-right alignment puts Q row zero at KV position kv_len-q_len.
        # The latest Q row ends at kv_len-1, so all intervening pages can be used.
        first = 0 if left < 0 else max(0, kv_len - q_len - left) // _PAGE_SIZE
        last = (kv_len + _PAGE_SIZE - 1) // _PAGE_SIZE
        slots = []
        for page in table[seq, first:last].tolist():
            if not 0 <= page < num_blocks:
                raise ValueError(f"Visible physical page {page} in request {seq} is out of range")
            slots.append(physical_to_slot.setdefault(page, len(physical_to_slot)))
        remapped[seq, first:last] = torch.tensor(slots, dtype=torch.int32, device="cpu")
    return torch.tensor(list(physical_to_slot), dtype=torch.int32, device="cpu"), remapped


def prepare_fp8_kv_staging(
    q, k_cache, *, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
    window_size=(511, 0), softcap=0.0,
) -> FP8KVStagingPlan:
    """Plan causal TND attention for BF16 Q and paged E4M3FN KV (D192, page 128).

    The window is (left, 0), with -1 for an unlimited left window. Page IDs
    may be out of order and shared by several logical pages or requests.
    A batch with at most 16 actual query rows per request uses the original
    mixed FP8 path and the original page indices. Its page table follows the
    same contract as `flash_attn_with_kvcache`.
    Snapshot the int32 lengths and table during this call. Prepare again when
    lengths, page assignments, or the geometry change. Setup reads metadata
    back to the host. Use `plan.forward(..., k_descale=..., v_descale=...)`
    with contiguous FP32 scales of shape [physical_pages, 128, kv_heads].
    """
    left, right = window_size
    if not isinstance(left, int) or not -1 <= left < 2**31 or right != 0:
        raise ValueError("FP8 KV staging requires a causal (left, 0) window with left >= -1")
    if (q.device.type != "npu" or q.dtype != torch.bfloat16 or q.ndim != 3
            or not q.shape[0] or q.shape[-1] != _HEAD_DIM):
        raise ValueError("FP8 KV staging requires NPU BF16 Q [total_q, q_heads, 192]")
    if (k_cache.device != q.device or k_cache.dtype != torch.float8_e4m3fn or k_cache.ndim != 4
            or k_cache.shape[1] != _PAGE_SIZE or k_cache.shape[3] != _HEAD_DIM):
        raise ValueError("FP8 KV staging requires NPU E4M3FN K/V [physical_pages, 128, kv_heads, 192]")
    num_blocks, _, kv_heads, _ = k_cache.shape
    if not 0 < num_blocks < 2**31 or not kv_heads or not q.shape[1] or q.shape[1] % kv_heads:
        raise ValueError("K/V must have pages and a positive head count dividing the Q head count")
    for tensor in (page_table, cache_seqlens, cu_seqlens_q):
        if tensor.device != q.device or tensor.dtype != torch.int32:
            raise ValueError("Page table and sequence metadata must be int32 on the query device")
    if page_table.ndim != 2 or not all(page_table.shape):
        raise ValueError("Page table must be a nonempty [batch, max_logical_pages] matrix")
    batch, width = page_table.shape
    if cache_seqlens.shape != (batch,) or cu_seqlens_q.shape != (batch + 1,):
        raise ValueError("KV lengths and Q cumulative lengths must have shapes [batch] and [batch+1]")
    if not isinstance(max_seqlen_q, int) or not 0 < max_seqlen_q < 2**31:
        raise ValueError("max_seqlen_q must be a positive int32")

    lengths = cache_seqlens.cpu().contiguous()
    prefixes = cu_seqlens_q.cpu().contiguous()
    q_offsets = prefixes.tolist()
    if q_offsets[0] != 0 or q_offsets[-1] != q.shape[0]:
        raise ValueError("Q cumulative lengths must start at zero and end at total_q")
    capacity = width * _PAGE_SIZE
    kv_lengths = lengths.tolist()
    if min(kv_lengths) < 0 or max(kv_lengths) > capacity:
        raise ValueError("A KV length is outside the supplied page table")
    q_lengths = [end - start for start, end in zip(q_offsets, q_offsets[1:])]
    max_query_length = max(q_lengths)
    if min(q_lengths) < 0 or max_query_length > max_seqlen_q:
        raise ValueError("A Q length is outside the supplied bounds")
    is_decode = max_query_length <= _MAX_DECODE_Q

    with torch.npu.device(q.device):
        if is_decode:
            source_pages = None
            table = page_table.clone(memory_format=torch.contiguous_format)
        else:
            pages_cpu, table_cpu = _compact_page_table(page_table, q_lengths, kv_lengths, left, num_blocks)
            source_pages = pages_cpu.to(q.device)
            table = table_cpu.to(q.device)
        lengths = lengths.to(q.device)
        prefixes = prefixes.to(q.device)
        metadata = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=max_seqlen_q, max_seqlen_k=table.shape[1] * _PAGE_SIZE,
            num_heads_q=q.shape[1], num_heads_kv=kv_heads, headdim=_HEAD_DIM,
            cache_seqlens=lengths, cu_seqlens_q=prefixes, page_size=_PAGE_SIZE,
            kv_dtype=k_cache.dtype if is_decode else q.dtype,
            causal=True, window_size=(left, 0), has_softcap=softcap > 0,
        )
    return FP8KVStagingPlan(
        tuple(q.shape), tuple(k_cache.shape), max_seqlen_q, (left, 0), softcap,
        source_pages, table, lengths, prefixes, metadata,
    )
