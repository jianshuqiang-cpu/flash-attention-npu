"""Host-side Ascend debug FIFO transport (Python port of RuntimeWrapper AscDebugFifo).

Allocates the CANN print FIFO, swaps workspace sentinels into launch args,
then D2H-decodes scalar / tensor records to stdout with the same text formats
e2e tests capture (``TLA printf:`` / ``DumpTensor:``).
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass
from typing import Callable

from ...execution import TlaRuntimeUnavailableError
from .ascend import check_acl_errors

DEBUG_PRINT_WORKSPACE_SENTINEL = int.from_bytes(b"TLA_PRNT", "big")
PRINT_TENSOR_WORKSPACE_SENTINEL = int.from_bytes(b"TLA_TPRN", "big")

_DEBUG_CORE_RECORDS = 108
_RING_BUFFER_BYTES = 1 << 20
_MAGIC = 0xAE86
_PRINT_TENSOR_DESCRIPTOR_NAMESPACE = 0x54500000
_PRINT_TENSOR_DESCRIPTOR_NAMESPACE_MASK = 0xFFFC0000
_GLOBAL_MEMORY_POSITION = 0
_UNIFIED_BUFFER_POSITION = 1

_FIFO_SCALAR = 1
_FIFO_TENSOR = 2
_FIFO_SHAPE = 3
_FIFO_BUF_IN = 8
_FIFO_BUF_OUT = 9

_HEAD_SIZE = 56
_READ_SIZE = 24
_WRITE_SIZE = 24
_PRINT_TLV_SIZE = 24
_PRINT_TENSOR_TLV_SIZE = 72
_PRINT_SHAPE_TLV_SIZE = 48
_TENSOR_PAYLOAD_ALIGNMENT = 32
# FIFO payload capacity leaves at most 262112 f32 values (see print_tensor.cpp).
_MAX_TENSOR_ELEMENTS = 262112
_PRINT_SLOT_BYTES = 8
_PRINT_FMT_OFFSET_BASE = 16

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _ring_buffer_offset() -> int:
    payload_offset = _HEAD_SIZE + _READ_SIZE + _PRINT_TENSOR_TLV_SIZE
    return _align_up(payload_offset, 32) - _PRINT_TENSOR_TLV_SIZE


@dataclass
class _FifoData:
    device_ptr: int
    region_size: int
    record_count: int
    launch_block_count: int
    mixed_handoff: bool
    block_length: int
    ring_buffer_offset: int
    ring_buffer_bytes: int


class AscDebugFifoError(TlaRuntimeUnavailableError):
    """Raised when FIFO open/close/decode fails."""


def _acl_malloc(size: int) -> int:
    import acl

    ptr, ret = acl.rt.malloc(int(size), _ACL_MEM_MALLOC_HUGE_FIRST)
    check_acl_errors(ret, "acl.rt.malloc", error_cls=AscDebugFifoError)
    return int(ptr)


def _acl_malloc_host(size: int) -> int:
    import acl

    ptr, ret = acl.rt.malloc_host(int(size))
    check_acl_errors(ret, "acl.rt.malloc_host", error_cls=AscDebugFifoError)
    return int(ptr)


def _acl_free(ptr: int) -> None:
    import acl

    if ptr:
        check_acl_errors(
            acl.rt.free(int(ptr)), "acl.rt.free", error_cls=AscDebugFifoError
        )


def _acl_free_host(ptr: int) -> None:
    import acl

    if ptr:
        check_acl_errors(
            acl.rt.free_host(int(ptr)), "acl.rt.free_host", error_cls=AscDebugFifoError
        )


def _acl_memcpy(dst: int, src: int, size: int, kind: int) -> None:
    import acl

    check_acl_errors(
        acl.rt.memcpy(int(dst), int(size), int(src), int(size), int(kind)),
        "acl.rt.memcpy",
        error_cls=AscDebugFifoError,
    )


def open_fifo(block_num: int, *, mixed_handoff: bool = False) -> _FifoData:
    ring_offset = _ring_buffer_offset()
    block_length = _align_up(ring_offset + _RING_BUFFER_BYTES + _WRITE_SIZE, 64)
    region_size = block_length * _DEBUG_CORE_RECORDS
    device_ptr = _acl_malloc(region_size)
    host_ptr = 0
    try:
        host_ptr = _acl_malloc_host(region_size)
        ctypes.memset(host_ptr, 0, region_size)
        for i in range(_DEBUG_CORE_RECORDS):
            record = host_ptr + i * block_length
            buf = (ctypes.c_char * block_length).from_address(record)
            head = struct.pack(
                "<IIIIHHIQQIIII",
                block_length,
                i,
                _DEBUG_CORE_RECORDS,
                _RING_BUFFER_BYTES,
                _MAGIC,
                0,
                0,
                device_ptr + i * block_length + ring_offset,
                0,
                0,
                0,
                0,
                0,
            )
            if len(head) != _HEAD_SIZE:
                raise AscDebugFifoError("internal FIFO head size mismatch")
            buf[0:_HEAD_SIZE] = head
            read = struct.pack("<IIQQ", _FIFO_BUF_OUT, 16, 0, 0)
            if len(read) != _READ_SIZE:
                raise AscDebugFifoError("internal FIFO read-info size mismatch")
            buf[_HEAD_SIZE : _HEAD_SIZE + _READ_SIZE] = read
            write_off = ring_offset + _RING_BUFFER_BYTES
            write = struct.pack("<IIQQ", _FIFO_BUF_IN, 16, 0, 0)
            if len(write) != _WRITE_SIZE:
                raise AscDebugFifoError("internal FIFO write-info size mismatch")
            buf[write_off : write_off + _WRITE_SIZE] = write
        _acl_memcpy(device_ptr, host_ptr, region_size, _ACL_MEMCPY_HOST_TO_DEVICE)
    except Exception:
        if host_ptr:
            try:
                _acl_free_host(host_ptr)
            except Exception:
                pass
        try:
            _acl_free(device_ptr)
        except Exception:
            pass
        raise
    _acl_free_host(host_ptr)
    return _FifoData(
        device_ptr=device_ptr,
        region_size=region_size,
        record_count=_DEBUG_CORE_RECORDS,
        launch_block_count=int(block_num),
        mixed_handoff=bool(mixed_handoff),
        block_length=block_length,
        ring_buffer_offset=ring_offset,
        ring_buffer_bytes=_RING_BUFFER_BYTES,
    )


def destroy_fifo(fifo: _FifoData | None) -> None:
    if fifo is None:
        return
    try:
        if fifo.device_ptr:
            _acl_free(fifo.device_ptr)
    finally:
        fifo.device_ptr = 0


def _args_to_u64_list(args: bytes) -> list[int]:
    if len(args) % 8 != 0:
        raise AscDebugFifoError(
            "debug workspace kernel arguments must be a multiple of 8 bytes"
        )
    return list(struct.unpack("<" + "Q" * (len(args) // 8), args))


def _u64_list_to_args(values: list[int]) -> bytes:
    return struct.pack("<" + "Q" * len(values), *values) if values else b""


def prepare_launch_args(
    args: bytes,
    *,
    expects_debug_fifo: bool,
    expects_print_tensor: bool | int,
    fifo_device_ptr: int,
) -> bytes:
    values = _args_to_u64_list(args)
    if expects_debug_fifo and expects_print_tensor:
        raise AscDebugFifoError(
            "scalar debug FIFO and native tensor print cannot share a launch"
        )
    if expects_debug_fifo:
        if not values or values[-1] != DEBUG_PRINT_WORKSPACE_SENTINEL:
            raise AscDebugFifoError(
                "debug print FIFO marker must occupy the final packed kernel argument"
            )
        values[-1] = int(fifo_device_ptr)
        return _u64_list_to_args(values)
    if expects_print_tensor:
        if not values or values[-1] != PRINT_TENSOR_WORKSPACE_SENTINEL:
            raise AscDebugFifoError(
                "tensor print FIFO marker must occupy the final packed kernel argument"
            )
        workspace = int(fifo_device_ptr)
        if int(expects_print_tensor) == 2:
            values[-1] = workspace
        else:
            values.pop()
            values.insert(0, workspace)
        return _u64_list_to_args(values)
    return args


def _write_stdout(text: str) -> None:
    os.write(1, text.encode("utf-8", errors="replace"))


def _decode_float16(bits: int) -> float:
    sign = (bits & 0x8000) << 16
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x03FF
    if exponent == 0:
        if mantissa == 0:
            result = sign
        else:
            exponent = 113
            while (mantissa & 0x0400) == 0:
                mantissa <<= 1
                exponent -= 1
            mantissa &= 0x03FF
            result = sign | (exponent << 23) | (mantissa << 13)
    elif exponent == 0x1F:
        result = sign | 0x7F800000 | (mantissa << 13)
    else:
        result = sign | ((exponent + 112) << 23) | (mantissa << 13)
    return struct.unpack("<f", struct.pack("<I", result & 0xFFFFFFFF))[0]


def _print_tensor_dtype(code: int) -> tuple[str, int] | None:
    return {
        0: ("float32", 4),
        1: ("float16", 2),
        2: ("int8", 1),
        3: ("int32", 4),
        4: ("uint8", 1),
        6: ("int16", 2),
        7: ("uint16", 2),
        8: ("uint32", 4),
    }.get(code)


def _decode_subblock(descriptor: int) -> int:
    if (descriptor & _PRINT_TENSOR_DESCRIPTOR_NAMESPACE_MASK) != (
        _PRINT_TENSOR_DESCRIPTOR_NAMESPACE
    ):
        return -2
    tag = (descriptor >> 16) & 0x3
    if tag == 0:
        return -1
    if tag in (1, 2):
        return int(tag - 1)
    return -2


def _is_supported_scalar_printf_format(fmt: bytes) -> bool:
    """Match RuntimeWrapper: literals plus ``%d`` / ``%u`` / ``%f`` / ``%%``."""
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != ord("%"):
            i += 1
            continue
        if i + 1 >= n:
            return False
        spec = fmt[i + 1]
        if spec not in (ord("%"), ord("d"), ord("u"), ord("f")):
            return False
        i += 2
    return True


def _format_scalar_body(fmt: bytes, args: bytes) -> str | None:
    """Format a scalar printf TLV body.

    Returns the body string on success, ``None`` if argument slots are
    malformed, or ``""`` if the format string is unsupported.
    """
    if not _is_supported_scalar_printf_format(fmt):
        return ""
    parts: list[str] = []
    arg_offset = 0
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != ord("%"):
            parts.append(chr(fmt[i]))
            i += 1
            continue
        spec = fmt[i + 1]
        i += 2
        if spec == ord("%"):
            parts.append("%")
            continue
        if arg_offset + _PRINT_SLOT_BYTES > len(args):
            return None
        if spec == ord("d"):
            (slot,) = struct.unpack_from("<Q", args, arg_offset)
            parts.append(str(ctypes.c_int32(slot & 0xFFFFFFFF).value))
        elif spec == ord("u"):
            (slot,) = struct.unpack_from("<Q", args, arg_offset)
            parts.append(str(ctypes.c_uint32(slot & 0xFFFFFFFF).value))
        elif spec == ord("f"):
            (value,) = struct.unpack_from("<f", args, arg_offset)
            parts.append(f"{float(value):f}")
        else:
            return ""
        arg_offset += _PRINT_SLOT_BYTES
    if arg_offset != len(args):
        return None
    return "".join(parts)


def _print_scalar_tlv(record: memoryview, total: int, core: int) -> bool:
    if total < _PRINT_TLV_SIZE:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV (short record)\n"
        )
        return True
    _typ, _length, block_idx, _reserved, fmt_offset = struct.unpack_from(
        "<IIIIQ", record, 0
    )
    if fmt_offset < _PRINT_SLOT_BYTES or (fmt_offset % _PRINT_SLOT_BYTES) != 0:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV (bad fmtOffset)\n"
        )
        return True
    fmt_start = _PRINT_FMT_OFFSET_BASE + fmt_offset
    if fmt_start >= total:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV "
            f"(fmtOffset out of bounds)\n"
        )
        return True
    args_start = _PRINT_FMT_OFFSET_BASE + _PRINT_SLOT_BYTES
    if fmt_start < args_start or ((fmt_start - args_start) % _PRINT_SLOT_BYTES) != 0:
        _write_stdout(
            f"TLA printf: core={core} malformed scalar printf TLV (argument layout)\n"
        )
        return True
    fmt_bytes = bytes(record[fmt_start:total]).split(b"\0", 1)[0]
    if not _is_supported_scalar_printf_format(fmt_bytes):
        return False
    body = _format_scalar_body(fmt_bytes, bytes(record[args_start:fmt_start]))
    if body is None:
        body = "<malformed scalar printf TLV: missing argument slot>"
    elif body == "":
        return False
    _write_stdout(f"TLA printf: core={core} block={block_idx} {body}\n")
    return True


def _print_scalar_records(host: memoryview, fifo: _FifoData) -> None:
    printed = False
    for i in range(fifo.record_count):
        record = host[i * fifo.block_length : (i + 1) * fifo.block_length]
        magic = struct.unpack_from("<H", record, 16)[0]
        if magic != _MAGIC:
            continue
        ring = record[fifo.ring_buffer_offset :]
        buf_offset = struct.unpack_from("<Q", ring, fifo.ring_buffer_bytes + 8)[0]
        written = min(int(buf_offset), fifo.ring_buffer_bytes)
        offset = 0
        while offset + 8 <= written:
            typ, length = struct.unpack_from("<II", ring, offset)
            total = 8 + length
            if length == 0 or total > written - offset:
                if typ == _FIFO_SCALAR:
                    _write_stdout(
                        f"TLA printf: core={i} malformed scalar printf TLV "
                        f"(length out of bounds)\n"
                    )
                    printed = True
                break
            if typ == _FIFO_SCALAR:
                printed = (
                    _print_scalar_tlv(ring[offset : offset + total], total, i)
                    or printed
                )
            offset += total
    if not printed:
        _write_stdout("TLA debug: no records captured\n")


def _format_tensor_value(data_type: int, payload: bytes, index: int) -> str:
    dtype = _print_tensor_dtype(data_type)
    assert dtype is not None
    _name, width = dtype
    off = index * width
    chunk = payload[off : off + width]
    if data_type == 0:
        return f"{struct.unpack_from('<f', chunk, 0)[0]:.9g}"
    if data_type == 1:
        bits = struct.unpack_from("<H", chunk, 0)[0]
        return f"{_decode_float16(bits):.9g}"
    if data_type == 2:
        return str(struct.unpack_from("<b", chunk, 0)[0])
    if data_type == 3:
        return str(struct.unpack_from("<i", chunk, 0)[0])
    if data_type == 4:
        return str(struct.unpack_from("<B", chunk, 0)[0])
    if data_type == 6:
        return str(struct.unpack_from("<h", chunk, 0)[0])
    if data_type == 7:
        return str(struct.unpack_from("<H", chunk, 0)[0])
    if data_type == 8:
        return str(struct.unpack_from("<I", chunk, 0)[0])
    raise AscDebugFifoError(f"unsupported tensor dtype code {data_type}")


def _render_tensor(tlv: memoryview, shape_tlv: memoryview, logical_block: int) -> None:
    data_type = struct.unpack_from("<I", tlv, 12)[0]
    desc = struct.unpack_from("<I", tlv, 16)[0]
    position = struct.unpack_from("<H", tlv, 24)[0]
    dump_size = struct.unpack_from("<I", tlv, 68)[0]
    shape_dim = struct.unpack_from("<I", shape_tlv, 8)[0]
    shape = struct.unpack_from("<8I", shape_tlv, 12)
    dtype = _print_tensor_dtype(data_type)
    if dtype is None:
        return
    name, width = dtype
    count = dump_size // width
    payload = bytes(tlv[_PRINT_TENSOR_TLV_SIZE : _PRINT_TENSOR_TLV_SIZE + dump_size])
    pos_name = "GM" if position == _GLOBAL_MEMORY_POSITION else "UB"
    parts = [f"DumpTensor: call={desc & 0xFFFF}, block={logical_block}, "]
    subblock = _decode_subblock(desc)
    if subblock >= 0:
        parts.append(f"subblock={subblock}, ")
    shape_text = ",".join(str(shape[i]) for i in range(shape_dim))
    parts.append(f"data_type={name}, position={pos_name}, shape=[{shape_text}] ")
    parts.append(f"dump_size={count} [")
    parts.append(
        ", ".join(_format_tensor_value(data_type, payload, i) for i in range(count))
    )
    parts.append("]\n")
    _write_stdout("".join(parts))


def _validate_tensor_tlv(tlv: memoryview, total: int, shape_tlv: memoryview) -> None:
    """Mirror RuntimeWrapper.cpp ``validate_tensor_tlv`` checks."""
    if total < _PRINT_TENSOR_TLV_SIZE:
        raise AscDebugFifoError("malformed tensor print FIFO: truncated tensor header")
    length = struct.unpack_from("<I", tlv, 4)[0]
    if length != total - 8:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor length does not match record"
        )
    data_type = struct.unpack_from("<I", tlv, 12)[0]
    desc = struct.unpack_from("<I", tlv, 16)[0]
    position = struct.unpack_from("<H", tlv, 24)[0]
    dim = struct.unpack_from("<I", tlv, 28)[0]
    tensor_shape = struct.unpack_from("<8I", tlv, 32)
    dump_size = struct.unpack_from("<I", tlv, 68)[0]
    dtype = _print_tensor_dtype(data_type)
    if dtype is None:
        raise AscDebugFifoError("malformed tensor print FIFO: unsupported tensor dtype")
    _name, width = dtype
    if position not in (_GLOBAL_MEMORY_POSITION, _UNIFIED_BUFFER_POSITION):
        raise AscDebugFifoError(
            "malformed tensor print FIFO: unsupported tensor position"
        )
    if dim != 0:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor dimension must be zero"
        )
    if _decode_subblock(desc) < -1:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: invalid tensor descriptor namespace"
        )
    if (
        dump_size == 0
        or dump_size % width != 0
        or dump_size > _MAX_TENSOR_ELEMENTS * width
    ):
        raise AscDebugFifoError("malformed tensor print FIFO: invalid tensor dump size")
    expected_total = _PRINT_TENSOR_TLV_SIZE + _align_up(
        dump_size, _TENSOR_PAYLOAD_ALIGNMENT
    )
    if total != expected_total:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor payload size does not match record"
        )
    if any(extent != 0 for extent in tensor_shape):
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor shape metadata must be zero"
        )
    if len(shape_tlv) != _PRINT_SHAPE_TLV_SIZE:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: missing or invalid tensor shape record"
        )
    shape_type, shape_length, shape_dim = struct.unpack_from("<III", shape_tlv, 0)
    shape_extents = struct.unpack_from("<8I", shape_tlv, 12)
    shape_reserved = struct.unpack_from("<I", shape_tlv, 44)[0]
    if (
        shape_type != _FIFO_SHAPE
        or shape_length != _PRINT_SHAPE_TLV_SIZE - 8
        or shape_reserved != 0
        or shape_dim not in (1, 2)
    ):
        raise AscDebugFifoError(
            "malformed tensor print FIFO: missing or invalid tensor shape record"
        )
    shape_elements = 1
    for index, extent in enumerate(shape_extents):
        if index < shape_dim:
            if extent == 0:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: tensor shape contains a zero extent"
                )
            shape_elements *= extent
        elif extent != 0:
            raise AscDebugFifoError(
                "malformed tensor print FIFO: tensor shape contains trailing metadata"
            )
    if shape_elements < dump_size // width:
        raise AscDebugFifoError(
            "malformed tensor print FIFO: tensor shape is smaller than the dump size"
        )


def _decode_tensor_records(host: memoryview, fifo: _FifoData) -> None:
    records: list[tuple[memoryview, memoryview, int]] = []
    for i in range(fifo.record_count):
        record = host[i * fifo.block_length : (i + 1) * fifo.block_length]
        magic = struct.unpack_from("<H", record, 16)[0]
        if magic != _MAGIC:
            raise AscDebugFifoError("malformed tensor print FIFO: invalid record magic")
        ring = record[fifo.ring_buffer_offset :]
        write_type, write_len, buf_offset = struct.unpack_from(
            "<IIQ", ring, fifo.ring_buffer_bytes
        )
        if write_type != _FIFO_BUF_IN or write_len != 16:
            raise AscDebugFifoError(
                "malformed tensor print FIFO: invalid write-control record"
            )
        if buf_offset > fifo.ring_buffer_bytes:
            raise AscDebugFifoError(
                "malformed tensor print FIFO: ring write offset exceeds 1 MiB capacity"
            )
        offset = 0
        written = int(buf_offset)
        pending_shape: memoryview | None = None
        while offset < written:
            if offset + 8 > written:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: truncated TLV header"
                )
            typ, length = struct.unpack_from("<II", ring, offset)
            total = 8 + length
            if length == 0 or offset + total > written:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: TLV length exceeds captured bytes"
                )
            chunk = ring[offset : offset + total]
            if typ == _FIFO_SHAPE:
                if pending_shape is not None:
                    raise AscDebugFifoError(
                        "malformed tensor print FIFO: shape record is missing its tensor record"
                    )
                if total != _PRINT_SHAPE_TLV_SIZE:
                    raise AscDebugFifoError(
                        "malformed tensor print FIFO: invalid shape record size"
                    )
                pending_shape = chunk
                offset += total
                continue
            if typ != _FIFO_TENSOR:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: unexpected record type"
                )
            if pending_shape is None:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: tensor record missing shape"
                )
            _validate_tensor_tlv(chunk, total, pending_shape)
            desc = struct.unpack_from("<I", chunk, 16)[0]
            block_idx = struct.unpack_from("<H", chunk, 26)[0]
            subblock = _decode_subblock(desc)
            logical = int(block_idx)
            if fifo.mixed_handoff and subblock >= 0:
                if block_idx % 2 != subblock:
                    raise AscDebugFifoError(
                        "malformed tensor print FIFO: tensor block index does not match mixed AIV subblock"
                    )
                logical = block_idx // 2
            if logical >= fifo.launch_block_count:
                raise AscDebugFifoError(
                    "malformed tensor print FIFO: tensor block index exceeds launch block_num"
                )
            records.append((pending_shape, chunk, logical))
            pending_shape = None
            offset += total
        if pending_shape is not None:
            raise AscDebugFifoError(
                "malformed tensor print FIFO: shape record is missing its tensor record"
            )
    if not records:
        _write_stdout("TLA debug: no records captured\n")
        return
    for shape_tlv, tlv, logical in records:
        _render_tensor(tlv, shape_tlv, logical)


def close_fifo(
    fifo: _FifoData,
    stream: int,
    *,
    tensor_only: bool,
) -> None:
    import acl

    host_ptr = 0
    try:
        check_acl_errors(
            acl.rt.synchronize_stream(int(stream)),
            "acl.rt.synchronize_stream(AscDebugFifo)",
            error_cls=AscDebugFifoError,
        )
        host_ptr = _acl_malloc_host(fifo.region_size)
        _acl_memcpy(
            host_ptr,
            fifo.device_ptr,
            fifo.region_size,
            _ACL_MEMCPY_DEVICE_TO_HOST,
        )
        host = (ctypes.c_char * fifo.region_size).from_address(host_ptr)
        view = memoryview(host).cast("B")
        if tensor_only:
            _decode_tensor_records(view, fifo)
        else:
            _print_scalar_records(view, fifo)
    finally:
        if host_ptr:
            try:
                _acl_free_host(host_ptr)
            except Exception:
                pass
        destroy_fifo(fifo)


def launch_with_debug_fifo(
    *,
    launch_kernel: Callable[[bytes], None],
    args: bytes,
    block_num: int,
    stream: int,
    expects_debug_fifo: bool,
    expects_print_tensor: bool | int,
) -> None:
    """Open FIFO if needed, rewrite args, invoke ``launch_kernel``, then close."""
    needs_fifo = bool(expects_debug_fifo) or bool(expects_print_tensor)
    if not needs_fifo:
        launch_kernel(args)
        return

    mixed = int(expects_print_tensor) == 2
    fifo = open_fifo(block_num, mixed_handoff=mixed)
    try:
        rewritten = prepare_launch_args(
            args,
            expects_debug_fifo=bool(expects_debug_fifo),
            expects_print_tensor=expects_print_tensor,
            fifo_device_ptr=fifo.device_ptr,
        )
        try:
            launch_kernel(rewritten)
        except Exception:
            destroy_fifo(fifo)
            raise
        close_fifo(
            fifo,
            stream,
            tensor_only=bool(expects_print_tensor),
        )
    except Exception:
        if fifo.device_ptr:
            try:
                destroy_fifo(fifo)
            except Exception:
                pass
        raise


__all__ = [
    "AscDebugFifoError",
    "DEBUG_PRINT_WORKSPACE_SENTINEL",
    "PRINT_TENSOR_WORKSPACE_SENTINEL",
    "launch_with_debug_fifo",
    "prepare_launch_args",
]
