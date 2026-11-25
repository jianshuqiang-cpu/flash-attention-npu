"""Compilation and execution support for Tla DSL kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .base_dsl.arch import (
    DEFAULT_NPU_ARCH,
    TlaKernelTarget,
    arch_scope_for_target as _arch_scope_for_target_impl,
    get_kernel_target as _get_kernel_target,
    parse_arch_scope as _parse_arch_scope_impl,
    resolve_npu_arch as _resolve_npu_arch,
)
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.compiler import _parse_compile_options_from_str
from .base_dsl.typing import Numeric
from .compiler_bridge import (
    BridgeLoweringError,
    BridgeUnavailableError,
    KernelAbiArgumentKind,
    KernelAbiIntegerSignedness,
    KernelAbiLayout,
    KernelAbiScalarCategory,
    KernelAbiScalarDescriptor,
    kernel_abi_from_dict,
    kernel_abi_to_dict,
    lower_tlair_module_to_mlir,
    resolve_bridge_extension_path,
)
from .types import dtype_size_bytes

# CATLASS_DSL_KEEP tokens: ir / ir-debug / kernel.
_KEEP_ALL_TOKENS: frozenset[str] = frozenset({"ir", "ir-debug", "kernel"})
_POINTER_ABI_SIZE = 8
_DEBUG_PRINT_WORKSPACE_SENTINEL_TEXT = b"TLA_PRNT"
_DEBUG_PRINT_WORKSPACE_SENTINEL = int.from_bytes(
    _DEBUG_PRINT_WORKSPACE_SENTINEL_TEXT, byteorder="big"
)
_PRINT_TENSOR_WORKSPACE_SENTINEL_TEXT = b"TLA_TPRN"
_PRINT_TENSOR_WORKSPACE_SENTINEL = int.from_bytes(
    _PRINT_TENSOR_WORKSPACE_SENTINEL_TEXT, byteorder="big"
)
# Cache compatibility tokens; change only when the corresponding ABI changes.
_DEBUG_PRINT_WORKSPACE_ABI_REVISION = "debug-print-workspace-i64-v1"
_PRINT_TENSOR_WORKSPACE_ABI_REVISION = (
    "print-tensor-workspace-i64-dynamic-shape-typed-multirecord-mixed-subblock"
)
_PRINT_TENSOR_HELPER_ABI_MARKER = b"__tla_print_tensor_abi"
_PRINT_TENSOR_MAX_BLOCKS = 1 << 16  # uint16_t block_idx
_PRINT_TENSOR_CORE_RECORDS = 108  # RuntimeWrapper::kDebugCoreRecords
_PRINT_TENSOR_FIFO_BYTES = 1 << 20  # kRingBufferBytes
_PRINT_TENSOR_SHAPE_HEADER_BYTES = 48  # sizeof(PrintShapeTlv)
_PRINT_TENSOR_HEADER_BYTES = 72  # sizeof(PrintTensorTlv)
_PRINT_TENSOR_PAYLOAD_ALIGNMENT = 32  # tensor payload alignment
_PRINT_TENSOR_MAX_F32_ELEMENTS = 262_112
_NATIVE_PRINT_TENSOR_RECORD_RE = re.compile(
    r"^DumpTensor: call=(?P<call>\d+), block=(?P<block>\d+), "
    r"(?:subblock=(?P<subblock>[01]), )?"
    r"data_type=(?P<dtype>[A-Za-z0-9_]+), "
    r"position=(?P<position>[A-Za-z0-9_]+), "
    r"shape=\[(?P<shape>[\d,\s]+)\] dump_size=(?P<count>\d+) "
    r"\[(?P<values>[^\]]*)\]$"
)
_FLOAT_LITERAL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)$",
    re.IGNORECASE,
)
_INTEGER_LITERAL_RE = re.compile(r"^[+-]?\d+$")
_HIVM_TEMPLATE_BITCODE_ATTRS = {
    "meta_op.aic.c310.bc": ("hivm.aic_bitcode", "hivm.aic_bitcode"),
    "meta_op.aiv.c310.bc": ("hivm.aiv_bitcode", "hivm.aiv_bitcode"),
}
_MAX_KERNEL_ABI_PAYLOAD_SIZE = 1 << 20
_ONLINE_CACHE_ABI_VERSION = 5


class TlaExecutionError(RuntimeError):
    """Base class for execution and toolchain failures."""


class TlaCompilerBridgeUnavailableError(TlaExecutionError):
    """Raised when the typed Tla compiler bridge cannot be resolved."""


class TlaBackendCompilerNotFoundError(TlaExecutionError):
    """Raised when the backend compiler cannot be resolved."""


class TlaKernelCompileError(TlaExecutionError):
    """Raised when kernel lowering or backend compilation fails."""

    def __init__(self, message: str, *, pass_ir_dump: str = "") -> None:
        super().__init__(message)
        self.pass_ir_dump = pass_ir_dump


class TlaRuntimeUnavailableError(TlaExecutionError):
    """Raised when Ascend runtime dependencies are unavailable."""


class TlaUnsupportedAbiError(TlaExecutionError):
    """Raised when attempting to execute an unsupported ABI shape."""


@dataclass(frozen=True)
class _PrintTensorDTypeSpec:
    mlir_token: str
    native_name: str
    numeric_kind: str
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True)
class _PrintTensorMetadata:
    shape: tuple[int, ...]
    count: int | None
    dtype: str
    position: str
    call: int = 0


@dataclass(frozen=True)
class _NativePrintTensorRecord:
    native_dtype: str
    position: str
    declared_count: int
    values_text: str
    shape: tuple[int, ...]
    call: int
    block: int
    subblock: int | None


_PRINT_TENSOR_DTYPES = {
    "f16": _PrintTensorDTypeSpec("f16", "float16", "float"),
    "f32": _PrintTensorDTypeSpec("f32", "float32", "float"),
    "i8": _PrintTensorDTypeSpec("i8", "int8", "integer", -128, 127),
    "i16": _PrintTensorDTypeSpec("i16", "int16", "integer", -32768, 32767),
    "i32": _PrintTensorDTypeSpec("i32", "int32", "integer", -2147483648, 2147483647),
    "u8": _PrintTensorDTypeSpec("ui8", "uint8", "integer", 0, 255),
    "u16": _PrintTensorDTypeSpec("ui16", "uint16", "integer", 0, 65535),
    "u32": _PrintTensorDTypeSpec("ui32", "uint32", "integer", 0, 4294967295),
}
_PRINT_TENSOR_DTYPES_BY_MLIR = {
    spec.mlir_token: token for token, spec in _PRINT_TENSOR_DTYPES.items()
}
_PRINT_TENSOR_DTYPES_BY_NATIVE = {
    spec.native_name: token for token, spec in _PRINT_TENSOR_DTYPES.items()
}


@dataclass(frozen=True)
class TlaKernelArtifact:
    cache_key: str
    cache_dir: Path
    tlair_mlir: str
    lowered_llvm: str
    entrypoint: str
    compiler_bridge_path: Path | None
    hivmc_path: Path
    kernel_binary_path: Path
    runtime: "TlaRuntimeOptions | None" = None
    pass_ir_dump: str = ""
    kernel_abi: KernelAbiLayout | None = None
    expects_print_tensor: bool = False
    expects_debug_fifo: bool = False
    logical_mixed_handoff: "_LogicalMixedHandoff | None" = None
    _artifact_metadata_prepared: bool = field(
        default=False, repr=False, compare=False, hash=False
    )
    _abi_packer: "_PreparedAbiPacker | None" = field(
        default=None, repr=False, compare=False, hash=False
    )
    _runtime_handle: tuple[int, int] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False
    )
    _runtime_handle_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        # Production compile/cache-load paths pass precomputed metadata. Keep
        # direct construction useful for tests and internal callers while still
        # establishing the same invariant at Artifact construction time.
        if not self._artifact_metadata_prepared:
            metadata = _analyze_artifact_static_metadata(
                self.tlair_mlir, self.lowered_llvm
            )
            object.__setattr__(
                self, "expects_print_tensor", metadata.expects_print_tensor
            )
            object.__setattr__(self, "expects_debug_fifo", metadata.expects_debug_fifo)
            object.__setattr__(
                self, "logical_mixed_handoff", metadata.logical_mixed_handoff
            )
            object.__setattr__(self, "_artifact_metadata_prepared", True)

        if self.kernel_abi is not None and self._abi_packer is None:
            expected_entrypoint = (
                self.logical_mixed_handoff.entrypoint
                if self.logical_mixed_handoff is not None
                else self.entrypoint
            )
            object.__setattr__(
                self,
                "_abi_packer",
                _prepare_abi_packer(
                    self.kernel_abi, expected_entrypoint=expected_entrypoint
                ),
            )


@dataclass(frozen=True)
class TlaExecutionResult:
    artifact: TlaKernelArtifact
    module_handle: int
    function_handle: int
    device: int


@dataclass(frozen=True)
class TlaRuntimeOptions:
    cache_enabled: bool = True
    cache_dir: Path | None = None
    force_recompile: bool = False
    kernel_mode: str = "aiv"
    # Placeholder until ``_runtime_options_from_lowered_mlir``; derived from
    # public ``--npu-arch`` default (not a separate Host knob).
    arch_scope: str = _arch_scope_for_target_impl(
        target_arch=_resolve_npu_arch(DEFAULT_NPU_ARCH),
        core_type="aiv",
    )
    #: When True, dump IR across the lowering pipeline (env ``CATLASS_DSL_PRINT_IR``).
    print_ir: bool = False


@dataclass(frozen=True)
class _KernelLaunchPlan:
    entrypoint: str
    kernel_mode: str
    block_num: int
    payload: bytes
    expects_debug_fifo: bool
    # 1 moves the workspace to arg 0 for pure kernels; 2 replaces the
    # trailing marker in place for mixed split kernels.
    expects_print_tensor: bool | int


@dataclass(frozen=True)
class _PreparedAbiSlot:
    argument: Any
    logical_index: int
    start: int
    end: int


@dataclass(frozen=True)
class _PreparedMemrefRun:
    logical_index: int
    start: int
    fields: tuple[str, ...]
    packer: struct.Struct


@dataclass(frozen=True)
class _PreparedAbiPacker:
    logical_count: int
    host_size: int
    slots: tuple[_PreparedAbiSlot, ...]
    memref_runs: tuple[_PreparedMemrefRun, ...]


@dataclass(frozen=True)
class _LogicalMixedHandoff:
    entrypoint: str
    user_arg_types: tuple[str, ...]


@dataclass(frozen=True)
class _ArtifactStaticMetadata:
    expects_print_tensor: bool
    expects_debug_fifo: bool
    logical_mixed_handoff: _LogicalMixedHandoff | None


_MEMORY_COMPILE_CACHE_LOCK = threading.RLock()
_MEMORY_COMPILE_CACHE: dict[str, TlaKernelArtifact] = {}
_NATIVE_PRINT_TENSOR_STDOUT_LOCK = threading.RLock()


def compile_kernel(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any],
    runtime: TlaRuntimeOptions,
    type_args: Sequence[Any] | None = None,
    decorator_location: DSLLocation | None = None,
) -> TlaKernelArtifact:
    lowered = BaseDSL()._lower(
        fn,
        kind=kind,
        options=dict(options),
        type_args=type_args,
        location=decorator_location,
    )
    tlair_mlir = lowered.asm(generic=True)
    entrypoint = _extract_entrypoint(tlair_mlir)
    compiler_bridge_path = resolve_bridge_extension_path()
    hivmc = _resolve_hivmc_a5()
    target = _resolve_kernel_target(runtime)
    cache_dir = runtime.cache_dir or _default_cache_dir()
    cache_key = _cache_key(
        tlair_mlir=tlair_mlir,
        entrypoint=entrypoint,
        runtime=runtime,
        compiler_bridge_path=compiler_bridge_path,
        hivmc=hivmc,
        target=target,
    )
    artifact_dir = cache_dir / cache_key
    manifest = artifact_dir / "manifest.json"

    if runtime.cache_enabled and not runtime.force_recompile:
        cached_memory = _get_memory_cached_artifact(cache_key)
        if cached_memory is not None:
            _copy_kept_artifacts(cached_memory)
            return cached_memory

    if runtime.cache_enabled and not runtime.force_recompile and manifest.exists():
        cached = _load_manifest(manifest)
        kernel_path = artifact_dir / str(cached["kernel_binary"])
        mlir_path = artifact_dir / str(cached["lowered_mlir"])
        cached_pass_dump = cached.get("pass_ir_dump")
        pass_dump_path = (
            artifact_dir / str(cached_pass_dump) if cached_pass_dump else None
        )
        if (
            _cache_manifest_has_current_workspace_abis(cached)
            and kernel_path.exists()
            and mlir_path.exists()
        ):
            lowered_llvm = mlir_path.read_text()
            static_metadata = _analyze_compiled_artifact_static_metadata(
                tlair_mlir, lowered_llvm
            )
            expected_abi_entrypoint = (
                static_metadata.logical_mixed_handoff.entrypoint
                if static_metadata.logical_mixed_handoff is not None
                else entrypoint
            )
            cached_kernel_abi, cached_abi_packer = _prepared_kernel_abi_from_manifest(
                cached, expected_entrypoint=expected_abi_entrypoint
            )
            artifact = TlaKernelArtifact(
                cache_key=cache_key,
                cache_dir=artifact_dir,
                tlair_mlir=tlair_mlir,
                lowered_llvm=lowered_llvm,
                entrypoint=entrypoint,
                compiler_bridge_path=compiler_bridge_path,
                hivmc_path=hivmc,
                kernel_binary_path=kernel_path,
                runtime=_runtime_options_from_lowered_mlir(
                    runtime,
                    lowered_llvm,
                    has_logical_mixed_handoff=(
                        static_metadata.logical_mixed_handoff is not None
                    ),
                ),
                pass_ir_dump=pass_dump_path.read_text()
                if pass_dump_path and pass_dump_path.exists()
                else "",
                kernel_abi=cached_kernel_abi,
                expects_print_tensor=static_metadata.expects_print_tensor,
                expects_debug_fifo=static_metadata.expects_debug_fifo,
                logical_mixed_handoff=static_metadata.logical_mixed_handoff,
                _artifact_metadata_prepared=True,
                _abi_packer=cached_abi_packer,
            )
            _set_memory_cached_artifact(artifact)
            _copy_kept_artifacts(artifact)
            return artifact

    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = artifact_dir / "lowered.mlir"
    pass_dump_path = artifact_dir / "pass-ir-dump.mlir"
    kernel_path = artifact_dir / "kernel.o"

    try:
        lowering_result = _run_tla_lowering_to_mlir(
            lowered_module=lowered.module,
            tlair_mlir=tlair_mlir,
            mlir_path=mlir_path,
            runtime=runtime,
        )
    except TlaKernelCompileError as exc:
        if exc.pass_ir_dump:
            _write_pass_ir_dump(pass_dump_path, exc.pass_ir_dump)
            raise TlaKernelCompileError(
                f"{exc}\npass IR dump: {pass_dump_path}",
                pass_ir_dump=exc.pass_ir_dump,
            ) from exc
        raise
    if lowering_result.pass_ir_dump:
        _write_pass_ir_dump(pass_dump_path, lowering_result.pass_ir_dump)
    lowered_llvm = mlir_path.read_text()
    static_metadata = _analyze_compiled_artifact_static_metadata(
        tlair_mlir, lowered_llvm
    )
    runtime_for_hivmc = _runtime_options_from_lowered_mlir(
        runtime,
        lowered_llvm,
        has_logical_mixed_handoff=(static_metadata.logical_mixed_handoff is not None),
    )
    kernel_abi = getattr(lowering_result, "kernel_abi", None)
    expected_abi_entrypoint = (
        static_metadata.logical_mixed_handoff.entrypoint
        if static_metadata.logical_mixed_handoff is not None
        else entrypoint
    )
    abi_packer = _prepare_compiled_abi_packer(
        kernel_abi, expected_entrypoint=expected_abi_entrypoint
    )
    hivmc_mlir_path, template_bitcode = _create_stamped_hivmc_input(
        mlir_path, runtime_for_hivmc
    )
    try:
        _run_checked(
            _build_hivmc_a5_command(
                compiler=hivmc,
                mlir_path=hivmc_mlir_path,
                kernel_path=kernel_path,
                runtime=runtime_for_hivmc,
                template_bitcode=template_bitcode,
            ),
            label="hivmc-a5",
            cwd=artifact_dir,
        )
    finally:
        if hivmc_mlir_path != mlir_path:
            hivmc_mlir_path.unlink(missing_ok=True)
    if not kernel_path.exists():
        raise TlaKernelCompileError(
            "hivmc-a5 completed but output kernel artifact was not "
            f"created at {kernel_path}"
        )

    manifest.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "debug_print_workspace_abi_revision": (
                    _DEBUG_PRINT_WORKSPACE_ABI_REVISION
                ),
                "print_tensor_workspace_abi_revision": (
                    _PRINT_TENSOR_WORKSPACE_ABI_REVISION
                ),
                "entrypoint": entrypoint,
                "kernel_binary": kernel_path.name,
                "lowered_mlir": mlir_path.name,
                "pass_ir_dump": pass_dump_path.name
                if lowering_result.pass_ir_dump
                else None,
                "compiler_bridge": (
                    str(compiler_bridge_path) if compiler_bridge_path else None
                ),
                "hivmc": str(hivmc),
                "arch_scope": runtime_for_hivmc.arch_scope,
                "kernel_abi": kernel_abi_to_dict(kernel_abi),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    artifact = TlaKernelArtifact(
        cache_key=cache_key,
        cache_dir=artifact_dir,
        tlair_mlir=tlair_mlir,
        lowered_llvm=lowered_llvm,
        entrypoint=entrypoint,
        compiler_bridge_path=compiler_bridge_path,
        hivmc_path=hivmc,
        kernel_binary_path=kernel_path,
        pass_ir_dump=lowering_result.pass_ir_dump,
        runtime=runtime_for_hivmc,
        kernel_abi=kernel_abi,
        expects_print_tensor=static_metadata.expects_print_tensor,
        expects_debug_fifo=static_metadata.expects_debug_fifo,
        logical_mixed_handoff=static_metadata.logical_mixed_handoff,
        _artifact_metadata_prepared=True,
        _abi_packer=abi_packer,
    )
    if runtime.cache_enabled:
        _set_memory_cached_artifact(artifact)
    _copy_kept_artifacts(artifact)
    return artifact


def _get_memory_cached_artifact(cache_key: str) -> TlaKernelArtifact | None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        artifact = _MEMORY_COMPILE_CACHE.get(cache_key)
    if artifact is None:
        return None
    if not artifact.kernel_binary_path.exists():
        _drop_memory_cached_artifact(cache_key)
        return None
    return artifact


def _set_memory_cached_artifact(artifact: TlaKernelArtifact) -> None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        _MEMORY_COMPILE_CACHE[artifact.cache_key] = artifact


def _drop_memory_cached_artifact(cache_key: str) -> None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        _MEMORY_COMPILE_CACHE.pop(cache_key, None)


def _checked_print_tensor_block_count(block_num: int) -> int:
    block_count = int(block_num)
    if block_count <= 0 or block_count > _PRINT_TENSOR_MAX_BLOCKS:
        raise TlaExecutionError(
            "tla.print_tensor 16-bit block identity requires a positive "
            f"block_num with at most {_PRINT_TENSOR_MAX_BLOCKS} blocks; "
            f"got {block_num}"
        )
    return block_count


def _print_tensor_native_wire_bytes(metadata: _PrintTensorMetadata) -> int:
    count = _PRINT_TENSOR_MAX_F32_ELEMENTS if metadata.count is None else metadata.count
    element_bytes = dtype_size_bytes(metadata.dtype)
    if element_bytes <= 0:
        raise TlaExecutionError(
            "tla.print_tensor FIFO capacity calculation encountered "
            f"unsupported dtype {metadata.dtype!r}"
        )
    payload_bytes = count * element_bytes
    aligned_payload_bytes = (
        payload_bytes + _PRINT_TENSOR_PAYLOAD_ALIGNMENT - 1
    ) // _PRINT_TENSOR_PAYLOAD_ALIGNMENT
    return (
        _PRINT_TENSOR_SHAPE_HEADER_BYTES
        + _PRINT_TENSOR_HEADER_BYTES
        + aligned_payload_bytes * _PRINT_TENSOR_PAYLOAD_ALIGNMENT
    )


def _validate_print_tensor_fifo_capacity(
    artifact: TlaKernelArtifact,
    block_num: int,
    *,
    helper_core: str,
    mixed: bool,
) -> None:
    block_count = _checked_print_tensor_block_count(block_num)
    records_per_block = 2 if helper_core == "aiv" and mixed else 1
    core_records = block_count * records_per_block
    if core_records > _PRINT_TENSOR_CORE_RECORDS:
        raise TlaExecutionError(
            "tla.print_tensor fixed 1 MiB FIFO workspace supports at most "
            f"{_PRINT_TENSOR_CORE_RECORDS} core records; got {core_records} "
            f"from {block_count} blocks"
        )
    metadata = _print_tensor_static_metadata_records(
        artifact.tlair_mlir, entrypoint=artifact.entrypoint
    )
    # Dynamic control flow can execute any site zero or multiple times. Keep
    # the native per-record bound, but do not reserve the FIFO for every
    # statically visible site.
    for item in metadata:
        record_bytes = _print_tensor_native_wire_bytes(item)
        if record_bytes > _PRINT_TENSOR_FIFO_BYTES:
            raise TlaExecutionError(
                "tla.print_tensor record exceeds the per-print FIFO capacity: "
                f"{record_bytes} bytes, capacity is {_PRINT_TENSOR_FIFO_BYTES} bytes"
            )


def _runtime_arg_values(arg: Any) -> list[int]:
    try:
        if hasattr(arg, "__c_pointers__"):
            return [int(ptr) for ptr in arg.__c_pointers__()]
        data_ptr = getattr(arg, "data_ptr", None)
        if callable(data_ptr):
            return [int(data_ptr())]
        if data_ptr is not None:
            return [int(data_ptr)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TlaUnsupportedAbiError(
            "Launch argument cannot provide exactly one concrete host value."
        ) from exc
    raise TlaUnsupportedAbiError(
        "Launch arguments must provide exactly one host value."
    )


def _align_payload(payload: bytearray, alignment: int) -> None:
    payload.extend(b"\0" * (-len(payload) % alignment))


def _scalar_storage_size(descriptor: KernelAbiScalarDescriptor) -> int:
    if descriptor.category is KernelAbiScalarCategory.INDEX:
        return 8
    return max(1, descriptor.bit_width // 8)


def _pack_scalar_argument(
    value: Any,
    descriptor: KernelAbiScalarDescriptor,
    mlir_type: str,
    storage_size: int,
) -> bytes:
    expected_size = _scalar_storage_size(descriptor)
    if storage_size != expected_size:
        raise TlaUnsupportedAbiError(
            f"kernel ABI scalar {mlir_type} declares unsupported storage size "
            f"{storage_size}; expected {expected_size}"
        )
    if not isinstance(value, Numeric):
        value_type = type(value)
        if (
            value_type is bool
            and descriptor.category is KernelAbiScalarCategory.INTEGER
            and descriptor.bit_width == 1
        ):
            return bytes((int(value),))
        if (
            value_type is int
            and descriptor.category is KernelAbiScalarCategory.INTEGER
            and descriptor.bit_width == 32
            and descriptor.integer_signedness is KernelAbiIntegerSignedness.SIGNLESS
        ):
            lower, upper = -(1 << 31), (1 << 31) - 1
            if value < lower or value > upper:
                raise TlaUnsupportedAbiError(
                    f"scalar value for {mlir_type} does not fit in its declared type"
                )
            return value.to_bytes(storage_size, byteorder="little", signed=True)
        if (
            value_type is float
            and descriptor.category is KernelAbiScalarCategory.FLOAT
            and descriptor.float_format is not None
            and descriptor.float_format.value == "f32"
        ):
            try:
                return struct.pack("<f", value)
            except (OverflowError, struct.error) as exc:
                raise TlaUnsupportedAbiError(
                    f"scalar value for {mlir_type} does not fit in its declared type"
                ) from exc
        raise TlaUnsupportedAbiError(
            f"plain Python {value_type.__name__} does not match kernel ABI type "
            f"{mlir_type}"
        )
    host_type = type(value)
    host_dtype = str(getattr(host_type, "dtype", "")).lower()
    host_is_float = bool(getattr(host_type, "is_float", False))
    type_matches = host_is_float == (
        descriptor.category is KernelAbiScalarCategory.FLOAT
    )
    if descriptor.category is KernelAbiScalarCategory.FLOAT:
        if descriptor.float_format is None:
            raise TlaUnsupportedAbiError(
                f"kernel ABI float scalar {mlir_type} has no format"
            )
        type_matches = type_matches and host_dtype == descriptor.float_format.value
    elif descriptor.category is KernelAbiScalarCategory.INDEX:
        type_matches = type_matches and host_dtype == "index"
    else:
        type_matches = (
            type_matches and int(getattr(host_type, "width", 0)) == descriptor.bit_width
        )
        if descriptor.integer_signedness is KernelAbiIntegerSignedness.SIGNED:
            type_matches = type_matches and bool(getattr(host_type, "signed", False))
        elif descriptor.integer_signedness is KernelAbiIntegerSignedness.UNSIGNED:
            type_matches = type_matches and not bool(getattr(host_type, "signed", True))
    if not type_matches:
        raise TlaUnsupportedAbiError(
            f"typed host scalar {host_dtype or type(value).__name__} does not match "
            f"kernel ABI type {mlir_type}"
        )
    if descriptor.category is not KernelAbiScalarCategory.FLOAT:
        host_value = getattr(value, "value", None)
        if not isinstance(host_value, (bool, int)):
            raise TlaUnsupportedAbiError(
                f"kernel ABI scalar {mlir_type} requires a concrete integer host value"
            )
        width = int(getattr(host_type, "width", storage_size * 8))
        if width == 1:
            lower, upper = 0, 1
        elif bool(getattr(host_type, "signed", False)):
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        else:
            lower, upper = 0, (1 << width) - 1
        if int(host_value) < lower or int(host_value) > upper:
            raise TlaUnsupportedAbiError(
                f"scalar value for {mlir_type} does not fit in its declared type"
            )
    values = _runtime_arg_values(value)
    if len(values) != 1:
        raise TlaUnsupportedAbiError(
            "each logical launch argument must provide exactly one host value"
        )
    bits = int(values[0])
    if bits < 0 or bits >= (1 << (storage_size * 8)):
        raise TlaUnsupportedAbiError(
            f"scalar value for {mlir_type} does not fit in {storage_size} bytes"
        )
    return bits.to_bytes(storage_size, byteorder="little", signed=False)


def _logical_launch_arg_count(layout: KernelAbiLayout) -> int:
    if not layout.arguments:
        return 0
    return 1 + max(
        (argument.index if argument.logical_index is None else argument.logical_index)
        for argument in layout.arguments
    )


def _prepare_abi_packer(
    layout: KernelAbiLayout | None, *, expected_entrypoint: str | None = None
) -> _PreparedAbiPacker:
    if expected_entrypoint is None:
        _validate_kernel_abi_layout(layout)
    else:
        _validate_kernel_abi_layout(layout, expected_entrypoint=expected_entrypoint)
    if layout is None:
        raise TlaUnsupportedAbiError("kernel ABI layout is missing")
    logical_count = _logical_launch_arg_count(layout)
    host_offsets: list[int] = []
    host_size = 0
    for argument in layout.arguments:
        host_alignment = 8 if argument.storage_size == 8 else 4
        host_size = (
            (host_size + host_alignment - 1) // host_alignment
        ) * host_alignment
        host_offsets.append(host_size)
        host_size += argument.storage_size
    host_size = ((host_size + 7) // 8) * 8
    if host_size > _MAX_KERNEL_ABI_PAYLOAD_SIZE:
        raise TlaUnsupportedAbiError(
            "kernel ABI host payload exceeds the supported maximum size"
        )
    all_slots = []
    for index, argument in enumerate(layout.arguments):
        logical_index = (
            argument.index if argument.logical_index is None else argument.logical_index
        )
        start = host_offsets[index]
        end = start + argument.storage_size
        all_slots.append(_PreparedAbiSlot(argument, logical_index, start, end))

    slots: list[_PreparedAbiSlot] = []
    memref_runs: list[_PreparedMemrefRun] = []
    index = 0
    while index < len(all_slots):
        slot = all_slots[index]
        if slot.argument.kind is not KernelAbiArgumentKind.MEMREF_FIELD:
            slots.append(slot)
            index += 1
            continue
        fields = []
        run_start = slot.start
        logical_index = slot.logical_index
        previous_end = slot.start
        while index < len(all_slots):
            candidate = all_slots[index]
            argument = candidate.argument
            if (
                argument.kind is not KernelAbiArgumentKind.MEMREF_FIELD
                or candidate.logical_index != logical_index
                or candidate.start != previous_end
            ):
                break
            if argument.field is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {argument.index} "
                    "requires a field name"
                )
            fields.append(argument.field)
            previous_end = candidate.end
            index += 1
        memref_runs.append(
            _PreparedMemrefRun(
                logical_index=logical_index,
                start=run_start,
                fields=tuple(fields),
                packer=struct.Struct("<" + "Q" * len(fields)),
            )
        )
    return _PreparedAbiPacker(
        logical_count, host_size, tuple(slots), tuple(memref_runs)
    )


def _pack_launch_args_prepared(
    args: Sequence[Any], packer: _PreparedAbiPacker
) -> bytes:
    if len(args) != packer.logical_count:
        raise TlaUnsupportedAbiError(
            "kernel launch argument count does not match ABI layout: "
            f"got {len(args)}, expected {packer.logical_count}"
        )
    payload = bytearray(packer.host_size)
    pointer_values: dict[int, list[int]] = {}
    for slot in packer.slots:
        argument = slot.argument
        logical_index = slot.logical_index
        value = args[logical_index]
        if argument.kind is KernelAbiArgumentKind.POINTER:
            if isinstance(value, Numeric) or type(value) in (bool, int, float):
                raise TlaUnsupportedAbiError(
                    f"kernel ABI argument {argument.index} requires a pointer"
                )
            values = pointer_values.get(logical_index)
            if values is None:
                values = _runtime_arg_values(value)
                pointer_values[logical_index] = values
            if len(values) != 1:
                raise TlaUnsupportedAbiError(
                    "each logical launch argument must provide exactly one host value"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported pointer storage size {argument.storage_size}"
                )
            pointer = int(values[0])
            if pointer < 0 or pointer >= (1 << (argument.storage_size * 8)):
                raise TlaUnsupportedAbiError(
                    f"pointer value does not fit in {argument.storage_size} bytes"
                )
            encoded = pointer.to_bytes(
                argument.storage_size, byteorder="little", signed=False
            )
        elif argument.kind is KernelAbiArgumentKind.SCALAR:
            if argument.scalar is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar argument {argument.index} has no scalar descriptor"
                )
            encoded = _pack_scalar_argument(
                value,
                argument.scalar,
                argument.mlir_type,
                argument.storage_size,
            )
        else:
            raise TlaUnsupportedAbiError(
                f"unsupported kernel ABI argument kind {argument.kind!r}"
            )
        payload[slot.start : slot.end] = encoded

    memref_fields: dict[int, Mapping[str, int]] = {}
    for run in packer.memref_runs:
        value = args[run.logical_index]
        fields = memref_fields.get(run.logical_index)
        if fields is None:
            builder = getattr(value, "build_memref_launch_fields", None)
            if not callable(builder):
                raise TlaUnsupportedAbiError(
                    "dynamic GM memref launch requires "
                    "Tensor.build_memref_launch_fields()"
                )
            fields = builder()
            memref_fields[run.logical_index] = fields
        missing = [field for field in run.fields if field not in fields]
        if missing:
            raise TlaUnsupportedAbiError(
                f"memref launch field {missing[0]!r} missing from tensor descriptor"
            )
        encoded_values = tuple(int(fields[field]) for field in run.fields)
        for field_name, field_value in zip(run.fields, encoded_values, strict=True):
            if field_value < 0 or field_value >= (1 << 64):
                raise TlaUnsupportedAbiError(
                    f"memref field {field_name!r} does not fit in 8 bytes"
                )
        run.packer.pack_into(payload, run.start, *encoded_values)
    return bytes(payload)


def _pack_launch_args(
    args: Sequence[Any], layout: KernelAbiLayout | None = None
) -> bytes:
    return _pack_launch_args_prepared(args, _prepare_abi_packer(layout))


def _append_debug_print_workspace_payload(
    payload: bytes,
    *,
    enabled: bool,
) -> bytes:
    if not enabled:
        return payload
    extension = bytearray(payload)
    _align_payload(extension, _POINTER_ABI_SIZE)
    extension.extend(
        _DEBUG_PRINT_WORKSPACE_SENTINEL.to_bytes(
            _POINTER_ABI_SIZE, byteorder="little", signed=False
        )
    )
    return bytes(extension)


def _validate_kernel_abi_layout(
    layout: KernelAbiLayout | None, *, expected_entrypoint: str | None = None
) -> None:
    if layout is None or layout.schema_version not in (3, 4):
        raise TlaUnsupportedAbiError(
            "A supported compiler-produced kernel ABI layout is required before launch."
        )
    if not layout.entrypoint.strip():
        raise TlaUnsupportedAbiError("kernel ABI layout entrypoint must be nonempty")
    if expected_entrypoint is not None and layout.entrypoint != expected_entrypoint:
        raise TlaUnsupportedAbiError(
            "kernel ABI layout entrypoint does not match launch artifact: "
            f"{layout.entrypoint!r} != {expected_entrypoint!r}"
        )
    if layout.total_size < 0:
        raise TlaUnsupportedAbiError("kernel ABI layout has an invalid total size")
    if layout.total_size > _MAX_KERNEL_ABI_PAYLOAD_SIZE:
        raise TlaUnsupportedAbiError(
            "kernel ABI payload exceeds the supported maximum size"
        )
    if layout.total_size % 8 != 0:
        raise TlaUnsupportedAbiError("kernel ABI total size must be rounded to 8 bytes")
    logical_count = _logical_launch_arg_count(layout)
    previous_end = 0
    for index, argument in enumerate(layout.arguments):
        if argument.index != index:
            raise TlaUnsupportedAbiError(
                "kernel ABI arguments must be ordered by contiguous index"
            )
        if argument.alignment != 4:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} must declare 4-byte alignment"
            )
        if argument.offset < previous_end or argument.offset % 4 != 0:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} offset must be ordered, "
                "non-overlapping, and 4-byte aligned"
            )
        if argument.storage_size <= 0:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} storage size must be positive"
            )
        logical_index = (
            argument.index if argument.logical_index is None else argument.logical_index
        )
        if logical_index < 0 or logical_index >= logical_count:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} logical_index {logical_index} is out of range"
            )
        if argument.kind is KernelAbiArgumentKind.POINTER:
            if argument.scalar is not None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI pointer argument {index} cannot have a scalar descriptor"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported pointer storage size {argument.storage_size}"
                )
        elif argument.kind is KernelAbiArgumentKind.MEMREF_FIELD:
            if argument.scalar is not None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {index} cannot have a scalar descriptor"
                )
            if argument.field is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {index} requires a field name"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported memref_field storage size {argument.storage_size}"
                )
        elif argument.kind is KernelAbiArgumentKind.SCALAR:
            if argument.scalar is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar argument {index} requires a scalar descriptor"
                )
            expected_size = _scalar_storage_size(argument.scalar)
            if argument.storage_size != expected_size:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar {argument.mlir_type} declares unsupported "
                    f"storage size {argument.storage_size}; expected {expected_size}"
                )
        else:
            raise TlaUnsupportedAbiError(
                f"unsupported kernel ABI argument kind {argument.kind!r}"
            )
        previous_end = argument.offset + argument.storage_size
        if previous_end > layout.total_size:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} storage does not fit in payload"
            )
    required_size = ((previous_end + 7) // 8) * 8
    if layout.total_size != required_size:
        raise TlaUnsupportedAbiError(
            "kernel ABI total size is not exactly sufficient for its arguments"
        )


def _prepare_compiled_abi_packer(
    layout: KernelAbiLayout | None, *, expected_entrypoint: str
) -> _PreparedAbiPacker:
    try:
        return _prepare_abi_packer(layout, expected_entrypoint=expected_entrypoint)
    except TlaUnsupportedAbiError as exc:
        raise TlaKernelCompileError(
            f"Invalid compiler-produced kernel ABI descriptor: {exc}"
        ) from exc


def _split_top_level_csv(text: str) -> list[str]:
    result: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    brace_depth = 0
    for index, char in enumerate(text):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth > 0:
            angle_depth -= 1
        elif char == "(" and angle_depth == 0:
            paren_depth += 1
        elif char == ")" and angle_depth == 0 and paren_depth > 0:
            paren_depth -= 1
        elif char == "{" and angle_depth == 0 and paren_depth == 0:
            brace_depth += 1
        elif char == "}" and angle_depth == 0 and paren_depth == 0 and brace_depth > 0:
            brace_depth -= 1
        elif char == "," and angle_depth == 0 and paren_depth == 0 and brace_depth == 0:
            item = text[start:index].strip()
            if item:
                result.append(item)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return result


def _find_matching_function_type_paren(text: str, start: int) -> int:
    angle_depth = 0
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth > 0:
            angle_depth -= 1
        elif char == "(" and angle_depth == 0:
            depth += 1
        elif char == ")" and angle_depth == 0:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _extract_named_func_args(mlir_text: str, func_name: str) -> list[str]:
    match = re.search(
        rf"(?:func\.func|\"func\.func\")\s+(?:private\s+)?@{re.escape(func_name)}\s*\(",
        mlir_text,
    )
    if not match:
        return []
    start = match.end() - 1
    end = _find_matching_function_type_paren(mlir_text, start)
    if end <= start:
        return []
    params = _split_top_level_csv(mlir_text[start + 1 : end])
    return params


def _mixed_handoff_user_arg_types(params: Sequence[str]) -> tuple[str, ...]:
    """Exclude compiler-owned print workspace operands from a mixed handoff.

    The split functions carry the same user ABI plus an internal workspace
    when scalar debug or tensor printing is present. That workspace is
    supplied by a marker-backed FIFO descriptor, never by a public launch
    argument.
    """

    user_arg_types: list[str] = []
    for param in params:
        if (
            "tla.debug_print.workspace" in param
            or "tla.print_tensor.workspace" in param
        ):
            continue
        if ":" in param:
            user_arg_types.append(param.split(":", 1)[1].strip())
        else:
            user_arg_types.append(param.strip())
    return tuple(user_arg_types)


def _extract_mixed_handoff_entrypoints(mlir_text: str) -> tuple[str, str] | None:
    names = re.findall(
        r"(?:func\.func|\"func\.func\")\s+@([A-Za-z_][A-Za-z0-9_]*)",
        mlir_text,
    )
    aic_names = [name for name in names if name.endswith("_mix_aic")]
    aiv_names = [name for name in names if name.endswith("_mix_aiv")]
    if len(aic_names) != 1 or len(aiv_names) != 1:
        return None
    return aic_names[0], aiv_names[0]


def _extract_logical_mixed_handoff(mlir_text: str) -> _LogicalMixedHandoff | None:
    entrypoints = _extract_mixed_handoff_entrypoints(mlir_text)
    if entrypoints is None:
        return None
    aic_name, aiv_name = entrypoints
    base_name = aic_name.removesuffix("_mix_aic")
    if aiv_name.removesuffix("_mix_aiv") != base_name:
        return None
    params = _extract_named_func_args(mlir_text, aic_name)
    if not params:
        raise TlaUnsupportedAbiError(
            "mixed handoff AIC split function must expose argument types"
        )
    return _LogicalMixedHandoff(base_name, _mixed_handoff_user_arg_types(params))


def _analyze_artifact_static_metadata(
    tlair_mlir: str, lowered_llvm: str
) -> _ArtifactStaticMetadata:
    mlir_text = lowered_llvm or tlair_mlir
    return _ArtifactStaticMetadata(
        expects_print_tensor="tla.print_tensor.workspace" in mlir_text,
        expects_debug_fifo="tla.debug_print.workspace" in mlir_text,
        logical_mixed_handoff=_extract_logical_mixed_handoff(mlir_text),
    )


def _analyze_compiled_artifact_static_metadata(
    tlair_mlir: str, lowered_llvm: str
) -> _ArtifactStaticMetadata:
    try:
        return _analyze_artifact_static_metadata(tlair_mlir, lowered_llvm)
    except TlaUnsupportedAbiError as exc:
        raise TlaKernelCompileError(
            f"Invalid compiler-produced launch metadata: {exc}"
        ) from exc


def _build_logical_mixed_handoff_launch_args(
    launch_args: Sequence[Any],
    arg_types: Sequence[str],
    kernel_abi: KernelAbiLayout | None,
) -> bytes:
    # Host still passes one object per logical kernel argument. Dynamic GM expands
    # each Tensor into many device params / memref_field slots in the ABI, so do
    # not compare against the raw split-function parameter count.
    expected = (
        _logical_launch_arg_count(kernel_abi)
        if kernel_abi is not None
        else len(arg_types)
    )
    if len(launch_args) != expected:
        raise TlaUnsupportedAbiError(
            "mixed handoff launch argument count does not match ABI layout: "
            f"got {len(launch_args)}, expected {expected}"
        )
    return _pack_launch_args(launch_args, kernel_abi)


def _capture_c_stdout(launch: Callable[[], None]) -> str:
    libc = ctypes.CDLL(None)
    with _NATIVE_PRINT_TENSOR_STDOUT_LOCK:
        sys.stdout.flush()
        libc.fflush(None)
        saved_stdout = os.dup(1)
        try:
            with tempfile.TemporaryFile(mode="w+b") as captured:
                os.dup2(captured.fileno(), 1)
                try:
                    launch()
                finally:
                    libc.fflush(None)
                    sys.stdout.flush()
                    os.dup2(saved_stdout, 1)
                captured.seek(0)
                return captured.read().decode("utf-8", errors="replace")
        finally:
            os.close(saved_stdout)


def _print_tensor_static_metadata_records(
    tlair_mlir: str, *, entrypoint: str | None = None
) -> tuple[_PrintTensorMetadata, ...]:
    function_text = (
        _mlir_entrypoint_text(tlair_mlir, entrypoint)
        if entrypoint is not None
        else tlair_mlir
    )
    op_matches = list(re.finditer(r'"tla\.print_tensor"', function_text))
    if not op_matches:
        raise TlaExecutionError(
            "tla.print_tensor requires static shape metadata; found 0 records"
        )
    constants = {
        match.group("result"): int(match.group("value"))
        for match in re.finditer(
            r'(?P<result>%[A-Za-z0-9_.$-]+)\s*=\s*"arith\.constant"\(\)'
            r"\s*<\{\s*value\s*=\s*(?P<value>\d+)\s*:\s*i64\s*\}>",
            function_text,
        )
    }
    op_ends = [match.start() for match in op_matches[1:]]
    op_ends.append(len(function_text))
    metadata = []
    for call, (match, end) in enumerate(zip(op_matches, op_ends)):
        op_text = function_text[match.start() : end]
        operand_match = re.match(
            r'"tla\.print_tensor"\(\s*[^,]+,\s*'
            r"(?P<length>%[A-Za-z0-9_.$-]+)\s*\)",
            op_text,
        )
        metadata.append(
            _parse_print_tensor_static_metadata(
                op_text,
                call=call,
                length=(
                    constants.get(operand_match.group("length"))
                    if operand_match is not None
                    else None
                ),
            )
        )
    return tuple(metadata)


def _mlir_entrypoint_text(mlir_text: str, entrypoint: str) -> str:
    function_matches = list(
        re.finditer(
            r'"(?:tla|func)\.func"\s*\('
            r"|(?:tla|func)\.func\s+@(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b",
            mlir_text,
        )
    )
    for index, match in enumerate(function_matches):
        end = (
            function_matches[index + 1].start()
            if index + 1 < len(function_matches)
            else len(mlir_text)
        )
        function_text = mlir_text[match.start() : end]
        name = match.group("name")
        if name is None:
            name_match = re.search(r'\bsym_name\s*=\s*"([^"]+)"', function_text)
            name = name_match.group(1) if name_match is not None else None
        if name == entrypoint:
            return function_text
    raise TlaExecutionError(
        f"tla.print_tensor entrypoint {entrypoint!r} is missing from the compiled artifact"
    )


def _parse_print_tensor_static_metadata(
    op_text: str, *, call: int, length: int | None = None
) -> _PrintTensorMetadata:
    shape_match = re.search(r"\bshape\s*=\s*array<i64:\s*([-\d,\s]+)>", op_text)
    if shape_match is None:
        raise TlaExecutionError(
            "tla.print_tensor static shape metadata is missing from the compiled artifact"
        )
    operand_match = re.search(r"!tla\.ptr<\s*([^,\s>]+)\s*,\s*(gm|ub)\s*,", op_text)
    if operand_match is None:
        raise TlaExecutionError(
            "tla.print_tensor operand type metadata is missing or malformed"
        )
    mlir_dtype = operand_match.group(1)
    position = operand_match.group(2).upper()
    shape = tuple(int(item.strip()) for item in shape_match.group(1).split(","))
    if len(shape) not in (1, 2) or any(extent == 0 or extent < -1 for extent in shape):
        raise TlaExecutionError(
            "tla.print_tensor static shape metadata contains an invalid extent"
        )
    length_match = re.search(r"\blength\s*=\s*(\d+)", op_text)
    if length is None and length_match is not None:
        length = int(length_match.group(1))
    if length is not None and (
        not 1 <= length <= _PRINT_TENSOR_MAX_F32_ELEMENTS
        or (all(extent > 0 for extent in shape) and length > math.prod(shape))
    ):
        raise TlaExecutionError(
            "tla.print_tensor static length metadata is invalid for the tensor shape"
        )
    dtype = _PRINT_TENSOR_DTYPES_BY_MLIR.get(mlir_dtype)
    if dtype is None:
        raise TlaExecutionError(
            f"tla.print_tensor runtime metadata has unsupported dtype {mlir_dtype!r}; "
            "supported: f16, f32, i8, i16, i32, ui8, ui16, ui32"
        )
    return _PrintTensorMetadata(
        shape=shape,
        count=length,
        dtype=dtype,
        position=position,
        call=call,
    )


def _print_tensor_decode_error(detail: str) -> TlaExecutionError:
    return TlaExecutionError(
        f"tla.print_tensor native initialization or decoding failed: {detail}"
    )


def _parse_native_print_tensor_records(output: str) -> list[_NativePrintTensorRecord]:
    records: list[_NativePrintTensorRecord] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("DumpTensor:"):
            continue
        match = _NATIVE_PRINT_TENSOR_RECORD_RE.fullmatch(line)
        if match is None:
            raise _print_tensor_decode_error("malformed native record")
        shape = tuple(int(item.strip()) for item in match.group("shape").split(","))
        records.append(
            _NativePrintTensorRecord(
                native_dtype=match.group("dtype"),
                position=match.group("position"),
                declared_count=int(match.group("count")),
                values_text=match.group("values"),
                shape=shape,
                call=int(match.group("call")),
                block=int(match.group("block")),
                subblock=(
                    int(match.group("subblock"))
                    if match.group("subblock") is not None
                    else None
                ),
            )
        )
    return records


def _print_tensor_integer_bounds(
    dtype: str, spec: _PrintTensorDTypeSpec
) -> tuple[int, int]:
    if spec.minimum is None or spec.maximum is None:
        raise TlaExecutionError(
            f"tla.print tensor dtype metadata is missing integer bounds (dtype={dtype})"
        )
    return spec.minimum, spec.maximum


def _decode_print_tensor_record_values(
    record: _NativePrintTensorRecord, *, dtype: str
) -> list[float | int]:
    spec = _PRINT_TENSOR_DTYPES[dtype]
    items = (
        [item.strip() for item in record.values_text.split(",")]
        if record.values_text.strip()
        else []
    )
    if len(items) != record.declared_count:
        raise _print_tensor_decode_error(
            f"payload count is {len(items)}, record declares {record.declared_count}",
        )
    if spec.numeric_kind == "float":
        if any(_FLOAT_LITERAL_RE.fullmatch(item) is None for item in items):
            raise _print_tensor_decode_error(f"invalid {dtype} numeric syntax")
        return [float(item) for item in items]
    if any(_INTEGER_LITERAL_RE.fullmatch(item) is None for item in items):
        raise _print_tensor_decode_error(f"invalid {dtype} numeric syntax")
    values = [int(item, 10) for item in items]
    minimum, maximum = _print_tensor_integer_bounds(dtype, spec)
    if any(value < minimum or value > maximum for value in values):
        raise _print_tensor_decode_error(
            f"{dtype} integer value outside [{minimum}, {maximum}]"
        )
    return values


def _decode_native_print_tensor_records(
    output: str,
    *,
    metadata: Sequence[_PrintTensorMetadata],
    block_count: int = 1,
    expected_subblocks: tuple[int | None, ...] = (None,),
) -> list[tuple[_PrintTensorMetadata, _NativePrintTensorRecord, list[float | int]]]:
    records = _parse_native_print_tensor_records(output)
    expected_metadata = {item.call: item for item in metadata}
    valid_blocks = set(range(block_count))
    valid_subblocks = set(expected_subblocks)
    record_values: dict[int, list[float | int]] = {}
    malformed_identities: set[tuple[int, int, int | None]] = set()
    for record in records:
        identity = (record.call, record.block, record.subblock)
        metadata_item = expected_metadata.get(record.call)
        dtype = (
            metadata_item.dtype
            if metadata_item is not None
            else _PRINT_TENSOR_DTYPES_BY_NATIVE.get(record.native_dtype)
        )
        if dtype is None:
            malformed_identities.add(identity)
            continue
        try:
            record_values[id(record)] = _decode_print_tensor_record_values(
                record, dtype=dtype
            )
        except TlaExecutionError:
            malformed_identities.add(identity)
    if malformed_identities:
        raise _print_tensor_decode_error(
            f"malformed records {_sort_print_tensor_identities(malformed_identities)}"
        )

    # A dynamic branch can produce no record and a loop can repeat one. Native
    # output is therefore best-effort: validate every observed record without
    # requiring a one-to-one correspondence with statically lowered calls.
    identities = [(record.call, record.block, record.subblock) for record in records]
    unexpected_identities = _sort_print_tensor_identities(
        {
            identity
            for identity in identities
            if (
                identity[0] not in expected_metadata
                or identity[1] not in valid_blocks
                or identity[2] not in valid_subblocks
            )
        }
    )
    if unexpected_identities:
        raise _print_tensor_decode_error(
            f"unexpected call/block/subblock identities {unexpected_identities}",
        )

    mismatches: dict[str, set[tuple[int, int, int | None]]] = {
        "native dtype": set(),
        "position": set(),
        "declared count": set(),
        "shape": set(),
    }
    for record in records:
        identity = (record.call, record.block, record.subblock)
        item = expected_metadata[record.call]
        if record.native_dtype != _PRINT_TENSOR_DTYPES[item.dtype].native_name:
            mismatches["native dtype"].add(identity)
        if record.position != item.position:
            mismatches["position"].add(identity)
        if item.count is not None and record.declared_count != item.count:
            mismatches["declared count"].add(identity)
        if (
            len(record.shape) != len(item.shape)
            or any(extent < 1 for extent in record.shape)
            or math.prod(record.shape) < record.declared_count
            or any(
                expected != -1 and expected != actual
                for expected, actual in zip(item.shape, record.shape, strict=True)
            )
        ):
            mismatches["shape"].add(identity)
    for label, mismatch in mismatches.items():
        if mismatch:
            raise _print_tensor_decode_error(
                f"unexpected {label} records {_sort_print_tensor_identities(mismatch)}"
            )

    return [
        (expected_metadata[record.call], record, record_values[id(record)])
        for record in records
    ]


def _sort_print_tensor_identities(
    identities: Iterable[tuple[int, int, int | None]],
) -> list[tuple[int, int, int | None]]:
    return sorted(
        identities,
        key=lambda identity: (
            identity[0],
            identity[1],
            -1 if identity[2] is None else identity[2],
        ),
    )


def _format_print_tensor_record(
    values: Sequence[float | int],
    *,
    shape: Sequence[int],
    dtype: str = "f32",
    call: int | None = None,
    block: int | None = None,
    position: str | None = None,
    subblock: int | None = None,
) -> str:
    spec = _PRINT_TENSOR_DTYPES.get(dtype)
    if spec is None:
        raise TlaExecutionError(
            f"tla.print tensor formatting failed (dtype={dtype}): unsupported dtype"
        )
    count = len(values)
    rendered_shape = ",".join(str(int(extent)) for extent in shape)
    if spec.numeric_kind == "float":
        rendered_values = ", ".join(str(float(value)) for value in values)
    else:
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise TlaExecutionError(
                f"tla.print tensor formatting failed (dtype={dtype}): "
                "integer dtype requires integer values"
            )
        minimum, maximum = _print_tensor_integer_bounds(dtype, spec)
        if any(value < minimum or value > maximum for value in values):
            raise TlaExecutionError(
                f"tla.print tensor formatting failed (dtype={dtype}): "
                f"integer value outside [{minimum}, {maximum}]"
            )
        rendered_values = ", ".join(str(value) for value in values)
    public_dtype = "float32" if dtype == "f32" else dtype
    identity = "" if call is None else f"call={call} block={block} "
    rendered_position = f"position={position} " if position is not None else ""
    rendered_subblock = f"subblock={subblock} " if subblock is not None else ""
    return (
        f"tla.print {identity}dtype={public_dtype} "
        f"{rendered_position}{rendered_subblock}"
        f"shape=[{rendered_shape}] count={count} "
        f"values=[{rendered_values}]"
    )


def runtime_options_from_kwargs(kwargs: Mapping[str, Any]) -> TlaRuntimeOptions:
    """Build runtime options from Host compile/launch kwargs.

    Public arch selection::

        tla.compile(fn, *args, options="--npu-arch 3510")

    AIC/AIV is not a Host compile knob; it is inferred from lowered MLIR.
    Cache / cache dir / force-recompile / IR dump use env vars
    (``CATLASS_DSL_CACHE``, ``CATLASS_DSL_CACHE_DIR``, ``CATLASS_DSL_FORCE_RECOMPILE``,
    ``CATLASS_DSL_PRINT_IR``, ``CATLASS_DSL_KEEP``, ``CATLASS_DSL_DUMP_DIR``).
    """
    parsed_options = _parse_compile_options_from_str(kwargs.get("options"))
    npu_arch = str(parsed_options.get("npu_arch", DEFAULT_NPU_ARCH))
    target_arch = _resolve_npu_arch(npu_arch)
    # Placeholder until ``_runtime_options_from_lowered_mlir`` reads core type.
    arch_scope = _arch_scope_for_target(target_arch=target_arch, core_type="aiv")
    _, core_type = _parse_arch_scope(arch_scope)
    cache_dir_env = os.getenv("CATLASS_DSL_CACHE_DIR")
    return TlaRuntimeOptions(
        cache_enabled=_env_truthy("CATLASS_DSL_CACHE", default="1"),
        cache_dir=(
            Path(cache_dir_env).expanduser().resolve() if cache_dir_env else None
        ),
        force_recompile=_env_truthy("CATLASS_DSL_FORCE_RECOMPILE", default="0"),
        kernel_mode=core_type,
        arch_scope=arch_scope,
        print_ir=_env_truthy("CATLASS_DSL_PRINT_IR", default="0"),
    )


def runtime_options_for_launch(runtime: TlaRuntimeOptions) -> TlaRuntimeOptions:
    if runtime.cache_dir is not None:
        return runtime
    if runtime.cache_enabled:
        return replace(runtime, cache_dir=_default_cache_dir())
    temp_dir = Path(tempfile.mkdtemp(prefix="tla-dsl-kernel-")).resolve()
    return replace(runtime, cache_enabled=False, cache_dir=temp_dir)


def _runtime_options_from_lowered_mlir(
    runtime: TlaRuntimeOptions,
    mlir_text: str,
    *,
    has_logical_mixed_handoff: bool,
) -> TlaRuntimeOptions:
    target_arch, core_type = _parse_arch_scope(runtime.arch_scope)
    kernel_mode = runtime.kernel_mode

    if has_logical_mixed_handoff:
        core_type = "aic"
        kernel_mode = "mix"
    elif (
        "hivm.module_core_type<AIC>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIC>" in mlir_text
    ):
        core_type = "aic"
        kernel_mode = "aic"
    elif (
        "hivm.module_core_type<AIV>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIV>" in mlir_text
    ):
        core_type = "aiv"
        kernel_mode = "aiv"

    if "dav-c310" in mlir_text or 'hacc.target<"Ascend950PR_9589">' in mlir_text:
        target_arch = "c310"

    arch_scope = _arch_scope_for_target(target_arch=target_arch, core_type=core_type)
    if runtime.kernel_mode == kernel_mode and runtime.arch_scope == arch_scope:
        return runtime
    return replace(
        runtime,
        kernel_mode=kernel_mode,
        arch_scope=arch_scope,
    )


def _extract_entrypoint(mlir_text: str) -> str:
    match = re.search(r"@([A-Za-z_][A-Za-z0-9_]*)\s*\(", mlir_text)
    if match:
        return match.group(1)
    sym_match = re.search(r'sym_name\s*=\s*"([A-Za-z_][A-Za-z0-9_]*)"', mlir_text)
    if sym_match:
        return sym_match.group(1)
    raise TlaExecutionError("Could not infer kernel entrypoint from lowered MLIR.")


def _cache_key(
    *,
    tlair_mlir: str,
    entrypoint: str,
    runtime: TlaRuntimeOptions,
    compiler_bridge_path: Path | None,
    hivmc: Path,
    target: TlaKernelTarget,
) -> str:
    key_payload = {
        "debug_print_workspace_abi_revision": _DEBUG_PRINT_WORKSPACE_ABI_REVISION,
        "print_tensor_workspace_abi_revision": _PRINT_TENSOR_WORKSPACE_ABI_REVISION,
        "cache_abi_version": _ONLINE_CACHE_ABI_VERSION,
        "entrypoint": entrypoint,
        "kernel_mode": runtime.kernel_mode,
        "arch_scope": runtime.arch_scope,
        "cce_arch": target.cce_arch,
        "compiler_bridge": str(compiler_bridge_path) if compiler_bridge_path else None,
        "hivmc": str(hivmc),
        "compiler_bridge_fingerprint": _tool_fingerprint(compiler_bridge_path),
        "hivmc_version": _tool_version(hivmc),
        "hivmc_fingerprint": _tool_fingerprint(hivmc),
        "mlir": tlair_mlir,
        "print_ir": runtime.print_ir,
    }
    return hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _tool_version(binary: Path) -> str:
    try:
        proc = subprocess.run(
            [str(binary), "--version"], check=False, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    text = (proc.stdout or "") + (proc.stderr or "")
    text = text.strip()
    if not text:
        return f"exit:{proc.returncode}"
    return text.splitlines()[0][:200]


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _tool_fingerprint(binary: Path | None) -> str:
    if binary is None:
        return "unresolved"
    try:
        stat = binary.stat()
    except OSError:
        return "missing"
    digest = hashlib.sha256()
    try:
        with binary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return f"stat:{stat.st_size}:{stat.st_mtime_ns}"
    return f"{stat.st_size}:{stat.st_mtime_ns}:{digest.hexdigest()}"


def _default_cache_dir() -> Path:
    cache = os.getenv("CATLASS_DSL_CACHE_DIR")
    if cache:
        return Path(cache).expanduser().resolve()
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return (Path(xdg) / "catlass").expanduser().resolve()
    return (Path.home() / ".cache" / "catlass").resolve()


def _write_pass_ir_dump(pass_dump_path: Path, text: str) -> None:
    """Write pass IR dump to the cache path; mirror to DUMP_DIR when set."""
    pass_dump_path.write_text(text)
    dump_dir_env = os.getenv("CATLASS_DSL_DUMP_DIR")
    if not dump_dir_env:
        return
    dump_dir = Path(dump_dir_env).expanduser().resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / pass_dump_path.name).write_text(text)


def _parse_keep_tokens(raw: str) -> frozenset[str]:
    """Parse ``CATLASS_DSL_KEEP`` into canonical artifact tokens.

    Tokens (comma-separated, case-insensitive):
      ir        — lowered MLIR after the Ascend pipeline (``*.mlir``)
      ir-debug  — TLA IR before HIVM lowering (``*.tlair.mlir``); also
                  ``*.pass-ir-dump.mlir`` when pass dump text is available
      kernel    — device object ``*.o``
      all       — expand to every token above

    Unknown tokens are ignored.
    """
    tokens = frozenset(t.strip().lower() for t in raw.split(",") if t.strip())
    if "all" in tokens:
        return _KEEP_ALL_TOKENS
    return tokens & _KEEP_ALL_TOKENS


def _copy_kept_artifacts(artifact: TlaKernelArtifact) -> None:
    """Copy selected compile artifacts to DUMP_DIR (or CWD) when KEEP is set."""
    tokens = _parse_keep_tokens(os.getenv("CATLASS_DSL_KEEP", ""))
    if not tokens:
        return
    dump_dir_env = os.getenv("CATLASS_DSL_DUMP_DIR")
    dump_dir = (
        Path(dump_dir_env).expanduser().resolve()
        if dump_dir_env
        else Path.cwd().resolve()
    )
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact.entrypoint or artifact.cache_key
    if "ir" in tokens:
        (dump_dir / f"{stem}.mlir").write_text(artifact.lowered_llvm)
    if "ir-debug" in tokens:
        (dump_dir / f"{stem}.tlair.mlir").write_text(artifact.tlair_mlir)
        if artifact.pass_ir_dump:
            (dump_dir / f"{stem}.pass-ir-dump.mlir").write_text(artifact.pass_ir_dump)
    if "kernel" in tokens and artifact.kernel_binary_path.exists():
        shutil.copy2(artifact.kernel_binary_path, dump_dir / f"{stem}.o")


def _parse_arch_scope(arch_scope: str) -> tuple[str, str]:
    try:
        return _parse_arch_scope_impl(arch_scope)
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _core_type_from_arch_scope(arch_scope: str) -> str:
    return _parse_arch_scope(arch_scope)[1]


def _arch_scope_for_target(*, target_arch: str, core_type: str) -> str:
    try:
        return _arch_scope_for_target_impl(target_arch=target_arch, core_type=core_type)
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _resolve_kernel_target(runtime: TlaRuntimeOptions) -> TlaKernelTarget:
    target_arch, core_type = _parse_arch_scope(runtime.arch_scope)
    try:
        return _get_kernel_target(
            target_arch=target_arch,
            core_type=core_type,
            arch_scope=runtime.arch_scope,
        )
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _resolve_hivmc_a5() -> Path:
    """Resolve ``hivmc-a5`` from PATH / ``ASCEND_HOME_PATH`` after ``set_env.sh``."""

    which = shutil.which("hivmc-a5")
    if which:
        return Path(which).resolve()
    ascend_home = os.getenv("ASCEND_HOME_PATH")
    if ascend_home:
        candidate = Path(ascend_home).expanduser().resolve() / "bin" / "hivmc-a5"
        if candidate.exists():
            return candidate
    raise TlaBackendCompilerNotFoundError(
        "hivmc-a5 not found on PATH. Source the CANN toolkit "
        "(`source .../ascend-toolkit/set_env.sh`) so `hivmc-a5` is available."
    )


def _build_hivmc_a5_command(
    *,
    compiler: Path,
    mlir_path: Path,
    kernel_path: Path,
    runtime: TlaRuntimeOptions,
    template_bitcode: str | None = None,
) -> list[str]:
    if template_bitcode is None:
        template_bitcode = _resolve_hivm_template_bitcode(runtime)
    command = [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
    ]
    if runtime.kernel_mode == "mix":
        command.extend(
            [
                "--disable-ffts",
                f"--link-aicore-bitcode={template_bitcode}",
                "-o",
                str(kernel_path),
            ]
        )
    else:
        command.extend(
            [
                "--disable-ffts",
                "--enable-hivm-compile=False",
                f"--link-aicore-bitcode={template_bitcode}",
                "-o",
                str(kernel_path),
            ]
        )
    return command


def _create_stamped_hivmc_input(
    mlir_path: Path, runtime: TlaRuntimeOptions
) -> tuple[Path, str | None]:
    """Stamp a private HIVMC input only when debug-print helpers are present.

    Ordinary kernels rely on ``--link-aicore-bitcode`` alone. Debug /
    ``print_tensor`` helpers also need module attrs ``hivm.aiv_bitcode`` /
    ``hivm.aic_bitcode`` (and optionally helper bitcode), so copy+stamp a
    private ``*.hivmc-input.mlir`` in those cases only.
    """
    compiler_text = mlir_path.read_text()
    if (
        "tla.debug_print.workspace" not in compiler_text
        and "tla.print_tensor.workspace" not in compiler_text
    ):
        return mlir_path, None
    compiler_input = mlir_path.with_name(f"{mlir_path.stem}.hivmc-input.mlir")
    compiler_input.write_text(compiler_text)
    template_bitcode = _resolve_hivm_template_bitcode(runtime)
    if "tla.print_tensor.workspace" in compiler_text:
        helper_core_type = (
            _mixed_print_tensor_helper_core(compiler_text)
            if runtime.kernel_mode == "mix"
            else _core_type_from_arch_scope(runtime.arch_scope)
        )
        helper = {
            "aic": ("Cube", "print_tensor.aic.c310.bc"),
            "aiv": ("Vector", "print_tensor.aiv.c310.bc"),
        }.get(helper_core_type)
        if helper is None:
            raise TlaRuntimeUnavailableError(
                "tla.print_tensor helper split could not be determined"
            )
        helper_dir, helper_name = helper
        probe_candidates = [
            build_dir / "bc" / helper_dir / helper_name
            for build_dir in _mlir_build_dirs()
        ]
        probe_bitcode = next(
            (path.resolve() for path in probe_candidates if path.exists()), None
        )
        if probe_bitcode is None:
            raise TlaRuntimeUnavailableError(
                f"C310 {helper_core_type} print tensor bitcode was not built"
            )
        helper_bytes = probe_bitcode.read_bytes()
        if _PRINT_TENSOR_HELPER_ABI_MARKER not in helper_bytes:
            raise TlaRuntimeUnavailableError(
                "C310 print tensor helper does not provide the required ABI marker"
            )
        template_bitcode = f"{template_bitcode},{probe_bitcode}"
    _stamp_hivm_template_bitcode_attrs(compiler_input, template_bitcode)
    return compiler_input, template_bitcode


def _mixed_print_tensor_helper_core(compiler_text: str) -> str:
    helper_call = re.search(r"\bcall\s+@_mlir_ciface_tla_print_tensor_", compiler_text)
    if helper_call is None:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper call is missing from lowered MLIR"
        )
    functions = list(
        re.finditer(
            r"func\.func(?:\s+private)?\s+@(?P<name>[^\s(]+)\s*\(",
            compiler_text,
        )
    )
    containing_function = [
        match for match in functions if match.start() < helper_call.start()
    ]
    if not containing_function:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper split could not be determined"
        )
    split = re.search(r"_mix_(aic|aiv)$", containing_function[-1].group("name"))
    if split is None:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper call is not inside a split function"
        )
    return split.group(1)


def _stamp_hivm_template_bitcode_attrs(mlir_path: Path, template_bitcode: str) -> None:
    text = mlir_path.read_text()
    additions: list[str] = []
    added_attr_names: set[str] = set()
    for raw_path in template_bitcode.split(","):
        path = Path(raw_path)
        attr = _HIVM_TEMPLATE_BITCODE_ATTRS.get(path.name)
        if attr is None or attr[0] in text or attr[0] in added_attr_names:
            continue
        added_attr_names.add(attr[0])
        additions.append(
            f"{attr[0]} = #hivm.{attr[1].removeprefix('hivm.')}<{json.dumps(str(path))}>"
        )
    if not additions:
        return
    add_text = ", ".join(additions)
    attributes_re = re.compile(
        r"(?P<prefix>\bmodule(?:\s+@[^\s{]+)?\s+attributes\s*\{)"
        r"(?P<body>.*?)"
        r"(?P<suffix>\}\s*\{)",
        re.DOTALL,
    )
    match = attributes_re.search(text)
    if match:
        body = match.group("body").strip()
        new_body = f"{add_text}, {body}" if body else add_text
        mlir_path.write_text(
            text[: match.start()]
            + match.group("prefix")
            + new_body
            + match.group("suffix")
            + text[match.end() :]
        )
        return

    match = re.search(r"(?P<prefix>\bmodule(?:\s+@[^\s{]+)?)(?P<brace>\s*\{)", text)
    if match:
        mlir_path.write_text(
            text[: match.start()]
            + match.group("prefix")
            + f" attributes {{{add_text}}}"
            + match.group("brace")
            + text[match.end() :]
        )


def _mlir_build_dirs() -> list[Path]:
    # .../python/tla_dsl/catlass/execution.py -> .../python/tla_dsl/csrc/mlir/build
    dsl_root = Path(__file__).resolve().parents[1]
    nested = dsl_root / "csrc" / "mlir" / "build"
    # Legacy ascend-catlass-DSL: .../python/tla_dsl/execution.py -> repo/mlir/build
    legacy = Path(__file__).resolve().parents[2] / "mlir" / "build"
    return [nested, legacy]


def _resolve_hivm_template_bitcode(runtime: TlaRuntimeOptions) -> str:
    candidates: list[Path] = []
    if runtime.kernel_mode == "mix":
        repo_aic_candidates: list[Path] = []
        aiv_candidates: list[Path] = []
        for build_dir in _mlir_build_dirs():
            repo_aic_candidates.append(build_dir / "bc" / "meta_op.aic.c310.bc")
            aiv_candidates.append(build_dir / "bc" / "meta_op.aiv.c310.bc")
        repo_aic = next(
            (path.resolve() for path in repo_aic_candidates if path.exists()), None
        )
        aiv_bc = next(
            (path.resolve() for path in aiv_candidates if path.exists()), None
        )
        if repo_aic is not None and aiv_bc is not None:
            return f"{repo_aic},{aiv_bc}"
        raise TlaRuntimeUnavailableError(
            "C310 mix HIVM bitcode not found. Expected DSL-built "
            "meta_op.aic.c310.bc and meta_op.aiv.c310.bc under the mlir build tree."
        )

    if _core_type_from_arch_scope(runtime.arch_scope) == "aic":
        for build_dir in _mlir_build_dirs():
            candidates.extend(
                [
                    build_dir / "meta_op.aic.c310.bc",
                    build_dir / "bc" / "meta_op.aic.c310.bc",
                ]
            )
    else:
        for build_dir in _mlir_build_dirs():
            candidates.append(build_dir / "bc" / "meta_op.aiv.c310.bc")
    existing = next((path.resolve() for path in candidates if path.exists()), None)
    if existing is not None:
        return str(existing)
    raise TlaRuntimeUnavailableError(
        "C310 HIVM bitcode not found. Build Tla DSL templates "
        "(meta_op.aic.c310.bc / meta_op.aiv.c310.bc) under the mlir build tree."
    )


def _run_checked(
    command: list[str], *, label: str, cwd: Path, stdin_text: str | None = None
) -> None:
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            input=stdin_text,
        )
    except subprocess.CalledProcessError as exc:
        raise TlaKernelCompileError(
            f"{label} failed with exit code {exc.returncode}\n"
            f"cmd: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc


def _run_tla_lowering_to_mlir(
    *,
    lowered_module: Any | None,
    tlair_mlir: str,
    mlir_path: Path,
    runtime: TlaRuntimeOptions | None = None,
) -> Any:
    try:
        return _run_typed_bridge_to_mlir(
            lowered_module=lowered_module,
            mlir_path=mlir_path,
            runtime=runtime,
        )
    except (TlaCompilerBridgeUnavailableError, TlaKernelCompileError):
        tla_compile = _resolve_tla_compile()
        if tla_compile is None:
            raise
        return _run_tla_compile_cli_to_mlir(
            tla_compile=tla_compile,
            tlair_mlir=tlair_mlir,
            mlir_path=mlir_path,
            runtime=runtime,
        )


def _run_typed_bridge_to_mlir(
    *,
    lowered_module: Any | None,
    mlir_path: Path,
    runtime: TlaRuntimeOptions | None = None,
) -> Any:
    if lowered_module is None:
        raise TlaCompilerBridgeUnavailableError(
            "Python runtime compilation requires a live MLIR module. "
            "String TLA MLIR lowering is not supported."
        )
    try:
        # Host surface is a single PRINT_IR switch; bridge still takes
        # pass-print flags — enable full pipeline dump when requested.
        print_ir = bool(runtime.print_ir) if runtime else False
        result = lower_tlair_module_to_mlir(
            lowered_module,
            mlir_print_ir_before=(),
            mlir_print_ir_after=(),
            mlir_print_ir_before_all=print_ir,
            mlir_print_ir_after_all=print_ir,
        )
    except BridgeUnavailableError as exc:
        raise TlaCompilerBridgeUnavailableError(str(exc)) from exc
    except BridgeLoweringError as exc:
        raise TlaKernelCompileError(
            f"In-process Tla compiler bridge failed.\nerror:\n{exc}",
            pass_ir_dump=exc.pass_ir_dump,
        ) from exc
    except Exception as exc:
        raise TlaKernelCompileError(
            f"In-process Tla compiler bridge failed.\nerror:\n{exc}"
        ) from exc
    mlir_path.write_text(result.lowered_mlir)
    return result


def _resolve_tla_compile() -> Path | None:
    candidates: list[Path] = []
    for build_dir in _mlir_build_dirs():
        candidates.extend(
            [
                build_dir / "TlaCompile",
                build_dir / "tools" / "tla-compile" / "TlaCompile",
            ]
        )
    which = shutil.which("TlaCompile")
    if which:
        candidates.append(Path(which).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _tla_compile_env() -> dict[str, str]:
    return os.environ.copy()


def _run_tla_compile_cli_to_mlir(
    *,
    tla_compile: Path,
    tlair_mlir: str,
    mlir_path: Path,
    runtime: TlaRuntimeOptions | None = None,
) -> Any:
    input_path = mlir_path.with_suffix(".tlair.mlir")
    input_path.write_text(tlair_mlir)
    cmd = [str(tla_compile), str(input_path), "-o", str(mlir_path)]
    print_requested = bool(runtime is not None and runtime.print_ir)
    if print_requested:
        cmd.append("--mlir-print-ir-before-all")
        cmd.append("--mlir-print-ir-after-all")
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=_tla_compile_env(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        stderr_message = "<captured in pass IR dump>" if print_requested else stderr
        raise TlaKernelCompileError(
            f"TlaCompile CLI fallback failed with exit code {exc.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{stderr_message}",
            pass_ir_dump=stderr if print_requested else "",
        ) from exc
    if not mlir_path.exists():
        raise TlaKernelCompileError(
            "TlaCompile CLI fallback completed but did not produce lowered MLIR at "
            f"{mlir_path}"
        )
    return type(
        "TlaLoweringResult",
        (),
        {
            "lowered_mlir": mlir_path.read_text(),
            "pass_ir_dump": completed.stderr if print_requested else "",
        },
    )()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise TlaExecutionError(f"Invalid cache manifest at {path}: {exc}") from exc


def _cache_manifest_has_current_debug_print_workspace_abi(
    manifest: Mapping[str, Any],
) -> bool:
    return (
        manifest.get("debug_print_workspace_abi_revision")
        == _DEBUG_PRINT_WORKSPACE_ABI_REVISION
    )


def _cache_manifest_has_current_print_tensor_workspace_abi(
    manifest: Mapping[str, Any],
) -> bool:
    return (
        manifest.get("print_tensor_workspace_abi_revision")
        == _PRINT_TENSOR_WORKSPACE_ABI_REVISION
    )


def _cache_manifest_has_current_workspace_abis(
    manifest: Mapping[str, Any],
) -> bool:
    return _cache_manifest_has_current_debug_print_workspace_abi(
        manifest
    ) and _cache_manifest_has_current_print_tensor_workspace_abi(manifest)


def _prepared_kernel_abi_from_manifest(
    manifest: Mapping[str, Any], *, expected_entrypoint: str | None = None
) -> tuple[KernelAbiLayout, _PreparedAbiPacker]:
    if "kernel_abi" not in manifest or manifest["kernel_abi"] is None:
        raise TlaKernelCompileError(
            "Cached artifact has no compiler-produced kernel_abi descriptor; "
            "the cache predates the current kernel ABI contract."
        )
    try:
        layout = kernel_abi_from_dict(manifest["kernel_abi"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TlaKernelCompileError(
            f"Invalid cached kernel ABI descriptor: {exc}"
        ) from exc
    if layout is None:
        raise TlaKernelCompileError(
            "Invalid cached kernel ABI descriptor: decoded to no layout"
        )
    try:
        packer = _prepare_abi_packer(
            layout,
            expected_entrypoint=(
                expected_entrypoint
                if expected_entrypoint is not None
                else str(manifest.get("entrypoint", ""))
            ),
        )
    except TlaUnsupportedAbiError as exc:
        raise TlaKernelCompileError(
            f"Invalid cached kernel ABI descriptor: {exc}"
        ) from exc
    return layout, packer


def _kernel_abi_from_manifest(
    manifest: Mapping[str, Any], *, expected_entrypoint: str | None = None
) -> KernelAbiLayout:
    layout, _ = _prepared_kernel_abi_from_manifest(
        manifest, expected_entrypoint=expected_entrypoint
    )
    return layout


def _env_truthy(name: str, *, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


from .catlass_dsl.ascend_jit_executor import (
    execute_kernel,
    _build_kernel_launch_plan,
    _mark_tensor_launch_args_uploaded,
)
from .base_dsl.runtime.ascend import (
    launch_kernel,
    load_acl,
    load_binary,
)
