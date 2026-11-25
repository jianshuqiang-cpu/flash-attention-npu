"""Typed helpers for running the Tla pass pipeline from Python MLIR modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, TYPE_CHECKING, Mapping, Sequence

from . import _tla_type_bridge

if TYPE_CHECKING:
    from mlir import ir as mlir_ir


class BridgeUnavailableError(RuntimeError):
    """Raised when the in-process compile bridge is unavailable."""


class BridgeLoweringError(RuntimeError):
    """Raised when the in-process bridge pipeline fails after producing IR dumps."""

    def __init__(self, message: str, *, pass_ir_dump: str = "") -> None:
        super().__init__(message)
        self.pass_ir_dump = pass_ir_dump


class KernelAbiArgumentKind(str, Enum):
    POINTER = "pointer"
    SCALAR = "scalar"
    MEMREF_FIELD = "memref_field"


class KernelAbiScalarCategory(str, Enum):
    INTEGER = "integer"
    INDEX = "index"
    FLOAT = "float"


class KernelAbiIntegerSignedness(str, Enum):
    SIGNLESS = "signless"
    SIGNED = "signed"
    UNSIGNED = "unsigned"


class KernelAbiFloatFormat(str, Enum):
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"


@dataclass(frozen=True)
class KernelAbiScalarDescriptor:
    category: KernelAbiScalarCategory
    bit_width: int
    integer_signedness: KernelAbiIntegerSignedness | None
    float_format: KernelAbiFloatFormat | None

    def __post_init__(self) -> None:
        if not isinstance(self.category, KernelAbiScalarCategory):
            raise ValueError("kernel ABI scalar descriptor has an invalid category")
        if self.integer_signedness is not None and not isinstance(
            self.integer_signedness, KernelAbiIntegerSignedness
        ):
            raise ValueError(
                "kernel ABI scalar descriptor has invalid integer signedness"
            )
        if self.float_format is not None and not isinstance(
            self.float_format, KernelAbiFloatFormat
        ):
            raise ValueError("kernel ABI scalar descriptor has an invalid float format")
        if self.category is KernelAbiScalarCategory.INTEGER:
            if self.bit_width not in (1, 8, 16, 32, 64):
                raise ValueError("kernel ABI integer scalar has unsupported bit width")
            if self.integer_signedness is None or self.float_format is not None:
                raise ValueError("kernel ABI integer scalar descriptor is incoherent")
        elif self.category is KernelAbiScalarCategory.INDEX:
            if (
                self.bit_width != 64
                or self.integer_signedness is not None
                or self.float_format is not None
            ):
                raise ValueError("kernel ABI index scalar descriptor is incoherent")
        elif self.category is KernelAbiScalarCategory.FLOAT:
            expected_width = {
                KernelAbiFloatFormat.F16: 16,
                KernelAbiFloatFormat.BF16: 16,
                KernelAbiFloatFormat.F32: 32,
            }.get(self.float_format)
            if expected_width != self.bit_width or self.integer_signedness is not None:
                raise ValueError("kernel ABI float scalar descriptor is incoherent")


@dataclass(frozen=True)
class KernelAbiArgument:
    index: int
    kind: KernelAbiArgumentKind
    scalar: KernelAbiScalarDescriptor | None
    mlir_type: str
    offset: int
    storage_size: int
    alignment: int
    logical_index: int | None = None
    field: str | None = None


@dataclass(frozen=True)
class KernelAbiLayout:
    schema_version: int
    entrypoint: str
    total_size: int
    arguments: tuple[KernelAbiArgument, ...]


def kernel_abi_from_dict(value: Mapping[str, Any] | None) -> KernelAbiLayout | None:
    if value is None:
        return None
    schema_version = int(value.get("schema_version", -1))
    if schema_version not in (3, 4):
        raise ValueError("Unsupported kernel ABI schema version.")
    arguments: list[KernelAbiArgument] = []
    for arg in value["arguments"]:
        try:
            kind = KernelAbiArgumentKind(str(arg["kind"]))
        except ValueError as exc:
            raise ValueError("Unsupported kernel ABI argument kind.") from exc
        scalar_value = arg.get("scalar")
        scalar = None
        if scalar_value is not None:
            try:
                scalar = KernelAbiScalarDescriptor(
                    category=KernelAbiScalarCategory(str(scalar_value["category"])),
                    bit_width=int(scalar_value["bit_width"]),
                    integer_signedness=(
                        None
                        if scalar_value.get("integer_signedness") is None
                        else KernelAbiIntegerSignedness(
                            str(scalar_value["integer_signedness"])
                        )
                    ),
                    float_format=(
                        None
                        if scalar_value.get("float_format") is None
                        else KernelAbiFloatFormat(str(scalar_value["float_format"]))
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid kernel ABI scalar descriptor.") from exc
        if kind is KernelAbiArgumentKind.POINTER and scalar is not None:
            raise ValueError(
                "Kernel ABI pointer argument cannot have a scalar descriptor."
            )
        if kind is KernelAbiArgumentKind.SCALAR and scalar is None:
            raise ValueError("Kernel ABI scalar argument requires a scalar descriptor.")
        if kind is KernelAbiArgumentKind.MEMREF_FIELD and scalar is not None:
            raise ValueError(
                "Kernel ABI memref_field argument cannot have a scalar descriptor."
            )
        storage_size = int(arg["storage_size"])
        if kind is KernelAbiArgumentKind.POINTER and storage_size != 8:
            raise ValueError("Kernel ABI pointer argument has invalid storage size.")
        if kind is KernelAbiArgumentKind.MEMREF_FIELD and storage_size != 8:
            raise ValueError(
                "Kernel ABI memref_field argument has invalid storage size."
            )
        field_value = arg.get("field")
        if kind is KernelAbiArgumentKind.MEMREF_FIELD:
            if field_value is None:
                raise ValueError("Kernel ABI memref_field requires a field name.")
            field = str(field_value)
        else:
            field = None
        if scalar is not None:
            expected_size = (
                8
                if scalar.category is KernelAbiScalarCategory.INDEX
                else max(1, scalar.bit_width // 8)
            )
            if storage_size != expected_size:
                raise ValueError(
                    "Kernel ABI scalar descriptor and storage size are inconsistent."
                )
        abi_index = int(arg["index"])
        logical_raw = arg.get("logical_index")
        logical_index = abi_index if logical_raw is None else int(logical_raw)
        arguments.append(
            KernelAbiArgument(
                index=abi_index,
                kind=kind,
                scalar=scalar,
                mlir_type=str(arg["mlir_type"]),
                offset=int(arg["offset"]),
                storage_size=storage_size,
                alignment=int(arg["alignment"]),
                logical_index=logical_index,
                field=field,
            )
        )
    return KernelAbiLayout(
        schema_version=schema_version,
        entrypoint=str(value["entrypoint"]),
        total_size=int(value["total_size"]),
        arguments=tuple(arguments),
    )


def kernel_abi_to_dict(layout: KernelAbiLayout | None) -> dict[str, Any] | None:
    if layout is None:
        return None
    return {
        "schema_version": layout.schema_version,
        "entrypoint": layout.entrypoint,
        "total_size": layout.total_size,
        "arguments": [
            {
                "index": arg.index,
                "logical_index": (
                    arg.index if arg.logical_index is None else arg.logical_index
                ),
                "kind": arg.kind.value,
                "field": arg.field,
                "scalar": (
                    None
                    if arg.scalar is None
                    else {
                        "category": arg.scalar.category.value,
                        "bit_width": arg.scalar.bit_width,
                        "integer_signedness": (
                            None
                            if arg.scalar.integer_signedness is None
                            else arg.scalar.integer_signedness.value
                        ),
                        "float_format": (
                            None
                            if arg.scalar.float_format is None
                            else arg.scalar.float_format.value
                        ),
                    }
                ),
                "mlir_type": arg.mlir_type,
                "offset": arg.offset,
                "storage_size": arg.storage_size,
                "alignment": arg.alignment,
            }
            for arg in layout.arguments
        ],
    }


@dataclass(frozen=True)
class TlaLoweringResult:
    """Result of lowering a live ``tla``-dialect MLIR module through the typed bridge."""

    lowered_mlir: str
    pass_ir_dump: str = ""
    kernel_abi: KernelAbiLayout | None = None


def lower_tlair_module_to_mlir(
    module: mlir_ir.Module,
    *,
    mlir_print_ir_before: Sequence[str] = (),
    mlir_print_ir_after: Sequence[str] = (),
    mlir_print_ir_before_all: bool = False,
    mlir_print_ir_after_all: bool = False,
) -> TlaLoweringResult:
    """Lower an existing MLIR module through the Tla pipeline.

    Pass-print arguments intentionally mirror the TlaCompile flag names while
    staying structured Python args instead of CLI strings.
    """

    ext = _load_bridge_extension()
    result = ext.lower_to_mlir(
        module,
        list(mlir_print_ir_before),
        list(mlir_print_ir_after),
        bool(mlir_print_ir_before_all),
        bool(mlir_print_ir_after_all),
    )
    pass_ir_dump = str(result.get("pass_ir_dump", ""))
    if result.get("success", True) is False:
        raise BridgeLoweringError(
            str(result.get("error", "Failed to run Tla pipeline.")),
            pass_ir_dump=pass_ir_dump,
        )
    return TlaLoweringResult(
        lowered_mlir=str(result["lowered_mlir"]),
        pass_ir_dump=pass_ir_dump,
        kernel_abi=kernel_abi_from_dict(result.get("kernel_abi")),
    )


def _load_bridge_extension() -> ModuleType:
    try:
        return _tla_type_bridge._load_bridge_extension()
    except _tla_type_bridge.TlaTypeBridgeUnavailableError as exc:
        raise BridgeUnavailableError(str(exc)) from exc


def resolve_bridge_extension_path() -> Path | None:
    return _tla_type_bridge._resolve_bridge_extension_path()


__all__ = [
    "BridgeLoweringError",
    "BridgeUnavailableError",
    "KernelAbiArgumentKind",
    "KernelAbiArgument",
    "KernelAbiFloatFormat",
    "KernelAbiIntegerSignedness",
    "KernelAbiLayout",
    "KernelAbiScalarCategory",
    "KernelAbiScalarDescriptor",
    "TlaLoweringResult",
    "kernel_abi_from_dict",
    "kernel_abi_to_dict",
    "lower_tlair_module_to_mlir",
    "resolve_bridge_extension_path",
]
