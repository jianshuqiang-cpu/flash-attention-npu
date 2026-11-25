"""Runtime type helpers and Ascend launch loader."""

from __future__ import annotations

from typing import Any

from .dlpack_types import (
    ASCEND_DEVICE_TYPES,
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLTensor,
)

__all__ = [
    "ASCEND_DEVICE_TYPES",
    "DLDataType",
    "DLDataTypeCode",
    "DLDevice",
    "DLDeviceType",
    "DLTensor",
    "launch_kernel",
    "load_binary",
]


def __getattr__(name: str) -> Any:
    if name in {"launch_kernel", "load_binary"}:
        from . import ascend as ascend_mod

        return getattr(ascend_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
