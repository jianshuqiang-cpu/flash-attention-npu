"""Tla device DSL launch layer."""

from __future__ import annotations

from typing import Any

__all__ = ["KernelLauncher"]


def __getattr__(name: str) -> Any:
    if name == "KernelLauncher":
        from .tla import KernelLauncher

        return KernelLauncher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
