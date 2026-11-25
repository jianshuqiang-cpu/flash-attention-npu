"""Tla-specific public-op semantics used by the top-level API facade."""

from __future__ import annotations

from typing import Any, Callable


def lower_copy(
    *,
    dst: Any,
    src: Any,
    loc: Any,
    copy_fn: Callable[..., Any],
) -> None:
    """Lower the public ``tla.copy`` facade to the Tla copy op."""

    copy_fn(dst, src, loc=loc)


__all__ = ["lower_copy"]
