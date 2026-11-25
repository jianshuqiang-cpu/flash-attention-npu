"""Ascend Catlass package root.

Preferred DSL import::

    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for source tree
    __version__ = "0.0.0"

# Submodules (import paths), not a flat DSL namespace.
from . import tla
from . import types
from . import runtime
from . import dsl
from . import core_api as core
from . import params
from .address_space import AddressSpace

__all__ = [
    "__version__",
    "tla",
    "types",
    "runtime",
    "dsl",
    "core",
    "params",
    "AddressSpace",
]
