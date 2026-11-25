"""Address-space enum and **architecture** tokens for ``!tla.ptr`` pointers.

:func:`~catlass.core_api.make_ptr` and :func:`~catlass.core_api.allocate` use
:class:`AddressSpace` to select the pointer memory space.

Pointer addrspace uses :class:`AddressSpace` only, as a :class:`~enum.IntEnum`; MLIR asm
keywords match **member names** (``generic``, ``gm``, ``l0c``, …) and are produced via
:meth:`AddressSpace.__str__`; parsing uses :meth:`AddressSpace.from_mlir_token`.
"""

from __future__ import annotations

import enum


class AddressSpace(enum.IntEnum):
    """Tla **pointer** address spaces; values match ``Tla_AddressSpaceEnum`` in ``Tla.td``.

    Member names are the ``!tla.ptr`` / ``stringifyAddressSpace`` keywords (slot ``0`` is ``generic``).
    """

    generic = 0
    gm = 1
    l1 = 2
    l0a = 3
    l0b = 4
    l0c = 5
    ub = 6

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_mlir_token(cls, s: str) -> AddressSpace:
        """Parse MLIR / ``!tla.ptr`` addrspace keyword (e.g. ``l0c``)."""
        t = str(s).strip().lower()
        try:
            return cls[t]
        except KeyError as exc:
            raise ValueError(f"unknown address space arch token: {s!r}") from exc


__all__ = [
    "AddressSpace",
]
