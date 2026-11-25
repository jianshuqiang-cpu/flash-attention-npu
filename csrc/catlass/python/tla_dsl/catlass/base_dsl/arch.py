"""Architecture metadata shared by Tla execution and allocator code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os


class Arch(str, Enum):
    """Internal toolchain arch id (hivmc ``dav-c310-*`` / bitcode layout).

    Public Host compile uses the Ascend chip name via ``options="--npu-arch 3510"``;
    ``3510`` maps here (see :func:`resolve_npu_arch`).
    """

    C310 = "c310"

    @classmethod
    def from_string(cls, value: str | "Arch") -> "Arch":
        if isinstance(value, cls):
            return value
        token = str(value).strip().lower()
        try:
            return cls(token)
        except ValueError:
            pass
        try:
            return cls(resolve_npu_arch(token))
        except ValueError as exc:
            supported = ", ".join(arch.value for arch in cls)
            raise ValueError(
                f"Unsupported target architecture {value!r}. Supported: {supported}."
            ) from exc


# Public ``--npu-arch`` token → internal Arch.value used by KERNEL_TARGETS / cce_arch.
_NPU_ARCH_TO_TARGET: dict[str, str] = {
    "3510": "c310",
}
DEFAULT_NPU_ARCH = os.environ.get("CATLASS_NPU_ARCH", "3510")


def resolve_npu_arch(npu_arch: str) -> str:
    """Map public ``--npu-arch`` (e.g. ``3510``) to internal target arch (``c310``)."""
    normalized = str(npu_arch).strip().lower()
    target = _NPU_ARCH_TO_TARGET.get(normalized)
    if target is None:
        supported = ", ".join(sorted(_NPU_ARCH_TO_TARGET) or ["3510"])
        raise ValueError(
            f"Unsupported --npu-arch={npu_arch!r}. Supported: {supported}."
        )
    return target


@dataclass(frozen=True)
class TlaKernelTarget:
    arch_scope: str
    target_arch: str
    core_type: str
    cce_arch: str


_LOCALMEM_CAPACITY_BYTES_c310 = {
    "L1": 512 * 1024,
    "L0A": 64 * 1024,
    "L0B": 64 * 1024,
    "L0C": 256 * 1024,
    "UB": 248 * 1024,
    "BIASBUF": 4 * 1024,
    "FIXBUF": 4 * 1024,
}

LOCALMEM_CAPACITY_BYTES: dict[Arch, dict[str, int]] = {
    Arch.C310: dict(_LOCALMEM_CAPACITY_BYTES_c310),
}

KERNEL_TARGETS: dict[tuple[Arch, str], TlaKernelTarget] = {
    (Arch.C310, "aiv"): TlaKernelTarget(
        arch_scope="aiv.c310",
        target_arch="c310",
        core_type="aiv",
        cce_arch="dav-c310-vec",
    ),
    (Arch.C310, "aic"): TlaKernelTarget(
        arch_scope="aic.c310",
        target_arch="c310",
        core_type="aic",
        cce_arch="dav-c310-cube",
    ),
}


def get_current_arch() -> Arch:
    return Arch.from_string(resolve_npu_arch(DEFAULT_NPU_ARCH))


def parse_arch_scope(arch_scope: str) -> tuple[str, str]:
    normalized = str(arch_scope).strip().lower()
    if "." not in normalized:
        raise ValueError(f"Unsupported arch_scope={arch_scope!r}.")
    core_type, target_arch = normalized.split(".", 1)
    target = KERNEL_TARGETS.get((Arch.from_string(target_arch), core_type))
    if target is None:
        supported = ", ".join(
            sorted(target.arch_scope for target in KERNEL_TARGETS.values())
        )
        raise ValueError(
            f"Unsupported arch_scope={arch_scope!r}. Supported: {supported}."
        )
    return target.target_arch, target.core_type


def arch_scope_for_target(*, target_arch: str | Arch, core_type: str) -> str:
    target = KERNEL_TARGETS.get((Arch.from_string(target_arch), str(core_type).lower()))
    if target is None:
        supported = ", ".join(
            sorted(target.arch_scope for target in KERNEL_TARGETS.values())
        )
        raise ValueError(
            f"Unsupported target_arch/core_type combination: {target_arch!r}/{core_type!r}. "
            f"Supported scopes: {supported}."
        )
    return target.arch_scope


def get_kernel_target(
    *, target_arch: str | Arch, core_type: str, arch_scope: str | None = None
) -> TlaKernelTarget:
    target = KERNEL_TARGETS.get((Arch.from_string(target_arch), str(core_type).lower()))
    if target is None:
        supported = ", ".join(
            sorted(target.arch_scope for target in KERNEL_TARGETS.values())
        )
        raise ValueError(
            f"Unsupported target_arch/core_type combination: {target_arch!r}/{core_type!r}. "
            f"Supported scopes: {supported}."
        )
    if arch_scope is not None and target.arch_scope != str(arch_scope).lower():
        raise ValueError(
            "Runtime target configuration is inconsistent: "
            f"target_arch={target_arch!r}, core_type={core_type!r}, arch_scope={arch_scope!r}."
        )
    return target


def get_localmem_capacity_bytes(mem_scope: str, arch: str | Arch | None = None) -> int:
    resolved_arch = get_current_arch() if arch is None else Arch.from_string(arch)
    capacities = LOCALMEM_CAPACITY_BYTES[resolved_arch]
    try:
        return capacities[mem_scope]
    except KeyError as exc:
        supported = ", ".join(sorted(capacities))
        raise ValueError(
            f"Unsupported local-memory scope {mem_scope!r}. Supported: {supported}."
        ) from exc


__all__ = [
    "Arch",
    "DEFAULT_NPU_ARCH",
    "TlaKernelTarget",
    "KERNEL_TARGETS",
    "LOCALMEM_CAPACITY_BYTES",
    "arch_scope_for_target",
    "get_current_arch",
    "get_kernel_target",
    "get_localmem_capacity_bytes",
    "parse_arch_scope",
    "resolve_npu_arch",
]
