"""Helpers for minimal mixed-kernel attribute emission.

This module only covers the smallest direct ``hivmc-a5`` attribute set needed
for the current mixed-kernel lowering work. It is intentionally independent of
MLIR Python bindings and runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixedKernelModuleAttrInputs:
    """Inputs needed to assemble minimal mixed-kernel module attributes."""

    target_name: str
    module_core_type: str
    target_system_spec: str

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("target_name must be non-empty.")
        if not self.module_core_type:
            raise ValueError("module_core_type must be non-empty.")
        if not self.target_system_spec:
            raise ValueError("target_system_spec must be non-empty.")


def target_system_spec_contains_arch(target_system_spec: str) -> bool:
    """Return whether the textual target-system-spec contains an ARCH entry."""

    return '#dlti.dl_entry<"ARCH",' in target_system_spec


def validate_mixed_kernel_target_system_spec(target_system_spec: str) -> None:
    """Validate that the minimal mixed-kernel target spec preserves ARCH."""

    if not target_system_spec_contains_arch(target_system_spec):
        raise ValueError("target_system_spec must contain an ARCH device-spec entry.")


def build_mixed_kernel_module_attrs(
    inputs: MixedKernelModuleAttrInputs,
) -> dict[str, str]:
    """Build the minimal mixed-kernel module attrs for the direct hivmc path."""

    validate_mixed_kernel_target_system_spec(inputs.target_system_spec)
    return {
        "dlti.target_system_spec": inputs.target_system_spec,
        "hacc.target": f'#hacc.target<"{inputs.target_name}">',
        "hivm.module_core_type": f"#hivm.module_core_type<{inputs.module_core_type}>",
    }


def build_mixed_kernel_entry_attrs() -> dict[str, str | bool]:
    """Build the minimal entry attrs required by the current mixed-kernel path."""

    return {
        "hacc.entry": True,
        "hacc.function_kind": "#hacc.function_kind<DEVICE>",
    }
