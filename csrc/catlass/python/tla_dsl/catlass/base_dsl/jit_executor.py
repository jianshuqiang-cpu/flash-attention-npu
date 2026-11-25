from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from ..execution import TlaExecutionResult, TlaKernelArtifact


@dataclass(frozen=True)
class TlaExecutionArgs:
    """Runtime ABI binder for TLA kernel launch arguments.

    Packing always requires a compiler-produced ``kernel_abi`` layout so
    Dynamic-GM / memref ABI stays signature-driven.

    Must not import ``catlass.execution`` at module load time: ``execution``
    re-exports ``execute_kernel`` from ``ascend_jit_executor``, which imports
    this class, so a top-level import here creates a circular import.
    """

    signature: Mapping[str, Any] | None = None
    kernel_abi: Any | None = None
    expected_arg_count: int | None = None
    abi_packer: Any | None = None

    def filter_runtime_signature(
        self, signature: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None:
        """Return the runtime-visible signature (constexpr stripping later)."""
        return self.signature if signature is None else signature

    def get_rectified_args(
        self, launch_args: Sequence[Any], **_kwargs: Any
    ) -> tuple[Any, ...]:
        """Normalize call args before packing (adapters + passthrough)."""
        from .runtime.jit_arg_adapters import (
            JitArgAdapterRegistry,
            _adapt_from_data_ptr,
        )
        from .typing import Numeric, as_numeric, is_constexpr_annotation

        rectified: list[Any] = []
        for arg in launch_args:
            if is_dataclass(arg) and not isinstance(arg, type):
                # Unpack a stdlib dataclass into one entry per **dynamic** field,
                # matching the frontend's scalar_group block args / ABI slots.
                # ``Constexpr`` fields are compile-time constants with no ABI slot.
                # Numerics and tensor-like fields (``__c_pointers__``) pass through;
                # plain scalars become Numerics via ``as_numeric``.
                for field in fields(arg):
                    if is_constexpr_annotation(field.type):
                        continue
                    value = getattr(arg, field.name)
                    if isinstance(value, Numeric) or hasattr(value, "__c_pointers__"):
                        rectified.append(value)
                    else:
                        rectified.append(as_numeric(value))
                continue
            if isinstance(arg, Numeric) or hasattr(arg, "__c_pointers__"):
                rectified.append(arg)
            else:
                adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
                if adapter is not None:
                    rectified.append(adapter(arg))
                else:
                    rectified.append(_adapt_from_data_ptr(arg))
        return tuple(rectified)

    def generate_launch_payload(self, launch_args: Sequence[Any]) -> bytes:
        """Pack ``launch_args`` into the Ascend host launch byte buffer."""
        from .. import execution as execution_mod

        if self.kernel_abi is None:
            raise execution_mod.TlaUnsupportedAbiError(
                "A compiler-produced kernel ABI layout is required before packing "
                "launch arguments."
            )
        rectified = self.get_rectified_args(launch_args)
        if (
            self.expected_arg_count is not None
            and len(rectified) != self.expected_arg_count
        ):
            raise execution_mod.TlaUnsupportedAbiError(
                "launch argument count does not match expected signature: "
                f"got {len(rectified)}, expected {self.expected_arg_count}"
            )
        if self.abi_packer is not None:
            return execution_mod._pack_launch_args_prepared(rectified, self.abi_packer)
        return execution_mod._pack_launch_args(rectified, self.kernel_abi)


@dataclass(frozen=True)
class TlaJitExecutor:
    """Callable wrapper around a compiled Tla kernel artifact."""

    artifact: TlaKernelArtifact

    def launch(
        self,
        *launch_args: Any,
        block_num: int | None = None,
        args: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> TlaExecutionResult:
        from ..execution import (
            TlaRuntimeUnavailableError,
            TlaUnsupportedAbiError,
            execute_kernel,
        )

        if launch_args and args is not None:
            raise TlaUnsupportedAbiError("Launch arguments specified multiple times.")
        if args is None:
            args = launch_args
        launch_kwargs = dict(kwargs)
        if block_num is None:
            block_num = 1
        if not isinstance(block_num, int):
            raise TlaUnsupportedAbiError("`block_num` must be an int.")
        launch_kwargs["block_num"] = int(block_num)
        runtime = self.artifact.runtime
        if runtime is None:
            raise TlaRuntimeUnavailableError(
                "Compiled artifact is missing runtime options and cannot be launched."
            )
        return execute_kernel(
            self.artifact,
            runtime=runtime,
            launch_args=tuple(args),
            launch_kwargs=launch_kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> TlaExecutionResult:
        return self.launch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.artifact, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TlaJitExecutor):
            return self.artifact == other.artifact
        return self.artifact == other


__all__ = ["TlaExecutionArgs", "TlaJitExecutor"]
