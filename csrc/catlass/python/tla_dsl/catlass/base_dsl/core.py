"""BaseDSL scaffold for the frontend rewrite series."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import TYPE_CHECKING, Any, Callable, Sequence

from mlir import ir as mlir_ir  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..execution_lowering import LoweredTlaIR


@dataclass(frozen=True)
class DSLLocation:
    """Represents source location information for DSL-originated IR."""

    filename: str
    lineno: int
    col_offset: int
    function_name: str


class BaseDSL:
    """Scaffold class for full_demo-style frontend orchestration."""

    def __init__(self) -> None:
        self.decorator_location: DSLLocation | None = None
        self.funcBody: Callable[..., Any] | None = None
        self._lowered: "LoweredTlaIR | None" = None

    def get_ir_location(self, location: DSLLocation | None = None) -> mlir_ir.Location:
        if location is None:
            location = self.decorator_location
        if location is None:
            return mlir_ir.Location.unknown()
        file_loc = mlir_ir.Location.file(
            location.filename, location.lineno, location.col_offset
        )
        return mlir_ir.Location.name(location.function_name, childLoc=file_loc)

    def _get_function_bound_args(
        self,
        sig: inspect.Signature,
        func_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> inspect.BoundArguments:
        try:
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
        except Exception as exc:  # pragma: no cover - pass-through for callers
            raise RuntimeError(
                f"Failed to bind arguments to function `{func_name}` with signature `{sig}`"
            ) from exc
        return bound_args

    def _canonicalize_args(
        self, sig: inspect.Signature, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.funcBody is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Function body is not set.")
        function_name = self.funcBody.__name__
        bound_args = self._get_function_bound_args(sig, function_name, *args, **kwargs)
        return bound_args.args, bound_args.kwargs

    def _check_arg_count(self, *args: Any, **kwargs: Any) -> inspect.Signature:
        if self.funcBody is None:
            raise RuntimeError("Function body is not set.")
        sig = inspect.signature(self.funcBody)
        function_name = self.funcBody.__name__
        bound_args = self._get_function_bound_args(sig, function_name, *args, **kwargs)
        for param in sig.parameters.values():
            if (
                param.default is inspect.Parameter.empty
                and param.name not in bound_args.arguments
            ):
                raise RuntimeError(
                    f"Missing required argument in `{function_name}`: '{param.name}'"
                )
        return sig

    def mangle_name(
        self, function_name: str, args: tuple[Any, ...], args_spec: inspect.FullArgSpec
    ) -> str:
        for spec_arg, arg in zip(args_spec.args, args):
            spec_ty = args_spec.annotations.get(spec_arg, None)
            if inspect.isclass(spec_ty):
                class_name = str(arg).replace("class", "").replace(" ", "")
                function_name = f"{function_name}_{class_name}"
            elif isinstance(arg, (list, tuple)):
                function_name = f"{function_name}_{'_'.join(map(str, arg))}"
            else:
                function_name = f"{function_name}_{arg}"
        unwanted_chars = r"'-![]#,.<>()\":{}=%?@;"
        translation_table = str.maketrans("", "", unwanted_chars)
        function_name = function_name.translate(translation_table)
        function_name = re.sub(r"0x[a-f0-9]{8,16}", "", function_name)
        function_name = re.sub(r"\s+", " ", function_name)
        function_name = function_name.replace(" ", "_").replace("\n", "_")
        function_name = function_name.replace("/", "_")
        return function_name[:180]

    def generate_original_ir(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate_mlir(*args, **kwargs)

    def lower_module(
        self,
        funcBody: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> "LoweredTlaIR":
        from ..execution_lowering import lower_jit_to_tlair_module_by_execution

        lowered = lower_jit_to_tlair_module_by_execution(
            funcBody,
            kind=kind,
            options=options or {},
            generic=generic,
            type_args=type_args,
            location=location,
        )
        self._lowered = lowered
        return lowered

    def generate_mlir(
        self,
        funcBody: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> str:
        return self.lower_module(
            funcBody,
            kind=kind,
            options=options or {},
            generic=generic,
            type_args=type_args,
            location=location,
        ).asm(generic=generic)

    def _func(
        self,
        funcBody: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> str:
        self.funcBody = funcBody
        self.decorator_location = location
        return self.generate_mlir(
            funcBody,
            kind=kind,
            options=options,
            generic=generic,
            type_args=type_args,
            location=location,
        )

    def _lower(
        self,
        funcBody: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> "LoweredTlaIR":
        self.funcBody = funcBody
        self.decorator_location = location
        return self.lower_module(
            funcBody,
            kind=kind,
            options=options,
            generic=generic,
            type_args=type_args,
            location=location,
        )


__all__ = ["DSLLocation", "BaseDSL"]
