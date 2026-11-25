"""Tla DSL decorators and lowering entry points."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
import inspect
from typing import Any, Callable, Mapping, Sequence

from . import runtime as _runtime
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.compiler import CompileCallable, compile
from .execution import TlaKernelArtifact


def _get_typed_call_args(args: Sequence[Any]) -> Sequence[Any] | None:
    inferred: list[Any] = []
    for arg in args:
        resolver = getattr(arg, "__get_mlir_types__", None)
        if callable(resolver):
            inferred.append(arg)
        elif is_dataclass(arg) and not isinstance(arg, type):
            # Plain stdlib ``@dataclass`` instances are unpacked into per-field
            # scalar kernel args; keep the instance so lowering can read fields.
            inferred.append(arg)
        elif isinstance(arg, (bool, int, float)):
            # Preserve host literals for ``tla.Constexpr[...]`` / numeric params.
            # Plain ints must not be erased to None or Constexpr lowering sees NoneType.
            inferred.append(arg)
        else:
            inferred.append(None)
    if all(item is None for item in inferred):
        return None
    return tuple(inferred)


from .catlass_dsl.tla import KernelLauncher


@dataclass
class TlaJitFunction:
    """Wrapper for Tla DSL JIT/kernels that can emit and execute Tla IR."""

    fn: Callable[..., Any]
    kind: str
    options: Mapping[str, Any]
    decorator_location: DSLLocation | None = None
    _mlir: str | None = None
    _base_dsl: BaseDSL | None = None
    _lowered: Any | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.kind == "kernel":
            return KernelLauncher(
                self, launch_kwargs=dict(kwargs), launch_args=tuple(args)
            )
        return self.fn(*args, **kwargs)

    def compile(
        self, *, type_args: Sequence[Any] | None = None, **kwargs: Any
    ) -> TlaKernelArtifact:
        runtime = _runtime.runtime_options_from_kwargs(kwargs)
        return _runtime.compile_kernel(
            self.fn,
            kind=self.kind,
            options=self.options,
            runtime=runtime,
            type_args=type_args,
            decorator_location=self.decorator_location,
        )

    @property
    def mlir(self) -> str:
        if self._mlir is None:
            base_dsl = self._base_dsl or BaseDSL()
            lowered = base_dsl._lower(
                self.fn,
                kind=self.kind,
                options=dict(self.options),
                location=self.decorator_location,
            )
            self._lowered = lowered
            self._mlir = lowered.asm()
            self._base_dsl = base_dsl
        return self._mlir

    def dump_mlir(self, *, type_args: Sequence[Any] | None = None) -> str:
        base_dsl = self._base_dsl or BaseDSL()
        if type_args is None and self._mlir is not None:
            return self._mlir
        lowered = base_dsl._lower(
            self.fn,
            kind=self.kind,
            options=dict(self.options),
            type_args=type_args,
            location=self.decorator_location,
        )
        mlir = lowered.asm()
        if type_args is None:
            self._mlir = mlir
            self._lowered = lowered
        self._base_dsl = base_dsl
        return mlir


def jit(fn: Callable[..., Any]) -> TlaJitFunction:
    """Decorate a helper for Tla DSL lowering."""

    _reject_async_dsl_function(fn, kind="jit")
    return TlaJitFunction(
        fn,
        kind="jit",
        options={},
        decorator_location=_capture_decorator_location(),
    )


def kernel(
    fn: Callable[..., Any] | None = None,
    *,
    auto_sync: str | None = None,
) -> TlaJitFunction | Callable[[Callable[..., Any]], TlaJitFunction]:
    """Decorate a Tla kernel entry point.

    By default, local synchronization remains explicit. ``auto_sync="v0"``
    enables the first version of automatic local mutex insertion.
    """

    if auto_sync not in (None, "v0"):
        raise ValueError(
            f"tla.kernel auto_sync must be 'v0' or None, got {auto_sync!r}"
        )

    def decorate(target: Callable[..., Any]) -> TlaJitFunction:
        if not callable(target):
            raise TypeError("tla.kernel expects a callable")
        _reject_async_dsl_function(target, kind="kernel")
        options = {} if auto_sync is None else {"auto_sync": auto_sync}
        return TlaJitFunction(
            target,
            kind="kernel",
            options=options,
            decorator_location=_capture_decorator_location(),
        )

    if fn is None:
        return decorate
    return decorate(fn)


def _reject_async_dsl_function(fn: Callable[..., Any], *, kind: str) -> None:
    unwrapped = inspect.unwrap(fn)
    if not (
        inspect.iscoroutinefunction(unwrapped) or inspect.isasyncgenfunction(unwrapped)
    ):
        return
    filename = inspect.getsourcefile(unwrapped) or "<unknown>"
    try:
        source_lines, lineno = inspect.getsourcelines(unwrapped)
    except (OSError, IOError, TypeError):
        source_lines, lineno = [], 1
    text = source_lines[0] if source_lines else None
    offset = len(text) - len(text.lstrip()) + 1 if text is not None else None
    error = SyntaxError(f"async Tla {kind} functions are not supported")
    error.filename = filename
    error.lineno = lineno
    error.offset = offset
    error.text = text
    raise error


def _capture_decorator_location() -> DSLLocation | None:
    frame = inspect.currentframe()
    if frame is None:
        return None
    caller = frame.f_back
    while caller is not None and caller.f_code.co_filename == __file__:
        caller = caller.f_back
    if caller is None:
        return None
    filename = caller.f_code.co_filename
    return DSLLocation(
        filename=filename,
        lineno=int(caller.f_lineno),
        col_offset=0,
        function_name=caller.f_code.co_name,
    )


__all__ = [
    "DSLLocation",
    "BaseDSL",
    "TlaJitFunction",
    "KernelLauncher",
    "CompileCallable",
    "compile",
    "jit",
    "kernel",
]
