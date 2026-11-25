from __future__ import annotations

import shlex
from typing import Any

from .jit_executor import TlaJitExecutor


def _parse_compile_options_from_str(options: Any) -> dict[str, str]:
    """Parse ``options="--npu-arch 3510"`` token string."""
    if options is None:
        return {}
    if not isinstance(options, str):
        raise TypeError(
            "compile options must be a string "
            '(e.g. options="--npu-arch 3510"), '
            f"got {type(options).__name__}"
        )
    text = options.strip()
    if not text:
        return {}
    tokens = shlex.split(text)
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and "=" in token:
            key, value = token[2:].split("=", 1)
            parsed[key.replace("-", "_")] = value
            index += 1
            continue
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise ValueError(f"Missing value for compile option {token!r}")
            parsed[key] = tokens[index + 1]
            index += 2
            continue
        raise ValueError(f"Unexpected token in compile options: {token!r}")
    return parsed


class CompileCallable:
    """Compile a Tla kernel and return a callable compiled executor."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> TlaJitExecutor:
        from ..execution import TlaUnsupportedAbiError
        from ..dsl import TlaJitFunction, _get_typed_call_args
        from ..catlass_dsl.tla import KernelLauncher

        if func is None:
            raise TlaUnsupportedAbiError("Function is not set or invalid.")

        if isinstance(func, KernelLauncher):
            func = func._fn
        if not isinstance(func, TlaJitFunction):
            raise TlaUnsupportedAbiError(
                "tla.compile expects a @tla.jit or @tla.kernel function."
            )
        type_args = kwargs.pop("type_args", None)
        if type_args is None and args:
            inferred = _get_typed_call_args(args)
            if inferred is not None:
                type_args = inferred
        return TlaJitExecutor(func.compile(type_args=type_args, **kwargs))


compile = CompileCallable()


__all__ = ["TlaJitExecutor", "CompileCallable", "compile"]
