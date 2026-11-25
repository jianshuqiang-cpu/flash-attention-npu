"""Tla kernel launch API."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Sequence

from .. import runtime as _runtime
from ..dsl import _get_typed_call_args
from ..execution import TlaExecutionResult, TlaUnsupportedAbiError

if TYPE_CHECKING:
    from ..dsl import TlaJitFunction


class KernelLauncher:
    """Runtime launch wrapper for ``@tla.kernel`` functions.

    Collects launch args / options, triggers host-side ``compile_kernel`` when
    needed, then calls ``execute_kernel``.
    """

    def __init__(
        self,
        fn: "TlaJitFunction",
        *,
        launch_kwargs: dict[str, Any] | None = None,
        launch_args: Sequence[Any] | None = None,
    ) -> None:
        self._fn = fn
        self._launch_kwargs = dict(launch_kwargs or {})
        self._launch_args = tuple(launch_args or ())
        self._runtime = None
        self._artifact = None
        type_args = (
            _get_typed_call_args(self._launch_args) if self._launch_args else None
        )
        should_eager_compile = (
            type_args is not None or not inspect.signature(self._fn.fn).parameters
        )
        if should_eager_compile:
            runtime = _runtime.runtime_options_for_launch(
                _runtime.runtime_options_from_kwargs(self._launch_kwargs)
            )
            self._runtime = runtime
            self._artifact = _runtime.compile_kernel(
                self._fn.fn,
                kind=self._fn.kind,
                options=self._fn.options,
                runtime=runtime,
                type_args=type_args,
                decorator_location=self._fn.decorator_location,
            )

    def launch(
        self,
        *,
        block_num: int | None = None,
        type_args: Sequence[Any] | None = None,
        args: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> TlaExecutionResult:
        launch_kwargs = {**self._launch_kwargs, **kwargs}
        if block_num is not None:
            launch_kwargs["block_num"] = block_num
        block_num = launch_kwargs.pop("block_num", 1)
        if not isinstance(block_num, int):
            raise TlaUnsupportedAbiError("`block_num` must be an int.")
        launch_args = self._launch_args
        if args is not None:
            if launch_args:
                raise TlaUnsupportedAbiError("`args` specified multiple times.")
            launch_args = tuple(args)
        if type_args is None and launch_args:
            type_args = _get_typed_call_args(launch_args)
        launch_kwargs["block_num"] = int(block_num)
        if (
            self._runtime is not None
            and "cache_dir" not in launch_kwargs
            and self._runtime.cache_dir is not None
            and not self._runtime.cache_enabled
        ):
            launch_kwargs["cache_dir"] = self._runtime.cache_dir
            launch_kwargs["cache"] = False
        runtime = _runtime.runtime_options_for_launch(
            _runtime.runtime_options_from_kwargs(launch_kwargs)
        )
        artifact = self._artifact
        if artifact is None or self._runtime != runtime:
            artifact = _runtime.compile_kernel(
                self._fn.fn,
                kind=self._fn.kind,
                options=self._fn.options,
                runtime=runtime,
                type_args=type_args,
                decorator_location=self._fn.decorator_location,
            )
            self._artifact = artifact
            self._runtime = runtime
        return _runtime.execute_kernel(
            artifact,
            runtime=runtime,
            launch_args=launch_args,
            launch_kwargs=launch_kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> TlaExecutionResult:
        return self.launch(args=args, **kwargs)


__all__ = ["KernelLauncher"]
