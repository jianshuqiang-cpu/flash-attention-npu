"""Runtime JIT argument adapters for host launch packing."""

from __future__ import annotations

from typing import Any, Callable


class _PointerLaunchArg:
    """Launch argument that exposes a single device pointer via ``__c_pointers__``."""

    __slots__ = ("_ptr",)

    def __init__(self, ptr: int) -> None:
        self._ptr = int(ptr)

    def __c_pointers__(self) -> list[int]:
        return [self._ptr]


class JitArgAdapterRegistry:
    """Map Python types to launch-time DSL objects with ``__c_pointers__``."""

    jit_arg_adapter_registry: dict[type[Any], Callable[[Any], Any]] = {}

    @classmethod
    def register_jit_arg_adapter(
        cls, python_type: type[Any]
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        """Register a JIT argument adapter for ``python_type``."""

        def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            if python_type in cls.jit_arg_adapter_registry:
                raise RuntimeError(
                    f"JIT argument adapter for {python_type} is already registered!"
                )
            cls.jit_arg_adapter_registry[python_type] = fn
            return fn

        return decorator

    @classmethod
    def clear(cls) -> None:
        cls.jit_arg_adapter_registry.clear()

    @classmethod
    def get_registered_adapter(cls, arg: object) -> Callable[[Any], Any] | None:
        """Return the registered adapter for ``type(arg)``, or ``None``."""
        return cls.jit_arg_adapter_registry.get(type(arg), None)


def _adapt_from_data_ptr(obj: Any) -> Any:
    """If ``obj`` exposes ``data_ptr``, wrap it as a pointer launch arg."""
    data_ptr = getattr(obj, "data_ptr", None)
    if callable(data_ptr):
        return _PointerLaunchArg(int(data_ptr()))
    if data_ptr is not None and not callable(data_ptr):
        return _PointerLaunchArg(int(data_ptr))
    return obj


__all__ = ["JitArgAdapterRegistry", "_PointerLaunchArg"]
