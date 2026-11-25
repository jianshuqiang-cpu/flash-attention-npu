"""Ascend NPU runtime helpers for kernel load and launch (PyACL / AscendCL).

Module-level ``load_binary`` / ``launch_kernel``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ...execution import TlaRuntimeUnavailableError

# acl_rt.h — PyACL does not export these macros; pass numeric literals.
_ACL_RT_BINARY_LOAD_OPT_MAGIC = 2
_ACL_RT_BINARY_MAGIC_ELF_AICORE = 0x43554245
_ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE = 0x41415246


@dataclass
class _LoadedKernel:
    bin_handle: int
    function_handle: int
    kernel_path: Path


def load_acl() -> ModuleType:
    """Import the Ascend ``acl`` Python package (early availability check)."""
    try:
        import acl
    except ImportError as exc:
        raise TlaRuntimeUnavailableError(
            "Failed to import `acl`. Ensure the Ascend Python runtime is installed."
        ) from exc
    return acl


def check_acl_errors(
    ret: object,
    op_name: str,
    *,
    error_cls: type[Exception] | None = None,
) -> None:
    """Raise if an ACL / AscendCL call did not return 0."""
    if int(ret) == 0:
        return
    cls = error_cls if error_cls is not None else TlaRuntimeUnavailableError
    raise cls(f"{op_name} failed with ret={int(ret)}")


def _acl_status(result: object) -> int:
    """Normalize PyACL return value to a status code (``ret`` or ``(value, ret)``)."""
    if isinstance(result, tuple):
        return int(result[-1])
    return int(result)


def _binary_load_options_for_mode(kernel_mode: str) -> list[dict[str, int]]:
    """Build PyACL ``binary_load_from_data`` options for ``kernel_mode``.

    Must use ``binary_load_from_data`` (not FromFile): FromFile rejects
    ``ACL_RT_BINARY_LOAD_OPT_MAGIC`` with 107000. See
    ``scripts/aclrt_poc/mre_binary_load_magic.cpp``.

    - aiv: VECTOR_CORE magic → AIV slot
    - aic: AICORE magic → AIC slot
    - mix: empty options; fat mix ELF already exposes both aic+aiv addrs
    """
    mode = (kernel_mode or "").strip().lower()
    if mode == "aiv":
        return [
            {
                "type": _ACL_RT_BINARY_LOAD_OPT_MAGIC,
                "value": _ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE,
            }
        ]
    if mode == "aic":
        return [
            {
                "type": _ACL_RT_BINARY_LOAD_OPT_MAGIC,
                "value": _ACL_RT_BINARY_MAGIC_ELF_AICORE,
            }
        ]
    if mode == "mix":
        return []
    return [
        {
            "type": _ACL_RT_BINARY_LOAD_OPT_MAGIC,
            "value": _ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE,
        }
    ]


def _register_kernel_binary(
    *, kernel_path: Path, fn_name: str, kernel_mode: str
) -> _LoadedKernel:
    """Load ``kernel.o`` via ``acl.rt.binary_load_from_data`` + mode magic."""
    import acl

    raw = kernel_path.read_bytes()
    host_buf = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
    opts = _binary_load_options_for_mode(kernel_mode)
    bin_handle, load_ret = acl.rt.binary_load_from_data(
        ctypes.addressof(host_buf), len(raw), opts
    )
    check_acl_errors(load_ret, "acl.rt.binary_load_from_data")
    fn_handle, get_ret = acl.rt.binary_get_function(int(bin_handle), fn_name)
    if int(get_ret) != 0:
        try:
            acl.rt.binary_unload(int(bin_handle))
        except Exception:
            pass
        check_acl_errors(get_ret, "acl.rt.binary_get_function")
    return _LoadedKernel(
        bin_handle=int(bin_handle),
        function_handle=int(fn_handle),
        kernel_path=kernel_path,
    )


def load_binary(*, name: str, kernel_path: Path, device: int) -> tuple[int, int]:
    """Load ``kernel.o`` and return ``(bin_handle, function_handle)``."""
    import acl

    fn_name, kernel_mode = name.split(maxsplit=1)
    check_acl_errors(acl.rt.set_device(int(device)), "acl.rt.set_device")
    resolved = kernel_path.resolve()
    loaded = _register_kernel_binary(
        kernel_path=resolved,
        fn_name=fn_name,
        kernel_mode=kernel_mode,
    )
    return loaded.bin_handle, loaded.function_handle


def launch_kernel(
    *,
    function: int,
    stream: int,
    block_num: int,
    args: bytes,
    expects_debug_fifo: bool = False,
    expects_print_tensor: bool | int = False,
) -> None:
    """Launch via PyACL ``kernel_args_*`` + ``launch_kernel_with_config`` (Host args)."""
    import acl

    from .ascend_debug_fifo import launch_with_debug_fifo

    block_num = int(block_num)

    def _acl_launch(launch_args: bytes) -> None:
        payload = launch_args or (b"\x00" * ctypes.sizeof(ctypes.c_uint64))
        host_buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        args_handle, init_ret = acl.rt.kernel_args_init(int(function))
        check_acl_errors(init_ret, "acl.rt.kernel_args_init")
        check_acl_errors(
            _acl_status(
                acl.rt.kernel_args_append(
                    args_handle, ctypes.addressof(host_buf), len(payload)
                )
            ),
            "acl.rt.kernel_args_append",
        )
        check_acl_errors(
            _acl_status(acl.rt.kernel_args_finalize(args_handle)),
            "acl.rt.kernel_args_finalize",
        )
        # PyACL: cfg must be a list (``None`` fails parse); reserve as ``0``.
        check_acl_errors(
            _acl_status(
                acl.rt.launch_kernel_with_config(
                    int(function),
                    int(block_num),
                    int(stream),
                    [],
                    args_handle,
                    0,
                )
            ),
            "acl.rt.launch_kernel_with_config",
        )

    needs_fifo = bool(expects_debug_fifo) or bool(expects_print_tensor)
    if needs_fifo:
        launch_with_debug_fifo(
            launch_kernel=_acl_launch,
            args=args,
            block_num=block_num,
            stream=int(stream),
            expects_debug_fifo=bool(expects_debug_fifo),
            expects_print_tensor=expects_print_tensor,
        )
        return

    _acl_launch(args)


__all__ = [
    "load_acl",
    "check_acl_errors",
    "load_binary",
    "launch_kernel",
]
