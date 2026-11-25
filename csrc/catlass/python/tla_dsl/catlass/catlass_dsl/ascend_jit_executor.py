"""Ascend kernel launch executor."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..base_dsl.jit_executor import TlaExecutionArgs
from ..execution import (
    TlaExecutionResult,
    TlaKernelArtifact,
    TlaRuntimeOptions,
    _KernelLaunchPlan,
    _POINTER_ABI_SIZE,
    _PRINT_TENSOR_WORKSPACE_SENTINEL,
    _align_payload,
    _append_debug_print_workspace_payload,
    _capture_c_stdout,
    _checked_print_tensor_block_count,
    _core_type_from_arch_scope,
    _decode_native_print_tensor_records,
    _format_print_tensor_record,
    _logical_launch_arg_count,
    _mixed_print_tensor_helper_core,
    _print_tensor_static_metadata_records,
    _validate_print_tensor_fifo_capacity,
)


def execute_kernel(
    artifact: TlaKernelArtifact,
    *,
    runtime: TlaRuntimeOptions,
    launch_args: Sequence[Any],
    launch_kwargs: Mapping[str, Any],
) -> TlaExecutionResult:
    from .. import execution as execution_mod

    # ACL is required for load/launch; fail early once here.
    execution_mod.load_acl()
    block_num = launch_kwargs["block_num"]
    device, stream = _resolve_launch_context(launch_kwargs)
    if launch_args:
        _mark_tensor_launch_args_uploaded(launch_args)
    plan = _build_kernel_launch_plan(
        artifact=artifact,
        runtime=runtime,
        launch_args=launch_args,
        block_num=block_num,
    )
    print_metadata = (
        _print_tensor_static_metadata_records(
            artifact.tlair_mlir, entrypoint=artifact.entrypoint
        )
        if plan.expects_print_tensor
        else None
    )

    with artifact._runtime_handle_lock:
        handles = artifact._runtime_handle
        if handles is None:
            handles = execution_mod.load_binary(
                name=f"{plan.entrypoint} {plan.kernel_mode}",
                kernel_path=artifact.kernel_binary_path,
                device=device,
            )
            object.__setattr__(artifact, "_runtime_handle", handles)
    binary_handle, function_handle = handles

    def launch() -> None:
        execution_mod.launch_kernel(
            function=function_handle,
            stream=int(stream),
            block_num=plan.block_num,
            args=plan.payload,
            expects_debug_fifo=plan.expects_debug_fifo,
            expects_print_tensor=plan.expects_print_tensor,
        )

    if print_metadata is None:
        launch()
    else:
        native_output = _capture_c_stdout(launch)
        print_block_count = _checked_print_tensor_block_count(plan.block_num)
        helper_core = (
            _mixed_print_tensor_helper_core(artifact.lowered_llvm)
            if plan.kernel_mode == "mix"
            else _core_type_from_arch_scope(runtime.arch_scope)
        )
        expected_subblocks: tuple[int | None, ...] = (
            (0, 1)
            if helper_core == "aiv" and plan.kernel_mode == "mix"
            else (0,)
            if helper_core == "aiv"
            else (None,)
        )
        decoded = _decode_native_print_tensor_records(
            native_output,
            metadata=print_metadata,
            block_count=print_block_count,
            expected_subblocks=expected_subblocks,
        )
        preserve_legacy_format = len(print_metadata) * print_block_count == 1
        for metadata, record, values in decoded:
            print(
                _format_print_tensor_record(
                    values,
                    shape=record.shape,
                    dtype=metadata.dtype,
                    call=None if preserve_legacy_format else metadata.call,
                    block=record.block,
                    position=(metadata.position if plan.kernel_mode == "mix" else None),
                    subblock=record.subblock,
                )
            )
    return TlaExecutionResult(
        artifact=artifact,
        module_handle=binary_handle,
        function_handle=function_handle,
        device=device,
    )


def _resolve_launch_context(launch_kwargs: Mapping[str, Any]) -> tuple[int, int]:
    from ..base_dsl.runtime.ascend_stream_adapter import (
        as_stream,
        current_device,
        current_stream,
    )

    device = int(current_device())
    raw_stream = launch_kwargs.get("stream")
    if raw_stream is None:
        return device, int(current_stream(device))
    return device, as_stream(raw_stream, device=device)


def _build_kernel_launch_plan(
    *,
    artifact: TlaKernelArtifact,
    runtime: TlaRuntimeOptions,
    launch_args: Sequence[Any],
    block_num: int,
) -> _KernelLaunchPlan:
    expects_print_tensor = artifact.expects_print_tensor
    if expects_print_tensor:
        helper_core = (
            _mixed_print_tensor_helper_core(artifact.lowered_llvm)
            if runtime.kernel_mode == "mix"
            else _core_type_from_arch_scope(runtime.arch_scope)
        )
        _validate_print_tensor_fifo_capacity(
            artifact,
            block_num,
            helper_core=helper_core,
            mixed=runtime.kernel_mode == "mix",
        )
    logical_mixed_handoff = artifact.logical_mixed_handoff
    abi_packer = artifact._abi_packer
    if abi_packer is None:
        raise TlaUnsupportedAbiError(
            "kernel ABI was not validated while constructing the compiled artifact"
        )
    if logical_mixed_handoff is not None and runtime.kernel_mode == "mix":
        # Host passes one object per *logical* argument. Dynamic-GM / memref
        # ABI expands each Tensor into many layout slots; ``user_arg_types`` is
        # the split-function MLIR param list (slot-level), so do not use its
        # length here. ``_pack_launch_args`` also checks via the same helper.
        expected_count = (
            _logical_launch_arg_count(artifact.kernel_abi)
            if artifact.kernel_abi is not None
            else len(logical_mixed_handoff.user_arg_types)
        )
        payload = TlaExecutionArgs(
            expected_arg_count=expected_count,
            kernel_abi=artifact.kernel_abi,
            abi_packer=abi_packer,
        ).generate_launch_payload(launch_args)
        payload = _append_debug_print_workspace_payload(
            payload, enabled=artifact.expects_debug_fifo
        )
        if expects_print_tensor:
            extension = bytearray(payload)
            _align_payload(extension, _POINTER_ABI_SIZE)
            extension.extend(
                _PRINT_TENSOR_WORKSPACE_SENTINEL.to_bytes(
                    _POINTER_ABI_SIZE, byteorder="little", signed=False
                )
            )
            payload = bytes(extension)
        return _KernelLaunchPlan(
            entrypoint=logical_mixed_handoff.entrypoint,
            kernel_mode="mix",
            block_num=int(block_num),
            payload=payload,
            expects_debug_fifo=artifact.expects_debug_fifo,
            expects_print_tensor=2 if expects_print_tensor else False,
        )
    payload = TlaExecutionArgs(
        kernel_abi=artifact.kernel_abi,
        abi_packer=abi_packer,
    ).generate_launch_payload(launch_args)
    payload = _append_debug_print_workspace_payload(
        payload, enabled=artifact.expects_debug_fifo
    )
    if expects_print_tensor:
        extension = bytearray(payload)
        _align_payload(extension, _POINTER_ABI_SIZE)
        extension.extend(
            _PRINT_TENSOR_WORKSPACE_SENTINEL.to_bytes(
                _POINTER_ABI_SIZE, byteorder="little", signed=False
            )
        )
        payload = bytes(extension)
    return _KernelLaunchPlan(
        entrypoint=artifact.entrypoint,
        kernel_mode=runtime.kernel_mode,
        block_num=int(block_num),
        payload=payload,
        expects_debug_fifo=artifact.expects_debug_fifo,
        expects_print_tensor=expects_print_tensor,
    )


def _mark_tensor_launch_args_uploaded(args: Sequence[Any]) -> None:
    for arg in args:
        if hasattr(arg, "prepare_for_launch") and callable(arg.prepare_for_launch):
            arg.prepare_for_launch()


__all__ = [
    "execute_kernel",
    "_build_kernel_launch_plan",
    "_mark_tensor_launch_args_uploaded",
]
