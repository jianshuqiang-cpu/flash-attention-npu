"""Execution-mode lowering that emits Tla MLIR directly while running Python frontend code."""

from __future__ import annotations

import dataclasses
import inspect
import linecache
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mlir import ir as mlir_ir  # type: ignore[assignment]

from . import _tla_type_bridge
from . import runtime
from . import tla_ast_decorators as ast_decorators
from .base_dsl.ast_preprocessor import (
    maybe_transform_for_lowering,
    reject_user_class_value,
    validate_language_boundaries,
)
from .base_dsl import DSLLocation
from .base_dsl.typing import Numeric, is_constexpr_annotation
from .tla.typing import Tensor


class TlaLoweringError(RuntimeError):
    """Raised when Tla DSL lowering fails."""


class UnsupportedExecutionLowering(RuntimeError):
    """Raised when execution-mode lowering cannot safely handle a function."""


_SOURCE_INFO_ATTR = "__tladsl_source_info__"


def _traceback_lineno_for_code(exc: BaseException, code: Any) -> int | None:
    tb = exc.__traceback__
    best: int | None = None
    while tb is not None:
        if tb.tb_frame.f_code is code:
            best = int(tb.tb_lineno)
        tb = tb.tb_next
    return best


def _format_execution_source_error(fn: Any, exc: Exception) -> str | None:
    info = getattr(fn, _SOURCE_INFO_ATTR, None)
    if not isinstance(info, dict):
        return None
    filename = str(info.get("filename") or "<unknown>")
    lineno = _traceback_lineno_for_code(exc, fn.__code__)
    if lineno is None:
        return None
    source = linecache.getline(filename, lineno).strip()
    message = (
        f"Execution-mode lowering failed while running `{fn.__name__}` "
        f"at {filename}:{lineno}"
    )
    if source:
        message += f"\n  source: {source}"
    message += f"\n  reason: {type(exc).__name__}: {exc}"
    return message


@dataclass
class LoweredTlaIR:
    """Structured result of execution-mode lowering to TLA MLIR (``tla`` dialect)."""

    context: mlir_ir.Context
    module: mlir_ir.Module
    generic: bool = False
    _asm: str | None = None

    def asm(self, *, generic: bool | None = None) -> str:
        emit_generic = self.generic if generic is None else bool(generic)
        if self._asm is None or emit_generic != self.generic:
            with self.context:
                self._asm = self.module.operation.get_asm(
                    print_generic_op_form=emit_generic,
                    assume_verified=False,
                )
            self.generic = emit_generic
        return self._asm


def lower_jit_to_tlair_by_execution(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any] | None = None,
    generic: bool = False,
    type_args: Sequence[Any] | None = None,
    location: DSLLocation | None = None,
) -> str:
    return lower_jit_to_tlair_module_by_execution(
        fn,
        kind=kind,
        options=options,
        generic=generic,
        type_args=type_args,
        location=location,
    ).asm(generic=generic)


def lower_jit_to_tlair_module_by_execution(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any] | None = None,
    generic: bool = False,
    type_args: Sequence[Any] | None = None,
    location: DSLLocation | None = None,
) -> LoweredTlaIR:
    auto_sync = (options or {}).get("auto_sync")
    if kind == "kernel" and auto_sync not in (None, "v0"):
        raise TlaLoweringError(
            f"kernel option auto_sync must be 'v0' or None, got {auto_sync!r}"
        )
    if kind != "kernel" and auto_sync is not None:
        raise TlaLoweringError("auto_sync is supported only for tla.kernel")
    validate_language_boundaries(fn)
    fn = maybe_transform_for_lowering(
        fn,
        internal_for=ast_decorators._internal_frontend_for,
        internal_region=runtime._internal_frontend_region,
        internal_if=ast_decorators._internal_frontend_if,
        internal_if_expr=ast_decorators._internal_frontend_if_expr,
        internal_bool_and=ast_decorators._internal_frontend_bool_and,
        internal_bool_or=ast_decorators._internal_frontend_bool_or,
        internal_bool_not=ast_decorators._internal_frontend_bool_not,
        internal_compare=ast_decorators._internal_frontend_compare,
        internal_any=ast_decorators._internal_frontend_any,
        internal_all=ast_decorators._internal_frontend_all,
        internal_bool=ast_decorators._internal_frontend_bool,
        internal_min=ast_decorators._internal_frontend_min,
        internal_max=ast_decorators._internal_frontend_max,
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    arg_names = [p.name for p in params]
    constexpr_names = {p.name for p in params if _is_constexpr_annotation(p.annotation)}
    call_args = _prepare_call_args(arg_names=arg_names, type_args=type_args)
    for name, value in zip(arg_names, call_args, strict=False):
        reject_user_class_value(value, context=f"kernel argument {name!r}")

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        _load_execution_dialects(ctx)
        arg_types = _resolve_execution_arg_types(
            fn=fn,
            arg_names=arg_names,
            arg_values=dict(zip(arg_names, call_args, strict=False))
            if call_args
            else None,
            ctx=ctx,
        )
        with mlir_ir.Location.unknown(ctx):
            module = mlir_ir.Module.create()
            with mlir_ir.InsertionPoint(module.body):
                fn_loc = _coerce_location(ctx, location)
                _build_tla_func(
                    fn=fn,
                    module=module,
                    fn_name=fn.__name__,
                    arg_names=arg_names,
                    constexpr_names=constexpr_names,
                    arg_types=arg_types,
                    call_args=call_args,
                    ctx=ctx,
                    fn_loc=fn_loc,
                    auto_sync=auto_sync,
                )
    lowered = LoweredTlaIR(context=ctx, module=module, generic=bool(generic))
    lowered._asm = module.operation.get_asm(
        print_generic_op_form=bool(generic),
        assume_verified=False,
    )
    return lowered


def _prepare_call_args(
    *, arg_names: Sequence[str], type_args: Sequence[Any] | None
) -> tuple[Any, ...]:
    if type_args is None:
        if arg_names:
            raise UnsupportedExecutionLowering(
                "Execution-mode lowering requires type_args for non-empty signatures."
            )
        return ()
    if len(type_args) != len(arg_names):
        raise TlaLoweringError(
            "type_args length must match function arguments: "
            f"expected {len(arg_names)}, got {len(type_args)}"
        )
    return tuple(type_args)


def _build_tla_func(
    *,
    fn: Any,
    module: mlir_ir.Module,
    fn_name: str,
    arg_names: Sequence[str],
    constexpr_names: set[str],
    arg_types: Mapping[str, Any],
    call_args: Sequence[Any],
    ctx: mlir_ir.Context,
    fn_loc: mlir_ir.Location,
    auto_sync: str | None,
) -> None:
    runtime_arg_names = [name for name in arg_names if name not in constexpr_names]

    # Dynamic GM host tensors enter as unified GM memref + originShape0/1 index args.
    from .core_api import (
        is_dynamic_gm_tensor_arg,
        _materialize_dynamic_gm_root_tensor_descriptor,
    )

    resolved_arg_types: dict[str, Any] = dict(arg_types)
    dynamic_gm_tensor_tys: dict[str, Any] = {}
    for name in runtime_arg_names:
        host = None
        for i, arg_name in enumerate(arg_names):
            if arg_name == name:
                host = call_args[i]
                break
        if host is not None and is_dynamic_gm_tensor_arg(host):
            tensor_ty = host.tla_tensor_type_descriptor()
            dynamic_gm_tensor_tys[name] = tensor_ty
            resolved_arg_types[name] = ("dynamic_gm", tensor_ty)

    bridge_ext = _tla_type_bridge._load_bridge_extension()
    index_ty = mlir_ir.IndexType.get(ctx)
    mlir_arg_types: list[Any] = []
    # name -> (memref_block_idx, origin0_idx, origin1_idx) or (single_idx,)
    block_slots: dict[str, tuple[int, ...]] = {}
    for name in runtime_arg_names:
        spec = resolved_arg_types.get(name)
        if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "dynamic_gm":
            tensor_ty = spec[1]
            gm_memref = bridge_ext.dynamic_gm_memref_type(tensor_ty.to_mlir_type(ctx))
            start = len(mlir_arg_types)
            mlir_arg_types.extend([gm_memref, index_ty, index_ty])
            block_slots[name] = (start, start + 1, start + 2)
        elif isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "scalar_group":
            # Dataclass args lower to one scalar block arg per field.
            group_types = spec[1]
            start = len(mlir_arg_types)
            mlir_arg_types.extend(_coerce_type(ctx, ty) for ty in group_types)
            block_slots[name] = tuple(range(start, start + len(group_types)))
        else:
            start = len(mlir_arg_types)
            mlir_arg_types.append(_coerce_type(ctx, resolved_arg_types.get(name)))
            block_slots[name] = (start,)

    fn_type = mlir_ir.FunctionType.get(mlir_arg_types, [])
    func_attrs = {
        "sym_name": mlir_ir.StringAttr.get(fn_name),
        "function_type": mlir_ir.TypeAttr.get(fn_type),
    }
    if auto_sync == "v0":
        func_attrs["tla.auto_sync"] = mlir_ir.StringAttr.get("v0")
    func_op = mlir_ir.Operation.create(
        "tla.func",
        attributes=func_attrs,
        regions=1,
        loc=fn_loc,
    )
    entry = func_op.regions[0].blocks.append(*mlir_arg_types)

    # Use distinct proxy objects for runtime parameters so names like `dim` resolve to block
    # SSA values, not type_args constants. Then make_shape(dim, 16) gets a dynamic `?` dim
    # while make_shape(4, 8, 16) keeps static dimensions from literals.
    class _ArgProxy:
        __slots__ = ()

        @property
        def ptr(self) -> Any:
            """``arg.ptr`` for kernel-argument proxies — mirrors ``_Tensor.ptr``.

            Resolves the proxy to its block-argument SSA value and emits
            ``tla.tensor_ptr`` so ``lhs.ptr + offset`` works in execution-mode
            lowering the same way as on a frontend ``_Tensor``.
            """
            from .base_dsl.op import _capture_user_loc
            from .core_api import _as_value, _emit_tensor_ptr

            loc = (
                _capture_user_loc()
                if runtime._current_frontend_state() is not None
                else None
            )
            return _emit_tensor_ptr(_as_value(self), loc)

        def _metadata(self, field: str) -> Any:
            from .core_api import _as_value, _tensor_metadata_field

            return _tensor_metadata_field(_as_value(self), field)

        @property
        def shape(self) -> Any:
            return self._metadata("shape")

        @property
        def stride(self) -> Any:
            return self._metadata("stride")

        @property
        def origin_shape(self) -> Any:
            return self._metadata("origin_shape")

        def __getitem__(self, crd: Any) -> Any:
            """Tensor indexing for kernel-argument proxies."""
            from .base_dsl.op import _capture_user_loc
            from .tla.tensor import _Tensor

            loc = (
                _capture_user_loc()
                if runtime._current_frontend_state() is not None
                else None
            )
            return _Tensor.__getitem__(self, crd, loc=loc)

        def __setitem__(self, crd: Any, data: Any) -> None:
            """Scalar store for kernel-argument proxies."""
            from .base_dsl.op import _capture_user_loc
            from .tla.tensor import _Tensor

            loc = (
                _capture_user_loc()
                if runtime._current_frontend_state() is not None
                else None
            )
            return _Tensor.__setitem__(self, crd, data, loc=loc)

    call_args_for_fn = list(call_args)
    arg_bindings: dict[int, tuple[Any, Any]] = {}
    category_bindings: dict[int, tuple[Any, Any]] = {}
    #: ``arg index -> (dataclass instance, block slots)`` — rebuilt inside the
    #: frontend emission context so the field Numerics auto-bind to their SSA.
    pending_dataclass_rebuilds: dict[int, tuple[Any, tuple[int, ...]]] = {}
    for i, name in enumerate(arg_names):
        if name in constexpr_names:
            continue
        host_arg = call_args[i]
        if _is_dataclass_instance(host_arg):
            _validate_dataclass_kernel_arg(host_arg)
            # Rebuilt later (inside the frontend emission context) from its
            # dynamic block args plus its compile-time ``Constexpr`` host values.
            pending_dataclass_rebuilds[i] = (host_arg, block_slots[name])
            continue
        slots = block_slots[name]
        ssa = entry.arguments[slots[0]]
        # Numeric host args must stay typed Numeric around the block SSA so
        # operators (``a + b``, ``a // 2``, …) lower via Numeric.__*__.
        if isinstance(host_arg, Numeric):
            num = type(host_arg)(ssa)
            call_args_for_fn[i] = num
            category_bindings[id(num)] = (num, "numeric")
            category_bindings[id(ssa)] = (ssa, "numeric")
        elif isinstance(host_arg, int) and not isinstance(host_arg, bool):
            # Kernel ``int`` args are i32 Numerics (not MLIR index / ArgProxy).
            from .base_dsl.typing import Int32

            num = Int32(ssa)
            call_args_for_fn[i] = num
            category_bindings[id(num)] = (num, "numeric")
            category_bindings[id(ssa)] = (ssa, "numeric")
        else:
            proxy = _ArgProxy()
            call_args_for_fn[i] = proxy
            arg_bindings[id(proxy)] = (proxy, ssa)
            # Dynamic GM proxies bind to tensor_desc after prologue (below).
            category = _category_from_type_like(
                ctx, arg_types.get(name) if name not in dynamic_gm_tensor_tys else None
            )
            if name in dynamic_gm_tensor_tys:
                category = "tensor"
            if category is not None:
                category_bindings[id(proxy)] = (proxy, category)
                category_bindings[id(ssa)] = (ssa, category)
    call_args_for_fn = tuple(call_args_for_fn)

    tensor_host_by_value: dict[Any, Any] = {}
    for pos, name in enumerate(arg_names):
        if name in constexpr_names:
            continue
        v = call_args[pos]
        if isinstance(v, Tensor) and name not in dynamic_gm_tensor_tys:
            tensor_host_by_value[entry.arguments[block_slots[name][0]]] = v

    # Dataclass tensor fields also map block SSA -> host tensor for metadata.
    for pos, name in enumerate(arg_names):
        if name in constexpr_names:
            continue
        host = call_args[pos]
        if not _is_dataclass_instance(host):
            continue
        constexpr_names = _dataclass_constexpr_field_names(host)
        slots = block_slots[name]
        slot_iter = iter(slots)
        for field in dataclasses.fields(host):
            if field.name in constexpr_names:
                continue
            field_value = getattr(host, field.name)
            if isinstance(field_value, Tensor):
                if is_dynamic_gm_tensor_arg(field_value):
                    raise TlaLoweringError(
                        f"dataclass field {name}.{field.name} is a dynamic-GM tensor, "
                        "which is not supported inside a dataclass; use a static tensor "
                        "or a top-level kernel argument"
                    )
                tensor_host_by_value[entry.arguments[next(slot_iter)]] = field_value
            else:
                next(slot_iter)

    with mlir_ir.InsertionPoint(entry):
        # Materialize dynamic GM root descriptors before the user body so
        # origin_shape/shape/stride reads hit side-table SSA (memref.dim).
        pending_dynamic_gm: list[tuple[Any, Any, dict[str, Any]]] = []
        for name in runtime_arg_names:
            tensor_ty = dynamic_gm_tensor_tys.get(name)
            if tensor_ty is None:
                continue
            mem_i, o0_i, o1_i = block_slots[name]
            desc, metadata = _materialize_dynamic_gm_root_tensor_descriptor(
                entry.arguments[mem_i],
                entry.arguments[o0_i],
                entry.arguments[o1_i],
                tensor_ty,
                loc=fn_loc,
            )
            # Rebind the matching ArgProxy to the tensor_desc result.
            for i, arg_name in enumerate(arg_names):
                if arg_name != name:
                    continue
                proxy = call_args_for_fn[i]
                arg_bindings[id(proxy)] = (proxy, desc)
                category_bindings[id(desc)] = (desc, "tensor")
                host = call_args[i]
                if isinstance(host, Tensor):
                    tensor_host_by_value[desc] = host
                pending_dynamic_gm.append((desc, tensor_ty, metadata))
                break

        with runtime._frontend_emission(
            arg_bindings=arg_bindings,
            category_bindings=category_bindings,
            tensor_host_by_value=tensor_host_by_value,
            module=module,
        ):
            from .core_api import (
                _register_tla_tensor_metadata,
                _register_tla_tensor_type,
            )

            # Descriptor emission ran before emission state existed; register now.
            for desc, tensor_ty, metadata in pending_dynamic_gm:
                _register_tla_tensor_type(desc, tensor_ty)
                _register_tla_tensor_metadata(desc, metadata)
            if pending_dataclass_rebuilds:
                from .core_api import _wrap_frontend_value

                call_args_for_fn = list(call_args_for_fn)
                for index, (instance, slots) in pending_dataclass_rebuilds.items():
                    # Rebuild the stdlib dataclass in field order: ``Constexpr``
                    # fields keep their host value (compile-time constant), dynamic
                    # fields consume one block arg wrapped in the matching frontend
                    # object (Numeric for scalars, _Tensor for tensors).
                    constexpr_names = _dataclass_constexpr_field_names(instance)
                    slot_iter = iter(slots)
                    field_names: list[str] = []
                    rebuilt_values: list[Any] = []
                    for field in dataclasses.fields(instance):
                        field_names.append(field.name)
                        if field.name in constexpr_names:
                            rebuilt_values.append(getattr(instance, field.name))
                        else:
                            rebuilt_values.append(
                                _wrap_frontend_value(entry.arguments[next(slot_iter)])
                            )
                    # kw_only dataclasses reject positional args; keyword
                    # reconstruction works for both kw_only and plain dataclasses.
                    kwargs = dict(zip(field_names, rebuilt_values))
                    if constexpr_names:
                        ro_cls = _constexpr_readonly_dataclass_cls(
                            type(instance), constexpr_names
                        )
                        call_args_for_fn[index] = ro_cls(**kwargs)
                    else:
                        call_args_for_fn[index] = type(instance)(**kwargs)
                call_args_for_fn = tuple(call_args_for_fn)
            try:
                fn(*call_args_for_fn)
            except runtime.TlaCoreAPIError:
                raise
            except TlaLoweringError:
                raise
            except ValueError:
                raise
            except ast_decorators.FrontendControlFlowLoweringError as exc:
                raise UnsupportedExecutionLowering(str(exc)) from exc
            except Exception as exc:
                message = _format_execution_source_error(fn, exc)
                if message is None:
                    message = f"Execution-mode lowering failed while running `{fn.__name__}`: {exc}"
                raise UnsupportedExecutionLowering(message) from exc
        mlir_ir.Operation.create("tla.return", loc=fn_loc)


def _coerce_location(
    ctx: mlir_ir.Context, location: DSLLocation | None
) -> mlir_ir.Location:
    if location is None:
        return mlir_ir.Location.unknown(ctx)
    if location.lineno <= 0:
        return mlir_ir.Location.unknown(ctx)
    file_loc = mlir_ir.Location.file(
        location.filename,
        int(location.lineno),
        int(location.col_offset),
        ctx,
    )
    return mlir_ir.Location.name(location.function_name, childLoc=file_loc, context=ctx)


def _coerce_type(ctx: mlir_ir.Context, type_like: Any) -> mlir_ir.Type:
    if isinstance(type_like, mlir_ir.Type):
        return type_like
    if type_like is None:
        raise TypeError(
            "execution lowering could not resolve a concrete runtime argument type"
        )
    if isinstance(type_like, str):
        with ctx:
            return Numeric.from_dtype_token(type_like).mlir_type(ctx)
    raise TypeError(
        "execution lowering expected mlir.ir.Type or a Tla element token; "
        f"got {type(type_like).__name__}"
    )


def _category_from_type_like(ctx: mlir_ir.Context, type_like: Any) -> str | None:
    from . import _tla_type_bridge

    if type_like is None:
        return None
    try:
        ty = _coerce_type(ctx, type_like)
    except Exception:
        return None
    if isinstance(ty, mlir_ir.IndexType):
        return "index"
    return _tla_type_bridge.tla_type_category(ty)


def _resolve_execution_arg_types(
    *,
    fn: Any,
    arg_names: Sequence[str],
    arg_values: Mapping[str, Any] | None,
    ctx: mlir_ir.Context,
) -> Mapping[str, Any]:
    resolved: dict[str, Any] = {}
    if arg_values is not None:
        for name, value in arg_values.items():
            mlir_types_getter = getattr(value, "__get_mlir_types__", None)
            if callable(mlir_types_getter):
                resolved_types = mlir_types_getter(ctx)
                if resolved_types:
                    if len(resolved_types) == 1:
                        resolved[name] = resolved_types[0]
                    else:
                        resolved[name] = ("scalar_group", tuple(resolved_types))
                    continue
            if _is_dataclass_instance(value):
                # Unpack a stdlib dataclass into one scalar type per field.
                resolved[name] = (
                    "scalar_group",
                    _resolve_dataclass_field_types(value, ctx),
                )
                continue
            if isinstance(value, bool):
                resolved[name] = "i1"
            elif isinstance(value, int):
                resolved[name] = "i32"
            elif isinstance(value, float):
                resolved[name] = "f32"
    return resolved


def _resolve_dataclass_field_types(value: Any, ctx: mlir_ir.Context) -> tuple[Any, ...]:
    """Resolve one MLIR type per **dynamic** field of a stdlib dataclass instance.

    Fields annotated ``tla.Constexpr[...]`` are compile-time constants: they
    produce no MLIR type and no kernel block arg. Mirrors ``_get_typed_call_args``:
    any other field value exposing ``__get_mlir_types__`` (``tla.*`` Numerics,
    ``tla.Tensor``, …) contributes its own type; plain ``bool``/``int``/``float``
    map to ``i1``/``i32``/``f32``.
    """
    constexpr_names = _dataclass_constexpr_field_names(value)
    field_types: list[Any] = []
    for field in dataclasses.fields(value):
        if field.name in constexpr_names:
            continue
        field_value = getattr(value, field.name)
        mlir_types_getter = getattr(field_value, "__get_mlir_types__", None)
        if callable(mlir_types_getter):
            resolved = mlir_types_getter(ctx)
            if not resolved:
                raise TlaLoweringError(
                    f"dataclass field {field.name!r} resolved to no MLIR type"
                )
            field_types.append(resolved[0])
        elif isinstance(field_value, bool):
            field_types.append("i1")
        elif isinstance(field_value, int):
            field_types.append("i32")
        elif isinstance(field_value, float):
            field_types.append("f32")
        else:
            raise TlaLoweringError(
                f"unsupported dataclass field {field.name!r} type "
                f"{type(field_value).__name__}; fields must expose __get_mlir_types__ "
                "(e.g. tla.Int32 / tla.Float32 / tla.Tensor) or be plain bool/int/float"
            )
    return tuple(field_types)


def _load_execution_dialects(ctx: mlir_ir.Context) -> None:
    _tla_type_bridge.load_tla_dialect(ctx)
    for dialect in ("arith", "scf", "memref"):
        ctx.dialects[dialect]


def _is_dataclass_instance(value: Any) -> bool:
    """True when ``value`` is a stdlib ``@dataclass`` instance (not the class)."""
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


#: stdlib ``@dataclass`` options that must keep their default values for a TLA
#: kernel-argument dataclass. Only ``frozen`` and ``kw_only`` may be customized.
#: The env's ``_DataclassParams`` may not record every option, so each check is
#: guarded by ``hasattr``.
_DATACLASS_DEFAULT_ONLY_PARAMS: tuple[tuple[str, object], ...] = (
    ("init", True),
    ("repr", True),
    ("eq", True),
    ("order", False),
    ("unsafe_hash", False),
    ("match_args", True),
    ("slots", False),
    ("weakref_slot", False),
)


def _validate_dataclass_kernel_arg(instance: Any) -> None:
    """Reject dataclasses whose stdlib options were customized beyond frozen/kw_only.

    The TLA frontend unpacks dataclass fields as kernel arguments and assumes the
    default dataclass semantics; options such as ``slots=True`` or ``init=False``
    would silently diverge from that, so they are rejected at compile time.
    """
    cls = type(instance)
    params = getattr(cls, "__dataclass_params__", None)
    if params is not None:
        for name, default in _DATACLASS_DEFAULT_ONLY_PARAMS:
            if hasattr(params, name) and getattr(params, name) != default:
                raise TlaLoweringError(
                    f"dataclass {cls.__name__} is used as a kernel argument but was "
                    f"declared with {name}={getattr(params, name)!r} (default "
                    f"{default!r}); only frozen= and kw_only= may be customized"
                )
    # ``slots=True`` is not recorded in ``_DataclassParams`` on some builds;
    # a slots dataclass defines ``__slots__`` on the class itself.
    if getattr(cls, "__slots__", None):
        raise TlaLoweringError(
            f"dataclass {cls.__name__} is used as a kernel argument but was declared "
            "with slots=True; only frozen= and kw_only= may be customized"
        )


def _dataclass_constexpr_field_names(value: Any) -> frozenset[str]:
    """Names of dataclass fields annotated ``tla.Constexpr[...]`` (compile-time)."""
    return frozenset(
        field.name
        for field in dataclasses.fields(value)
        if is_constexpr_annotation(field.type)
    )


_CONSTEXPR_RO_CLASS_CACHE: dict[tuple[Any, tuple[str, ...]], type] = {}


def _constexpr_readonly_dataclass_cls(
    cls: type, constexpr_names: frozenset[str]
) -> type:
    """Return a subclass of ``cls`` whose ``tla.Constexpr`` fields are read-only.

    The first write to a field (during ``__init__``) is allowed; any later
    assignment to a constexpr field raises ``AttributeError``, so compile-time
    constants cannot be reassigned inside a kernel body. ``super().__setattr__``
    preserves frozen-dataclass semantics for non-constexpr fields.
    """
    key = (cls, tuple(sorted(constexpr_names)))
    cached = _CONSTEXPR_RO_CLASS_CACHE.get(key)
    if cached is not None:
        return cached

    def __setattr__(self: Any, name: str, value: Any) -> None:
        if name in constexpr_names and name in self.__dict__:
            raise AttributeError(
                f"{cls.__name__}.{name} is a tla.Constexpr field and is read-only "
                "(compile-time constant)"
            )
        cls.__setattr__(self, name, value)

    ro_cls = type(f"{cls.__name__}ConstexprRO", (cls,), {"__setattr__": __setattr__})
    _CONSTEXPR_RO_CLASS_CACHE[key] = ro_cls
    return ro_cls


def _is_constexpr_annotation(annotation: Any) -> bool:
    if annotation is inspect._empty:
        return False
    return is_constexpr_annotation(annotation)


__all__ = [
    "TlaLoweringError",
    "LoweredTlaIR",
    "UnsupportedExecutionLowering",
    "lower_jit_to_tlair_by_execution",
    "lower_jit_to_tlair_module_by_execution",
]
