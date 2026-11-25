"""Runtime executors for Tla AST-preprocessed frontend control flow."""

from __future__ import annotations

import builtins
import contextlib
import contextvars
import linecache
import operator
import types
from dataclasses import fields
from typing import Any, Callable, Iterator

from . import runtime as _runtime
from .base_dsl.ast_helpers import FrontendRange
from .base_dsl.utils import tree_utils

TlaCoreAPIError = _runtime.TlaCoreAPIError
_capture_caller_location = _runtime._capture_caller_location
_coerce_bool_value = _runtime._coerce_bool_value
_const_i1 = _runtime._const_i1
_resolve_frontend_bound_value = _runtime._resolve_frontend_bound_value
_SOURCE_INFO_ATTR = "__tladsl_source_info__"
_IN_DYNAMIC_LAZY_REGION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tla_in_dynamic_lazy_region", default=False
)


@contextlib.contextmanager
def _dynamic_lazy_region() -> Iterator[None]:
    token = _IN_DYNAMIC_LAZY_REGION.set(True)
    try:
        yield
    finally:
        _IN_DYNAMIC_LAZY_REGION.reset(token)


def _internal_unknown_effect_call(thunk: Callable[[], Any]) -> Any:
    if not _IN_DYNAMIC_LAZY_REGION.get():
        return thunk()
    info = _source_info_for(thunk) or {}
    error = SyntaxError("trace-time effect unknown for call in runtime-lazy operand")
    error.filename = str(info.get("filename") or "<unknown>")
    error.lineno = int(info.get("lineno") or 0)
    error.offset = int(info.get("col_offset") or 0) + 1
    error.text = str(info.get("source") or "")
    raise error


_SAFE_ATOMIC_TYPES = (type(None), bool, int, float, complex, str, bytes)
_SAFE_CONTAINER_TYPES = (tuple, list, dict, set, frozenset, range)
_TRUSTED_DSL_TYPES: frozenset[type] = frozenset()


def _exact_class(value: Any) -> type:
    """Return an object's class without invoking an overridden protocol."""

    return object.__getattribute__(value, "__class__")


def _register_trusted_dsl_types(types_: frozenset[type]) -> None:
    """Freeze genuine DSL value types once core_api has initialized."""

    global _TRUSTED_DSL_TYPES
    if _TRUSTED_DSL_TYPES:
        raise RuntimeError("trusted DSL types are already registered")
    _TRUSTED_DSL_TYPES = types_


def _is_exact_builtin_value(value: Any) -> bool:
    """Whether basic operations on the object cannot reach user overrides."""

    return _exact_class(value) in (*_SAFE_ATOMIC_TYPES, *_SAFE_CONTAINER_TYPES, slice)


def _is_safe_python_value(value: Any, seen: set[int] | None = None) -> bool:
    """Whether Python may inspect *value* without dispatching to user code."""

    value_class = _exact_class(value)
    if value_class in _SAFE_ATOMIC_TYPES:
        return True
    if value_class is slice:
        return all(
            item is None or _exact_class(item) is int
            for item in (value.start, value.stop, value.step)
        )
    if value_class not in _SAFE_CONTAINER_TYPES:
        return False
    seen = seen or set()
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    items = value.items() if value_class is dict else value
    if value_class is dict:
        return all(
            _is_safe_python_value(key, seen) and _is_safe_python_value(item, seen)
            for key, item in items
        )
    return all(_is_safe_python_value(item, seen) for item in items)


def _is_trusted_dsl_value(value: Any) -> bool:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    if isinstance(value, mlir_ir.Value):
        return True
    if _runtime._resolve_frontend_bound_category(value) is not None:
        return True
    return _exact_class(value) in _TRUSTED_DSL_TYPES


def _raise_unknown_protocol(kind: str, value: Any) -> None:
    raise SyntaxError(
        f"trace-time effect unknown for {kind} on {_exact_class(value).__name__} "
        "in runtime-lazy operand"
    )


def _internal_lazy_attribute(value: Any, name: str) -> Any:
    if not _IN_DYNAMIC_LAZY_REGION.get():
        return getattr(value, name)
    if not (_is_trusted_dsl_value(value) or _is_exact_builtin_value(value)):
        _raise_unknown_protocol("attribute read", value)
    return getattr(value, name)


def _internal_lazy_subscript(value: Any, index: Any) -> Any:
    if not _IN_DYNAMIC_LAZY_REGION.get():
        return value[index]
    if _is_trusted_dsl_value(value):

        def is_safe_dsl_index(candidate: Any) -> bool:
            if _exact_class(candidate) is tuple:
                return all(is_safe_dsl_index(item) for item in candidate)
            return _is_safe_python_value(candidate) or _is_trusted_dsl_value(candidate)

        if not is_safe_dsl_index(index):
            _raise_unknown_protocol("subscription index", index)
        return value[index]
    sequence_types = (tuple, list, str, bytes, range)
    if _exact_class(value) in sequence_types and (
        _exact_class(index) is int
        or (_exact_class(index) is slice and _is_safe_python_value(index))
    ):
        return value[index]
    if (
        _exact_class(value) is dict
        and _is_safe_python_value(value)
        and _is_safe_python_value(index)
    ):
        return value[index]
    _raise_unknown_protocol("subscription", value)


_LAZY_BINOPS = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mult": operator.mul,
    "MatMult": operator.matmul,
    "Div": operator.truediv,
    "FloorDiv": operator.floordiv,
    "Mod": operator.mod,
    "Pow": operator.pow,
    "LShift": operator.lshift,
    "RShift": operator.rshift,
    "BitOr": operator.or_,
    "BitXor": operator.xor,
    "BitAnd": operator.and_,
}
_LAZY_UNARYOPS = {
    "Invert": operator.invert,
    "UAdd": operator.pos,
    "USub": operator.neg,
}


def _internal_lazy_binop(op: str, left: Any, right: Any) -> Any:
    operation = _LAZY_BINOPS[op]
    if _IN_DYNAMIC_LAZY_REGION.get() and not (
        (_is_safe_python_value(left) and _is_safe_python_value(right))
        or (_is_trusted_dsl_value(left) and _is_trusted_dsl_value(right))
        or (_is_trusted_dsl_value(left) and _is_safe_python_value(right))
        or (_is_safe_python_value(left) and _is_trusted_dsl_value(right))
    ):
        _raise_unknown_protocol(f"binary {op}", left)
    return operation(left, right)


def _internal_lazy_unary(op: str, value: Any) -> Any:
    operation = _LAZY_UNARYOPS[op]
    if _IN_DYNAMIC_LAZY_REGION.get() and not (
        _is_safe_python_value(value) or _is_trusted_dsl_value(value)
    ):
        _raise_unknown_protocol(f"unary {op}", value)
    return operation(value)


class FrontendControlFlowLoweringError(RuntimeError):
    """Raised when an AST-generated control-flow helper fails during lowering."""


def _source_info_for(fn: Callable[..., Any]) -> dict[str, Any] | None:
    info = getattr(fn, _SOURCE_INFO_ATTR, None)
    return info if isinstance(info, dict) else None


def _traceback_lineno_for_code(exc: BaseException, code: types.CodeType) -> int | None:
    tb = exc.__traceback__
    best: int | None = None
    while tb is not None:
        if tb.tb_frame.f_code is code:
            best = int(tb.tb_lineno)
        tb = tb.tb_next
    return best


def _format_control_flow_error(fn: Callable[..., Any], exc: Exception) -> str | None:
    info = _source_info_for(fn)
    if info is None:
        return None
    filename = str(info.get("filename") or "<unknown>")
    fallback_lineno = int(info.get("lineno") or 0)
    helper_lineno = _traceback_lineno_for_code(exc, fn.__code__)
    lineno = helper_lineno if helper_lineno is not None else fallback_lineno
    if lineno <= 0:
        lineno = fallback_lineno
    source = linecache.getline(filename, lineno).strip()
    if not source:
        source = str(info.get("source") or "")
    construct = str(info.get("construct") or "control flow")
    region = str(info.get("region") or "region")
    message = (
        f"Execution-mode lowering failed in {construct} {region} at {filename}:{lineno}"
    )
    if source:
        message += f"\n  source: {source}"
    message += f"\n  reason: {_exact_class(exc).__name__}: {exc}"
    return message


def _wrap_control_flow_exception(
    fn: Callable[..., Any], exc: Exception
) -> Exception | None:
    if isinstance(exc, FrontendControlFlowLoweringError):
        return exc
    message = _format_control_flow_error(fn, exc)
    if message is None:
        return None
    try:
        from .execution_lowering import TlaLoweringError
    except ImportError:  # pragma: no cover - defensive for partial imports
        TlaLoweringError = ()  # type: ignore[assignment]
    if isinstance(exc, TlaCoreAPIError):
        return TlaCoreAPIError(message)
    if TlaLoweringError and isinstance(exc, TlaLoweringError):
        return TlaLoweringError(message)
    return FrontendControlFlowLoweringError(message)


def _call_with_control_flow_source(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except FrontendControlFlowLoweringError:
        raise
    except Exception as exc:
        wrapped = _wrap_control_flow_exception(fn, exc)
        if wrapped is None:
            raise
        raise wrapped from exc


class ScfGenerator:
    """Shared SCF construction helper for AST-preprocessed Tla control flow."""

    @staticmethod
    def _normalize_region_result_to_list(region_result: Any) -> list[Any]:
        if region_result is None:
            return []
        if isinstance(region_result, list):
            return region_result
        return [region_result]

    def scf_execute_dynamic(
        self,
        *,
        op_type_name: str,
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
        mix_iter_arg_names: list[str] | tuple[str, ...],
        create_op_func: Callable[[list[Any]], Any],
        region_builders: list[
            Callable[
                [
                    Any,
                    list[Any],
                    list[Any],
                    tuple[list[Any], list[str]],
                    list[Any] | tuple[Any, ...],
                    int,
                ],
                list[Any] | tuple[Any, ...],
            ]
        ],
        initial_ir_values: list[Any] | None = None,
        initial_pytree_def: tuple[list[Any], list[str]] | None = None,
        initial_ir_types: list[Any] | None = None,
        result_tensor_type_metadata: list[Any] | None = None,
        block_term_op_builder: dict[Callable[..., Any], Callable[..., None]]
        | None = None,
    ) -> Any:
        from mlir import ir as mlir_ir  # type: ignore[assignment]
        from mlir.dialects import scf  # type: ignore[import-not-found]
        from . import core_api as _core_api

        if initial_ir_values is None or initial_pytree_def is None:
            ir_values, pytree_def = _core_api.unpack_to_irvalue(
                mix_iter_args,
                op_type_name,
                full_write_args_count,
                mix_iter_arg_names,
            )
        else:
            ir_values = initial_ir_values
            pytree_def = initial_pytree_def
        expected_types = (
            initial_ir_types
            if initial_ir_types is not None
            else [value.type for value in ir_values]
        )
        custom_terminators = block_term_op_builder or {}

        def unpack_and_validate_region_values(region_result: Any) -> list[Any]:
            region_result_list = self._normalize_region_result_to_list(region_result)
            region_values, yield_pytree_def = _core_api.unpack_to_irvalue(
                region_result_list,
                op_type_name,
                full_write_args_count,
                mix_iter_arg_names,
            )
            if pytree_def[0] != yield_pytree_def[0]:
                if len(region_values) != len(expected_types):
                    name = _dynamic_region_structure_name(
                        pytree_def, yield_pytree_def, mix_iter_arg_names
                    )
                    raise TlaCoreAPIError(
                        f"Dynamic {op_type_name} region result {name} has "
                        "incompatible carried value structure"
                    )
                for actual, expected_type in zip(
                    region_values, expected_types, strict=False
                ):
                    if str(actual.type) != str(expected_type):
                        name = _dynamic_region_leaf_name(yield_pytree_def, 0)
                        raise TlaCoreAPIError(
                            f"Dynamic {op_type_name} region result {name} has type "
                            f"{actual.type}, expected {expected_type}"
                        )
            else:
                for index, (actual, expected_type) in enumerate(
                    zip(region_values, expected_types, strict=False)
                ):
                    if str(actual.type) != str(expected_type):
                        name = _dynamic_region_leaf_name(yield_pytree_def, index)
                        raise TlaCoreAPIError(
                            f"Dynamic {op_type_name} region result {name} has type "
                            f"{actual.type}, expected {expected_type}"
                        )
            return region_values

        op = create_op_func(ir_values)
        for index, builder in enumerate(region_builders):
            region = op.regions[index]
            block = region.blocks[0]
            with mlir_ir.InsertionPoint(block):
                region_result = builder(
                    op,
                    list(block.arguments),
                    ir_values,
                    pytree_def,
                    mix_iter_args,
                    full_write_args_count,
                )
                custom_terminator = custom_terminators.get(builder)
                if custom_terminator is not None:
                    custom_terminator(
                        region_result,
                        pytree_def,
                        expected_types,
                        mix_iter_args,
                        full_write_args_count,
                    )
                else:
                    region_values = unpack_and_validate_region_values(region_result)
                    scf.YieldOp(region_values)
        final_results = _core_api.pack_from_irvalue(
            list(op.results),
            pytree_def,
            mix_iter_args,
            full_write_args_count,
            tensor_type_metadata=result_tensor_type_metadata,
        )
        return tree_utils.return_carried_values(final_results)


def _internal_frontend_bool_and(*values: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    from .base_dsl.typing import Bool

    if not values:
        return True
    if all(isinstance(value, bool) for value in values):
        return all(values)
    current = _coerce_bool_value(values[0])
    for value in values[1:]:
        rhs = _coerce_bool_value(value)
        op = mlir_ir.Operation.create(
            "arith.andi", operands=[current, rhs], results=[current.type]
        )
        current = op.results[0]
    return Bool(current)


def _is_tensor_handle(value: Any) -> bool:
    """Whether a value owns a tla tensor's descriptor.

    The single place that decides what a tensor *is* for carrier analysis. A
    new frontend tensor representation has to be added here, or it silently
    becomes carriable again.
    """
    from .types import TlaTensor
    from .tla.tensor import _Tensor as _FrontendTensor

    return isinstance(value, (TlaTensor, _FrontendTensor))


def _find_tensor_leaves(value: Any, path: str, found: list[str]) -> None:
    """Collect the paths of every tensor inside ``value``.

    Mirrors the node cases of ``flatten_frontend_if_tree`` -- but tests for a
    tensor *before* descending, which is why it cannot reuse that function. A
    tensor satisfies ``is_frontend_if_dynamic_expression``, so flattening
    decomposes it into its descriptor SSA fields and no leaf is ever a tensor;
    a leaf-level isinstance check against the flattened result passes on
    exactly the trees this is meant to reject.
    """
    if _is_tensor_handle(value):
        found.append(path)
        return
    if isinstance(value, (tuple, list)):
        for index, element in enumerate(value):
            _find_tensor_leaves(element, f"{path}[{index}]", found)
        return
    if isinstance(value, dict):
        for key in value:
            _find_tensor_leaves(value[key], f"{path}[{key!r}]", found)
        return
    if tree_utils.is_frontend_if_dataclass_instance(value):
        for field in fields(value):
            _find_tensor_leaves(
                getattr(value, field.name), f"{path}.{field.name}", found
            )


# Method-only captures are restricted to objects from these packages: a method
# on one of them emits IR, rather than mutating host state at trace time.
_TLA_HANDLE_MODULES = frozenset({"catlass", "mlir"})


def _reject_host_captures(
    names: tuple[str, ...] | None,
    values: tuple[Any, ...] | None,
    construct: str,
) -> None:
    """Reject calling a method on a host object inside a runtime construct.

    A name the body only invokes a method on is not a carrier, so it is not
    rebound per iteration -- and the body is traced exactly once. For a TLA
    handle that is correct: ``tile.store(...)`` emits an op into the region and
    the op runs every iteration on device. For an ordinary Python object it is
    silently wrong: ``values.reverse()`` mutates the host list once, at trace
    time, and the emitted IR is independent of the runtime trip count.

    The two are indistinguishable syntactically -- whether a method touches
    device memory is a property of the object -- so the runtime value decides.

    A capture need not *be* a handle, only reach one: ``state.tile.store(...)``
    invokes on ``state``, and the method that emits the op belongs to the tile
    inside it, so a structure holding a handle is accepted too. That is as far
    as a value-based rule goes -- mutating a host container that happens to
    hold handles (``tiles.reverse()``) still traces once and is not caught
    here.
    """
    for name, value in zip(names or (), values or ()):
        root = (type(value).__module__ or "").split(".")[0]
        if root in _TLA_HANDLE_MODULES:
            continue
        reached: list[str] = []
        _find_tensor_leaves(value, name, reached)
        if reached:
            continue
        raise TlaCoreAPIError(
            f"tla.{construct}: {name!r} is a {type(value).__name__}, and the "
            f"body calls a method on it. Only tla handles may be used that "
            f"way: the body is traced once, so a method on a tla handle emits "
            f"an op that runs every iteration, while a method on a host object "
            f"mutates it once at trace time and the loop count never reaches "
            f"the generated code. Hoist the call out of the construct, or "
            f"carry the value the method would have produced."
        )


def _internal_frontend_for(
    range_value: Any,
    body_fn: Callable[..., Any],
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
    captured_names: tuple[str, ...] | None = None,
    captured_values: tuple[Any, ...] | None = None,
) -> Any:
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    if not isinstance(range_value, FrontendRange):
        raise TlaCoreAPIError(
            "for loops over tla.range require frontend range metadata"
        )
    all_names = tree_utils.normalize_frontend_if_carried_names(
        carried_names, len(carried_values)
    )
    _reject_host_captures(captured_names, captured_values, "range")
    loop_values = carried_values
    loop_names = all_names

    carried_specs = [tree_utils.frontend_if_tree_spec(value) for value in loop_values]
    _, carried_pytree_def = _core_api.unpack_to_irvalue(
        loop_values, "for", len(loop_values), loop_names
    )
    carried_leaf_names = carried_pytree_def[1]

    mlir_loc = _capture_caller_location()
    # ``as_numeric`` + promote to Int32 for ``scf.for`` bounds/IV.
    from .base_dsl.typing import Int32, as_numeric

    start = as_numeric(range_value.start).to(Int32).ir_value()
    end = as_numeric(range_value.end).to(Int32).ir_value()
    step = as_numeric(range_value.step).to(Int32).ir_value()

    generator = ScfGenerator()

    def create_for_op(ir_values: list[Any]) -> Any:
        return scf.ForOp(start, end, step, ir_values, loc=mlir_loc)

    def build_body(
        _op: Any,
        block_args: list[Any],
        _ir_values: list[Any],
        _pytree_def: tuple[list[Any], list[str]],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        _full_write_args_count: int,
    ) -> list[Any]:
        loop_args = _core_api.pack_from_irvalue(
            block_args[1:], carried_pytree_def, loop_values, len(loop_values)
        )
        # ``as_numeric(induction_variable)``; IV SSA is already i32.
        body_result = _call_with_control_flow_source(
            body_fn, as_numeric(block_args[0]), *list(loop_args)
        )
        return tree_utils.extract_frontend_if_yields(
            body_result,
            loop_values,
            carried_specs,
            loop_names,
            carried_leaf_names,
            "for",
        )

    results = generator.scf_execute_dynamic(
        op_type_name="for",
        mix_iter_args=loop_values,
        full_write_args_count=len(loop_values),
        mix_iter_arg_names=loop_names,
        create_op_func=create_for_op,
        region_builders=[build_body],
    )

    return results


def _while_execute_dynamic(
    while_before_block: Callable[..., Any],
    while_after_block: Callable[..., Any],
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
    captured_names: tuple[str, ...] | None = None,
    captured_values: tuple[Any, ...] | None = None,
    full_write_args_count: int | None = None,
) -> Any:
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    if _runtime._has_enclosing_region("vec.func"):
        raise TlaCoreAPIError(
            "while loops are not currently supported inside tla.vec.func()"
        )
    carried_count = (
        len(carried_values)
        if full_write_args_count is None or full_write_args_count == 0
        else full_write_args_count
    )
    carried_names_tuple = tree_utils.normalize_frontend_if_carried_names(
        carried_names, len(carried_values)
    )
    _reject_host_captures(captured_names, captured_values, "while")
    initial_ir_values, carried_pytree_def = _core_api.unpack_to_irvalue(
        carried_values, "while", carried_count, carried_names_tuple
    )
    expected_types = [value.type for value in initial_ir_values]

    def create_while_op(ir_values: list[Any]) -> Any:
        while_op = scf.WhileOp([value.type for value in ir_values], ir_values)
        while_op.before.blocks.append(*[value.type for value in ir_values])
        while_op.after.blocks.append(*[value.type for value in ir_values])
        return while_op

    def before_builder(
        _op: Any,
        block_args: list[Any],
        _ir_values: list[Any],
        pytree_def: tuple[list[Any], list[str]],
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> Any:
        before_args = _core_api.pack_from_irvalue(
            block_args, pytree_def, mix_iter_args, full_write_args_count
        )
        return _call_with_control_flow_source(while_before_block, *before_args)

    def before_terminator(
        cond_and_results: Any,
        pytree_def: tuple[list[Any], list[str]],
        expected_types: list[Any],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> None:
        cond, before_results = _normalize_while_before_result(cond_and_results)
        ir_cond = (
            _const_i1(int(bool(cond)))
            if _is_safe_python_value(cond)
            else _coerce_bool_value(cond)
        )
        result_values, result_pytree_def = _core_api.unpack_to_irvalue(
            before_results,
            "while",
            full_write_args_count,
            carried_names_tuple,
        )
        _validate_dynamic_while_results(
            result_values,
            result_pytree_def,
            pytree_def,
            expected_types,
            "condition",
        )
        scf.ConditionOp(ir_cond, result_values)

    def after_builder(
        _op: Any,
        block_args: list[Any],
        _ir_values: list[Any],
        pytree_def: tuple[list[Any], list[str]],
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> Any:
        after_args = _core_api.pack_from_irvalue(
            block_args, pytree_def, mix_iter_args, full_write_args_count
        )
        return _call_with_control_flow_source(while_after_block, *after_args)

    return ScfGenerator().scf_execute_dynamic(
        op_type_name="while",
        mix_iter_args=carried_values,
        full_write_args_count=carried_count,
        mix_iter_arg_names=carried_names_tuple,
        create_op_func=create_while_op,
        region_builders=[before_builder, after_builder],
        initial_ir_values=initial_ir_values,
        initial_pytree_def=carried_pytree_def,
        initial_ir_types=expected_types,
        block_term_op_builder={before_builder: before_terminator},
    )


def _normalize_while_before_result(result: Any) -> tuple[Any, list[Any]]:
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise TlaCoreAPIError(
            "Dynamic while condition block must return (condition, carried_values)"
        )
    cond = result[0]
    carried = result[1]
    if carried is None:
        return cond, []
    if isinstance(carried, list):
        return cond, carried
    if isinstance(carried, tuple):
        return cond, list(carried)
    return cond, [carried]


def _validate_dynamic_while_results(
    actual_values: list[Any],
    actual_pytree_def: tuple[list[Any], list[str]],
    expected_pytree_def: tuple[list[Any], list[str]],
    expected_types: list[Any],
    region_name: str,
) -> None:
    # Bare int/float/bool are promoted via as_numeric before specs are built.
    actual_specs = actual_pytree_def[0]
    expected_specs = expected_pytree_def[0]
    specs_ok = len(actual_specs) == len(expected_specs) and all(
        actual == expected
        for actual, expected in zip(actual_specs, expected_specs, strict=True)
    )
    if not specs_ok:
        name = _dynamic_region_structure_name(
            expected_pytree_def, actual_pytree_def, tuple()
        )
        raise TlaCoreAPIError(
            f"Dynamic while {region_name} region result {name} has incompatible "
            "carried value structure"
        )
    if len(actual_values) != len(expected_types):
        raise TlaCoreAPIError(
            f"Dynamic while {region_name} region produced "
            f"{len(actual_values)} value(s), expected {len(expected_types)}"
        )
    for index, (actual, expected_type) in enumerate(
        zip(actual_values, expected_types, strict=False)
    ):
        if str(actual.type) != str(expected_type):
            name = _dynamic_region_leaf_name(actual_pytree_def, index)
            raise TlaCoreAPIError(
                f"Dynamic while {region_name} region result {name} has type "
                f"{actual.type}, expected {expected_type}"
            )


def _dynamic_region_structure_name(
    expected_pytree_def: tuple[list[Any], list[str]],
    actual_pytree_def: tuple[list[Any], list[str]],
    carried_names: list[str] | tuple[str, ...],
) -> str:
    expected_specs = expected_pytree_def[0]
    actual_specs = actual_pytree_def[0]
    for index, (expected, actual) in enumerate(
        zip(expected_specs, actual_specs, strict=False)
    ):
        if expected != actual:
            if index < len(carried_names):
                return repr(carried_names[index])
            expected_leaf_names = expected_pytree_def[1]
            if expected_leaf_names:
                return repr(expected_leaf_names[0].split("[", 1)[0].split(".", 1)[0])
            actual_leaf_names = actual_pytree_def[1]
            if actual_leaf_names:
                return repr(actual_leaf_names[0].split("[", 1)[0].split(".", 1)[0])
            return f"at index {index}"
    return "values"


def _dynamic_region_leaf_name(
    pytree_def: tuple[list[Any], list[str]], index: int
) -> str:
    leaf_names = pytree_def[1]
    if index < len(leaf_names):
        return repr(leaf_names[index])
    return f"at index {index}"


def _internal_frontend_bool_or(*values: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    from .base_dsl.typing import Bool

    if not values:
        return False
    if all(isinstance(value, bool) for value in values):
        return any(values)
    current = _coerce_bool_value(values[0])
    for value in values[1:]:
        rhs = _coerce_bool_value(value)
        op = mlir_ir.Operation.create(
            "arith.ori", operands=[current, rhs], results=[current.type]
        )
        current = op.results[0]
    return Bool(current)


def _internal_frontend_bool_not(value: Any) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    from .base_dsl.typing import Bool

    if _is_safe_python_value(value):
        return not bool(value)
    if not _is_trusted_dsl_value(value):
        _raise_unknown_protocol("truth testing", value)
    operand = _coerce_bool_value(value)
    one = _const_i1(1)
    op = mlir_ir.Operation.create(
        "arith.xori", operands=[operand, one], results=[operand.type]
    )
    return Bool(op.results[0])


def _internal_frontend_bool(value: Any) -> Any:
    from .base_dsl.typing import Bool

    if isinstance(value, bool):
        return bool(value)
    return Bool(_coerce_bool_value(value))


def _internal_frontend_any(iterable: Any) -> Any:
    values = list(iterable)
    if not values:
        return False
    if all(isinstance(value, bool) for value in values):
        return any(values)
    return _internal_frontend_bool_or(*values)


def _internal_frontend_all(iterable: Any) -> Any:
    values = list(iterable)
    if not values:
        return True
    if all(isinstance(value, bool) for value in values):
        return all(values)
    return _internal_frontend_bool_and(*values)


def _internal_frontend_min(*values: Any) -> Any:
    return _internal_frontend_minmax("min", *values)


def _internal_frontend_max(*values: Any) -> Any:
    return _internal_frontend_minmax("max", *values)


def _internal_frontend_minmax(kind: str, *values: Any) -> Any:
    flat_values = _flatten_minmax_values(values)
    if not flat_values:
        raise TlaCoreAPIError(f"{kind}() expected at least one argument")
    if len(flat_values) == 1:
        return flat_values[0]
    if not any(_is_dynamic_index_like(value) for value in flat_values):
        op = builtins.min if kind == "min" else builtins.max
        return op(*flat_values)

    current = flat_values[0]
    for value in flat_values[1:]:
        current = _select_minmax_numeric(kind, current, value)
    return current


def _flatten_minmax_values(values: tuple[Any, ...]) -> list[Any]:
    if len(values) == 1 and not _is_dynamic_index_like(values[0]):
        try:
            return list(values[0])
        except TypeError:
            return [values[0]]
    flat_values: list[Any] = []
    for value in values:
        if _is_dynamic_index_like(value):
            flat_values.append(value)
            continue
        if isinstance(value, (str, bytes)):
            flat_values.append(value)
            continue
        try:
            flat_values.extend(value)
        except TypeError:
            flat_values.append(value)
    return flat_values


def _is_dynamic_index_like(value: Any) -> bool:
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    from .base_dsl.typing import Numeric

    if isinstance(value, Numeric):
        return isinstance(value.value, mlir_ir.Value)
    resolved = _resolve_frontend_bound_value(value)
    return isinstance(resolved, mlir_ir.Value) or isinstance(value, mlir_ir.Value)


def _select_minmax_numeric(kind: str, left: Any, right: Any) -> Any:
    from mlir.dialects import arith  # type: ignore[import-not-found]

    from .base_dsl.typing import Numeric, as_numeric

    lhs = left if isinstance(left, Numeric) else as_numeric(left)
    rhs = right if isinstance(right, Numeric) else as_numeric(right)
    if _exact_class(lhs) is not _exact_class(rhs):
        raise TlaCoreAPIError(
            f"{kind}() Numeric operands must share a type, "
            f"got {_exact_class(lhs).__name__} and {_exact_class(rhs).__name__}"
        )
    if isinstance(lhs.value, (int, bool)) and isinstance(rhs.value, (int, bool)):
        pick_left = (
            int(lhs.value) < int(rhs.value)
            if kind == "min"
            else int(lhs.value) > int(rhs.value)
        )
        return lhs if pick_left else rhs
    cond = (lhs < rhs) if kind == "min" else (lhs > rhs)
    selected = arith.SelectOp(
        _coerce_bool_value(cond),
        lhs.ir_value(),
        rhs.ir_value(),
    ).result
    return _exact_class(lhs)(selected)


def _internal_frontend_compare(
    left: Any, comparators: tuple[Any, ...], ops: tuple[str, ...]
) -> Any:
    if len(comparators) != len(ops):
        raise TlaCoreAPIError(
            "Comparison metadata mismatch: "
            f"{len(comparators)} comparator(s), {len(ops)} operator(s)"
        )
    current = left
    results: list[Any] = []
    for op, comparator in zip(ops, comparators, strict=False):
        results.append(_internal_frontend_compare_pair(current, comparator, op))
        current = comparator
    if not results:
        return True
    # A single comparison returns its result directly (``Bool`` Numeric for
    # scalar comparisons, or a MaskSSA for vector comparisons). Equivalent to the
    # bool-and path for one element, and lets vector masks pass through.
    if len(results) == 1:
        return results[0]
    if all(isinstance(result, bool) for result in results):
        return all(results)
    return _internal_frontend_bool_and(*results)


def _internal_frontend_compare_pair(left: Any, right: Any, op: str) -> Any:
    from mlir.dialects import arith  # type: ignore[import-not-found]

    if (
        op not in {"is", "is not"}
        and _IN_DYNAMIC_LAZY_REGION.get()
        and not (
            (_is_safe_python_value(left) and _is_safe_python_value(right))
            or (_is_trusted_dsl_value(left) and _is_trusted_dsl_value(right))
            or (_is_trusted_dsl_value(left) and _is_safe_python_value(right))
            or (_is_safe_python_value(left) and _is_trusted_dsl_value(right))
        )
    ):
        _raise_unknown_protocol(f"comparison {op}", left)
    if op == "==":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.eq, op)
    if op == "!=":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.ne, op)
    if op == "<":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.slt, op)
    if op == "<=":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.sle, op)
    if op == ">":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.sgt, op)
    if op == ">=":
        return _compare_index_or_python(left, right, arith.CmpIPredicate.sge, op)
    if op == "is":
        return left is right
    if op == "is not":
        return left is not right
    if op == "in":
        return left in right
    if op == "not in":
        return left not in right
    raise TlaCoreAPIError(f"Unsupported comparison operator: {op}")


def _compare_index_or_python(left: Any, right: Any, predicate: Any, op: str) -> Any:
    del predicate  # Numeric / as_numeric path emits typed cmpi; unused for bare Values.
    from mlir import ir as mlir_ir  # type: ignore[assignment]

    from .base_dsl.typing import Numeric

    _PY_COMPARE = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    def _python_compare(a: Any, b: Any) -> Any:
        fn = _PY_COMPARE.get(op)
        if fn is None:
            raise TlaCoreAPIError(f"Unsupported comparison operator: {op}")
        return fn(a, b)

    # Numeric SSA binds an ``ir.Value``; check the wrapper *before* resolving so
    # we use typed ``Numeric.__lt__``/… instead of index coercion.
    # ``if value < 0`` then emits element-typed ``arith.cmpi`` (e.g. i32).
    # Bare MLIR values (rare) go through ``as_numeric``.
    if isinstance(left, Numeric) or isinstance(right, Numeric):
        return _python_compare(left, right)

    lhs = _resolve_frontend_bound_value(left)
    rhs = _resolve_frontend_bound_value(right)
    if lhs is None:
        lhs = left
    if rhs is None:
        rhs = right
    if isinstance(lhs, mlir_ir.Value) or isinstance(rhs, mlir_ir.Value):
        from .base_dsl.typing import as_numeric

        return _python_compare(as_numeric(lhs), as_numeric(rhs))
    return _python_compare(left, right)


def _internal_frontend_if(
    condition: Any,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any] | None,
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
    captured_names: tuple[str, ...] | None = None,
    captured_values: tuple[Any, ...] | None = None,
) -> Any:
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    carried_names_tuple = tree_utils.normalize_frontend_if_carried_names(
        carried_names, len(carried_values)
    )
    carried_specs = [
        tree_utils.frontend_if_tree_spec(value) for value in carried_values
    ]
    if _is_safe_python_value(condition):
        selected = then_fn if condition else else_fn
        if selected is None:
            return tree_utils.return_carried_values(carried_values)
        result = _call_with_control_flow_source(selected, *carried_values)
        return tree_utils.normalize_frontend_if_result_with_names(
            result, carried_values, carried_names_tuple, carried_specs
        )

    if not _is_trusted_dsl_value(condition):
        _raise_unknown_protocol("truth testing", condition)

    cond = _coerce_bool_value(condition)
    _reject_host_captures(captured_names, captured_values, "if")
    carried_mlir, carried_pytree_def = _core_api.unpack_to_irvalue(
        carried_values, "if", len(carried_values), carried_names_tuple
    )
    carried_leaf_names = carried_pytree_def[1]
    has_else = else_fn is not None or bool(carried_mlir)

    def create_if_op(ir_values: list[Any]) -> Any:
        return scf.IfOp(cond, [value.type for value in ir_values], hasElse=has_else)

    def then_builder(
        _op: Any,
        _block_args: list[Any],
        ir_values: list[Any],
        pytree_def: tuple[list[Any], list[str]],
        mix_iter_args: list[Any] | tuple[Any, ...],
        full_write_args_count: int,
    ) -> list[Any]:
        then_args = _core_api.pack_from_irvalue(
            ir_values, pytree_def, mix_iter_args, full_write_args_count
        )
        then_result = _call_with_control_flow_source(then_fn, *then_args)
        return tree_utils.extract_frontend_if_yields(
            then_result,
            carried_values,
            carried_specs,
            carried_names_tuple,
            carried_leaf_names,
            "then",
        )

    region_builders: list[
        Callable[
            [
                Any,
                list[Any],
                list[Any],
                tuple[list[Any], list[str]],
                list[Any] | tuple[Any, ...],
                int,
            ],
            list[Any] | tuple[Any, ...],
        ]
    ] = [then_builder]
    if has_else:

        def else_builder(
            _op: Any,
            _block_args: list[Any],
            ir_values: list[Any],
            pytree_def: tuple[list[Any], list[str]],
            mix_iter_args: list[Any] | tuple[Any, ...],
            full_write_args_count: int,
        ) -> list[Any]:
            if else_fn is None:
                return ir_values
            else_args = _core_api.pack_from_irvalue(
                ir_values, pytree_def, mix_iter_args, full_write_args_count
            )
            else_result = _call_with_control_flow_source(else_fn, *else_args)
            return tree_utils.extract_frontend_if_yields(
                else_result,
                carried_values,
                carried_specs,
                carried_names_tuple,
                carried_leaf_names,
                "else",
            )

        region_builders.append(else_builder)

    return ScfGenerator().scf_execute_dynamic(
        op_type_name="if",
        mix_iter_args=carried_values,
        full_write_args_count=len(carried_values),
        mix_iter_arg_names=carried_names_tuple,
        create_op_func=create_if_op,
        region_builders=region_builders,
    )


def _internal_frontend_if_expr(
    condition: Any, true_fn: Callable[[], Any], false_fn: Callable[[], Any]
) -> Any:
    from mlir import ir as mlir_ir  # type: ignore[assignment]
    from mlir.dialects import scf  # type: ignore[import-not-found]
    from . import core_api as _core_api

    if _is_safe_python_value(condition):
        selected = true_fn if bool(condition) else false_fn
        return _call_with_control_flow_source(selected)

    if not _is_trusted_dsl_value(condition):
        _raise_unknown_protocol("truth testing", condition)

    cond = _coerce_bool_value(condition)

    execution_region = scf.ExecuteRegionOp(result=[])
    execution_region.region.blocks.append()
    with mlir_ir.InsertionPoint(execution_region.region.blocks[0]):
        with _dynamic_lazy_region():
            true_probe = _call_with_control_flow_source(true_fn)
        true_mlir, result_pytree_def = _core_api.unpack_to_irvalue(
            [true_probe], "if expression", 1, ["if expression"]
        )
        result_spec = result_pytree_def[0][0]
        true_leaf_names = result_pytree_def[1]
        with _dynamic_lazy_region():
            false_probe = _call_with_control_flow_source(false_fn)
        false_mlir, false_pytree_def = _core_api.unpack_to_irvalue(
            [false_probe], "if expression", 1, ["if expression"]
        )
        false_spec = false_pytree_def[0][0]
        false_leaf_names = false_pytree_def[1]
        result_types = [value.type for value in true_mlir]
        result_tensor_type_metadata = _core_api._collect_tla_tensor_type_metadata(
            true_mlir
        )
        result_type_names = [str(value_type) for value_type in result_types]
        _validate_if_expr_branch(
            false_mlir,
            false_spec,
            false_leaf_names,
            result_spec,
            result_type_names,
            true_leaf_names,
            "else",
        )
    execution_region.operation.erase()

    def create_if_op(_ir_values: list[Any]) -> Any:
        return scf.IfOp(cond, result_types, hasElse=True)

    def then_builder(
        _op: Any,
        _block_args: list[Any],
        _ir_values: list[Any],
        _pytree_def: tuple[list[Any], list[str]],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        _full_write_args_count: int,
    ) -> list[Any]:
        with _dynamic_lazy_region():
            true_result = _call_with_control_flow_source(true_fn)
        true_values, true_pytree_def = _core_api.unpack_to_irvalue(
            [true_result], "if expression", 1, ["if expression"]
        )
        true_spec = true_pytree_def[0][0]
        true_names = true_pytree_def[1]
        _validate_if_expr_branch(
            true_values,
            true_spec,
            true_names,
            result_spec,
            result_type_names,
            true_leaf_names,
            "then",
        )
        return true_values

    def else_builder(
        _op: Any,
        _block_args: list[Any],
        _ir_values: list[Any],
        _pytree_def: tuple[list[Any], list[str]],
        _mix_iter_args: list[Any] | tuple[Any, ...],
        _full_write_args_count: int,
    ) -> list[Any]:
        with _dynamic_lazy_region():
            false_result = _call_with_control_flow_source(false_fn)
        false_values, false_pytree_def = _core_api.unpack_to_irvalue(
            [false_result], "if expression", 1, ["if expression"]
        )
        false_spec = false_pytree_def[0][0]
        false_names = false_pytree_def[1]
        _validate_if_expr_branch(
            false_values,
            false_spec,
            false_names,
            result_spec,
            result_type_names,
            true_leaf_names,
            "else",
        )
        return false_values

    return ScfGenerator().scf_execute_dynamic(
        op_type_name="if",
        mix_iter_args=[None],
        full_write_args_count=1,
        mix_iter_arg_names=["if expression"],
        create_op_func=create_if_op,
        region_builders=[then_builder, else_builder],
        initial_ir_values=[None] * len(result_types),
        initial_pytree_def=result_pytree_def,
        initial_ir_types=result_types,
        result_tensor_type_metadata=result_tensor_type_metadata,
    )


def _validate_if_expr_branch(
    actual_values: list[Any],
    actual_spec: tree_utils.FrontendIfTreeSpec,
    actual_names: list[str],
    expected_spec: tree_utils.FrontendIfTreeSpec,
    expected_type_names: list[str],
    expected_names: list[str],
    branch_name: str,
) -> None:
    if actual_spec != expected_spec:
        raise TlaCoreAPIError(
            f"Conditional expression {branch_name} branch has incompatible structure"
        )
    if len(actual_values) != len(expected_type_names):
        raise TlaCoreAPIError(
            f"Conditional expression {branch_name} branch returned "
            f"{len(actual_values)} value(s), expected {len(expected_type_names)}"
        )
    for index, (actual, expected_type_name) in enumerate(
        zip(actual_values, expected_type_names)
    ):
        if str(actual.type) != expected_type_name:
            leaf_name = actual_names[index] if index < len(actual_names) else None
            if leaf_name is None and index < len(expected_names):
                leaf_name = expected_names[index]
            suffix = f" for {leaf_name!r}" if leaf_name is not None else ""
            raise TlaCoreAPIError(
                f"Conditional expression {branch_name} branch result{suffix} has "
                f"type {actual.type}, expected {expected_type_name}"
            )
