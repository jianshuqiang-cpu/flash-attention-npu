"""PyTree helpers for Tla frontend control-flow lowering.

This mirrors the role of ``base_dsl/utils/tree_utils.py`` for
branch-carried values in dynamic control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from ...runtime import TlaCoreAPIError


@dataclass(frozen=True)
class FrontendIfTreeSpec:
    kind: str
    keys: tuple[Any, ...] = ()
    children: tuple["FrontendIfTreeSpec", ...] = ()
    node_type: type[Any] | None = None
    metadata: Any = field(default=None, compare=False, repr=False)


def normalize_frontend_if_result_with_names(
    result: Any,
    carried_values: tuple[Any, ...],
    carried_names: tuple[str, ...],
    carried_specs: list[FrontendIfTreeSpec],
) -> Any:
    if not carried_values:
        return None
    if result is None:
        return return_carried_values(carried_values)
    values = frontend_if_outer_result_values(result, carried_values, carried_names)
    for index, (actual, expected_spec) in enumerate(zip(values, carried_specs)):
        actual_spec = frontend_if_tree_spec(actual)
        if actual_spec != expected_spec:
            raise TlaCoreAPIError(
                "Dynamic if branch result "
                f"{format_frontend_if_carried_name(carried_names, index)} has "
                "incompatible structure"
            )
    return return_carried_values(tuple(values))


def extract_frontend_if_yields(
    result: Any,
    carried_values: tuple[Any, ...],
    carried_specs: list[FrontendIfTreeSpec],
    carried_names: tuple[str, ...],
    carried_leaf_names: list[str],
    branch_name: str,
) -> list[Any]:
    if not carried_values:
        return []
    if result is None:
        result_values = carried_values
    else:
        result_values = frontend_if_outer_result_values(
            result, carried_values, carried_names, branch_name
        )
    mlir_values: list[Any] = []
    actual_leaf_names: list[str] = []
    for index, (actual, expected_spec, name) in enumerate(
        zip(result_values, carried_specs, carried_names)
    ):
        actual_leaves, actual_spec, leaf_names = flatten_frontend_if_tree(actual, name)
        if actual_spec != expected_spec:
            raise TlaCoreAPIError(
                f"Dynamic if {branch_name} branch result "
                f"{format_frontend_if_carried_name(carried_names, index)} has "
                "incompatible structure"
            )
        mlir_values.extend(actual_leaves)
        actual_leaf_names.extend(leaf_names)
    expected_mlir, _ = flatten_frontend_if_carried_mlir_values(
        carried_values, carried_names
    )
    for index, (actual, expected_value) in enumerate(zip(mlir_values, expected_mlir)):
        if str(actual.type) != str(expected_value.type):
            leaf_name = (
                actual_leaf_names[index]
                if index < len(actual_leaf_names)
                else carried_leaf_names[index]
            )
            raise TlaCoreAPIError(
                f"Dynamic if {branch_name} branch result {leaf_name!r} has type "
                f"{actual.type}, expected {expected_value.type}"
            )
    return mlir_values


def frontend_if_outer_result_values(
    result: Any,
    carried_values: tuple[Any, ...],
    carried_names: tuple[str, ...],
    branch_name: str | None = None,
) -> tuple[Any, ...]:
    values = result if isinstance(result, (list, tuple)) else (result,)
    if len(values) != len(carried_values):
        prefix = (
            f"Dynamic if {branch_name} branch" if branch_name else "Dynamic if branch"
        )
        raise TlaCoreAPIError(
            f"{prefix} returned {len(values)} value(s), "
            f"expected {len(carried_values)} for carried values: "
            f"{format_frontend_if_carried_names(carried_names)}"
        )
    return tuple(values)


def flatten_frontend_if_carried_mlir_values(
    carried_values: tuple[Any, ...], carried_names: tuple[str, ...]
) -> tuple[list[Any], list[str]]:
    mlir_values: list[Any] = []
    leaf_names: list[str] = []
    for value, name in zip(carried_values, carried_names):
        leaves, _, names = flatten_frontend_if_tree(value, name)
        mlir_values.extend(leaves)
        leaf_names.extend(names)
    return mlir_values, leaf_names


def flatten_frontend_if_tree(
    value: Any, name: str
) -> tuple[list[Any], FrontendIfTreeSpec, list[str]]:
    from ... import core_api as _core_api
    from ..typing import as_numeric

    # Bare int/float/bool enter SCF as Numeric.
    # ``bool`` is a subclass of ``int`` — check it before ``type is int``.
    if isinstance(value, bool) or type(value) is int or isinstance(value, float):
        value = as_numeric(value)
    if isinstance(value, tuple):
        leaves: list[Any] = []
        leaf_names: list[str] = []
        children: list[FrontendIfTreeSpec] = []
        for index, element in enumerate(value):
            child_leaves, child_spec, child_names = flatten_frontend_if_tree(
                element, f"{name}[{index}]"
            )
            leaves.extend(child_leaves)
            leaf_names.extend(child_names)
            children.append(child_spec)
        return (
            leaves,
            FrontendIfTreeSpec("tuple", children=tuple(children)),
            leaf_names,
        )
    if isinstance(value, list):
        leaves = []
        leaf_names = []
        children = []
        for index, element in enumerate(value):
            child_leaves, child_spec, child_names = flatten_frontend_if_tree(
                element, f"{name}[{index}]"
            )
            leaves.extend(child_leaves)
            leaf_names.extend(child_names)
            children.append(child_spec)
        return (
            leaves,
            FrontendIfTreeSpec("list", children=tuple(children)),
            leaf_names,
        )
    if isinstance(value, dict):
        leaves = []
        leaf_names = []
        children = []
        keys = tuple(value.keys())
        for key in keys:
            child_leaves, child_spec, child_names = flatten_frontend_if_tree(
                value[key], f"{name}[{key!r}]"
            )
            leaves.extend(child_leaves)
            leaf_names.extend(child_names)
            children.append(child_spec)
        return (
            leaves,
            FrontendIfTreeSpec("dict", keys=keys, children=tuple(children)),
            leaf_names,
        )
    if is_frontend_if_dataclass_instance(value):
        leaves = []
        leaf_names = []
        children = []
        field_names = tuple(field.name for field in fields(value))
        for field_name in field_names:
            child_leaves, child_spec, child_names = flatten_frontend_if_tree(
                getattr(value, field_name), f"{name}.{field_name}"
            )
            leaves.extend(child_leaves)
            leaf_names.extend(child_names)
            children.append(child_spec)
        return (
            leaves,
            FrontendIfTreeSpec(
                "dataclass",
                keys=field_names,
                children=tuple(children),
                node_type=type(value),
            ),
            leaf_names,
        )
    if is_frontend_if_dynamic_expression(value):
        values = [
            _core_api._as_branch_value(item) for item in value.__extract_mlir_values__()
        ]
        return (
            values,
            FrontendIfTreeSpec(
                "dynamic_expression",
                keys=tuple(range(len(values))),
                node_type=type(value),
                metadata=value,
            ),
            [f"{name}[{index}]" for index in range(len(values))],
        )
    return [_core_api._as_branch_value(value)], FrontendIfTreeSpec("leaf"), [name]


def frontend_if_tree_spec(value: Any) -> FrontendIfTreeSpec:
    from ..typing import as_numeric, Numeric

    # Same host-scalar rule as flatten_frontend_if_tree.
    if isinstance(value, bool) or type(value) is int or isinstance(value, float):
        value = as_numeric(value)
    if isinstance(value, tuple):
        return FrontendIfTreeSpec(
            "tuple", children=tuple(frontend_if_tree_spec(item) for item in value)
        )
    if isinstance(value, list):
        return FrontendIfTreeSpec(
            "list", children=tuple(frontend_if_tree_spec(item) for item in value)
        )
    if isinstance(value, dict):
        keys = tuple(value.keys())
        return FrontendIfTreeSpec(
            "dict",
            keys=keys,
            children=tuple(frontend_if_tree_spec(value[key]) for key in keys),
        )
    if is_frontend_if_dataclass_instance(value):
        field_names = tuple(field.name for field in fields(value))
        return FrontendIfTreeSpec(
            "dataclass",
            keys=field_names,
            children=tuple(
                frontend_if_tree_spec(getattr(value, field_name))
                for field_name in field_names
            ),
            node_type=type(value),
        )
    if is_frontend_if_dynamic_expression(value):
        # Host Numerics must not call ``__extract_mlir_values__`` (needs MLIR ctx).
        n = 1 if isinstance(value, Numeric) else len(value.__extract_mlir_values__())
        return FrontendIfTreeSpec(
            "dynamic_expression",
            keys=tuple(range(n)),
            node_type=type(value),
            metadata=value,
        )
    return FrontendIfTreeSpec("leaf")


def rebuild_frontend_if_carried_values(
    leaves: list[Any], specs: list[FrontendIfTreeSpec]
) -> tuple[Any, ...]:
    values: list[Any] = []
    cursor = 0
    for spec in specs:
        value, cursor = rebuild_frontend_if_tree(leaves, cursor, spec)
        values.append(value)
    if cursor != len(leaves):
        raise TlaCoreAPIError("Dynamic if produced extra carried result values")
    return tuple(values)


def rebuild_frontend_if_tree(
    leaves: list[Any], cursor: int, spec: FrontendIfTreeSpec
) -> tuple[Any, int]:
    if spec.kind == "leaf":
        if cursor >= len(leaves):
            raise TlaCoreAPIError("Dynamic if produced too few carried result values")
        return leaves[cursor], cursor + 1
    if spec.kind == "dynamic_expression":
        from ... import core_api as _core_api

        if spec.metadata is None:
            raise TlaCoreAPIError(
                "Dynamic if custom carried value lost reconstruction metadata"
            )
        value_count = len(spec.keys)
        next_cursor = cursor + value_count
        if next_cursor > len(leaves):
            raise TlaCoreAPIError("Dynamic if produced too few carried result values")
        values = [
            _core_api._as_branch_value(value) for value in leaves[cursor:next_cursor]
        ]
        return (
            spec.metadata.__new_from_mlir_values__(values),
            next_cursor,
        )
    values: list[Any] = []
    for child in spec.children:
        value, cursor = rebuild_frontend_if_tree(leaves, cursor, child)
        values.append(value)
    if spec.kind == "tuple":
        return tuple(values), cursor
    if spec.kind == "list":
        return values, cursor
    if spec.kind == "dict":
        return dict(zip(spec.keys, values)), cursor
    if spec.kind == "dataclass":
        if spec.node_type is None:
            raise TlaCoreAPIError(
                "Dynamic if dataclass carried value lost type metadata"
            )
        return spec.node_type(**dict(zip(spec.keys, values))), cursor
    raise TlaCoreAPIError(f"Unknown dynamic if carried value structure: {spec.kind}")


def is_frontend_if_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def is_frontend_if_dynamic_expression(value: Any) -> bool:
    return all(
        hasattr(value, attr)
        for attr in ("__extract_mlir_values__", "__new_from_mlir_values__")
    )


def normalize_frontend_if_carried_names(
    carried_names: tuple[str, ...] | list[str] | None, expected_count: int
) -> tuple[str, ...]:
    if carried_names is None:
        return tuple(str(index) for index in range(expected_count))
    names = tuple(str(name) for name in carried_names)
    if len(names) != expected_count:
        raise TlaCoreAPIError(
            "Dynamic if carried_names metadata has "
            f"{len(names)} name(s), expected {expected_count}"
        )
    return names


def format_frontend_if_carried_name(names: tuple[str, ...], index: int) -> str:
    if index < len(names) and not names[index].isdigit():
        return f"{names[index]!r}"
    return str(index)


def format_frontend_if_carried_names(names: tuple[str, ...]) -> str:
    display = [name if not name.isdigit() else f"result {name}" for name in names]
    return ", ".join(display) if display else "<none>"


def return_carried_values(values: tuple[Any, ...] | list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return tuple(values)
