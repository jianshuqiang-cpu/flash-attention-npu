#!/usr/bin/env python3
"""Generate TLA DSL Kernel API reference docs from ``core_api.py`` (AST only).

API docstrings carry ``Directory:`` plus Description / Parameters / Constraints /
Example. Section order and blurbs live in ``DIRECTORY_SECTIONS`` below.
Writes English Markdown to ``docs/en/kernel_api_reference.md`` only.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent


PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # python/tla_dsl
CORE_API_PATH = PACKAGE_ROOT / "catlass" / "core_api.py"
TENSOR_API_PATH = PACKAGE_ROOT / "catlass" / "tla" / "tensor.py"
OUTPUT_PATH = PACKAGE_ROOT / "docs" / "en" / "kernel_api_reference.md"

# Drop from prototypes: MLIR location plumbing, not a user-facing API argument.
HIDDEN_PARAMETERS = frozenset({"loc"})
SECTION_LABELS = ("Description", "Parameters", "Constraints", "Example")
ROOT_TITLE = "API Reference"

# (directory path, section intro). Order defines the TOC.
DIRECTORY_SECTIONS: list[tuple[str, str]] = [
    (
        "Basic Data Types and Operations",
        "Construction and views for front-end structured values such as "
        "Shape / Coord / Stride / Layout / Tensor, plus pointer helpers.",
    ),
    (
        "Data Movement",
        "Tensor copies between on-chip and global memory, and UB register load/store.",
    ),
    ("Matrix Compute", "Cube-side matrix multiply-accumulate (`tla.mmad`)."),
    (
        "Vector Compute",
        "Compute and mask ops on the register-vector path; usually must be "
        "called inside `tla.vec.func()`.",
    ),
    ("Vector Compute / Mask Compute", "Mask creation and tail-mask updates."),
    (
        "Vector Compute / Basic Arithmetic",
        "Element-wise arithmetic and unary math ops. `VectorSSA` overloads "
        "`+` / `-` / `*` / `/` for `add` / `sub` / `mul` / `div` when no "
        "`mask=` is needed.",
    ),
    ("Vector Compute / Logical Compute", "Bitwise and logical ops on Mask / Vector."),
    (
        "Vector Compute / Compare and Select",
        "Vector compares that produce masks, and masked select.",
    ),
    (
        "Vector Compute / Data Fill",
        "Constant fill and lane-index sequence construction.",
    ),
    (
        "Vector Compute / Discrete and Aggregate",
        "Gather elements from a UB tensor by index.",
    ),
    (
        "Vector Compute / Data Rearrange",
        "Interleave / deinterleave and related lane reshuffles.",
    ),
    ("Vector Compute / Data Compress", "Compress valid lanes under a mask."),
    (
        "Sync Control",
        "In-core / cross-core flags, pipe barriers, mutexes, and local-memory "
        "barriers.",
    ),
    (
        "System Variable Access",
        "Architecture attributes on `tla.arch` (layout tags, pipe identifiers, "
        "block helpers, etc.).",
    ),
    ("Resource Management", "On-chip scratch allocation via `allocate`."),
    (
        "Debug APIs",
        "In-kernel scalar / tensor debug printing.",
    ),
    (
        "Scopes and Control Flow",
        "Cube / Vector / `vec.func` regions and kernel-side loop ranges.",
    ),
]
DIRECTORY_ORDER = [path for path, _ in DIRECTORY_SECTIONS]
DIRECTORY_INTROS = dict(DIRECTORY_SECTIONS)


@dataclass(frozen=True)
class DocNode:
    title: str
    apis: tuple[str, ...] = ()
    children: tuple["DocNode", ...] = ()
    intro: str = ""


@dataclass
class ParamInfo:
    name: str
    annotation: str
    kind: str  # positional | var_positional | keyword_only | var_keyword
    default: str | None


@dataclass
class APIEntry:
    name: str
    qualified_name: str
    source_line: int | None
    params: list[ParamInfo] = field(default_factory=list)
    returns: str = ""
    docstring: str = ""
    is_class: bool = False
    is_namespace: bool = False
    source_path: Path | None = None


class MissingAPIDocError(RuntimeError):
    """Public API is missing required English docstring sections."""


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _unparse(node: ast.AST | None, *, missing: str = "Any") -> str:
    if node is None:
        return missing
    try:
        return ast.unparse(node).replace("typing.", "")
    except Exception:
        return missing


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _decorator_basename(dec: ast.expr) -> str | None:
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return None


def _has_dsl_user_op(decorators: list[ast.expr]) -> bool:
    return any(_decorator_basename(d) == "dsl_user_op" for d in decorators)


def _params_from_args(args: ast.arguments) -> list[ParamInfo]:
    params: list[ParamInfo] = []
    pos_args = list(args.posonlyargs) + list(args.args)
    default_offset = len(pos_args) - len(args.defaults)

    for index, arg in enumerate(pos_args):
        default = None
        if index >= default_offset:
            default = _unparse(args.defaults[index - default_offset], missing="...")
        params.append(
            ParamInfo(arg.arg, _unparse(arg.annotation), "positional", default)
        )

    if args.vararg is not None:
        params.append(
            ParamInfo(
                args.vararg.arg,
                _unparse(args.vararg.annotation),
                "var_positional",
                None,
            )
        )

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        default = _unparse(default_node, missing="...") if default_node else None
        params.append(
            ParamInfo(arg.arg, _unparse(arg.annotation), "keyword_only", default)
        )

    if args.kwarg is not None:
        params.append(
            ParamInfo(
                args.kwarg.arg,
                _unparse(args.kwarg.annotation),
                "var_keyword",
                None,
            )
        )

    return [p for p in params if p.name not in HIDDEN_PARAMETERS]


def _function_entry(
    name: str,
    node: ast.FunctionDef,
    *,
    qualified_name: str | None = None,
    source_path: Path | None = None,
    drop_self: bool = False,
) -> APIEntry:
    params = _params_from_args(node.args)
    if drop_self:
        params = [p for p in params if p.name != "self"]
    return APIEntry(
        name=name,
        qualified_name=qualified_name or f"catlass.core_api.{name}",
        source_line=node.lineno,
        params=params,
        returns=_unparse(node.returns, missing="") if node.returns else "",
        docstring=ast.get_docstring(node) or "",
        source_path=source_path,
    )


def _parse_all_names(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return {
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    return set()


def _unary_op_template(tree: ast.Module) -> APIEntry | None:
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_make_unary_op":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.FunctionDef)
                and child.name == "_unary"
                and _has_dsl_user_op(child.decorator_list)
            ):
                return _function_entry("_unary", child)
    return None


def _doc_from_make_unary_call(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "doc":
            return _const_str(kw.value) or ""
    if len(call.args) >= 2:
        return _const_str(call.args[1]) or ""
    return ""


# ---------------------------------------------------------------------------
# Collect API entries
# ---------------------------------------------------------------------------


def _collect_core_functions(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_vec_func" and _has_dsl_user_op(node.decorator_list):
            entry = _function_entry("vec.func", node)
            entry.qualified_name = "catlass.core_api._vec_func"
            entries["vec.func"] = entry
            continue
        if node.name not in exported:
            continue
        if node.name == "print" or _has_dsl_user_op(node.decorator_list):
            entries[node.name] = _function_entry(node.name, node)
    return entries


def _collect_unary_aliases(
    tree: ast.Module, exported: set[str], template: APIEntry | None
) -> dict[str, APIEntry]:
    if template is None:
        return {}
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or target.id not in exported:
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_make_unary_op"
        ):
            continue
        name = target.id
        entries[name] = APIEntry(
            name=name,
            qualified_name=f"catlass.core_api.{name}",
            source_line=node.lineno,
            params=list(template.params),
            returns=template.returns,
            docstring=_doc_from_make_unary_call(value),
        )
    return entries


def _collect_arch_namespace(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    if "arch" not in exported:
        return {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "arch"
            and target.attr == "__doc__"
        ):
            return {
                "arch": APIEntry(
                    name="arch",
                    qualified_name="catlass.core_api.arch",
                    source_line=node.lineno,
                    docstring=_const_str(value) or "",
                    is_namespace=True,
                )
            }
    return {
        "arch": APIEntry(
            name="arch",
            qualified_name="catlass.core_api.arch",
            source_line=None,
            docstring="",
            is_namespace=True,
        )
    }


def _collect_tensor_methods(path: Path) -> dict[str, APIEntry]:
    """Document Tensor methods that are ``@dsl_user_op`` and have ``Directory:``."""
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tensor_cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in {"Tensor", "_Tensor"}
        ),
        None,
    )
    if tensor_cls is None:
        return {}

    entries: dict[str, APIEntry] = {}
    for node in tensor_cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _has_dsl_user_op(node.decorator_list):
            continue
        doc = ast.get_docstring(node) or ""
        if directory_path(doc) is None:
            continue
        entries[f"Tensor.{node.name}"] = _function_entry(
            f"Tensor.{node.name}",
            node,
            qualified_name=f"catlass.tla.tensor.{tensor_cls.name}.{node.name}",
            source_path=path.resolve(),
            drop_self=True,
        )
    return entries


def parse_core_api(path: Path) -> dict[str, APIEntry]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    exported = _parse_all_names(tree)
    entries: dict[str, APIEntry] = {}
    entries.update(_collect_core_functions(tree, exported))
    entries.update(_collect_unary_aliases(tree, exported, _unary_op_template(tree)))
    entries.update(_collect_arch_namespace(tree, exported))
    entries.update(_collect_tensor_methods(TENSOR_API_PATH))
    return entries


# ---------------------------------------------------------------------------
# Docstring / TOC
# ---------------------------------------------------------------------------


def directory_path(doc: str) -> str | None:
    """Return the ``Directory:`` path, or None if missing."""
    if not doc:
        return None
    match = re.search(r"(?m)^Directory:\s*(.+?)\s*$", dedent(doc).strip())
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split("/") if p.strip()]
    return " / ".join(parts) if parts else None


def docstring_sections(doc: str) -> dict[str, str]:
    """Split a docstring into the four fixed English sections."""
    sections = {label: "" for label in SECTION_LABELS}
    if not doc:
        return sections
    text = re.sub(r"(?m)^\s*Examples\s*:", "Example:", dedent(doc).strip())
    # Only match horizontal space after the colon. A trailing ``\s*`` would also
    # consume the next line's leading indent and leave sibling bullets nested
    # (``- a`` then ``    - b``).
    hits = [
        (label, m.start(), m.end())
        for label in SECTION_LABELS
        for m in [re.search(rf"(?m)^[ \t]*{label}[ \t]*:", text)]
        if m
    ]
    hits.sort(key=lambda item: item[1])
    for i, (label, _start, content_start) in enumerate(hits):
        content_end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        body = dedent(text[content_start:content_end]).strip()
        if label == "Example":
            code = re.search(r"```(?:python)?\n(.*?)```", body, flags=re.S)
            sections[label] = dedent(code.group(1)).strip() if code else body
        else:
            sections[label] = body
    return sections


def require_docs(entries: dict[str, APIEntry]) -> None:
    problems: list[str] = []
    for name in sorted(entries):
        entry = entries[name]
        loc = entry.qualified_name + (
            f" (approx. L{entry.source_line})" if entry.source_line else ""
        )
        if directory_path(entry.docstring) is None:
            problems.append(f"- `{name}` @ {loc}: missing `Directory:`")
        sections = docstring_sections(entry.docstring)
        for label in SECTION_LABELS:
            if not sections[label]:
                problems.append(f"- `{name}` @ {loc}: missing or empty `{label}:`")
    if problems:
        raise MissingAPIDocError(
            "Incomplete API docs. Add `Directory:` plus docstring sections "
            "`Description:` / `Parameters:` / `Constraints:` / `Example:` "
            "in source:\n" + "\n".join(problems)
        )


def _path_prefixes(path: str) -> list[str]:
    parts = path.split(" / ")
    return [" / ".join(parts[:i]) for i in range(1, len(parts) + 1)]


def build_doc_tree(entries: dict[str, APIEntry]) -> DocNode:
    """Group APIs by ``Directory:`` and order sections via ``DIRECTORY_SECTIONS``."""
    leaf_apis: dict[str, list[str]] = {}
    first_line: dict[str, int] = {}
    ranked = {path: idx for idx, path in enumerate(DIRECTORY_ORDER)}

    # Within a leaf: definition order (file path, then line). No per-API catalog.
    pending = sorted(
        (
            (
                str(entry.source_path or CORE_API_PATH),
                entry.source_line or 0,
                name,
                directory_path(entry.docstring),
            )
            for name, entry in entries.items()
        )
    )
    for _path, line, name, directory in pending:
        if not directory:
            continue
        leaf_apis.setdefault(directory, []).append(name)
        first_line.setdefault(directory, line)

    path_set: set[str] = set()
    for leaf in leaf_apis:
        path_set.update(_path_prefixes(leaf))
    for path in DIRECTORY_INTROS:
        path_set.update(_path_prefixes(path))

    def order_key(path: str) -> tuple[int, int, str]:
        if path in ranked:
            return (0, ranked[path], path)
        prefix = f"{path} / "
        child_ranks = [ranked[p] for p in ranked if p.startswith(prefix)]
        if child_ranks:
            return (0, min(child_ranks), path)
        return (1, first_line.get(path, 10**9), path)

    def build_node(path: str) -> DocNode:
        title = path.rsplit(" / ", 1)[-1]
        prefix = f"{path} / "
        child_titles = sorted(
            {
                candidate[len(prefix) :]
                for candidate in path_set
                if candidate.startswith(prefix)
                and " / " not in candidate[len(prefix) :]
            },
            key=lambda t: order_key(prefix + t),
        )
        children = tuple(build_node(prefix + t) for t in child_titles)
        return DocNode(
            title=title,
            apis=() if children else tuple(leaf_apis.get(path, ())),
            children=children,
            intro=DIRECTORY_INTROS.get(path, ""),
        )

    top = sorted({p for p in path_set if " / " not in p}, key=order_key)
    return DocNode(title=ROOT_TITLE, children=tuple(build_node(p) for p in top))


# ---------------------------------------------------------------------------
# Render Markdown
# ---------------------------------------------------------------------------


def format_signature(entry: APIEntry) -> str:
    if entry.is_namespace:
        return f"tla.{entry.name}"
    if entry.is_class:
        return f"class tla.{entry.name}"

    parts: list[str] = []
    saw_kwonly = False
    for param in entry.params:
        if param.kind == "keyword_only" and not saw_kwonly:
            parts.append("*")
            saw_kwonly = True
        if param.kind == "var_positional":
            piece = f"*{param.name}: {param.annotation}"
        elif param.kind == "var_keyword":
            piece = f"**{param.name}: {param.annotation}"
        else:
            piece = f"{param.name}: {param.annotation}"
            if param.default is not None:
                piece += f" = {param.default}"
        parts.append(piece)

    callee = (
        f"tile.{entry.name.split('.', 1)[1]}"
        if entry.name.startswith("Tensor.")
        else f"tla.{entry.name}"
    )
    sig = f"{callee}({', '.join(parts)})"
    return f"{sig} -> {entry.returns}" if entry.returns else sig


def source_markdown(entry: APIEntry, docs_dir: Path) -> str:
    if entry.source_line is None:
        return f"**Source:** `{entry.qualified_name}`"
    src = entry.source_path or CORE_API_PATH
    rel = Path(os_path_rel(src, docs_dir))
    return (
        f"**Source:** [`{entry.qualified_name}`]({rel.as_posix()}#L{entry.source_line})"
    )


def os_path_rel(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)


def _anchor(title: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff\- ]", "", title).strip().replace(" ", "-").lower()


def _numbered_title(number: str | None, title: str) -> str:
    if not number:
        return title
    return f"{number} {title}" if "." in number else f"{number}. {title}"


def render_entry(entry: APIEntry, heading_level: int, docs_dir: Path) -> str:
    sections = docstring_sections(entry.docstring)
    for label in SECTION_LABELS:
        if not sections[label]:
            raise MissingAPIDocError(
                f"`{entry.name}` missing {label}:; please update {entry.qualified_name}"
            )
    h = "#" * heading_level
    return "\n".join(
        [
            f"{h} `{entry.name}`",
            "",
            source_markdown(entry, docs_dir),
            "",
            "Description:",
            "",
            sections["Description"],
            "",
            "Prototype:",
            "",
            "```python",
            format_signature(entry),
            "```",
            "",
            "Parameters:",
            "",
            sections["Parameters"],
            "",
            "Constraints:",
            "",
            sections["Constraints"],
            "",
            "Example:",
            "",
            "```python",
            sections["Example"],
            "```",
            "",
            "---",
            "",
        ]
    )


def _iter_apis(node: DocNode) -> list[str]:
    names = list(node.apis)
    for child in node.children:
        names.extend(_iter_apis(child))
    return names


def render_toc(tree: DocNode) -> str:
    lines = ["## Table of Contents", ""]

    def walk(node: DocNode, number: str) -> None:
        heading = _numbered_title(number, node.title)
        indent = "  " * number.count(".")
        lines.append(f"{indent}- [{heading}](#{_anchor(heading)})")
        for idx, child in enumerate(node.children, start=1):
            walk(child, f"{number}.{idx}")

    for idx, child in enumerate(tree.children, start=1):
        walk(child, str(idx))
    lines.append("")
    return "\n".join(lines)


def render_doc_node(
    node: DocNode,
    entries: dict[str, APIEntry],
    heading_level: int,
    docs_dir: Path,
    section_number: str,
) -> str:
    parts = [f"{'#' * heading_level} {_numbered_title(section_number, node.title)}", ""]
    if node.intro:
        parts.extend([node.intro, ""])
    for name in node.apis:
        if name in entries:
            parts.append(render_entry(entries[name], heading_level + 1, docs_dir))
    for idx, child in enumerate(node.children, start=1):
        parts.append(
            render_doc_node(
                child,
                entries,
                heading_level + 1,
                docs_dir,
                f"{section_number}.{idx}",
            )
        )
    return "\n".join(parts).rstrip() + "\n"


def generate(*, docs_dir: Path | None = None) -> str:
    if not CORE_API_PATH.is_file():
        raise FileNotFoundError(f"core_api not found: {CORE_API_PATH}")

    docs_dir = docs_dir or OUTPUT_PATH.parent
    entries = parse_core_api(CORE_API_PATH)
    require_docs(entries)
    tree = build_doc_tree(entries)
    used = set(_iter_apis(tree))
    leftovers = sorted(
        (e for name, e in entries.items() if name not in used),
        key=lambda e: (e.source_line or 0, e.name),
    )

    header = [
        "<!--",
        "This file is generated by python/tla_dsl/tools/generate_api_reference.py.",
        "Do not edit manually. Update docstrings/Examples in catlass/core_api.py",
        "(or the defining module for imported types) instead.",
        "-->",
        "",
        "# TLA DSL Kernel API Reference",
        "",
        "This document describes the **TLA DSL kernel-side Core APIs** "
        "(typically imported as `import catlass.tla as tla`). "
        "It covers data structures, compute / sync helpers, on-chip resources, "
        "and debug printing. See `docs/zh/framework_integration.md` for Host "
        "tensor binding.",
        "",
        "Interface descriptions and examples come from each op's source docstring "
        "(`Directory:` plus `Description:` / `Parameters:` / `Constraints:` / `Example:`).",
        "",
        "All APIs must be called inside a `@tla.kernel`-decorated kernel function body.",
        "",
        "---",
        "",
        render_toc(tree),
        "---",
        "",
    ]

    body = [
        render_doc_node(child, entries, 2, docs_dir, str(idx))
        for idx, child in enumerate(tree.children, start=1)
    ]
    if leftovers:
        body.append("## Other Core APIs\n")
        body.append(
            "APIs still exported by the current source but not yet filed under "
            "the directory tree above.\n"
        )
        body.extend(render_entry(e, 3, docs_dir) for e in leftovers)

    return "\n".join(header + body).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH.relative_to(PACKAGE_ROOT)})",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PACKAGE_ROOT / args.output

    try:
        content = generate(docs_dir=output.parent)
    except MissingAPIDocError as exc:
        print(exc)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"generated {output.relative_to(PACKAGE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
