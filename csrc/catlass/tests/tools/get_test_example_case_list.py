#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compute the test example case list affected by a difflist.

Pure-stdlib scanner: reads the changed-file list (difflist) and prints, one per
line, the numbered example directories (e.g. ``00_basic_matmul``) that must be
re-tested:

  - a change under ``examples/NN_name/...`` affects ``NN_name`` directly;
  - a change under ``include/...`` affects every example that (transitively)
    includes that header (reverse closure of the #include graph).

Usage:
    python3 tests/tools/get_test_example_case_list.py < changes.txt
    python3 tests/tools/get_test_example_case_list.py --difflist changes.txt
"""

import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_INCLUDE_DIR = os.path.join(_REPO_ROOT, "include")
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")

# Only double-quote #include "..." is parsed; angle-bracket includes are ignored.
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
# A directory name is "numbered" when it starts with digits + underscore.
_NUMBERED_EXAMPLE_RE = re.compile(r"^\d+_.+")
_HEADER_EXT_RE = re.compile(r"\.hpp$")


def _iter_hpp_files():
    """Yield (absolute_path, node_id) for every .hpp under include/.

    node_id is the path relative to include/ (posix), e.g. "catlass/gemm/block/block_mmad.hpp".
    """
    for root, _dirs, files in os.walk(_INCLUDE_DIR):
        for name in files:
            if _HEADER_EXT_RE.search(name):
                abs_path = os.path.join(root, name)
                node_id = os.path.relpath(abs_path, _INCLUDE_DIR).replace(os.sep, "/")
                yield abs_path, node_id


def _iter_numbered_examples():
    """Yield (abs_dir, example_dir_name) for each numbered example directory."""
    if not os.path.isdir(_EXAMPLES_DIR):
        return
    for name in sorted(os.listdir(_EXAMPLES_DIR)):
        full = os.path.join(_EXAMPLES_DIR, name)
        if os.path.isdir(full) and _NUMBERED_EXAMPLE_RE.match(name):
            yield full, name


def _parse_includes(file_path):
    """Return the double-quoted include targets found in file_path."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return _INCLUDE_RE.findall(f.read())
    except OSError:
        return []


def _build_reverse_map():
    """Return (reverse_map, header_nodes).

    reverse_map: header node_id -> set of nodes that directly include it,
    where a node is either a header node_id or a numbered example directory name.
    """
    header_nodes = {}
    for abs_path, node_id in _iter_hpp_files():
        header_nodes[node_id] = abs_path

    reverse = {}
    for node_id, abs_path in header_nodes.items():
        for inc in _parse_includes(abs_path):
            if inc in header_nodes:
                reverse.setdefault(inc, set()).add(node_id)

    example_names = set()
    for _abs_dir, name in _iter_numbered_examples():
        example_names.add(name)
        for root, _dirs, files in os.walk(os.path.join(_EXAMPLES_DIR, name)):
            for fname in files:
                if not (fname.endswith((".cpp", ".hpp"))):
                    continue
                for inc in _parse_includes(os.path.join(root, fname)):
                    if inc in header_nodes:
                        reverse.setdefault(inc, set()).add(name)
    return reverse, header_nodes


def _affected_examples(changed_headers, reverse_map):
    """Return sorted example names affected by the changed header node_ids.

    BFS over the reverse include map: every example node reachable from a
    changed header (transitively via intermediate headers) is affected.
    """
    affected = set()
    visited = set()
    queue = list(changed_headers)
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for includer in reverse_map.get(node, ()):
            if _NUMBERED_EXAMPLE_RE.match(includer):
                affected.add(includer)
            elif includer not in visited:
                queue.append(includer)
    return sorted(affected)


def _normalize_path(path):
    """Normalize a difflist line to a repo-relative posix path."""
    return path.replace(os.sep, "/").lstrip("./")


def get_test_example_case_list(changed_files):
    """Return the sorted affected example names for the given changed-file list."""
    reverse_map, header_nodes = _build_reverse_map()
    affected = set()

    for line in changed_files:
        norm = _normalize_path(line)
        if not norm:
            continue
        # examples/NN_name/... -> that example itself
        m = re.match(r"^examples/(\d+_.+?)/", norm)
        if m and _NUMBERED_EXAMPLE_RE.match(m.group(1)):
            affected.add(m.group(1))
            continue
        # include/...hpp -> reverse closure over the #include graph
        if norm.startswith("include/"):
            node_id = norm[len("include/") :]
            if node_id in header_nodes:
                affected.update(_affected_examples([node_id], reverse_map))
    return sorted(affected)


def main(argv=None):
    """CLI entry: read the difflist and print affected example names."""
    argv = argv if argv is not None else sys.argv[1:]
    difflist_path = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--difflist", "-d") and i + 1 < len(argv):
            difflist_path = argv[i + 1]
            i += 2
        elif not a.startswith("-"):
            difflist_path = a
            i += 1
        else:
            i += 1

    if difflist_path:
        with open(difflist_path, "r", encoding="utf-8", errors="replace") as f:
            changed_files = [ln.strip() for ln in f if ln.strip()]
    else:
        changed_files = [ln.strip() for ln in sys.stdin if ln.strip()]

    for name in get_test_example_case_list(changed_files):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
