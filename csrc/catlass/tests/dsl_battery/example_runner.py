#!/usr/bin/env python3
"""Replay end-to-end examples in-process, for the validation battery.

The battery used to spawn one ``python examples/...`` per case. Each of those
processes pays the same fixed cost -- importing torch_npu and bringing the
device up -- which dwarfs the case itself: on an Ascend950PR box a basic_mmad
case costs ~23 s as its own process but ~1 s as an extra case in a process that
is already up.

Nothing here manages the runtime. The examples call torch.npu.set_device()
themselves and no longer initialize or finalize the ACL context, so cases simply
share whatever the first one brought up.

So the examples are not modified at all. Every one exposes ``main()`` parsing
``sys.argv``, so a case is just an argv: import the example module once, and call
its main() with the argv the shell used to pass on the command line.

See test_battery.py for the case list, and conftest.py for the session fixture.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from typing import Any

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[2]
    / "python" / "tla_dsl" / "examples" / "end_to_end"
)

_TLA_DSL_ROOT = EXAMPLES_DIR.parents[1]

_MODULE_CACHE: dict[Path, Any] = {}


class _SiblingFinder(importlib.abc.MetaPathFinder):
    """Resolve one example directory's sibling modules, and nothing else.

    Run as ``python debug_print/debug_print_mixed.py``, an example's own
    directory is sys.path[0], so ``from debug_print import DTYPE_SPECS`` finds
    the sibling debug_print.py. Imported from here it would not.

    Prepending that directory to sys.path would fix it and cause a worse
    problem: the entry outlives the case, and every .py beside it then shadows
    same-named modules for the rest of the session -- an example called
    types.py or queue.py would break the interpreter for every later case. So
    resolve siblings through this finder instead, installed as a fallback only
    while the example is being imported.

    The stdlib check below is belt-and-braces: as a fallback finder we are only
    reached once normal resolution has failed, so a stdlib name cannot get here
    anyway. It keeps the guarantee if the finder is ever moved earlier.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def find_spec(
        self, name: str, path: Any = None, target: Any = None
    ) -> Any | None:
        if path is not None or name in sys.stdlib_module_names:
            return None
        candidate = self.directory / f"{name}.py"
        if not candidate.is_file():
            return None
        return importlib.util.spec_from_file_location(name, candidate)


@contextlib.contextmanager
def _sibling_imports(directory: Path) -> Any:
    """Make `directory`'s modules importable by bare name, for this block only."""

    finder = _SiblingFinder(directory)
    # Appended, not prepended: the finder is a *fallback*, consulted only after
    # the builtin, frozen and sys.path finders have all failed. That makes it
    # structurally unable to shadow the stdlib, site-packages or anything else --
    # a prepended finder would sit ahead of the entire import machinery.
    sys.meta_path.append(finder)

    # Make the `examples` shared-helpers package importable
    # (`from examples.end_to_end.common import ...`).
    _root = str(_TLA_DSL_ROOT)
    _root_added = _root not in sys.path
    if _root_added:
        sys.path.append(_root)

    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        if _root_added:
            try:
                sys.path.remove(_root)
            except ValueError:
                pass


def load_example(path: Path) -> Any:
    """Import an example module once and reuse it for every later case."""

    cached = _MODULE_CACHE.get(path)
    if cached is not None:
        return cached
    name = f"_tla_case_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"example_runner: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with _sibling_imports(path.parent):
        spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise SystemExit(f"example_runner: {path} has no main(); it cannot be batched")
    _MODULE_CACHE[path] = module
    return module



def run_case(argv: list[str]) -> int:
    """Run one case by replaying its argv through the example's own main()."""

    script = Path(argv[0])
    module = load_example(script)
    saved_argv = sys.argv
    sys.argv = list(argv)
    try:
        # A cached module may still import a sibling lazily, inside run().
        with _sibling_imports(script.parent):
            rc = module.main()
    except SystemExit as exc:  # some examples exit rather than return
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = saved_argv
    return int(rc or 0)


