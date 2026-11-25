"""Fixtures for the end-to-end DSL validation battery (the RDV gate).

These tests drive real kernels on a real NPU and take minutes, so they are NOT
part of a plain `pytest tests/` run: they are skipped unless explicitly asked
for, with --run-battery or TLA_DSL_RUN_BATTERY=1 (which tests/run_dsl_test.sh
sets).

Do NOT run this suite under pytest-xdist. Every case shares one process, and
paying the device bring-up and the torch_npu import once instead of per case is
the entire point -- `-n auto` would put each worker in its own process and hand
back the minutes this suite exists to save. Several cases are also order
dependent, so a shuffling plugin would make them flaky.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("dsl-battery")
    group.addoption(
        "--run-battery",
        action="store_true",
        default=False,
        help="run the on-device DSL validation battery (needs an NPU)",
    )
    group.addoption(
        "--device", action="store", type=int, default=0, help="NPU device id"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "npu: needs a real NPU device")

    # Enforce the two properties this suite depends on, rather than trusting a
    # comment: every case runs in ONE process, in declaration order.
    if _battery_requested(config):
        if hasattr(config, "workerinput") or config.getoption("numprocesses", None):
            raise pytest.UsageError(
                "the DSL battery must run in a single process: every case shares "
                "the process that first brought the device up. Drop -n (xdist)."
            )
        if config.pluginmanager.hasplugin("randomly") and config.getoption(
            "randomly_reorganize", True
        ):
            raise pytest.UsageError(
                "the DSL battery must run in declaration order; pass -p no:randomly."
            )


def _battery_requested(config: pytest.Config) -> bool:
    return bool(config.getoption("--run-battery")) or (
        os.environ.get("TLA_DSL_RUN_BATTERY") == "1"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _battery_requested(config):
        return
    skip = pytest.mark.skip(
        reason="on-device battery; pass --run-battery or set TLA_DSL_RUN_BATTERY=1"
    )
    for item in items:
        if "npu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--device"))
