"""The end-to-end DSL validation battery -- the RDV gate, as pytest cases.

Every case replays an example's own main() with a given argv, inside the one
process, so the device bring-up and the torch_npu import are paid once. The
examples are not modified in any way; a case is just an argv.

Adding coverage means adding to the tables below. Keep them as data: the point of
this file is that the whole gate is readable in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator
from pathlib import Path
from typing import Iterator

import pytest

pytestmark = pytest.mark.npu

MMAD_SHAPES = (("333", "444", "555"), ("1", "2", "3"))
BIG_SHAPES = (("3200", "4096", "257"), ("8192", "2906", "1027"))
BATCH_SHAPES = (("1", "333", "444", "555"), ("1", "1", "2", "3"), ("8", "320", "333", "444"))
MMAD_LAYOUTS = (("row", "row"), ("row", "col"), ("col", "row"), ("col", "col"))
#: (dtype-a/b, dtype-c)
MMAD_TRIPLES = (
    ("f16", "f32"),
    ("f16", "f16"),
    ("bf16", "f32"),
    ("bf16", "bf16"),
    ("f32", "f32"),
)
MMAD_MNK = ("--m", "333", "--n", "444", "--k", "555")
MMAD_ROW_ROW = ("--layout-a", "row", "--layout-b", "row")

VADD_DTYPES = ("i8", "i16", "i32", "f16", "f32")
VADD_MODES = ("", "--use-mutex", "--use-mutex-with", "--use-atomic-add")

COMPARE_MASK_OPS = (
    "vector_vector_lt", "vector_vector_le", "vector_vector_gt", "vector_vector_ge",
    "vector_vector_eq", "vector_vector_ne", "vector_scalar_gt", "vector_scalar_ge",
    "masked_vector_vector_lt", "cmp_masked_fused", "static_dynamic_lt",
)

EVG_OPS = (
    "add",
    "add_ub",
    "bias",
    "leaky_relu",
    "sigmoid",
    "silu",
    "tanh",
)

def _mmad_cases(
    op_name: str,
    op_script: str,
    device: int,
    **kwargs,
) -> Iterator[tuple[str, list[list[str]]]]:
    """Matmul-like cases generator
    
    Iterate over all shapes, layouts and dtype triples
    """
    for m, n, k in kwargs.get("shapes", MMAD_SHAPES):
        for la, lb in kwargs.get("layouts", MMAD_LAYOUTS):
            for dab, dc in kwargs.get("triples", MMAD_TRIPLES):
                yield (
                    f"{op_name}-{m}x{n}x{k}-{la}{lb}-{dab}-{dc}",
                    [[
                        op_script,
                        "--m", m, "--n", n, "--k", k,
                        "--layout-a", la, "--layout-b", lb,
                        "--dtype-a", dab, "--dtype-b", dab, "--dtype-c", dc,
                        "--device", str(device)
                    ]]
                )

def _cases(device: int) -> Iterator[tuple[str, list[list[str]]]]:
    """Yield (test id, [argv, ...]) for every case in the battery.

    A case is usually one argv. It is a *sequence* only where a later invocation
    depends on an earlier one -- see the basic_mixed compile-then-cache-reuse
    case, which must stay a single test so the two halves can never be split up
    or reordered.
    """

    dev = ["--device", str(device)]

    # --- basic_mmad: the flag-sync layout x dtype matrix ---
    yield from _mmad_cases(
        "mmad",
        "basic_mmad/basic_matmul.py",
        device,
    )

    for script, label in (
        ("basic_matmul_mutex.py", "mutex"),
        ("basic_matmul_mutex_with.py", "mutex-with"),
        ("basic_matmul_auto_sync.py", "auto-sync"),
    ):
        yield (
            f"mmad-{label}",
            [[
                f"basic_mmad/{script}", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", "f16", "--dtype-b", "f16", "--dtype-c", "f32", *dev,
            ]],
        )

    for dtype in ("f16", "bf16", "f32"):
        yield (
            f"mmad-atomic-add-{dtype}",
            [[
                "basic_mmad/basic_matmul_atomic_add.py", *MMAD_MNK, *MMAD_ROW_ROW,
                "--dtype-a", dtype, "--dtype-b", dtype, "--dtype-c", "f32", *dev,
            ]],
        )

    yield ("mmad-ptr", [["basic_mmad/basic_mmad_ptr.py", *dev]])

    # --- basic_vadd: every dtype under each sync mode ---
    for mode in VADD_MODES:
        for dtype in VADD_DTYPES:
            label = mode.lstrip("-") or "flag-sync"
            yield (
                f"vadd-{label}-{dtype}",
                [[
                    "basic_vadd/basic_vadd.py", "--dtype", dtype, *dev,
                    *([mode] if mode else []),
                ]],
            )

    # --- basic_mixed ---
    # One test on purpose: the second invocation exercises cache reuse and is
    # only meaningful straight after the one that populates the cache.
    yield (
        "mixed-compile-then-cache-reuse",
        [
            ["basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
            ["basic_mixed/basic_mixed.py", *dev, "--block-num", "1"],
        ],
    )
    yield (
        "mixed-mutex",
        [["basic_mixed/basic_mixed_mutex.py", *dev, "--block-num", "1"]],
    )
    yield ("mixed-ub2l1", [["basic_mixed/basic_mixed_ub2l1.py", *dev]])
    yield ("mixed-store-zN", [["basic_mixed/basic_mixed_store_zN.py", *dev]])
    # m=64 is fractal-aligned (multiple of 16); m=50 exercises the zNUnAlign M
    # axis, where the dest leaf[0] is the runtime row count and the stride varies.
    for m in ("64", "50"):
        yield (
            f"mixed-store-zNUnAlign-m{m}",
            [["basic_mixed/basic_mixed_store_zNUnAlign.py", *dev, "--m", m]],
        )
    yield ("mixed-fixpipe-nz2dn", [["basic_mixed/basic_mixed_fixpipe_nz2dn.py", *dev]])

    # --- mixed-core handshake and standalone control-flow probes ---
    yield (
        "flash-attention-infer",
        [["flash_attention_infer/flash_attention_infer.py", *dev]],
    )

    # --- multi_core_splitk_matmul and tail_multi_core_splitk_matmul ---
    for script in ("multi_core_splitk_matmul.py", "tail_multi_core_splitk_matmul.py"):
        for m, n, k in MMAD_SHAPES + BIG_SHAPES:
            yield (
                f"{script}-{m}x{n}x{k}",
                [[f"multi_core_splitk_matmul/{script}", "--m", m, "--n", n, "--k", k, *dev]],
            )

    # --- basic_mmad_streamk: a streamK example for workload balance ---
    for m, n, k in MMAD_SHAPES + BIG_SHAPES:
        yield (
            f"mmad-streamk-{m}x{n}x{k}",
            [["basic_mmad_streamk/basic_mmad_streamk.py", "--m", m, "--n", n, "--k", k, *dev]],
        )

    # --- batched_matmul ---
    for b, m, n, k in BATCH_SHAPES:
        yield (
            f"batched-matmul-{b}x{m}x{n}x{k}",
            [["batched_matmul/batched_matmul.py", "--batch", b, "--m", m, "--n", n, "--k", k, *dev]],
        )

    # --- grouped_matmul_slice_m: grouped matmul example ---
    yield (
        "grouped-matmul-slice-m",
        [["grouped_matmul_slice_m/grouped_matmul_slice_m.py", *dev],
         ["grouped_matmul_slice_m/grouped_matmul_slice_m.py", 
            "--groups", "3", "--m", "768", "--n", "333", "--k", "333", *dev]]
    ) 

    # --- basic_mmad_epilogue: multiple epilogue examples ---
    for op in EVG_OPS:
        if op in ("add_ub", "tanh"):
            # f32 (dtype-c) only examples
            triples = (
                ("f16", "f32"),
                ("f32", "f32"),
            )
        else:
            triples = (
                ("f16", "f32"),
                ("f32", "f32"),
                ("f16", "f16"),
                ("bf16", "f16"),
            )
        for m, n, k in MMAD_SHAPES:
            for dab, dc in triples:
                yield (
                    f"mmad-epilogue-{op.replace('_', '-')}-{m}x{n}x{k}-{dab}-{dc}",
                    [[
                        f"basic_mmad_epilogue/matmul_{op}.py",
                        "--m", m, "--n", n, "--k", k,
                        "--dtype-a", dab, "--dtype-b", dab, "--dtype-c", dc,
                        *dev
                    ]],
                )

    # --- vector_ops: each of these sweeps or batches many kernels internally ---
    yield (
        "vector-masked-binary",
        [["vector_ops/masked_binary.py", "masked_binary", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-bitwise-ops",
        [["vector_ops/bitwise_ops.py", "bitwise_ops", "--sweep", "--shapes", "400", *dev]],
    )
    yield (
        "vector-binary-op",
        [[
            "vector_ops/binary_op.py", "--batch-run",
            "add", "sub", "mul", "div", "max", "min", "add_unalign", "add_brc_b32",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield ("vector-reduction-ops", [["vector_ops/reduction_ops.py", "--batch-run", *dev]])
    yield (
        "vector-load-store-scalar-after-reduction",
        [["vector_ops/load_and_store_scalar_after_reduction.py", *dev]],
    )
    yield (
        "vector-compare-mask",
        [[
            "vector_ops/compare_mask.py", "--batch-run", *COMPARE_MASK_OPS,
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-unary-ops",
        [[
            "vector_ops/unary_ops.py", "--batch-run",
            "exp", "log", "sqrt", "abs", "neg", "masked_unary", "masked_abs", "masked_neg",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-arange-op",
        [[
            "vector_ops/arange_op.py", "--batch-run", "increase", "decrease",
            "--shape", "400", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-interleave-op",
        [[
            "vector_ops/interleave_op.py", "--batch-run", "interleave", "deinterleave",
            "--shape", "512", "--batch-size", "4", *dev,
        ]],
    )
    yield (
        "vector-load-dintlv-op",
        [["vector_ops/load_dintlv_op.py", "dintlv_b32", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-us-b8-op",
        [["vector_ops/load_us_b8_op.py", "us_b8", "--sweep", "--shapes", "512", *dev]],
    )
    yield (
        "vector-load-store-mask",
        [["vector_ops/load_store_mask.py", "load_store_mask", "--all-dtypes", *dev]],
    )
    yield (
        "vector-store-pack",
        [["vector_ops/store_pack.py", "store_pack", "--all-dtypes", *dev]],
    )
    yield (
        "vector-squeeze-op",
        [["vector_ops/squeeze_op.py", "squeeze", "--sweep", "--shapes", "64", *dev]],
    )
    yield (
        "vector-register-control-flow",
        [["vector_ops/register_control_flow.py", "register_carriers", *dev]],
    )
    yield (
        "vector-cast-multi",
        [["vector_ops/cast_multi.py", "cast_multi", "--shape", "256", *dev]],
    )
    yield (
        "vector-gather",
        [["vector_ops/gather_op.py", "--run", *dev]],
    )


def battery_cases(device: int = 0) -> list[tuple[str, list[list[str]]]]:
    """The whole battery as data. Used for the tests, and to audit coverage."""

    return list(_cases(device))


def _ids() -> list[str]:
    return [case_id for case_id, _ in battery_cases()]


@pytest.mark.parametrize("case_id", _ids())
def test_case(case_id: str, device: int) -> None:
    import example_runner

    argvs = dict(battery_cases(device))[case_id]
    for argv in argvs:
        script = example_runner.EXAMPLES_DIR / argv[0]
        assert script.is_file(), f"no such example: {argv[0]}"
        rc = example_runner.run_case([str(script), *argv[1:]])
        assert rc == 0, f"{argv[0]} {' '.join(argv[1:])} exited {rc}"
