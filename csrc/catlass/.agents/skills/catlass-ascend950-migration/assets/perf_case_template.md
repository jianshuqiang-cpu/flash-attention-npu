# Performance cases — `<operator family>`, `<shape family / group configuration>`

`mig.py init` stages this template as `<run-dir>/perf_cases.md` when `--perf-cases` is not
supplied. It is a report artifact, not a gate: fill only measurements actually taken and leave
anything unmeasured blank. An existing campaign table is preserved.

Rename the heading for the migrated unit. If a grouped or otherwise parameterized axis changes
the problem, state it in the heading. Add one results table per shape family; use the same column
schema in each table.

## Test environment

Record one row for every measured configuration. A ratio is comparable only when its rows give
enough context to judge it. Record material differences in `notes`; do not estimate missing
values.

`A5 l0c` means the same A5 unit after the non-TLA `L0C→UB` rewrite.

| config | NPU model | CANN version | driver | dtype | layouts (A / B / C) | measured by | date | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2/A3 |  |  |  |  |  |  |  |  |
| A5 |  |  |  |  |  |  |  |  |
| A5 l0c |  |  |  |  |  |  |  |  |

## Manual measurement validation

1. Profile every recorded launch with `msprof op` and an explicit `--output=` directory under the
   run directory. Retain the profiler output that supplied each table value.
2. Before transcribing a duration, manually confirm that the profiler output names the intended
   kernel and configuration. Record the kernel total duration from that output in `us`.
3. Before transcribing `aic` or `aiv`, inspect the `PipeUtilization.csv` header in that profiler
   output. Use the column names present there; do not assume a CANN-version-specific spelling.
4. Reduce `aic` and `aiv` separately over only rows with a numeric value for that core type.
   `NA` is not zero and must not enter a reduction. Validate the numeric-row population against
   the profiler's own block information before accepting the result.
5. Check the reported clock information in the profiler output. Record a clock mismatch in
   `note`; do not use that sample for a ratio.
6. One launch is one sample. If a table cell reduces multiple launches, state the reduction in
   `note` and apply the same rule to comparable configurations.

## Results

| # | M | N | K | A2/A3 us | A2/A3 aic | A2/A3 aiv | A5 us | A5 aic | A5 aiv | A5 l0c us | A5 l0c aic | A5 l0c aiv | speedup | speedup_l0c | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Replace placeholder rows with campaign shapes. Keep `#` dense and one-based in each table. If the
unit uses another shape vocabulary, rename `M`, `N`, or `K` and define it in the heading.

## Filling rules

- `speedup` = `A2/A3 us` ÷ `A5 us`; `speedup_l0c` = `A2/A3 us` ÷ `A5 l0c us`. A value above one
  means the A5 configuration is faster.
- Leave a ratio blank until both inputs were measured in this campaign for the same problem.
- An empty cell means **not measured**. Do not use `0` or `-` as a spacer.
- Fill every field that declared hardware access can measure, and leave the rest blank:
  - A5 access only: fill A5 fields, and A5-l0c fields only after that rewrite is applied.
  - A2/A3 access only: fill A2/A3 fields.
  - Neither: keep the table empty and report that no comparison was measured.
- A performance value is never an accuracy claim. Accuracy comes only from the migrated example's
  CPU-golden comparison in this campaign.

## Grouped families

For grouped workloads, inspect the unit's actual group-list generation before deriving any
throughput metric. A displayed shape may be an upper bound rather than the extent computed. Do
not derive FLOPS, bandwidth, or arithmetic intensity from a bound. Timing ratios remain useful
only when both configurations used the same generated problem; record the seed, grouping inputs,
and any actual extent needed to establish that fact in `note`.