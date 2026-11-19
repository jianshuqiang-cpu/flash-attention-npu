#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash run_950_aswt_ab.sh BASELINE_CHECKOUT CANDIDATE_CHECKOUT ORIGINAL_D192_PY NEW_OUTPUT_DIR [PHYSICAL_NPU_ID=0]'
    echo 'Checks O and LSE on both builds, then profiles D192 and seven CATLASS-derived FA3 cases (A/B/B/A).'
    exit 0
fi

baseline=$(realpath "${1:?baseline checkout}")
candidate=$(realpath "${2:?ASWT checkout}")
decode=$(realpath "${3:?path to original profile_950_decode_d192.py}")
output=${4:?new output directory}
physical=${5:-0}
runner=$(dirname "$(realpath "$0")")/profile_950_aswt.py
mkdir "$output"
output=$(realpath "$output")
export ASCEND_RT_VISIBLE_DEVICES="$physical" PYTHONNOUSERSITE=1
cases=$(python "$runner" --list | cut -d: -f1)
shopt -s nullglob
baseline_lib=("$baseline"/flash_attn_npu_3_950*.so)
candidate_lib=("$candidate"/flash_attn_npu_3_950*.so)
if [[ ${#baseline_lib[@]} != 1 || ${#candidate_lib[@]} != 1 ]]; then
    echo 'Expected exactly one in-place FA3 950 extension per checkout.' >&2
    exit 1
fi
sha256sum "${baseline_lib[0]}" "${candidate_lib[0]}" "$decode" > "$output/identity.txt"
mapfile -t hashes < <(cut -d' ' -f1 "$output/identity.txt")
if [[ ${hashes[0]} == "${hashes[1]}" ]]; then
    echo 'A and B have identical binaries; check the build flags.' >&2
    exit 1
fi

for shape in $cases; do
    for check_label in A B; do
        checkout=$candidate
        [[ $check_label != A ]] || checkout=$baseline
        python -u "$runner" --checkout "$checkout" --case "$shape" --check-only 2>&1 | tee "$output/$shape-check-$check_label.log"
    done
    for label in A1 B1 B2 A2; do
        checkout=$candidate
        [[ $label != A* ]] || checkout=$baseline
        round="$output/$shape-$label"
        mkdir "$round"
        timeout 10 npu-smi info -t proc-mem -i "$physical" > "$round/before.txt"
        grep -q 'No process in device' "$round/before.txt"
        command=(python -u "$runner" --checkout "$checkout" --case "$shape")
        if [[ $shape == decode-d192 ]]; then
            command=(python -u "$decode" --checkout "$checkout" --fa 3 --num-splits 1 --device 0 --flush-l2)
        fi
        msprof --output="$round" --task-time=on --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
            --aicpu=on --python-path="$(command -v python)" "${command[@]}" > "$round/msprof.log" 2>&1 &
        collector=$!
        while kill -0 "$collector" 2>/dev/null; do
            timeout 10 npu-smi info -t proc-mem -i "$physical" >> "$round/processes.log" || echo 'query failed' >> "$round/processes.log"
            sleep 1
        done
        wait "$collector"
        timeout 10 npu-smi info -t proc-mem -i "$physical" > "$round/after.txt"
        grep -q 'No process in device' "$round/after.txt"
        python "$runner" --summary "$round" | tee "$round/summary.json"
        expected=${hashes[1]}
        [[ $label != A* ]] || expected=${hashes[0]}
        python -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["sha256"] == sys.argv[2] else "Loaded binary differs from identity.txt")' "$round/summary.json" "$expected"
    done
done
python "$runner" --compare "$output" > "$output/comparison.csv.tmp"
mv "$output/comparison.csv.tmp" "$output/comparison.csv"
cat "$output/comparison.csv"
