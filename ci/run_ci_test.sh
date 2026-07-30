#!/usr/bin/env bash
# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# 阶段2: NPU 自检 + 安装 + 测试 (容器内执行, 需要 NPU, 已加锁)
#   1. NPU 可用性自检
#   2. python setup.py install (复用阶段1 build/ 产物, 快速安装)
#   3. import 校验
#   4. 按 ci/example_st_cases.json 跑 Example ST (pytest)
#
# 由 ci/run_ci_container.sh 阶段2通过 docker run 调用 (绑卡 + 加锁)。
#
# 环境变量 (由 run_ci_container.sh 注入):
#   ASCEND_RT_VISIBLE_DEVICES   宿主机物理卡号
#   CI_MODE                     quick|full
#   CI_RUN_EXAMPLE_ST           true|false
#   CI_EXAMPLE_CASE_FILTER      只跑指定 case name (逗号分隔)
#   CI_CONTAINER_DEVICE         容器内逻辑设备号 (默认 0)

set -euo pipefail

REPO_ROOT="$(pwd)"
CASES_JSON="$REPO_ROOT/ci/example_st_cases.json"
DEVICE="${CI_CONTAINER_DEVICE:-0}"

# git safe.directory (容器内 root 操作宿主机 runner 用户的目录, 会触发 dubious ownership)
git config --global --add safe.directory "$REPO_ROOT"

log() { printf '[CI-test] %s\n' "$*"; }
die() { printf '[CI-test][ERROR] %s\n' "$*" >&2; exit 1; }

log "repo=$REPO_ROOT device=$DEVICE mode=${CI_MODE:-quick}"
log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
log "test phase start: $(date '+%Y-%m-%d %H:%M:%S')"

# ---------- 1. NPU 自检 (需要卡) ----------
command -v python3 >/dev/null 2>&1 || die "python3 not found in container"
python3 - <<'PY' || die "torch_npu not functional inside container (check --privileged / driver mount)"
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torch_npu device_count:", torch_npu.npu.device_count())
assert torch_npu.npu.device_count() >= 1, "device_count==0; --privileged or driver mount missing?"
PY

# ---------- 2. 安装 (复用 build/ 产物, 不重新编译) ----------
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
log "python setup.py install --skip-build (reuse build/ artifacts)"
python3 setup.py install --skip-build

log "import check"
python3 - <<'PY'
import flash_attn_npu_v2
print("flash_attn_npu_v2", flash_attn_npu_v2.__version__)
PY

# ---------- 3. Example ST ----------
if [ "${CI_RUN_EXAMPLE_ST:-true}" != "true" ]; then
  log "CI_RUN_EXAMPLE_ST!=true, skip Example ST"
  exit 0
fi

command -v pytest >/dev/null 2>&1 || pip install pytest --quiet
command -v jq >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq jq >/dev/null)

MODE="${CI_MODE:-quick}"
FILTER="${CI_EXAMPLE_CASE_FILTER:-}"

run_case() {
  local name="$1" file="$2" kfilter="$3" args="$4"
  log ">>> case=$name file=$file k=${kfilter:-<none>} args=$args"
  set +e
  # shellcheck disable=SC2086
  python3 -m pytest "$file" $args ${kfilter:+-k "$kfilter"}
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    die "case=$name FAILED (pytest rc=$rc)"
  fi
  log "<<< case=$name OK"
}

select_cases() {
  jq -r '.cases[] | select(.enabled==true) | "\(.name)|\(.test_file)|\(.test_filter // "")|\(.pytest_args // "")"' "$CASES_JSON"
}

filter_cases() {
  if [ -z "$FILTER" ]; then
    cat
  else
    local IFS_save="$IFS"
    IFS=','
    local -a want=($FILTER)
    IFS="$IFS_save"
    local line name w
    while IFS= read -r line; do
      name="${line%%|*}"
      for w in "${want[@]}"; do
        if [ "$name" = "$w" ]; then
          printf '%s\n' "$line"
          break
        fi
      done
    done
  fi
}

log "running Example ST (mode=$MODE filter=${FILTER:-<none>})"
selected="$(select_cases | filter_cases)"
if [ "$MODE" = "quick" ]; then
  selected="$(printf '%s\n' "$selected" | grep -v -iE '^[a-z0-9_]*bwd' || true)"
fi
log "selected cases:"
printf '%s\n' "$selected" | sed 's/^/  /'

if [ -z "$selected" ]; then
  die "no Example ST case selected (check ci/example_st_cases.json or CI_EXAMPLE_CASE_FILTER)"
fi

while IFS='|' read -r name file kfilter args; do
  [ -z "$name" ] && continue
  run_case "$name" "$file" "$kfilter" "$args"
done <<< "$selected"

log "all Example ST cases passed"
log "test phase end: $(date '+%Y-%m-%d %H:%M:%S')"
