#!/usr/bin/env bash
# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# 容器内执行脚本, 由 ci/run_ci_container.sh 通过 docker run 调用。
# 步骤:
#   1. 清理历史日志
#   2. git submodule update --init
#   3. FLASH_ATTN_FORCE_BUILD=TRUE pip install -e .
#   4. 按 ci/example_st_cases.json 跑 Example ST (pytest)
#
# 环境变量 (由 run_ci_container.sh 注入):
#   ASCEND_RT_VISIBLE_DEVICES   宿主机物理卡号 (容器内映射成逻辑 0)
#   CI_MODE                     quick|full
#   CI_RUN_EXAMPLE_ST           true|false
#   CI_EXAMPLE_CASE_FILTER      只跑指定 case name (逗号分隔)
#   CI_CONTAINER_DEVICE         容器内逻辑设备号 (默认 0)

set -euo pipefail

REPO_ROOT="$(pwd)"
CASES_JSON="$REPO_ROOT/ci/example_st_cases.json"
DEVICE="${CI_CONTAINER_DEVICE:-0}"

log() { printf '[CI-inside] %s\n' "$*"; }
die() { printf '[CI-inside][ERROR] %s\n' "$*" >&2; exit 1; }

log "repo=$REPO_ROOT device=$DEVICE mode=${CI_MODE:-quick}"
log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

# ---------- 0. 基础自检 ----------
command -v python3 >/dev/null 2>&1 || die "python3 not found in container"
python3 - <<'PY' || die "torch_npu not functional inside container (check --privileged / driver mount)"
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torch_npu device_count:", torch_npu.npu.device_count())
assert torch_npu.npu.device_count() >= 1, "device_count==0; --privileged or driver mount missing?"
PY

# ---------- 1. 清理历史日志 ----------
bash "$REPO_ROOT/ci/cleanup_ci_logs.sh" || log "cleanup_ci_logs.sh returned non-zero (ignored)"

# ---------- 2. 子模块 ----------
log "init submodules: csrc/catlass csrc_AscendC950/catlass"
git submodule update --init --recursive csrc/catlass csrc_AscendC950/catlass

# ---------- 3. 编译安装整包 ----------
# 默认构建 v2 + v3 两个 backend (BUILD_VERSION=all)
export FLASH_ATTN_FORCE_BUILD=TRUE
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
log "pip install -e . (FLASH_ATTN_BUILD_VERSION=${FLASH_ATTN_BUILD_VERSION:-all})"
pip install -e . --no-build-isolation

log "import check"
python3 - <<'PY'
import flash_attn_npu
print("flash_attn_npu", flash_attn_npu.__version__)
PY

# ---------- 4. Example ST ----------
if [ "${CI_RUN_EXAMPLE_ST:-true}" != "true" ]; then
  log "CI_RUN_EXAMPLE_ST!=true, skip Example ST"
  exit 0
fi

command -v pytest >/dev/null 2>&1 || pip install pytest --quiet
command -v jq >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq jq >/dev/null)

# quick 模式默认只跑前向类用例, full 跑全部 enabled=true
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

# 读取 JSON 中 enabled=true 的用例
select_cases() {
  jq -r '.cases[] | select(.enabled==true) | "\(.name)\t\(.test_file)\t\(.test_filter // "")\t\(.pytest_args // "")"'
}

# 按 FILTER 过滤
filter_cases() {
  if [ -z "$FILTER" ]; then
    cat
  else
    local IFS_save="$IFS"
    IFS=','
    local -a want=($FILTER)
    IFS="$IFS_save"
    local pattern
    pattern="$(IFS='|'; echo "${want[*]}")"
    grep -E "^($pattern)"'\t'
  fi
}

log "running Example ST (mode=$MODE filter=${FILTER:-<none>})"
selected="$(select_cases | filter_cases)"
if [ "$MODE" = "quick" ]; then
  selected="$(printf '%s\n' "$selected" | grep -v -iE '^[a-z0-9_]*bwd' || true)"
fi

if [ -z "$selected" ]; then
  die "no Example ST case selected (check ci/example_st_cases.json or CI_EXAMPLE_CASE_FILTER)"
fi

while IFS=$'\t' read -r name file kfilter args; do
  [ -z "$name" ] && continue
  run_case "$name" "$file" "$kfilter" "$args"
done <<< "$selected"

log "all Example ST cases passed"
