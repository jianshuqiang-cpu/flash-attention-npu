#!/usr/bin/env bash
# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# CI 容器入口 (两阶段):
#   阶段1 (编译, 不加锁): docker run 不绑卡, 跑 ci/run_ci_build.sh
#     - git submodule update --init
#     - python setup.py build  (产物在 build/, 通过 volume 持久化)
#   阶段2 (测试, 加锁): 选卡 + flock 加锁 + docker run 绑卡, 跑 ci/run_ci_test.sh
#     - NPU 自检
#     - python setup.py install (复用 build/ 产物)
#     - pytest Example ST
#
# 环境变量:
#   CI_MODE                  (默认 quick)  quick|full
#   CI_RUN_EXAMPLE_ST        (默认 true)   是否跑 Example ST
#   CI_EXAMPLE_CASE_FILTER   (默认空)      只跑指定 case name (逗号分隔)
#   CI_CONTAINER_DEVICE      (默认 0)      容器内逻辑设备号
#   CI_DOCKER_PRIVILEGED     (默认 true)   是否带 --privileged
#   CI_DOCKER_IMAGE          (默认 fla-npu-ci:8.5.0-910b)
#   CI_SKIP_BUILD            (默认 false)  true=跳过阶段1 (已有 build/ 产物)
#   CI_NPU_LOCK_DIR          (默认 /tmp)
#   CI_NPU_LOCK_WAIT_SECONDS (默认 14400)  0 表示一直等
#   CI_NPU_LOCK_RETRY_SECONDS(默认 10)
#   ASCEND_RT_VISIBLE_DEVICES               手动指定宿主机物理卡时跳过自动选卡

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CI_MODE="${CI_MODE:-quick}"
CI_RUN_EXAMPLE_ST="${CI_RUN_EXAMPLE_ST:-true}"
CI_EXAMPLE_CASE_FILTER="${CI_EXAMPLE_CASE_FILTER:-}"
CI_CONTAINER_DEVICE="${CI_CONTAINER_DEVICE:-0}"
CI_DOCKER_PRIVILEGED="${CI_DOCKER_PRIVILEGED:-true}"
CI_DOCKER_IMAGE="${CI_DOCKER_IMAGE:-fla-npu-ci:8.5.0-910b}"
CI_SKIP_BUILD="${CI_SKIP_BUILD:-false}"
CI_NPU_LOCK_DIR="${CI_NPU_LOCK_DIR:-/tmp}"
CI_NPU_LOCK_WAIT_SECONDS="${CI_NPU_LOCK_WAIT_SECONDS:-14400}"
CI_NPU_LOCK_RETRY_SECONDS="${CI_NPU_LOCK_RETRY_SECONDS:-10}"

log() { printf '[CI] %s\n' "$*"; }
die() { printf '[CI][ERROR] %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found"
docker image inspect "$CI_DOCKER_IMAGE" >/dev/null 2>&1 || die "docker image $CI_DOCKER_IMAGE not found; load or build it first"

# ---------- docker run 公共参数 ----------
docker_mount_args() {
  local mount_args=()
  for path in \
    /usr/local/dcmi \
    /usr/local/bin/npu-smi \
    /usr/local/Ascend/driver/lib64 \
    /usr/local/Ascend/driver/version.info \
    /etc/ascend_install.info; do
    if [ -e "$path" ]; then
      mount_args+=(-v "$path:$path")
    fi
  done
  printf '%s\n' "${mount_args[@]}"
}

privileged_args=()
if [ "$CI_DOCKER_PRIVILEGED" = "true" ]; then
  privileged_args+=(--privileged)
fi

# ---------- 阶段1: 编译 (不加锁, 不绑卡) ----------
run_build_phase() {
  log "=== Phase 1: build (no NPU lock) ==="
  docker run --rm \
    "${privileged_args[@]}" \
    --network host \
    --ipc host \
    -v "$REPO_ROOT:/workspace/flash-attention-npu" \
    -e FLASH_ATTN_BUILD_VERSION="${FLASH_ATTN_BUILD_VERSION:-all}" \
    -w /workspace/flash-attention-npu \
    "$CI_DOCKER_IMAGE" \
    bash -lc 'bash ci/run_ci_build.sh'
}

# ---------- 阶段2: 测试 (加锁 + 绑卡) ----------
# NPU 候选 id 列表
get_candidates() {
  bash "$SCRIPT_DIR/detect_npu.sh" --candidates 2>/dev/null \
    | sed -n 's/^  - id=\([0-9]\+\) .*/\1/p'
}

run_docker_test() {
  local device_id="$1"
  local mount_args=()
  while IFS= read -r m; do
    [ -n "$m" ] && mount_args+=("$m")
  done < <(docker_mount_args)

  log "starting test container: image=$CI_DOCKER_IMAGE physical_device=$device_id -> container_device=$CI_CONTAINER_DEVICE mode=$CI_MODE"
  docker run --rm \
    "${privileged_args[@]}" \
    --network host \
    --ipc host \
    -v "$REPO_ROOT:/workspace/flash-attention-npu" \
    "${mount_args[@]}" \
    -e ASCEND_RT_VISIBLE_DEVICES="$device_id" \
    -e CI_MODE="$CI_MODE" \
    -e CI_RUN_EXAMPLE_ST="$CI_RUN_EXAMPLE_ST" \
    -e CI_EXAMPLE_CASE_FILTER="$CI_EXAMPLE_CASE_FILTER" \
    -e CI_CONTAINER_DEVICE="$CI_CONTAINER_DEVICE" \
    -e FLASH_ATTN_BUILD_VERSION="${FLASH_ATTN_BUILD_VERSION:-all}" \
    -w /workspace/flash-attention-npu \
    "$CI_DOCKER_IMAGE" \
    bash -lc 'bash ci/run_ci_test.sh'
}

acquire_lock_and_run_test() {
  local device_id="$1"
  local lockfile="$CI_NPU_LOCK_DIR/fla-npu-ci-npu-${device_id}.lock"
  mkdir -p "$CI_NPU_LOCK_DIR"
  exec 9>"$lockfile"
  if ! flock -n 9; then
    return 1
  fi
  log "acquired NPU lock: $lockfile (physical device=$device_id)"
  trap 'flock -u 9' EXIT
  run_docker_test "$device_id"
  return $?
}

# ---------- 主流程 ----------
main() {
  local total_start
  total_start="$(date +%s)"
  log "CI start: $(date '+%Y-%m-%d %H:%M:%S')"

  # 阶段1: 编译
  if [ "$CI_SKIP_BUILD" = "true" ]; then
    log "CI_SKIP_BUILD=true, skip phase 1 (assume build/ exists)"
  else
    run_build_phase
    log "=== Phase 1 done ==="
  fi

  # 阶段2: 选卡 + 加锁 + 测试
  log "=== Phase 2: test (with NPU lock) ==="
  local start_ts waited
  start_ts="$(date +%s)"
  waited=0

  while true; do
    local cands
    cands="$(get_candidates)"
    if [ -z "$cands" ]; then
      die "no candidate NPU detected; run 'bash ci/detect_npu.sh --summary' to check"
    fi

    while IFS= read -r id; do
      [ -z "$id" ] && continue
      if acquire_lock_and_run_test "$id"; then
        local total_end
        total_end="$(date +%s)"
        log "CI end: $(date '+%Y-%m-%d %H:%M:%S') (total=$((total_end - total_start))s)"
        exit 0
      fi
      log "NPU $id locked, trying next candidate..."
    done <<< "$cands"

    if [ "$CI_NPU_LOCK_WAIT_SECONDS" != "0" ]; then
      waited=$(( $(date +%s) - start_ts ))
      if [ "$waited" -ge "$CI_NPU_LOCK_WAIT_SECONDS" ]; then
        die "All detected NPU devices are locked; waited ${waited}s >= CI_NPU_LOCK_WAIT_SECONDS=${CI_NPU_LOCK_WAIT_SECONDS}. Giving up."
      fi
    fi
    log "All detected NPU devices are locked; retrying in ${CI_NPU_LOCK_RETRY_SECONDS}s."
    sleep "$CI_NPU_LOCK_RETRY_SECONDS"
  done
}

main "$@"
