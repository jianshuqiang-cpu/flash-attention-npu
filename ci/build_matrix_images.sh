#!/usr/bin/env bash
# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# 读取 ci/build_matrix.tsv, 为每个 combo (CANN×torch/torch_npu) 构建专用镜像
# fla-npu-matrix:<name> (由 ci/Dockerfile.matrix 参数化构建)。
# 构建一次后, ci/run_ci_matrix.sh 直接 docker run 这些镜像并发编译。
#
# 用法:
#   bash ci/build_matrix_images.sh                 # 构建全部 combo
#   bash ci/build_matrix_images.sh <name1> <name2> # 只构建指定 combo
#
# 环境变量:
#   MATRIX_FILE           (默认 ci/build_matrix.tsv)
#   IMAGE_PREFIX          (默认 fla-npu-matrix)
#   BUILD_MATRIX_PARALLEL (默认 false) true=并发构建

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MATRIX_FILE="${MATRIX_FILE:-$REPO_ROOT/ci/build_matrix.tsv}"
IMAGE_PREFIX="${IMAGE_PREFIX:-fla-npu-matrix}"
PARALLEL="${BUILD_MATRIX_PARALLEL:-false}"

log() { printf '[matrix-images] %s\n' "$*"; }
die() { printf '[matrix-images][ERROR] %s\n' "$*" >&2; exit 1; }

[ -f "$MATRIX_FILE" ] || die "matrix file not found: $MATRIX_FILE"
command -v docker >/dev/null 2>&1 || die "docker not found"

WANT=("$@")

# 输出 combo 行 (跳过 # 和空行)
read_combos() {
  awk -F'|' '/^[[:space:]]*#/ || /^[[:space:]]*$/ {next} {print}' "$MATRIX_FILE"
}

build_one() {
  local line="$1" name base_image py_tag torch_ver torch_npu_ver torch_npu_rel
  IFS='|' read -r name base_image py_tag torch_ver torch_npu_ver torch_npu_rel <<< "$line"
  [ -n "$name" ] || return 0
  if [ "${#WANT[@]}" -gt 0 ]; then
    local hit=false w
    for w in "${WANT[@]}"; do [ "$w" = "$name" ] && { hit=true; break; }; done
    [ "$hit" = "true" ] || return 0
  fi
  log "building $IMAGE_PREFIX:$name (base=$base_image torch=$torch_ver torch_npu=$torch_npu_ver py=$py_tag)"
  if docker build -t "$IMAGE_PREFIX:$name" \
        --build-arg BASE_IMAGE="$base_image" \
        --build-arg PY_TAG="$py_tag" \
        --build-arg TORCH_VER="$torch_ver" \
        --build-arg TORCH_NPU_VER="$torch_npu_ver" \
        --build-arg TORCH_NPU_RELEASE="$torch_npu_rel" \
        -f "$SCRIPT_DIR/Dockerfile.matrix" "$SCRIPT_DIR"; then
    log "  -> $IMAGE_PREFIX:$name OK"
    return 0
  else
    log "  -> $IMAGE_PREFIX:$name FAILED"
    return 1
  fi
}

overall=0
if [ "$PARALLEL" = "true" ]; then
  pids=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    build_one "$line" &
    pids+=($!)
  done < <(read_combos)
  for p in "${pids[@]}"; do wait "$p" || overall=1; done
else
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    build_one "$line" || overall=1
  done < <(read_combos)
fi

if [ "$overall" -ne 0 ]; then
  die "one or more matrix images failed to build"
fi
log "done. built images:"
docker images --format '  {{.Repository}}:{{.Tag}}' | grep "^  ${IMAGE_PREFIX}:" || true
