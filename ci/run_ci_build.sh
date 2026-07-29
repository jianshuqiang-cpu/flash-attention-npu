#!/usr/bin/env bash
# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# 阶段1: 编译 (容器内执行, 不需要 NPU, 不加锁)
#   1. git submodule update --init
#   2. python setup.py build  (产物在 build/, 通过 volume 持久化供阶段2复用)
#
# 由 ci/run_ci_container.sh 阶段1通过 docker run 调用 (不绑卡)。

set -euo pipefail

REPO_ROOT="$(pwd)"

log() { printf '[CI-build] %s\n' "$*"; }
die() { printf '[CI-build][ERROR] %s\n' "$*" >&2; exit 1; }

log "repo=$REPO_ROOT (build phase, no NPU needed)"
log "build phase start: $(date '+%Y-%m-%d %H:%M:%S')"

command -v python3 >/dev/null 2>&1 || die "python3 not found in container"

# ---------- 1. 子模块 ----------
log "init submodules: csrc/catlass csrc_AscendC950/catlass"
git submodule update --init --recursive csrc/catlass csrc_AscendC950/catlass

# ---------- 2. 编译 (python setup.py build) ----------
export FLASH_ATTN_FORCE_BUILD=TRUE
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
log "python setup.py build (FLASH_ATTN_BUILD_VERSION=${FLASH_ATTN_BUILD_VERSION:-all})"
python3 setup.py build

log "build phase done (artifacts in build/)"
log "build phase end: $(date '+%Y-%m-%d %H:%M:%S')"
