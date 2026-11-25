#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: build.sh [--release] [--clean]

Options:
  --release    CMAKE_BUILD_TYPE=Release
  --clean      Remove all build artifacts and rebuild from scratch
  (default)    CMAKE_BUILD_TYPE=Debug and editable installation

Environment variables (auto-detected where possible):
  ASCEND_HOME_PATH                  CANN / ascend-toolkit root (REQUIRED)
  CATLASS_DSL_PREBUILT_ASCENDNPU_IR     AscendNPU-IR build root (REQUIRED)
                                    e.g. /path/to/AscendNPU-IR
EOF
}

mode="debug"
do_clean=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      mode="release"
      ;;
    --clean)
      do_clean=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

# ============================================================================
# 0. Clean build artifacts
# ============================================================================

if [[ $do_clean -eq 1 ]]; then
  echo "==> Cleaning build artifacts..."
  rm -rf build/ csrc/mlir/build/ dist/ *.egg-info .pytest_cache/
  rm -f catlass/_tla_type_bridge_native*.so
  echo "Clean complete."
fi

# ============================================================================
# 1. Validate required environment variables
# ============================================================================

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ERROR: ASCEND_HOME_PATH is not set." >&2
  echo "  Source set_env.sh from your CANN installation, e.g.:" >&2
  echo '    source /usr/local/Ascend/cann/set_env.sh' >&2
  exit 1
fi

if [[ -z "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-}" ]]; then
  CATLASS_DSL_PREBUILT_ASCENDNPU_IR="${repo_root}/3rdparty/AscendNPU-IR"
  echo "==> CATLASS_DSL_PREBUILT_ASCENDNPU_IR not set, using repo default: ${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}"
fi
# Export so child processes (setup.py / pip build subprocesses) can see it.
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR

# ============================================================================
# 2. Auto-derive MLIR paths (consumed by setup.py / CMake / generate_tla_python_bindings)
#    Layout: $CATLASS_DSL_PREBUILT_ASCENDNPU_IR/build/install/{include,lib/cmake/{mlir,llvm}}
# ============================================================================

npu_ir_install="${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install"

if [[ ! -d "$npu_ir_install" ]]; then
  echo "ERROR: AscendNPU-IR install prefix not found: $npu_ir_install" >&2
  echo "  Build AscendNPU-IR first, or fix CATLASS_DSL_PREBUILT_ASCENDNPU_IR." >&2
  exit 1
fi

export MLIR_TBLGEN_INCLUDE_DIR="${npu_ir_install}/include"
export MLIR_DIR="${npu_ir_install}/lib/cmake/mlir"
export LLVM_DIR="${npu_ir_install}/lib/cmake/llvm"

# Allow Python to import mlir_core and other AscendNPU-IR-provided Python packages
mlir_core="${npu_ir_install}/python_packages/mlir_core"
if [[ -d "$mlir_core" ]]; then
  export PYTHONPATH="${mlir_core}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "==> ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "==> CATLASS_DSL_PREBUILT_ASCENDNPU_IR=${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}"
echo "==> MLIR_TBLGEN_INCLUDE_DIR=${MLIR_TBLGEN_INCLUDE_DIR}"
echo "==> MLIR_DIR=${MLIR_DIR}"

# ============================================================================
# 3. 检查 C++ 源文件是否过期（仅 dev 模式，自动触发 cmake 重新配置）
# ============================================================================

if [[ "$mode" == "debug" ]]; then
  tla_compile_bin="csrc/mlir/build/tools/tla-compile/TlaCompile"
  if [[ -f "$tla_compile_bin" ]]; then
    stale=$(find csrc/mlir/ -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.td' -o -name '*.inc' \) \
      -newer "$tla_compile_bin" -print -quit)
    if [[ -n "$stale" ]]; then
      echo "==> Source file newer than tla-compile: ${stale#csrc/mlir/}"
      echo "==> Removing cmake build stamp to force reconfigure."
      rm -f csrc/mlir/build/CMakeCache.txt
    fi
  fi
fi

# ============================================================================
# 4. Build

if [[ "$mode" == "release" ]]; then
  export CMAKE_BUILD_TYPE="Release"
  python -m pip wheel . -w dist/
  echo "Release build complete."
  echo "Wheels: ${repo_root}/dist/"
else
  export CMAKE_BUILD_TYPE="Debug"
  export CMAKE_BUILD_DIR="csrc/mlir/build"
  python setup.py build_ext --inplace
  # PEP 517 metadata generation leaves an in-tree *.egg-info directory. Start
  # pip outside the project and isolate its Python process from PYTHONPATH so
  # that source metadata cannot shadow the site-packages installation.
  (
    cd "$repo_root/.."
    python -I -m pip install -e "$repo_root" --no-deps
  )
  echo "Debug build and install complete."
fi
