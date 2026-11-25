#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-$(which python3)}"
echo "Using python: $($PYTHON --version 2>&1) at $PYTHON"

clear_caches() {
    # JIT cache priority (kernels/jit/jit_compiler.cpp):
    #   1. $CATLASS_JIT_CACHE_DIR  (env override)
    #   2. $HOME/.cache/catlass/jit_cache  (default, with <version> subdir)
    #   3. /tmp/catlass_jit  (fallback when HOME is unset)
    if [ -n "${CATLASS_JIT_CACHE_DIR:-}" ] && [ -d "$CATLASS_JIT_CACHE_DIR" ]; then
        echo "    \$CATLASS_JIT_CACHE_DIR=$CATLASS_JIT_CACHE_DIR"
        rm -rf "$CATLASS_JIT_CACHE_DIR"
    fi
    local home_jit="$HOME/.cache/catlass/jit_cache"
    if [ -d "$home_jit" ]; then
        echo "    $home_jit"
        rm -rf "$home_jit"
    fi
    if [ -d /tmp/catlass_jit ]; then
        echo "    /tmp/catlass_jit (fallback)"
        rm -rf /tmp/catlass_jit
    fi
    echo "    $1/.pytest_cache"
    rm -rf "$1/.pytest_cache"
    echo "    $1/tests/__pycache__"
    rm -rf "$1/tests/__pycache__"
    echo "    $1/tests/test.log"
    rm -rf "$1/tests/test.log"
    echo "    $1/.ruff_cache"
    rm -rf "$1/.ruff_cache"
}

# 1. 清理旧的编译产物
echo "============================================"
echo "Step 1: Cleaning old build artifacts..."
echo "============================================"
rm -rf optest/build optest/dist optest/*.egg-info optest/_skbuild

echo ""
echo "Cleaning runtime caches before build..."
clear_caches "optest"

# 2. 编译 optest
echo ""
echo "============================================"
echo "Step 2: Building optest..."
echo "============================================"

export CC=$(command -v gcc)
export CXX=$(command -v g++)

echo "Using CC=$CC, CXX=$CXX"
pip install scikit-build-core
cd "$SCRIPT_DIR/optest"
bash build.sh

# 3. 安装 optest（从 dist）
echo ""
echo "============================================"
echo "Step 3: Installing optest from dist..."
echo "============================================"
cd "$SCRIPT_DIR/optest"
WHEEL_FILE=$(ls -1 dist/*.whl | head -1)
echo "Installing: $WHEEL_FILE"
pip install "$WHEEL_FILE" --force-reinstall --no-deps

# 4. 运行测试
echo ""
echo "============================================"
echo "Step 4: Cleaning runtime caches before run..."
echo "============================================"
clear_caches "optest"

echo ""
echo "============================================"
echo "Step 5: Running tests..."
echo "============================================"
# JIT 日志全开：0=None, 1=Info, 2=Debug（详细 compile/cache/mem hit 等）
export CATLASS_JIT_LOG_LEVEL=2
cd "$SCRIPT_DIR/optest"
python3 -m pytest tests/ -v

# 6. 卸载 optest
echo ""
echo "============================================"
echo "Step 6: Uninstalling optest..."
echo "============================================"
pip uninstall torch-catlass -y

# 7. 收尾清理
echo ""
echo "============================================"
echo "Step 7: Cleaning runtime caches after run..."
echo "============================================"
clear_caches "optest"

echo ""
echo "============================================"
echo "All steps completed successfully!"
echo "============================================"