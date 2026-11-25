#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_DIR="$PROJECT_ROOT/python/catlass_cppgen/tests"

# ------------------------------------------------------------------
# resolve python
# ------------------------------------------------------------------
if command -v pyenv &>/dev/null && pyenv which python &>/dev/null; then
    PYTHON="$(pyenv which python)"
else
    PYTHON="python3"
fi

# 1. 编译 catlass_cppgen
echo ""
echo "============================================"
echo "Step 1: Building catlass_cppgen..."
echo "============================================"
cd "$PROJECT_ROOT/python/catlass_cppgen"
"$PYTHON" -m pip install build
"$PYTHON" -m build
whl_file=$(python -m build 2>&1 | grep "Successfully built" | grep -oE "[^ ]+\.whl")

# 2. 安装 catlass_cppgen
echo ""
echo "============================================"
echo "Step 2: Installing catlass_cppgen from dist..."
echo "============================================"
"$PYTHON" -m pip install "dist/$whl_file" --force-reinstall --no-deps

# 3. 执行所有catlass_cppgen的测试例
echo ""
echo "============================================"
echo "Step 3: Preparing test all testcases..."
echo "============================================"

PASSED=0
FAILED=0
test_files=()
while IFS= read -r -d '' file; do
    test_files+=("$file")
done < <(find "$TEST_DIR" -name "test_*.py" -print0 | sort -z)

if [ ${#test_files[@]} -eq 0 ]; then
    echo "No test files found under $TEST_DIR"
    exit 1
fi

for test_file in "${test_files[@]}"; do
    rel_path="${test_file#$TEST_DIR/}"
    echo "--- $rel_path ---"
    if "$PYTHON" "$test_file"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] $rel_path"
    fi
    echo ""
done

# 4. 清理环境
echo ""
echo "============================================"
echo "Step 4: Cleaning and uninstalling catlass_cppgen..."
echo "============================================"
cd "$PROJECT_ROOT/python/catlass_cppgen"
rm -rf ./dist ./build ./catlass_cppgen.egg-info
"$PYTHON" -m pip uninstall catlass_cppgen -y
"$PYTHON" -m pip uninstall build -y

echo ""
echo "============================================"
echo "All steps completed successfully!"
echo " SUMMARY: $PASSED passed, $FAILED failed"
echo "============================================"

exit $FAILED
