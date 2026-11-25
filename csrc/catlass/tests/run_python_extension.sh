#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Build/install/test runner for python extension + torch lib.
# Both targets share the same code path and run together.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
BUILD_SH="$PROJECT_ROOT/scripts/build.sh"
OUTPUT_DIR="$PROJECT_ROOT/output"

# --- pyext: build wheel -> install -> test -> uninstall ---
run_pyext() {
    echo ""
    echo "=== [python_extension] build + install + test ==="
    bash "$BUILD_SH" --clean python_extension || return 1

    wheel=$(find "$OUTPUT_DIR/python_extension" -name 'torch_catlass-*.whl' 2>/dev/null | head -1)
    if [ -z "$wheel" ]; then
        echo "    wheel not found under $OUTPUT_DIR/python_extension"
        return 1
    fi
    pip install "$wheel" || return 1
    python3 "$SCRIPT_DIR/test_python_extension.py" || return 1
    pip uninstall torch_catlass -y >/dev/null 2>&1 || true
    return 0
}

# --- torch_lib: build so -> test (test_torch_lib.py loads the so directly) ---
run_torch_lib() {
    echo ""
    echo "=== [torch_lib] build + test ==="
    bash "$BUILD_SH" --clean torch_library || return 1
    python3 "$SCRIPT_DIR/test_torch_lib.py" || return 1
    return 0
}

FAILED=0
for target in pyext torch_lib; do
    if run_${target}; then
        echo "    [OK] ${target}"
    else
        echo "    [FAIL] ${target}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "${FAILED}" -eq 0 ]; then
    echo "All python extension tests passed."
else
    echo "SUMMARY: ${FAILED} target(s) failed."
fi
echo "============================================================"
exit "${FAILED}"
