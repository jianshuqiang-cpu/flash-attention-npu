# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# CI 镜像: torch 2.7.1 + Ascend PyTorch v26.1.0-beta.1 (torch_npu-2.7.1.post5) + triton-ascend 3.2.0
#
# 构建前必须确认:
#   1. BASE_IMAGE 是带 Ascend CANN toolkit (含 bisheng 编译器 + ascendcl/tiling_api/platform 库 + tikcpp 头) 的镜像
#      本仓 setup.py 用 bisheng 编译扩展, 因此 BASE_IMAGE 必须含完整 CANN toolkit, 不能只有 runtime。
#   2. TORCH_NPU_WHEEL_URL 指向 Ascend PyTorch v26.1.0-beta.1 对应的 torch_npu wheel (GitCode release 页)
#      该 wheel 已包含 torchnpugen, 不要再拉 op-plugin 重新编译。
#   3. CANN toolkit 路径通过 ASCEND_TOOLKIT_HOME 传给 setup.py, 默认 /usr/local/Ascend/ascend-toolkit/latest
#
# 构建:
#   docker build -t fla-npu-ci:8.5.0-910b \
#     --build-arg TORCH_NPU_WHEEL_URL=<你的 torch_npu wheel URL> \
#     -f ci/Dockerfile .

# 带完整 CANN toolkit 的基础镜像 (含 bisheng / ascendcl / tiling_api / tikcpp 头)
# 默认占位: 请替换为实际可用的 CANN toolkit aarch64 镜像
ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel
FROM ${BASE_IMAGE}

# Python 版本 (与 torch_npu wheel 匹配)
ARG PYTHON_VERSION=3.10
# torch_npu wheel (Ascend PyTorch v26.1.0-beta.1 对应 torch_npu-2.7.1.post5)
# 必填: 指向 GitCode release 页对应 aarch64 / py3.10 / torch2.7.1 的 wheel
ARG TORCH_NPU_WHEEL_URL
ARG TORCH_NPU_WHEEL_NAME=torch_npu-2.7.1.post5-cp310-cp310-linux_aarch64.whl

# CANN toolkit home (镜像内路径)
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/runtime/lib64:${LD_LIBRARY_PATH:-}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${PATH:-}

# 1. 基础工具
RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
      apt-get update -qq && apt-get install -y -qq \
        git curl wget jq build-essential cmake ninja-build \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v yum >/dev/null 2>&1; then \
      yum install -y -q git curl wget jq gcc-c++ make cmake ninja-build \
        && yum clean all; \
    fi

# 2. Python 依赖
RUN python3 --version || true
RUN python3 -m pip install --upgrade pip wheel setuptools

# torch 2.7.1 (CPU/aarch64 wheel; torch_npu 会接管 NPU dispatch)
RUN python3 -m pip install --no-cache-dir torch==2.7.1

# torch_npu (Ascend PyTorch v26.1.0-beta.1)
RUN if [ -z "${TORCH_NPU_WHEEL_URL:-}" ]; then \
      echo "[Dockerfile] ERROR: TORCH_NPU_WHEEL_URL build-arg is required"; \
      echo "[Dockerfile]        set it to the GitCode release wheel URL for v26.1.0-beta.1"; \
      exit 1; \
    fi; \
    python3 -m pip install --no-cache-dir "${TORCH_NPU_WHEEL_URL}/${TORCH_NPU_WHEEL_NAME}"

# triton-ascend (与社区版 triton 互斥, 只保留 triton-ascend 提供的 triton 模块)
RUN python3 -m pip install --no-cache-dir triton-ascend==3.2.0
RUN python3 -m pip uninstall -y triton || true

# 3. 编译/测试所需 Python 包
RUN python3 -m pip install --no-cache-dir \
      pybind11 \
      einops \
      packaging \
      psutil \
      pytest \
      numpy

# 4. 校验 torch / torch_npu 可 import (不触卡, 仅模块可用性)
RUN python3 - <<'PY'
import torch, torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
PY

WORKDIR /workspace/flash-attention-npu
