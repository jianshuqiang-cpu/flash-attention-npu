# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# CI 镜像: 基于本地已有的 CANN toolkit 镜像 (py3.12),
#          叠加本仓 CI 所需的编译/测试工具 + torch 2.9.0 + torch_npu 2.9.0.post2。
#
# 构建:
#   docker build -t fla-npu-ci:9.1.0 -f ci/Dockerfile .

# 服务器本地已有的 CANN toolkit 镜像 (aarch64, 910b, ubuntu22.04, py3.12)
# 已预装: CANN 9.1.0-beta.1 toolkit (含 bisheng 编译器, ascendcl, tiling_api 等)
ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel
FROM ${BASE_IMAGE}

# CANN toolkit home (镜像内路径, 官方镜像已设置, 这里显式声明以防缺失)
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/runtime/lib64:${LD_LIBRARY_PATH:-}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${PATH:-}

# 1. 基础工具 (官方镜像通常已有 git, 这里补齐确保齐全)
RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
      apt-get update -qq && apt-get install -y -qq \
        git curl wget jq build-essential cmake ninja-build \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v yum >/dev/null 2>&1; then \
      yum install -y -q git curl wget jq gcc-c++ make cmake ninja-build \
        && yum clean all; \
    fi

# 2. Python 包管理工具
RUN python3 -m pip install --upgrade pip wheel setuptools

# 3. torch 2.9.0 (cpu 版, aarch64, cp312)
#    从 PyTorch 官方 wheel 仓库下载
RUN cd /tmp && \
    wget -q https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl && \
    python3 -m pip install --no-cache-dir torch-2.9.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl && \
    rm -f torch-2.9.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl

# 4. torch_npu 2.9.0.post2 (aarch64, cp312)
#    从 GitCode release 页下载 (对应 CANN 26.0.0 / 9.0.0, 向下兼容 CANN 9.1.0-beta.1)
RUN cd /tmp && \
    wget -q https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.9.0/torch_npu-2.9.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl && \
    python3 -m pip install --no-cache-dir torch_npu-2.9.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl && \
    rm -f torch_npu-2.9.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl

# 5. 编译/测试所需 Python 包
#    本仓不依赖 triton, 不装 triton-ascend
RUN python3 -m pip install --no-cache-dir \
      pybind11 \
      einops \
      packaging \
      psutil \
      pytest \
      numpy

# 6. 校验 torch / torch_npu 可 import (不触卡, 仅模块可用性)
RUN python3 - <<'PY'
import torch, torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
PY

WORKDIR /workspace/flash-attention-npu
