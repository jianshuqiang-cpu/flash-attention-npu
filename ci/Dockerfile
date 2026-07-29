# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# CI 镜像: 基于官方 torch_npu 镜像 (torch 2.7.1 + torch_npu 2.7.1.post4 + CANN 9.0.0)
#          叠加本仓 CI 所需的编译/测试工具。
#
# 方案 B: 直接用 Ascend 官方 torch-npu 镜像, torch/torch_npu/CANN 均已预装,
#         无需 TORCH_NPU_WHEEL_URL, 无需 Python 版本匹配。
#         本仓 setup.py 用 bisheng 编译 C++ 扩展, 不依赖 triton, 无需 triton-ascend。
#
# 构建:
#   docker build -t fla-npu-ci:8.5.0-910b -f ci/Dockerfile .

# Ascend 官方 torch-npu 镜像 (aarch64, 910b, ubuntu22.04, py3.11)
# 已预装: torch 2.7.1 + torch_npu 2.7.1.post4 + CANN 9.0.0 toolkit (含 bisheng)
# 如需切换芯片/系统/Python 版本, 参考:
#   https://www.hiascend.com/developer/ascendhub/detail/torch-npu
# 可选 tag 示例:
#   2.7.1.post4-910b-ubuntu22.04-py3.11   (默认)
#   2.7.1.post4-910b-openeuler24.03-py3.11
#   2.7.1.post4-a3-ubuntu22.04-py3.11
ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11
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

# 3. 编译/测试所需 Python 包 (torch/torch_npu 已由基础镜像提供, 不重装)
#    本仓不依赖 triton, 不装 triton-ascend
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
