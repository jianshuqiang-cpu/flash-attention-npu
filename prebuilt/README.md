# 预编译产物（可选）

同一份 `.so` 已放在包根目录 `flash-attention-npu/flash_attn_npu_3_950.cpython-312-x86_64-linux-gnu.so`
（`flash_attn_npu_3/__init__.py` 需要它在 `sys.path` 上，放在根目录即可直接 import）。

- 构建于：Ascend950PR_957b / CANN 9.1.0 / Python 3.12 / torch 2.10.0+cpu / torch_npu 2.10.0.post4
- 构建命令：`FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_JOBS=2 python setup.py build_ext --inplace`
- sha256：`e30f43157f18fc9502e1d310cae3d2a8...`（完整值见包根目录 SHA256SUMS）
- 仅在同版本 CANN/torch_npu 环境下可直接使用；否则按根目录 README 重新编译。
- 对应源码即本包 v10（测试 139 passed / 86 skipped）。
