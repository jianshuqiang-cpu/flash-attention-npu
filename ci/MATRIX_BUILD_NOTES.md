# 遗留问题:多 CANN × torch 并发编译矩阵

> 状态:**已搁置**(2026-07-31)。待提供兼容矩阵后继续。脚手架已建好,不影响现有 CI。

## 目标

在不同 CANN 版本(8.5.0 / 9.0.0 / 9.1.0)× 不同 torch/torch_npu 版本(2.7.0~2.10.0)的
docker 容器中并发编译本仓,检查代码在各版本组合下能否编译通过(编译/语法/兼容问题)。
torch 与 torch_npu 同版本(README 约定)。

## 已建文件(均在工作区,未提交)

| 文件 | 作用 |
|---|---|
| `ci/build_matrix.tsv` | combo 清单。格式 `name\|base_image\|py_tag\|torch_ver\|torch_npu_ver\|torch_npu_release`,已用 9.1.0+2.9.0 一行播种 |
| `ci/Dockerfile.matrix` | 参数化镜像(ARG: BASE_IMAGE/PY_TAG/TORCH_VER/TORCH_NPU_VER/TORCH_NPU_RELEASE),装 torch(cpu aarch64)+torch_npu(gitcode)+依赖 |
| `ci/build_matrix_images.sh` | 读 tsv,为每个 combo 构建 `fla-npu-matrix:<name>` |
| `ci/run_ci_matrix.sh` | 并发编译驱动:预初始化子模块一次 -> 每 combo 一容器并发 `python setup.py build --build-base=/tmp/build` + `FLASH_ATTN_SKIP_SUBMODULE_INIT=1` -> 汇总 per-combo pass/fail |
| `.github/workflows/npu_ci_matrix.yml` | `workflow_dispatch` 手动触发(独立于主 CI,不影响 /run-npu-ci) |
| `setup.py` | 加了 `FLASH_ATTN_SKIP_SUBMODULE_INIT` 守卫(并发时跳过 git submodule,避免 .git/config 写竞争;默认行为不变) |

设计决策:预构建每 combo 专用镜像(用户选择);独立 workflow_dispatch 手动跑(矩阵重,不挂每次 PR);
并发安全靠 `--build-base` 隔离产物 + 子模块预初始化一次。

## 待办(恢复时)

1. **提供兼容矩阵**:在 AscendHub 核对各 CANN 的 `base_image` tag、在 gitcode 核对每个 torch_npu 的
   `torch_npu_ver`(wheel 文件名版本串,如 `2.9.0.post2`)和 `torch_npu_release`(release tag,如
   `v26.0.0-pytorch2.9.0`),按格式追加到 `ci/build_matrix.tsv`。
2. **验证两个假设**(我无法在此环境联网核实):
   - wheel URL 模式:`torch-<ver>%2Bcpu-<py>-<py>-manylinux_2_28_aarch64.whl`(pytorch.org)、
     `torch_npu-<ver>-<py>-<py>-manylinux_2_28_aarch64.whl`(gitcode)。某版本 manylinux tag 或文件名不同会在 build 镜像时 404。
   - CANN 路径:Dockerfile 硬编码 `/usr/local/Ascend/ascend-toolkit/latest`(9.1.0 可用);8.5.0/9.0.0 镜像若不同需改。

## 恢复步骤

```bash
# 1. 用已知 combo 验证链路
bash ci/build_matrix_images.sh 910b-cann9.1-torch2.9
bash ci/run_ci_matrix.sh          # 看 build/matrix-logs/<name>.log + 结果表

# 2. 补全矩阵行后
bash ci/build_matrix_images.sh
CI_MATRIX_MAX_JOBS=3 bash ci/run_ci_matrix.sh
```

## 备注

- 之前基于误读做的"按扩展并发编译"已回退;高优先级 CI 修复已提交为 `d4171ed fix`。
- 矩阵文件未提交,在 `git status` 里是 untracked。需要保留时记得提交(或我帮忙提到一个分支)。
