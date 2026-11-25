# Docker 安装

Docker 配置基于与目标设备、驱动和 Python 版本匹配的 CANN 基础镜像。

项目镜像会安装 LLVM 工具链、Python 依赖、`torch`、`torch-npu` 和 AscendNPU-IR。镜像不包含 CATLASS 源码，启动时从宿主机挂载。

## 1. 构建镜像

```bash
# /path/to/catlass 需替换为你 clone 的 CATLASS 仓库实际路径
cd /path/to/catlass/python/tla_dsl
# 国内环境推荐追加 --default-mirror 使用镜像源加速（镜像源不可达时去掉）：
#   bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12 --default-mirror
bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12
```

基础镜像名称只是示例。请从 [AscendHub](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884) 选择与设备、驱动和 Python 版本匹配的 CANN 镜像，并将完整名称作为第一个参数。输出镜像为 `ascend-catlass-dsl:<基础镜像 tag>`。

脚本支持替换软件源、指定 LLVM 版本、调整构建并发数和目标平台：

```bash
bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12 --help
```

AscendNPU-IR 构建耗时和资源占用较高。内存或磁盘 I/O 有限时，通过 `--build-jobs` 降低并发数。

## 2. 启动编译容器

只进行 DSL 构建时，不需要挂载 NPU 设备：

```bash
# 镜像名与 tag 需与第 1 节构建产物（ascend-catlass-dsl:<基础镜像 tag>）一致，此处仅为例示
# 容器名可自行指定，多人共用宿主机时避免重名
docker run \
    --rm \
    --name ascend-catlass-dsl-dev \
    -v /path/to/catlass:/workspace/catlass \
    -w /workspace/catlass/python/tla_dsl \
    -it ascend-catlass-dsl:9.1.0-950-ubuntu22.04-py3.12 bash
```

容器内的关键环境：

```bash
# 以下命令均无输出、退出码为 0 即表示环境就绪；失败时退出码非零并输出错误信息
test -n "${ASCEND_HOME_PATH}"
test -n "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}"
test -f "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/lib/cmake/mlir/MLIRConfig.cmake"
python -c "import mlir"
```

[编译与测试](../01_build_and_test.md)说明项目构建和各类测试入口。

## 3. 启动上板容器

运行 NPU 端到端示例时，需要透传设备节点和宿主机驱动文件：

```bash
docker run \
    --rm \
    --name ascend-catlass-dsl-dev \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /path/to/catlass:/workspace/catlass \
    -w /workspace/catlass/python/tla_dsl \
    -it ascend-catlass-dsl:9.1.0-950-ubuntu22.04-py3.12 bash
```

设备节点和驱动路径以宿主机的 CANN/驱动安装为准。容器内可通过 `npu-smi info` 检查设备，再运行 [NPU 端到端示例](../01_build_and_test.md#5-npu-端到端示例)。
