# Self-hosted Runner 注册指南 (910B / 950)

本仓 CI 使用 self-hosted runner，按架构分两类机器：

| 架构 | 机器标签 | Docker 镜像 | 适用 job |
|------|---------|------------|---------|
| aarch64 (910B) | `self-hosted, linux, arm64, npu, flash-attention-npu` | `fla-npu-ci:9.1.0` | 910B CI |
| x86_64 (950) | `self-hosted, linux, x64, npu, flash-attention-npu` | `fla-npu-matrix:950-cann9.1-torch2.9` | 950 CI |

CI 采用**动态选卡**（按 `task_count` 选最空闲的卡，不锁卡），支持多 runner 共享同一台机器的 NPU。要实现多 PR 并行，需在同一台机器注册多个 runner 进程。

---

## 前置条件

每台机器上需提前准备好：

1. **Ascend 驱动 + npu-smi** 可用：`npu-smi info` 能正常输出
2. **Docker** 可用，且已 load 对应镜像：
   - 910B：`docker load -i fla-npu-ci-9.1.0.tar.gz`（确认 `docker images | grep fla-npu-ci`）
   - 950：`docker load -i fla-npu-matrix-950.tar.gz`（确认 `docker images | grep fla-npu-matrix`）
3. **注册 token**：GitHub 仓库 → Settings → Actions → Runners → New self-hosted runner，选对应架构，复制 token（时效约 1 小时，过期重新生成）

---

## 方式一：用脚本自动注册（推荐）

仓库自带注册脚本 [ci/setup_self_hosted_runner.sh](ci/setup_self_hosted_runner.sh)，自动识别架构、下载 runner、配置标签、安装 systemd 服务。

### 910B 机器注册第 N 个 runner

```bash
# 以普通用户执行 (不要用 root), 需有 sudo 权限用于安装 systemd 服务
# RUNNER_ROOT 每个 runner 用不同目录, 避免冲突
# token 从 GitHub Settings -> Actions -> Runners 获取

# 第 1 个 runner
RUNNER_ROOT=/home/j00574704/flash-attention-npu-ci/actions-runner \
RUNNER_NAME=ubuntu-flash-attention-npu \
  bash ci/setup_self_hosted_runner.sh \
    --url https://github.com/jianshuqiang-cpu/flash-attention-npu \
    --token <TOKEN_1>

# 第 2 个 runner (多 PR 并行用)
RUNNER_ROOT=/home/j00574704/flash-attention-npu-ci/actions-runner-2 \
RUNNER_NAME=ubuntu-flash-attention-npu-2 \
  bash ci/setup_self_hosted_runner.sh \
    --url https://github.com/jianshuqiang-cpu/flash-attention-npu \
    --token <TOKEN_2>

# 第 3 个 runner (按需)
RUNNER_ROOT=/home/j00574704/flash-attention-npu-ci/actions-runner-3 \
RUNNER_NAME=ubuntu-flash-attention-npu-3 \
  bash ci/setup_self_hosted_runner.sh \
    --url https://github.com/jianshuqiang-cpu/flash-attention-npu \
    --token <TOKEN_3>
```

脚本会自动：
- 检测架构（910B 为 arm64）→ 标签设为 `linux,arm64,npu,flash-attention-npu`
- 下载 runner v2.319.0（已下载则复用）
- `config.sh --unattended --replace` 非交互配置
- `sudo ./svc.sh install && start` 装 systemd 服务开机自启

### 950 机器注册第 N 个 runner

```bash
# 950 为 x86_64, 脚本自动设标签 linux,x64,npu,flash-attention-npu

# 第 1 个 runner
RUNNER_ROOT=/home/npu_user7/jianshuqiang/flash-attention-npu-ci/actions-runner \
RUNNER_NAME=sz-blue-950pr-13-241 \
  bash ci/setup_self_hosted_runner.sh \
    --url https://github.com/jianshuqiang-cpu/flash-attention-npu \
    --token <TOKEN_1>

# 第 2 个 runner
RUNNER_ROOT=/home/npu_user7/jianshuqiang/flash-attention-npu-ci/actions-runner-2 \
RUNNER_NAME=sz-blue-950pr-13-241-2 \
  bash ci/setup_self_hosted_runner.sh \
    --url https://github.com/jianshuqiang-cpu/flash-attention-npu \
    --token <TOKEN_2>
```

---

## 方式二：手动注册（调试用）

如果脚本失败或需自定义，可手动执行：

### 910B (arm64) 注册第 2 个 runner

```bash
# 1. 创建独立目录
RUNNER_ROOT=/home/j00574704/flash-attention-npu-ci/actions-runner-2
mkdir -p "$RUNNER_ROOT"
cd "$RUNNER_ROOT"

# 2. 下载解压 runner (arm64)
RUNNER_VERSION=2.319.0
curl -fL -o actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz \
  https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz
tar xzf actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz

# 3. 配置 (token 从 GitHub 获取)
./config.sh \
  --url https://github.com/jianshuqiang-cpu/flash-attention-npu \
  --token <TOKEN> \
  --name ubuntu-flash-attention-npu-2 \
  --labels "self-hosted,linux,arm64,npu,flash-attention-npu" \
  --replace \
  --unattended \
  --work "_work"

# 4. 安装 systemd 服务 (用 sudo, 服务名会带 runner 目录名区分)
sudo ./svc.sh install
sudo ./svc.sh start
```

### 950 (x64) 注册第 2 个 runner

```bash
RUNNER_ROOT=/home/npu_user7/jianshuqiang/flash-attention-npu-ci/actions-runner-2
mkdir -p "$RUNNER_ROOT"
cd "$RUNNER_ROOT"

# 下载解压 runner (x64)
RUNNER_VERSION=2.319.0
curl -fL -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
  https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
tar xzf actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# 配置 (标签含 x64, 区分 950)
./config.sh \
  --url https://github.com/jianshuqiang-cpu/flash-attention-npu \
  --token <TOKEN> \
  --name sz-blue-950pr-13-241-2 \
  --labels "self-hosted,linux,x64,npu,flash-attention-npu" \
  --replace \
  --unattended \
  --work "_work"

# 安装 systemd 服务
sudo ./svc.sh install
sudo ./svc.sh start
```

---

## 验证

注册后在 GitHub 仓库 → Settings → Actions → Runners 页面应看到：

- 910B 机器：多个 runner，状态为绿色 Idle，标签含 `arm64`
- 950 机器：多个 runner，状态为绿色 Idle，标签含 `x64`

在机器上验证：

```bash
# 910B: 看所有 runner 服务
sudo systemctl list-units --type=service | grep actions.runner

# 910B: 看 runner 进程数 (每个 runner 一个 Runner.Worker)
ps aux | grep Runner.Worker | grep -v grep

# 950: 同上
```

---

## 运维操作

### 查看状态
```bash
# 单个 runner
sudo ./svc.sh status

# 所有 runner 服务
sudo systemctl list-units --type=service | grep actions.runner
```

### 重启
```bash
cd <RUNNER_ROOT>
sudo ./svc.sh stop
sudo ./svc.sh start
```

### 删除 runner
```bash
cd <RUNNER_ROOT>
# 1. 先在 GitHub 页面 Remove, 或用 token remove
./config.sh remove --token <TOKEN>
# 2. 卸载服务
sudo ./svc.sh uninstall
# 3. 删除目录
cd ..
rm -rf <RUNNER_ROOT>
```

### 升级 runner 版本
```bash
cd <RUNNER_ROOT>
sudo ./svc.sh stop
# 备份 .credentials 和 .env
cp -r .credentials .env /tmp/runner-backup/
# 下载新版覆盖
RUNNER_VERSION=2.320.0
curl -fL -o actions-runner-linux-<arch>-${RUNNER_VERSION}.tar.gz \
  https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-<arch>-${RUNNER_VERSION}.tar.gz
tar xzf actions-runner-linux-<arch>-${RUNNER_VERSION}.tar.gz
# 恢复凭证
cp -r /tmp/runner-backup/.credentials /tmp/runner-backup/.env .
sudo ./svc.sh start
```

---

## CI 动态选卡机制

注册多 runner 后，CI 会自动并行：

1. 多个 PR 同时触发 → GitHub 分发给不同 runner 进程
2. 每个 runner 跑 `ci/run_ci_container.sh`，调用 `ci/detect_npu.sh` 探测卡状态
3. 选 `task_count < 阈值` 且 HBM 空闲的最优卡（按 task_count 升序）
4. docker 绑卡运行，不持锁，其他人可共享同卡

### 选卡阈值（按架构区分，在 workflow yml 配置）

| 架构 | require_healthy | max_tasks | 理由 |
|------|----------------|-----------|------|
| 950 | true | 6 | Alarm 卡多需排除；128GB HBM 可放宽 |
| 910B | false | 4 | 卡稳定；64GB HBM 保守 |

### 所有卡忙时

runner 会等待（默认每 30s 重探，最长等 600s），有卡空闲即自动接上。超时则报错，PR 可重新触发 CI。

---

## 常见问题

### Q: job 卡在 "Waiting for a runner to pick up this job"

- runner 离线 → GitHub Runners 页面看是否灰色 Offline，机器上 `sudo ./svc.sh start`
- runner 都在忙（单 runner 单并发）→ 注册更多 runner，或等当前 job 跑完
- 标签不匹配 → 确认 runner 标签含 job 要求的全部标签（`self-hosted, linux, arm64/x64, npu, flash-attention-npu`）

### Q: runner 显示 Busy 但 Actions 页面无运行中的 job

僵尸占用，重启 runner：
```bash
cd <RUNNER_ROOT>
sudo ./svc.sh stop
ps aux | grep -E "Runner|runsvc" | grep -v grep  # 有残留就 kill
sudo ./svc.sh start
```

### Q: config.sh 报 "must not run with sudo/root"

GitHub runner 拒绝 root 运行。用普通用户执行 `config.sh`，仅在 `./svc.sh install` 时用 sudo。

### Q: 两个 runner 选中同一张卡

正常现象（动态选卡是软共享）。两个 docker 绑同一卡，NPU 支持多进程并发，靠 `max_tasks` 阈值限流。若担心过载，调小 `CI_NPU_MAX_TASKS`（在 workflow yml 的 matrix env 里改）。
