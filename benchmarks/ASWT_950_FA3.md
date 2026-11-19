# 交接文档：Ascend 950 FA3 forward 的滑窗自适应分核（ASWT）

**交接状态（2026-11-19）：候选实现和两版二进制已就绪，Host 调度覆盖通过；950 真机精度和性能均未验收，默认关闭。** 接手人首先确认有一张可独占的 Ascend 950，再按下文用同一设备验证原版 A 与候选 B。不要把任务数变化或文档中的 GEMM 加速比当成 FA3 实测收益。

本机直接使用 `SWAT/flash-attention-npu` 工作树即可。若使用 `fa3-aswt-handoff.tar.gz` 换机交接，可在固定基线版本的**干净候选 checkout** 根目录解包，执行 `git apply --check fa3-aswt.patch && git apply fa3-aswt.patch`；不要在已有改动的本机候选或基线 A 上再次打补丁。包内含此文档、三份性能脚本、Host 测试和生产代码补丁，不含预编译二进制、设备数据或 CATLASS 子模块。

## 设计来源：从 GEMM 到 attention

用户给的图中，方格是计算块，数字是分配到的核编号，不是矩阵值。传统行/列优先、错位分核和 SWAT 比较的是**同一轮各核去读哪些数据**。SWAT 让邻近计算块在相近时间运行，目的是提高共享 L2 中数据被再次使用的机会。这里的“滑窗”仅指**调度顺序**，不是 attention 的滑窗 mask，不会改变一个 query 能看见哪些 key。

[CATLASS ASWT 设计原文](https://catlass.readthedocs.io/zh-cn/latest/2_Design/01_kernel_design/05_aswt/) 在 S 形滑窗之外，还对不足一轮的剩余基本块做自适应切分：原文 20 核示例中，9 个尾块拆成 18 块。固定版原文见 `csrc/catlass/docs/zh/2_Design/01_kernel_design/05_aswt.md`，参考算法见 `csrc/catlass/include/catlass/gemm/block/block_scheduler_aswt.hpp`。原文比较的是 **GEMM Swizzle 与 GEMM ASWT**；本实验比较的是 **未修改的 FA3 与 FA3 ASWT**，两组基线不同。

迁移的关键是只拆相互独立的输出行。不同 query token 或 query 头可以独立算出完整的 O 和 LSE；若沿同一行的 `Sk` 拆分，各片的 softmax 最大值和归一化和需要重新合并，那是另一种算法。本改动保留每行原有的完整 KV 循环，不引入 FlashDecode（FD）及 combine。

## 实现边界

- 在一个 batch 内，把相互独立的 `(query 序列块, query 头块)` 排成网格；按 4 行宽的条带蛇形访问，最后一条吸收不足 4 行的余数。同一 KV 头的多个 query 块有机会在相近时间访问相同 K/V。单行或单列不做蛇形重排。
- 完整轮不拆。若尾轮任务数不为零且不超过核数一半，在不增加轮数的前提下拆成 2 或 4 份。Prefill 按 32 个 query token 对齐拆；单 token GQA decode 则拆同一个 KV 头内合并的 query 头，粒度为 8 个头（8 个 FP32 LSE 是 32 字节）。不拆 KV 序列、不改变 KV 遍历方向、不生成部分 softmax、无需 combine。短尾切出的空片不做访存。
- 仅限 `B=1`、dense BSND、无 mask、非 FD 的 FP16/BF16 forward，且 KV 长度上界至少 1024、`Dv` 是 16 的倍数、LSE 切分满足对齐。保留原来的 tile 上限和 Host/AICPU metadata ABI；不改变 paged/TND/BNSD、causal/SWA、FP8、backward 或 FA4。
- 对 D192，原本一个核合并同一 KV 头的 16 个 query 头；候选将它分给两个核，各算 8 个头。核数增加的代价是二者都要读完整 K/V，Cube 对小 M 的对齐也不会随有效行数等比缩小。**不能由 4→8 个任务推断 2× 加速或 L2 命中率上升。**

| 改动位置 | 接手时重点看什么 |
|---|---|
| `setup.py` | 仅 950 v3 构建识别 `FLASH_ATTN_950_FA3_ASWT=1`，传入 `-DFA950_ENABLE_ASWT=1`。不是运行时开关。 |
| `csrc/ascend950/flash_attn_npu_3/fwd_aswt.h` | `MakeAswtPlan` 从任务数、核数和对齐粒度选 `parts=1/2/4`；`AswtPlan::Map` 把分核槽位还原为原块和片号；`AswtBoundary` 计算片的半开区间。 |
| `csrc/ascend950/flash_attn_npu_3/fwd_kernel.cpp` | `useAswt` 检查格式、mask、FD、长度和对齐；AIC/AIV 使用同一槽位映射。子片更新 query 序列或头的起点及长度，使 Q/O/LSE 使用对应偏移，空片跳过。 |
| `tests/host/test_fa950_aswt.cpp` | 直接包含生产调度头文件，检查覆盖、不重叠、轮数不增加及图例；不模拟 Cube 计算或设备访存。 |

## 验证场景

下表是生产调度函数在 **假设 32 个 Cube 核** 时的 Host 输出，不是设备计数。两个不改变分核的形状也是回归对照。GEMM 维度按 QK 单头解释为 `(Sq,Sk,Dq)`；`K=1024/2048` 超出 FA3 接口上限，保留原 `M,N` 和总宽度，映射为多头 D512。各头独立 softmax，不能把映射当成原始 GEMM，也不能对比两者的微秒数。

| 来源 `(M,N,K)` | FA3 `(Sq,Sk,Hq,Hkv,D)` | 原任务 → 有效子任务 | 说明 |
|---|---|---:|---|
| 指定 D192 decode | `1,65535,64,4,192` | 4 → 8 | GQA 头切分 |
| `1024,1024,1024` | `1024,1024,2,2,512` | 16 → 32 | 多头映射 |
| `2048,2048,256` | `2048,2048,1,1,256` | 32 → 32 | 单头 QK 原尺寸；整轮，不拆 |
| `2208,2048,512` | `2208,2048,1,1,512` | 18 → 18 | 单头 QK 原尺寸；超过半轮，不拆 |
| `2208,2048,1024` | `2208,2048,2,2,512` | 36 → 45 | 多头映射；48 个槽位中 3 个为空 |
| `1024,2368,512` | `1024,2368,1,1,512` | 8 → 32 | 单头 QK 原尺寸 |
| `1024,2368,1024` | `1024,2368,2,2,512` | 16 → 32 | 多头映射 |
| `1024,2368,2048` | `1024,2368,4,4,512` | 32 → 32 | 多头映射；整轮，只重排 |

## 本次结果（2026-11-19）

- A 为干净的 `f3ac020243b508ca7d3ac2e1035fa299c292cbb5`，B 为同一版本加本补丁。CATLASS 均为 `769cd40a8716b28650b6bebb08db4834eea4462f`。两套 950 v3 扩展均由 CANN 9.2.0-beta.2 编译并链接。A/B 二进制 SHA-256 分别为 `0334931da49eccdf9cbf465c6e63ab3e3e29f855e7d11e69956b640ce7e89653`、`efbca9852c2b72711b5a288b864432526e9537780f5730d66c08f8b1b556d1e3`。
- Host C++/UBSan：1,296 组调度均覆盖全部输出且不重叠，另核对了图中 13×19 蛇形的拐点、20 核尾轮 9→18，以及表中 8 个输入。8 个场景均生成了有限的 CPU `O` 和 `LSE`；这些不是 NPU 精度验证。
- 该主机 PCI 可见加速卡，但没有 `/dev/davinci*`，`npu-smi info` 报 `dcmi module initialize failed. ret is -8005`，`torch.npu.device_count()` 为 0。指定原始 D192 脚本在设备初始化阶段失败；原始 GEMM 参考程序也未通过 ACL 设备初始化。**接受的 NPU 样本为 0；没有 A/B 耗时、加速比、L2 命中率或设备精度结论。** 构建、摘要、设备探测、CPU 与失败日志保存在此次 checkout 的同级目录 `../aswt-validation-f3ac020/`。

## 1. 构建和设备预检

从本 checkout 的根目录执行。需要匹配的 torch/torch_npu 环境、CANN 环境和一张空闲 950；`baseline` 必须是相同 revision 的未修改 checkout，两者均需初始化固定版本 CATLASS 子模块。

```bash
source /usr/local/Ascend/cann/set_env.sh
export PATH=/opt/miniconda3/envs/hyh/bin:$PATH
set -eo pipefail
export FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_JOBS=10
baseline=$(realpath ../aswt-validation-f3ac020/baseline)
candidate=$PWD
git -C "$baseline" rev-parse HEAD; git -C "$candidate" rev-parse HEAD
git -C "$baseline" submodule status csrc/catlass; git -C "$candidate" submodule status csrc/catlass
(cd "$baseline" && FLASH_ATTN_950_FA3_ASWT=0 python setup.py build_ext --force --inplace)
FLASH_ATTN_950_FA3_ASWT=1 python setup.py build_ext --force --inplace
card=0  # 物理卡号，后续命令使用同一张卡
export ASCEND_RT_VISIBLE_DEVICES="$card" PYTHONNOUSERSITE=1
npu-smi info -t proc-mem -i "$card"
python -c 'import torch, torch_npu; assert torch.npu.device_count() == 1; name=torch.npu.get_device_name(0); print(name); assert "Ascend950" in name'
command -v msprof
```

换机时，`baseline` 可指向任一**干净**的上述 FA3 基线 checkout；`candidate` 指向同一基线加本改动的独立 checkout。分别运行 `git submodule update --init --recursive` 准备依赖。调整 CANN 与 Python 环境路径；历史环境为 Python 3.12、torch `2.12.0+cu130`、torch_npu `2.12.0.post2`。板卡不可见、类型不匹配或已有其他任务时先停，不要改用 CPU 计时。要回退，直接加载 A；或关闭构建开关并强制重编 B，再用新 Python 进程加载。

## 2. 正确性：先验调度，再验 O 和 LSE

**无卡可做：** 从仓库根目录执行。不要传 `-DNDEBUG`，测试使用 C++ `assert`。

```bash
host_test=$(mktemp -d)/test_fa950_aswt
g++ -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=undefined -fno-sanitize-recover=undefined \
    -Icsrc/ascend950/flash_attn_npu_3 tests/host/test_fa950_aswt.cpp -o "$host_test"
"$host_test"
```

预期首行 `1296 schedules: complete, disjoint output coverage`，随后打印上表 8 例的 `base,slots,nonempty`。该测试通过只能说明每个输出位置在调度模型中恰好被覆盖一次。

**真机必做：** 以下仅跑数值，不采性能；若直接执行第 3 节，其入口已包含这些检查。沿用第 1 节的环境、`baseline` 和 `candidate`。

```bash
runner="$candidate/benchmarks/profile_950_aswt.py"
python "$runner" --list
for checkout in "$baseline" "$candidate"; do
    while IFS=: read -r shape rest; do
        python "$runner" --checkout "$checkout" --case "$shape" --device 0 --check-only || exit 1
    done < <(python "$runner" --list)
done
```

每例固定种子生成 FP16 输入，调用真实 950 FA3 `flash_attn_func(..., num_splits=1, return_attn_probs=True)`。核对 O/LSE 形状，用仓内 `ref_flash_attention_pair` 和 `assert_fa_close` 分别判精度。两版 × 8 例应全部输出 `O and LSE passed` 并正常退出；D192 原性能脚本自带的 finite 检查不能替代这一步。该矩阵不验 BF16、反向、FD 或其他回退路径，不能据此外推。

## 3. 性能：一条命令采集 A/B/B/A

`benchmarks/run_950_aswt_ab.sh` 是总入口；`benchmarks/profile_950_aswt.py` 提供七个映射场景、数值检查及 trace 解析；`benchmarks/profile_950_decode_d192.py` 是指定 D192 原脚本的逐字节副本（SHA-256 `0b5d844f11b42b4ac5eba23bdfbf4082e657a6336c7b1da55a540570ff2d95e8`）。在原机器优先直接传入原文件；换机可传入随附副本。使用 Bash，输出目录必须尚不存在、父目录已存在。

```bash
original=/home/jianshuqiang/repo/JY_Perf/Jy_ATK_TEST/flash-attention-npu/benchmarks/profile_950_decode_d192.py
if [[ ! -f "$original" ]]; then original="$candidate/benchmarks/profile_950_decode_d192.py"; fi
out="$candidate/../aswt-rerun-$(date +%Y%m%d-%H%M%S)"
bash "$candidate/benchmarks/run_950_aswt_ab.sh" "$baseline" "$candidate" "$original" "$out" "$card"
cat "$out/comparison.csv"
```

- **采什么：** 同一物理卡（进程内逻辑卡 0），逐例 A1/B1/B2/A2，各 10 个窗口。D192 直接运行原脚本 `--fa 3 --num-splits 1 --flush-l2`；七个映射场景沿用相同的 256 MiB 缓冲区刷缓存、同步和 ArgMax 分界策略。测的是该刷缓存条件下的耗时，不代表所有线上缓存状态。
- **怎么算：** 从 `op_summary_*.csv` 按 `Task Start Time(us)` 排序，相邻 ArgMax 之间取唯一名字含 `FAInfer` 的主核任务（如 `FAInferDn`）的 `Task Duration(us)`。只含这一项，不加 `ArgMax`、刷缓存、元数据、Host 或其他 kernel。`baseline_us` 为 A1+A2 共 20 个值的中位数，`candidate_us` 同理；`speedup = baseline_us / candidate_us`，大于 1 表示 B 快。不是端到端时延，也不是 L2 命中率。
- **留下什么：** `identity.txt` 是两版 `.so` 及原脚本哈希；`*-check-{A,B}.log` 留数值结果；每个 `<case>-<轮次>/` 保留 `msprof.log`、设备进程快照、原始 profiler 导出和 `summary.json`。全矩阵通过后才发布 `comparison.csv`。
- **何时拒收：** 数值失败、A/B 库相同、加载哈希不符、采样观察到其他 PID、设备查询失败、异常退出、导出不完整、ArgMax 数不是 11 或某窗口目标任务数不等于 1，均不计入比较。保留失败目录，修复后使用新目录重跑。逐例看 A1/A2、B1/B2 的波动；若优化差值与轮间噪声相当，报告“未证实收益”。`PipeUtilization` 和时延都不能证明 L2 命中率提高，要归因需另采平台支持的 L2 计数器。

## 附：原尺寸 GEMM 检查（不是 FA3 性能验收）

若要在**原始 GEMM** 上检查全部七个三元组，而非运行注意力映射，可用固定 CATLASS 子模块的例 46。它采用 FP16 输入、FP32 输出、文档的 L1/L0 tile 和 ASWT，但含 Fixpipe 优化，**不是**文档的 Swizzle/ASWT 单变量对照，也不用于归因 FA3：

```bash
(cd csrc/catlass && bash scripts/build.sh 46_ascend950_matmul_fixpipe_opti -DCATLASS_ARCH=3510)
set -eo pipefail
for shape in $(python benchmarks/profile_950_aswt.py --list | sed -n 's/^\([0-9][0-9x]*\):.*/\1/p'); do
    IFS=x read -r m n k <<< "$shape"
    csrc/catlass/output/bin/46_ascend950_matmul_fixpipe_opti "$m" "$n" "$k" 0 | grep 'Compare success\.'
done
```
