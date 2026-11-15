# Ascend950 FA3 forward 迁入 FA4

日期：2026-11-15。实现目录：`csrc/ascend950/flash_attn_npu_4`。

## 范围与接口

将当前工作区 FA3 的 forward 资源规划、搬运、mask、输出和 FlashDecode 实现迁入 FA4。FA3 参考库 SHA256 为 `f9b310b778f16657fbd67ed97486d1742893419dfd406ef915496bf16bef1edc`；迁移没有修改 FA3 源码或库。没有重新加入用户已放弃的长 Q GQA/MQA 尾块合并。

FA4 保留现有 `flash_attn_func`、`flash_attn_varlen_func`、`get_scheduler_metadata`、`flash_attn_combine` 接口及其返回约定。补充 Ascend950 的 `flash_attn_with_kvcache`，签名对应 FA3 支持的 forward 子集，实际调用 `flash_attn_npu_4_950.fwd`。不经过 FA3 库。

本次支持普通 FP16/BF16 的 `Dq=Dk=Dv=512`，以及已有的 D≤256 路径。D512 不支持 FP8、非对称 Dq/Dv、qv、KV append、rotary、cache 索引重排、attention_chunk、pack_gqa 或 backward。native Python入口对qv明确拒绝。已有 FA4 FP8 D128 路径保留。910 后端不在本次范围内。

## 核心迁移

| 部件 | 实现 | FA4 适配 |
|---|---|---|
| 资源规划 | `fwd_resource_plan.h`、`fwd_resource.hpp` | 保持 Cube/AIV 各自分配；FA4 另需256B reduction scratch |
| Host/metadata tiling | `tiling.cpp`、`fwd_tiling_policy.h`、`fa_metadata.aicpu` | 普通类型使用同一 Q tile/resource policy；FP8维持原Q128策略 |
| QK/PV 搬运 | `qk_matmul.hpp`、`qk_matmul_DN.hpp`、`pv_matmul.hpp` | 单 head strided Q、paged K/V 独立页/token/head stride |
| Softmax | `online_softmax.hpp` | D512位置 causal/SWA mask；保留FA4 threshold=4 baseline freeze |
| O/LSE | `rescale_o.hpp`、`init_outputs.hpp` | D512四片消费、整块GO归一化、独立LSE scratch、空行初始化 |
| FD combine | `fd_combine.hpp` | D512容量、标准task几何；保留FA4 runtime allowFlashDecode |
| Python/Host | `mha_fwd.cpp`、`flash_attn_npu_interface_950.py` | exact D512校验、dense物理KV extent、合法非连续paged输入和TND适配 |

### D512 的局部存储与 GM 访问

Q/KV tile 均为128，QK完整计算D512。L1为480KiB：Q 128KiB、K 128KiB、三槽P 96KiB、V 128KiB。每个AIV的共享UB计划为242.25KiB，另加FA4的256B归约scratch，总计242.5KiB，小于248KiB容量。

Score保留两槽，P UB只有一槽。PV每次把128列结果送入同一LO槽，AIV通过四轮ready/free逐片累加到完整512列FP32 GO。普通FA的Score/P/GO不经GM中转；FD使用原有partial O/LSE workspace及combine。没有增加为D拆分而反复读取Q或Score/P中间结果的GM路径。

### FA4 算法与运行时 ABI

继续使用 `FAInferStaticTilingData` 和 `FAInferRuntimeData` 的静态/运行时分离语义。所有forward路径省去独立runtime GM tensor及其AICPU写入、事件和kernel读取。普通FP16/BF16沿用FA3的基本标量传参：scale/softcap、numBlocks、maxNumBlocks，以及将page size与FD permission合并的blockConfig；只有SWA传递16B窗口值。FP8保留已有的按值runtime参数。kernel入口恢复完整运行时数据，metadata字节布局保持不变，metadata仍为只读调度数据，不同调用各自持有runtime值。无需每次forward合并或修改共享metadata。

FA4普通M-major Softmax在完整64行AIV块上对新max先减4，再与旧baseline取max，控制baseline更新；部分行块使用标准在线max，匹配其直接rescale路径。DN保留原来的baseline规则。固定阈值4直接编译在VF中，不为每次Softmax多传一个scalar参数。普通FA在完整64行AIV块上使用 `DmAllOne` 判断是否可以跳过乘法；较短块直接执行rescale。FlashDecode使用编译期 `CheckDm=false` 的直接rescale版本，去掉整个Scalar判定路径；普通FA保留条件rescale。两种baseline都计算同一attention语义，使用原精度标准验收。

普通与FP8共用一份 `FAInferRuntimeData`，删除已经不再使用的runtime GM指针。`FAIKernelParams`的64位Host POD布局从296B降为240B；该数值只表示对象布局，不作为device stack spilling或GM流量的实测结论。

## 验证与复现

使用 `/opt/miniconda3/envs/hyh/bin/python`，CANN环境由 `/usr/local/Ascend/ascend-toolkit/set_env.sh` 导入，物理卡7映射到逻辑0。记录实际库路径、SHA、原始profile、正常退出码和设备进程快照。

强制只构建FA4，避免旧header缓存：

```bash
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v4 \
FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_BUILD_JOBS=8 \
FLASH_ATTN_FORCE_REBUILD=TRUE \
/opt/miniconda3/envs/hyh/bin/python setup.py build_ext --inplace
```

功能测试入口：`batchtest/fa3_to_fa4_950_20261115/run_tests.py --stage d512` 与 `--stage legacy`。复用FP64 golden、dtype baseline和原精度阈值，禁止执行backward。

性能入口：

```bash
/opt/miniconda3/envs/hyh/bin/python \
  batchtest/fa3_to_fa4_950_20261115/benchmark.py \
  --dim 512 --output batchtest/fa4_compare_d512
```

`--dim 256`为原JD的30个独立输入；D512另加36个BSND/TND/paged、GQA/MQA、Q129/257/513用例。所有性能调用均为 `num_splits=0`。`torch_npu.profiler` warmup=5、active=5，比较主kernel的5次中位数。Host、metadata、辅助kernel、wall/Event耗时不混入主指标。输出目录必须是新目录。

该入口复用JD的输入、抽样CPU精度校验、parser和计时。原 `JD_compare_fa3_fused.py` 仍只支持FA3及FIA/fusion，不能通过 `--pr-root` 加载FA4。

最终功能为775 passed、188 skipped。性能批量首采为93/96在±5%内，3项超界均为FA4更快。验收对全部28项短decode/FD统一使用同输入相邻配对、每实现50个样本，其余68项保留批量首采；按此协议96/96在±5%内，原始两组数据完整保留于报告，不跨模式平均。配对采集入口为 `paired_perf.py --dim 512 --case decode-fp16 --output <新目录> --build-record build_attempt14.json --calls-per-step 10 --batch-cases`；可重复指定 `--case`。

同步静态审计保留全部候选，完整输出见证据目录的 `SYNC_CANDIDATES.md`、`sync_final_native.json`、`flow_final_native.json`。正则扫描不能证明所有模板/宏/跨文件生命周期正确，不将功能或性能通过视为完整同步证明。

FA4原生支持的metadata + 显式split路径继续保留，D512新增对应实测；没有沿用FA3中针对该组合的限制。最终功能计数、逐例性能表、异常项和库身份见 `batchtest/fa3_to_fa4_950_20261115/REPORT.md`。
