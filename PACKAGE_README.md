# FP8(E4M3) KV 混合存储注意力 — 交付包（Ascend 950 / FA3）

本包是 **PR #466**（<https://github.com/MinghuasLab/flash-attention-npu/pull/466>）在
commit `b0e5e56`（"Pipeline paired FP8 KV prefill on Ascend 950 from a single wave"）上的
完整源码快照，外加一套可直接运行的 PyTorch 精度/性能验证样例，用于同事同步复现。

```
flash-attention-npu-fp8-mixed-kv/
├── PACKAGE_README.md              ← 本文件：编译 / 测试 / 验证步骤
├── pr466.diff                     ← PR #466 相对 upstream main (f3ac020) 的 diff
├── pr466_graphmode.diff           ← 含本次"入图"补全的完整 diff（PR + 本包新增改动）
├── fp8_kv_pytorch_samples/        ← 最小可运行样例（精度自检 + 性能对比）
│   ├── README.md
│   └── fp8_kv_attention_demo.py
├── evidence/                      ← 本机实测证据
│   ├── build.log                  ← 一次成功编译的日志（含编译器/版本头）
│   ├── pytest_mixed_kv.log        ← 混存 KV 精度测试：109 passed
│   ├── measured_results.md        ← 精度/性能实测表
│   └── scripts/                   ← 本次测量用的脚本（benchmark / profiler 探针）
└── （源码）csrc/ flash_attn_npu_3/ tests/ benchmarks/ setup.py ...
```

**未包含（为控制体积）**：`.git` 目录、`csrc/catlass` 子模块（约 2 GB）、编译产物
（`build*/`、`*.so`、`dist/`、`__pycache__`）、以及本机的 profiler 输出目录。
因此**首次编译前必须先补 catlass 子模块**，见下文第 2 步。

---

## 1. 环境要求

| 项 | 要求 | 实测环境 |
|---|---|---|
| 硬件 | Ascend 950（`Ascend950PR`），建议独占一张空闲卡 | Ascend950PR_957b |
| CANN | 9.1.0 | `/usr/local/Ascend/cann-9.1.0` |
| Python | 3.12 | 3.12.13 |
| torch / torch_npu | 2.10 系列 | 2.10.0+cpu / 2.10.0.post4 |

> 该 fp8 混存路径**只在 Ascend 950 上实现**（`flash_attn_npu_3` / v3 / DAV_3510）。
> 910 上只有普通 bf16/fp16 路径。

## 2. 编译（约 10 分钟）

```bash
cd flash-attention-npu-fp8-mixed-kv

# 2.1 补 catlass 子模块（本包未收录，约 2 GB；tag 必须为 v2.0.0）
git clone -b v2.0.0 https://gitcode.com/cann/catlass.git csrc/catlass
#   —— 若已有 catlass v2.0.0 本地副本，直接拷贝到 csrc/catlass 亦可

# 2.2 编译（只编译 950 的 v3 扩展）
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_JOBS=2 \
  python setup.py build_ext --inplace
```

产物：仓库根目录 `flash_attn_npu_3_950.cpython-312-x86_64-linux-gnu.so`（约 56 MB）。

注意事项：
- `FLASH_ATTN_BUILD_JOBS` 建议 **2**。默认会开 13 路并行编译，在共享机上曾把 bisheng
  编译器挤到崩溃（现象：`Compilation failed for flash_api.cpp`，但日志里只有 warning）。
- 编译完成后**新开一个 Python 进程**再跑测试（避免旧 .so 被缓存）。

## 3. 测试

### 3.1 精度测试（PR 自带，109 项）

```bash
cd flash-attention-npu-fp8-mixed-kv
export PYTHONPATH=$PWD
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> python -m pytest tests/test_flash_attn_npu_v3_mixed_kv.py -q
# 期望：109 passed, 14 warnings   （本机 16~30 s）
```

覆盖内容：TND/BSND、causal/滑窗、多 batch、FlashDecode split、页表乱序、非 2 幂形状、
非连续 scale、超出 4 个 KV 头的 scale 回退路径等；参考值由 CPU FP32 计算。

可选的长上下文测试（1M context，CPU 参考较慢，默认 skip）：

```bash
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> FA950_RUN_LONG_KV=1 \
  python -m pytest tests/test_flash_attn_npu_v3_mixed_kv_long.py -q
```

### 3.2 精度/性能样例（本包提供，step5 形状）

```bash
export PYTHONPATH=$PWD
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> python fp8_kv_pytorch_samples/fp8_kv_attention_demo.py acc
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> python fp8_kv_pytorch_samples/fp8_kv_attention_demo.py perf
```

本机实测（Ascend950PR / CANN 9.1.0）：

精度（sq=128、kv=1024、16:1 GQA、因果 + 滑窗 512）：

| 对比 | 相对误差 | 含义 |
|---|---|---|
| fp8 输出 vs bf16 输出（原始 KV） | 3.715e-02 | **量化误差**（e4m3 KV，per-token-per-head scale） |
| fp8 输出 vs CPU 参考（量化后 KV） | 2.133e-03 | **实现误差**（kernel vs CPU，同一份量化 KV） |
| bf16 输出 vs CPU 参考（未量化 KV） | 2.130e-03 | bf16 自身误差 |

性能（交替 A/B、6 轮中位数、同一份逻辑 KV）：

| 场景 | bf16 ms | fp8 ms | fp8/bf16 |
|---|---|---|---|
| prefill 4096 query / 8192 KV，SWA512 | 0.373 | 0.366 | 0.98× |
| prefill 4096 / 8192，全注意力 | 1.550 | 1.389 | **0.90×** |
| decode 12×4 / kv8192，SWA512 | 0.202 | 0.199 | 0.98× |
| decode 12×4 / kv131072，SWA512 | 0.208 | 0.207 | 1.00× |
| decode 12×4 / kv131072，全注意力 | 1.475 | 0.890 | **0.60×（快 1.7×）** |

> 绝对耗时随卡况浮动，**只比较同一张卡上的 fp8/bf16 相对值**。

## 4. 接口约束（不满足会直接报错）

- **Q 必须 bfloat16**；fp16 的 Q 不支持该混存路径。
- **K/V 必须 `torch.float8_e4m3fn` 且是分页缓存** `(num_blocks, 128, heads_kv, 192)`；
  dense 布局的 fp8 KV 不支持。
- `head_dim(Dqk=Dv) = 192`，`page_size = 128`。
- `k_descale` / `v_descale`：FP32、同设备、contiguous、`(num_blocks, 128, heads_kv)`，
  per-token-per-head 缩放。
- 不支持 append-KV；`num_splits` 传 0。`scheduler_metadata` 现已支持（见下节）。
- 不传 `scheduler_metadata` 时仍走 host tiling（行为与 PR 一致）。

## 5. 入图（scheduler_metadata / NPUGraph）

PR #466 的混存 KV 路径原本强制 host tiling，因此**不能入图**（host tiling 在 capture 期间不允许）。
本包已补全：

```python
from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata

# 1) 按 shape 预生成 AICPU tiling（kv_dtype 声明 KV 是 fp8，即混合 KV）
md = get_scheduler_metadata(
    batch_size=12, max_seqlen_q=4, max_seqlen_k=page_table.shape[1] * 128,
    num_heads_q=16, num_heads_kv=1, headdim=192,
    cache_seqlens=cache_seqlens, qkv_dtype=torch.bfloat16,
    kv_dtype=torch.float8_e4m3fn, cu_seqlens_q=cu_seqlens_q,
    page_size=128, causal=True, window_size=(511, 0), num_splits=0,
)
# 2)（可选）长上下文全注意力 decode 会启用 FlashDecode，其启动核数/workspace 随
#    cache_seqlens 变化，需要在 **capture 之外** 用 prepare_fd_metadata 做一次 host tiling：
from flash_attn_npu_3 import prepare_fd_metadata
md = prepare_fd_metadata(md, q, k_cache, v_cache, cache_seqlens, page_table=page_table,
                         cu_seqlens_q=cu_seqlens_q, max_seqlen_q=4, causal=True,
                         window_size=(511, 0), return_softmax_lse=True, num_splits=0)
# （滑窗 / prefill 不需要这一步；不调用时 FD 形态会退回普通解码，结果正确但更慢）

# 3) eager 或 graph 内都传 scheduler_metadata（混存 KV 现在接受）
with torch.npu.graph(g):
    out, lse, *_ = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=..., page_table=...,
        cu_seqlens_q=..., max_seqlen_q=4, k_descale=..., v_descale=..., causal=True,
        window_size=(511, 0), num_splits=0, return_softmax_lse=True,
        scheduler_metadata=md)
```

验证结果（`tests/test_flash_attn_npu_v3_mixed_kv_graph.py`，6 passed；含 paired prefill 与 FlashDecode 用例）：
metadata 与 host tiling 位级一致、NPUGraph 捕获+重放与 eager 一致、CPU 参考精度通过。

端到端收益（同一张卡、交替 A/B）：

| 场景 | host tiling | + metadata | 提升 | 位级一致 |
|---|---|---|---|---|
| prefill 4096/8192 SWA512 | 0.365 ms | 0.252 ms | **1.45x** | 是 |
| prefill 4096/8192 全注意力 | 1.334 ms | 1.218 ms | **1.10x** | 是 |
| decode 12x4/8192 SWA512 | 0.192 ms | 0.076 ms | **2.52x** | 是 |
| decode 12x4/131072 SWA512 | 0.198 ms | 0.085 ms | **2.34x** | 是 |
| decode 12x4/32768 全注意力（FD） | 0.355 ms | 0.234 ms | **1.52x** | 是 |
| decode 12x4/131072 全注意力（FD） | 0.816 ms | 0.696 ms | **1.17x** | 是 |

> 全部场景位级一致（max|diff|=0），且 NPUGraph 捕获/重放同样一致（含 FlashDecode 形态）。
> 滑窗（step5 生产形态）是收益场景：去掉每次约 0.17 ms 的 host tiling。
>
> metadata 与 `cache_seqlens` 的关系：
> * **非 FlashDecode 形态**（滑窗、prefill）：tiling 与 KV 长度无关，metadata 可跨 KV 长度复用，
>   实测 2048/4096/8192 与 host tiling 逐位一致 —— 稳态 decode 只需构建一次，无需每步重建；
> * **FlashDecode 形态**（长上下文全注意力 decode）：tiling 含分核/workspace，KV 长度变化时
>   需重建 metadata 并重跑 `prepare_fd_metadata`；
> * metadata 构建本身已不再做设备级同步（decode 形态零同步，prefill 仅当前 stream 同步），
>   详见 `evidence/metadata_sync_removal.md`。


## 5b. 与优化前 bf16 路径的对比

以 **优化前 bf16 路径**（bf16 KV + host tiling，即 PR #466 `b0e5e56` 原始行为）为基线：

| 场景 | bf16 优化前 wall | v7 wall | v7/bf16 | NPUGraph replay | graph/bf16 |
|---|---|---|---|---|---|
| prefill 4096/8192 SWA512 | 0.351 ms | **0.233 ms** | **1.5x** | 0.202 ms | 1.74x |
| prefill 4096/8192 全注意力 | 1.503 ms | **1.194 ms** | **1.26x** | 1.167 ms | 1.29x |
| decode 12x4/8192 SWA512 | 0.179 ms | **0.057 ms** | **3.12x** | 0.033 ms | 5.43x |
| decode 12x4/131072 SWA512 | 0.175 ms | **0.058 ms** | **3.04x** | 0.033 ms | 5.30x |
| decode 12x4/32768 全注意力（FD） | 0.473 ms | **0.234 ms** | **2.02x** | 0.190 ms | 2.49x |
| decode 12x4/131072 全注意力（FD） | 1.337 ms | **0.700 ms** | **1.91x** | 0.664 ms | 2.01x |

- 滑窗形态 kernel 很小（decode 约 0.014 ms），端到端主要剩 launch/同步与调用开销 →
  生产部署建议用 NPUGraph（本包支持，见 5 节），decode 滑窗可到 0.032 ms；
- 全注意力形态 fp8 带宽优势显现（FD decode kernel 约 960 GB/s）；
- bf16 路径无回归（独立 worktree 重编译对比，±3% 噪声内）。
详细数据与复现脚本见 `evidence/vs_bf16_baseline.md`；第二轮 host 开销优化与选核竞态修复见
`evidence/host_overhead_round2.md`；完整对比表（含 kernel 与精度口径）见 `evidence/vs_bf16_v7.md`。

## 5c. 稳态 decode 的 lean 入口（v9）

`flash_attn_with_kvcache` 每次调用约 22 us 花在通用包装/参数转换上，而 SWA decode 的 kernel
只有 ~14 us。为此新增最小开销入口：

```python
from flash_attn_npu_3 import flash_attn_with_kvcache_lean

out, lse = flash_attn_with_kvcache_lean(
    q, k_cache, v_cache, cache_seqlens, page_table,
    cu_seqlens_q=cu_seqlens_q, max_seqlen_q=4,
    scheduler_metadata=md,           # FD 形态先用 prepare_fd_metadata(md, ...)
    k_descale=k_descale, v_descale=v_descale,
    causal=True, window_size=(511, 0), return_softmax_lse=True,
)
```

契约：paged cache、张量连续、传入 metadata、不支持 append-KV / rotary / 显式 num_splits。
与公共 API 逐位一致（有专门用例覆盖，含 NPUGraph 重放）。

| 场景 | bf16 优化前 | 公共 API + metadata | lean | NPUGraph | lean/bf16 | graph/bf16 |
|---|---|---|---|---|---|---|
| prefill 4096/8192 SWA512 | 0.330 ms | 0.217 ms | **0.208 ms** | 0.199 ms | 1.59x | 1.66x |
| prefill 4096/8192 全注意力 | 1.474 ms | 1.184 ms | **1.173 ms** | 1.164 ms | 1.26x | 1.27x |
| decode 12x4/8192 SWA512 | 0.173 ms | 0.045 ms | **0.036 ms** | 0.030 ms | **4.81x** | **5.73x** |
| decode 12x4/131072 SWA512 | 0.171 ms | 0.045 ms | **0.036 ms** | 0.030 ms | **4.82x** | **5.75x** |
| decode 12x4/32768 全注意力（FD） | 0.437 ms | 0.214 ms | **0.205 ms** | 0.190 ms | 2.14x | 2.29x |
| decode 12x4/131072 全注意力（FD） | 1.270 ms | 0.683 ms | **0.688 ms** | 0.682 ms | 1.85x | 1.86x |

decode SWA 的 36 us 已接近 eager 下限（device 下发延迟 16 us + kernel 14 us + sync 5 us），
NPUGraph replay 30 us 即该下限。时间构成与下一步（SWA 的 FD 分核）见
`evidence/decode_host_optimize.md`。

## 6. 已知性能边界（供判断是否适用）

- **长上下文全注意力 decode**：受益最大（KV 字节减半，已到显存带宽 ~90%），约快 1.7×。
- **全注意力 prefill**：约快 10%。
- **滑窗（step5 生产形态，窗口 512）**：端到端持平。device 只需 0.02–0.19 ms，
  剩余约 0.17 ms 是每次调用的 host tiling 开销；量化的带宽优势被淹没。
  若要在该形态继续提速，应优化 host tiling 缓存，而不是 kernel。

## 7. 版本与溯源

| 项 | 值 |
|---|---|
| 上游仓库 | `https://github.com/MinghuasLab/flash-attention-npu` |
| PR | #466 |
| 本包 commit | `b0e5e561785c0f666760efc40d7a8486af1159eb` |
| 对比基线 | `origin/main` = `f3ac020243b508ca7d3ac2e1035fa299c292cbb5` |
| diff | `pr466_graphmode.diff`（`git diff origin/main`，含 PR + 入图补全） |

本次验证环境：Ascend950PR_957b / CANN 9.1.0 / Python 3.12.13 / torch 2.10.0+cpu /
torch_npu 2.10.0.post4；单卡单并发。
