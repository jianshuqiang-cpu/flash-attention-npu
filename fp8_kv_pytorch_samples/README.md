# FP8(E4M3) KV 分页注意力 — PyTorch 调用样例（Ascend 950 / FA3）

对应 PR：<https://github.com/MinghuasLab/flash-attention-npu/pull/466>

本目录给出一份可直接运行的 PyTorch 样例，方便其他同事在自己机器上同步验证**精度**与**性能**。
样例本身只有两个函数调用面：`flash_attn_npu_3.flash_attn_with_kvcache(...)`（唯一的 fp8 入口）
和可选的 bf16 基线（同样的调用、只是 K/V 用 bf16）。

## 1. 环境

| 项 | 要求 |
|---|---|
| 硬件 | Ascend 950（`Ascend950PR`），建议独占一张卡 |
| CANN | 9.1.0（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`） |
| Python | 3.12 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |

## 2. 编译安装

```bash
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu && git fetch origin pull/466/head && git checkout FETCH_HEAD
git submodule update --init --recursive          # 需要 csrc/catlass
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_JOBS=2 \
  python setup.py build_ext --inplace            # 产物：flash_attn_npu_3_950*.so
```

> 共享机上 `FLASH_ATTN_BUILD_JOBS` 建议给 2：默认 13 路并行编译曾把 bisheng 挤爆。
> 编译约 10 分钟。装好后**新开进程**再跑样例。

## 3. 接口约束（不满足直接报错）

- **Q 必须是 `bfloat16`**；fp16 的 Q 走不了混存路径。
- **K/V 必须是 `torch.float8_e4m3fn`，且必须是分页缓存**：
  `(num_blocks, 128, num_kv_heads, 192)`。dense 布局的 fp8 KV 不支持
  （会报 `mixed BF16/FP8 KV currently requires paged cache, page size 128 and Dqk=Dv=192`）。
- `Dqk = Dv = 192`，`page_size = 128`。
- **`k_descale` / `v_descale`**：FP32、同设备、contiguous，形状 `(num_blocks, 128, num_kv_heads)`；
  每个元素缩放「一个物理 token × 一个 KV 头」（per-token-per-head，最细的常规粒度）。
- 不支持 append-KV（边写边算）。
- `num_splits` 建议传 0，让 host 选 tiling（混存路径强制 host tiling；
  正因如此 `scheduler_metadata` 不要传）。

核心调用就这一段：

```python
out, lse = flash_attn_with_kvcache(
    q,                                  # (total_tokens, heads_q, 192) bfloat16
    k_cache, v_cache,                   # (num_blocks, 128, heads_kv, 192) float8_e4m3fn
    cache_seqlens=seqlens,              # (batch,) int32，每条序列实际 KV 长度
    page_table=page_table,              # (batch, pages) int32
    cu_seqlens_q=cu_seqlens_q,          # (batch+1,) int32
    max_seqlen_q=sq,
    k_descale=k_descale, v_descale=v_descale,   # (num_blocks, 128, heads_kv) fp32
    causal=True, window_size=(511, 0),  # step5 FA3 的 SWA 配置；全注意力传 (-1, -1)
    num_splits=0, return_softmax_lse=True,
)
```

## 4. 运行

```bash
export PYTHONPATH=/path/to/flash-attention-npu
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> python fp8_kv_attention_demo.py acc
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> python fp8_kv_attention_demo.py perf
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> python fp8_kv_attention_demo.py all
```

## 5. 预期结果（Ascend950PR / CANN 9.1.0 实测）

### 精度（`acc` 模式，sq=128、kv=1024、16:1 GQA、因果+滑窗 512）

```
fp8  输出 vs bf16 输出（原始 KV）  : 3.715e-02   <- 量化误差
fp8  输出 vs CPU 参考（量化后 KV） : 2.133e-03   <- 实现误差
bf16 输出 vs CPU 参考（未量化 KV） : 2.130e-03   <- bf16 自身误差
```

- **量化误差 ≈ 3.7%**：完全来自 KV 被量化成 e4m3（每 token 每头一个 scale），任何实现都一样。
- **实现误差 ≈ 0.2%**：kernel 与 CPU 高精度参考（同一份量化 KV）的差异，说明实现本身没问题。
- 两者独立：换 bf16 KV 也还是 0.2% 的实现误差。

### 性能（`perf` 模式，交替 A/B，6 轮中位数，同一份逻辑 KV）

| 场景 | bf16 (ms) | fp8 (ms) | fp8/bf16 |
|---|---|---|---|
| prefill q4096/kv8192 SWA512 | 0.373 | 0.366 | 0.98× |
| prefill q4096/kv8192 full | 1.550 | 1.389 | **0.90×** |
| decode 12×4/kv8192 SWA512 | 0.202 | 0.199 | 0.98× |
| decode 12×4/kv131072 SWA512 | 0.208 | 0.207 | 1.00× |
| decode 12×4/kv131072 **full** | 1.475 | 0.890 | **0.60×（快 1.7×）** |

结论：
- **全注意力长上下文 decode 明显更快**（KV 字节减半，已到显存带宽的 ~90%）；
- **全注意力 prefill 快 ~10%**；
- **滑窗（step5 生产形态）端到端持平**：device 只占 0.02–0.19 ms，剩下约 0.17 ms 是每次调用的
  host tiling，量化的带宽优势被淹没了。

> 绝对耗时会随卡况/占用浮动，**请只比同一张卡上的 fp8/bf16 相对值**。

## 6. 常见报错

| 报错 | 原因 |
|---|---|
| `mixed BF16/FP8 KV currently requires paged cache, page size 128 and Dqk=Dv=192` | K/V 用了 dense 布局，或 head_dim ≠ 192、page_size ≠ 128 |
| `key and value must both be rank 4` | K/V 不是 `(num_blocks, 128, heads_kv, 192)` 的分页形状 |
| `Tensors must have dtype ...` | Q 不是 bf16，或 K/V 不是 `float8_e4m3fn` |
| descale 相关报错 | `k_descale`/`v_descale` 不是 FP32 / 形状不是 `(num_blocks,128,heads_kv)` / 不连续 |
| 结果全 0 或异常 | `cache_seqlens` 与 `page_table` 的行数/页数不匹配，或 `cu_seqlens_q` 不等于 `arange(batch+1)*sq` |
