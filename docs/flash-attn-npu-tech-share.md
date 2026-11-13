# flash-attention-npu：生态友好的昇腾注意力算子

> 对内技术分享 · **约 10 页**（模板：`26年新版课件参考模板.pptx`）  
> 配套 PPT：`flash-attn-npu-tech-share-final.pptx`（基于 `26年新版课件参考模板.pptx`；若旧文件被占用请用本文件）  
> 原则：**不删减要点**，按主题合并到更少页；表述强调共通设计语言，不宣称「完全一样」。

## 表述口径

| 避免 | 改用 |
|------|------|
| 和 FA 完全一样 / 100% 兼容 / drop-in 无差异 | 沿用主流入口名与参数习惯；迁移通常只需调整 import 与选型版本 |
| 我们就是 FA 的 NPU 版 | 参照主流 FlashAttention 的核心设计，面向昇腾实现 |
| 隐瞒能力缺口 | 用特性矩阵诚实标出代际 / 平台差异 |

**叙事**：用法共通 → 架构共通 → 设计共通 → 能力边界 → 落地迁移。

---

## 第 1 页 · 封面

**标题**：flash-attention-npu：生态友好的昇腾注意力算子  
**副标题**：对内技术分享 · 可移植的用法与设计  
**脚注**：基于 CANN / CATLASS · Ascend 910 / 950

**讲稿**：今天不讲「我们复制了谁」，而讲「为什么主流 FlashAttention 用户能较快迁到 NPU」。

---

## 第 2 页 · 背景、定位与价值

### 迁移痛点
- 业界栈围绕 `flash_attn*` 习惯组织代码
- 换硬件时若接口重学，模型改动面大
- 痛点是缺熟悉的调用表面，不是有没有注意力内核
- 目标：适配成本从「重写」降为「选型 + 验证」
- 口径：不说「完全一样」，讲共通设计语言与可移植

### 项目定位
- 参照主流 FlashAttention 核心设计
- 基于 CATLASS / CANN 面向昇腾实现
- 提供一致调用习惯，便于模型迁移
- 并为后续 NPU 优化留演进空间

### 一句话价值
- **生态友好**：熟悉的 API 表面
- **可移植**：改 import + 选代际包
- **可演进**：910/950 分后端；v2/v3/v4 分代覆盖同一套名字

**讲稿**：三列一起讲完「为什么做、做成什么样、用户带走什么」。

---

## 第 3 页 · 目录

1. 用法共通  
2. 架构与设计共通  
3. 能力边界与审计  
4. 迁移实践与总结  

---

## 第 4 页 · 用法共通（一）：入口与最小迁移

### 入口对照（完整）

| 主流常见入口 | npu (v2) | npu_3 | npu_4 |
|--------------|----------|-------|-------|
| `flash_attn_func` | ✅ | ✅ | ✅ (910) |
| `flash_attn_varlen_func` | ✅ | ✅ | ✅ |
| `flash_attn_with_kvcache` | ✅ | ✅ | 常经 varlen+page_table |
| `flash_attn_combine` | — | ✅ | ✅ |
| `get_scheduler_metadata` | ✅ | ✅ | ✅ (910) |
| packed 系列 | ✅ 导出 | 部分未导出 | — |

### 最小迁移

```python
# 主流
from flash_attn import flash_attn_func
# 昇腾
from flash_attn_npu import flash_attn_func  # 或 flash_attn_npu_3 / _4
out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
# pip install flash-attn-npu --no-build-isolation
```

**讲稿**：强调包名是 `flash-attn-npu`；调用点形状尽量不动。

---

## 第 5 页 · 用法共通（二）：参数语义与分代选型

### 参数与语义（完整）

| 类别 | 共通习惯 |
|------|----------|
| 布局 | 稠密 BSND；变长 TND + `cu_seqlens` |
| 掩码 | `causal` 右下角；`window_size=(L,R)`，`-1` 无限 |
| 头结构 | MQA/GQA：`nheads % nheads_k == 0` |
| 训练 | dropout(偏v2)、deterministic、softcap、alibi(偏v2) |
| 推理 | kvcache；`block_table`(v2) / `page_table`(v3/v4) |
| 统计量 | `softmax_lse`；`return_attn_probs` / `return_lse` |

### 分代选型（完整）

| 代际包 | 贴近 | 适合场景 | 提示 |
|--------|------|----------|------|
| `flash_attn_npu` | FA2 | 训练、dropout/ALiBi、packed | 仅 910 |
| `flash_attn_npu_3` | FA3 | metadata/combine/paged | 950 以前向为主 |
| `flash_attn_npu_4` | FA4 | 统一 varlen/dense、LSE | 部分 FA4 扩展未实现 |

**讲稿**：先问 GPU 上用哪一代，再选包；不要跨代硬套参数。

---

## 第 6 页 · 架构共通

### 三层栈
用户应用/框架 → `flash_attn_npu*` Python 包 → C++ 扩展 → CATLASS + CANN

### 包与设备分发
- `flash_attn_npu/`：910 · FA2 表面  
- `flash_attn_npu_3/` / `_4/`：运行时按设备名分发 910 / 950  
- 构建：`FLASH_ATTN_BUILD_VERSION`、`FLASH_ATTN_BUILD_NPU`  
- 运行：`torch_npu.npu.get_device_name()`  
- 可移植也包括 910↔950，入口名稳定  

---

## 第 7 页 · 设计共通

### 算法抽象
- 按块加载 K/V，online softmax  
- 前向 `out` + `softmax_lse`；反向消费 LSE  
- 数学目标：`softmax(QKᵀ·scale)·V`  
- split-KV + `flash_attn_combine`  

### 内核流水线
`qk_matmul → online_softmax(/quant) → pv_matmul → rescale_o`  
工程习惯：autogen、编译期开关、FlashDecode/num_splits  

---

## 第 8 页 · 能力边界与对齐审计

### 特性矩阵（完整）

| 特性 | v2 | v3 | v4 | 备注 |
|------|----|----|----|------|
| Causal / SWA / GQA | ✅ | ✅ | ✅ | 主路径 |
| varlen / paged KV | ✅ | ✅ | ✅ | 页表命名有代际差异 |
| Backward | ✅ | ✅ | ✅910 | 950：v3/v4 以前向为主 |
| Dropout / ALiBi | ✅ | — | — | 训练偏 v2 |
| Softcap | ✅ | ✅ | ✅ | 950 部分路径不支持 |
| metadata / FD / combine | 部分 | ✅ | ✅ | 推理增强 |
| RoPE in kvcache | — | — | — | 明确未支持 |
| FA4 score_mod 等 | — | — | 未实现 | 签名可接受 |

### PARAM_USAGE
对照主流指定 commit；`✅/⚠️/❌/👻/➖`；签名能传 ≠ 内核会用。

---

## 第 9 页 · 迁移实践

### 生态钩子
`torch.compile`、NPU Graph、非连续输入、精度/确定性反向。

### Checklist
1. 锁定 GPU 侧 FA 代际与关键入口  
2. 选型 `flash_attn_npu*`，确认 910/950  
3. 对照 README 矩阵 + `PARAM_USAGE.md`  
4. 改 import（必要时 `block_table`↔`page_table`）  
5. 回归精度、边角、图模式/compile；缺口（如 RoPE）拆 host  

---

## 第 10 页 · 总结与讨论

- 用法 / 架构 / 设计三板斧回顾  
- 边界用矩阵与审计文档管理  
- 讨论：训练 vs 推理优先级；950 训练/FP8；框架统一适配层  

**结束语**：生态友好，是让昇腾注意力说社区听得懂的语言；可移植，是把硬件差异关在后端与文档里。

---

## 第 11 页 · Thank you（模板固定页）

保留课件模板感谢页与社区信息。

---

## 附录 · 文件

| 材料 | 路径 |
|------|------|
| 模板 | `docs/26年新版课件参考模板.pptx` |
| 本大纲 | `docs/flash-attn-npu-tech-share.md` |
| PPT | `docs/flash-attn-npu-tech-share-final.pptx` |
| 生成脚本 | `docs/generate_tech_share_pptx.py` |
