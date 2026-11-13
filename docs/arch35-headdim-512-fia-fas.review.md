# arch35 HeadDim 512 文档源码校验

> 状态：已根据本报告修正[当前文档](arch35-headdim-512-fia-fas.md)。本报告记录修改前的问题，下文的“原文位置”链接指向[修改前备份](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md)。

校验日期：2026-11-05。原文：[arch35-headdim-512-fia-fas.md](arch35-headdim-512-fia-fas.md)。源码：本机 `D:/code/FA/ops-transformer`，提交 `75b6beb460eb32644a6c87fa5f9d76b756d17b4d`，校验时该仓工作区干净。

修改前结论：**旧文不完全符合当前代码。最主要的问题是把当前 Host 不接受的普通 GQA D=512 重构分支，作为 FIA 实际执行路径，并据此建立了错误的 FIA/FAS 内存对照。** FAS 的 768 模板、GM 落点及 Vec2 每次 16 行等结果成立；MLA 主体流程成立，但 P/V 偏移解释相反。

本次为静态源码校验，覆盖原文全部章节，并向前追查 Host 能力门禁、模板选择，向后追查 Cube 分块、Fixpipe 和 Vector 更新。修改前原文的 71 个 Markdown 源码链接涉及 14 个文件，文件均存在、起始行号均未越界；这不代表链接处足以支持其关联结论。未执行 CANN 编译、仿真或 NPU 数值/性能测试。初次校验时尚未修改原文；随后已按用户要求修正文档，算子代码未修改。

## 1. [P1] 普通 FIA GQA D=512 当前不会选中重构 kernel

原文位置：[第 2 节](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L184)、[对照表](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L513)、[小结第 2 项](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L561)。

原文从 Config 表存在 D=512 和 `isReconstructTemp=true` 的 kernel 分支，推导出普通 D=512 使用 GQA 重构路径。但当前 `IsCapableSparseLayoutCheckGqa()` 有明确门禁：

```cpp
if (fiaInfo_->qkHeadDim != 128 && fiaInfo_->mlaMode != MlaMode::ROPE_COMBINE_D128) {
    return false;
}
```

普通 GQA 的 `mlaMode=NO_MLA`、`qkHeadDim=512` 会返回 false。[门禁源码](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp#L141)。`DoTiling()` 在不具备能力时返回 `GRAPH_PARAM_INVALID`，[注册器继续尝试后续模板](../../ops-transformer/attention/common/op_host/fia_tiling_templates_registry.h#L110)，而不是执行该类的 `DoOpTiling()`。[基类执行顺序](../../ops-transformer/attention/common/op_host/fia_tiling_base.h#L58)。

普通非量化 GQA D=512 在满足公共输入检查、其他 tiling 条件时走兜底 `FusedInferAttentionScoreTilingImpl`，该模板明确设置 `isReconstructTemp=false`，继而进入 FAS 共用的 infer kernel。[兜底注册](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp#L2164)、[设置 false](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp#L1484)、[Aligned512 分发](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_arch35_noquant.h#L115)。

**建议修正：** 将第 2 节主体改为当前实际兜底路径；GQA 重构 D=512 代码只能作为“已有模板/分支，但当前 Host 门禁不接受普通 GQA D=512”的单独说明。

## 2. [P1] Host 的 sOuterFactor=64 不等于 Cube S1=64

原文位置：[Host 切分解释](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L186)、[128 KB 的推算](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L261)、[对照表 S1 行](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L515)。

`AdjustSinnerAndSouter()` 设置的是 Host `sOuterFactor_`；`UpdateTilingKeyConfig()` 计算：

```cpp
auto sOuter = sOuterFactor_ * platformInfo_.cvRatio;
```

在本文采用的 `cvRatio=2` 下，64 对应 **Cube S1=128**，不是 64。[换算](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp#L593)、[匹配的 128/128/512/512 Config](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp#L78)。这只是对该 Host 代码的静态推导，受第 1 项门禁限制，不能表示普通 D=512 实际进入重构模板。

实际兜底路径同样默认 `sOuterFactor_=64`、`sInnerFactor_=128`，并乘 `cvRatio` 后选择 Cube S1=128、S2=128。[兜底切块](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp#L361)、[换算](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp#L1329)、[Config](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp#L1363)。

因此，原文不能用“FIA Cube S1=64，而 FAS Cube S1=128”解释普通 D=512 的差异。真正使用 Cube M=64 的本文路径是 MLA，其 Host `sOuterFactor_=32`。[MLA Host](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp#L263)、[MLA Cube static_assert](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h#L103)。

另需区分单 AIC 的 L0C 结果与单 AIV 的 UB 结果：重构 GQA 的 UB 分配含 `mBaseSize / CV_RATIO`，Fixpipe 使用 `dualDstCtl=1`，按 M 拆给两个 AIV。[UB 大小](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h#L166)、[Fixpipe 参数](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h#L538)。

## 3. [P2] 重构 GQA 的 V 搬运不能由缓冲分配注释推导为 128 宽 slab

原文位置：[共同策略](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L62)、[重构 GQA BMM2](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L243)、[K/V 对照行](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L518)。

重构 GQA 确实为 D>256 分配 `s2BaseSize*l1BaseD` 的 V 缓冲；但当前 `IterateBmm2()` 直接调用 `IterateBmm2l0Split()`，后者执行 `CopyValueTile()`，而该函数传入完整 `constInfo.dSizeV`。它没有调用另一个已定义的 `IterateBmm2L1SplitN()`。[分配](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h#L155)、[全宽 CopyValueTile](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h#L340)、[实际入口](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h#L494)、[调用 CopyValueTile](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h#L554)。

`MatmulN<...,128,...>` 表示其内部按 N 拆 L1→L0 计算，不足以证明 GM→L1 的 V 搬运也按 128 切。由于第 1 项门禁排除了普通 D=512，此处记录的是原文与现有分支实现的矛盾，不把假设强行进入该分支后的行为作为已发生的运行错误。

与此不同，FAS 和 FIA 兜底共用 Cube 的 `IterateBmm2()` 在 D/DV 模板超过 256 时，明确调用 `IterateBmm2L1SplitN()`，每次搬 `S2×128`，这条描述有实际调用链支持。[入口](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L827)、[N 循环](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L679)。

## 4. [P2] MLA 的 P/V 偏移解释写反

原文位置：[第 3.5 节末句](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L391)。

原文称 `s2BaseSize * VDsize` 把左矩阵指到 V 区。实际 `MatmulN` 的第一个参数是左矩阵 P，第二个参数是右矩阵 V：

```text
共享槽起始地址                    → V，128×512
共享槽 + s2BaseSize * VDsize      → P，64×128
```

Vector 把 softmax 的 P 写到 `s2BaseSize*dVBaseSize + ...`，Cube 从同一偏移读取左矩阵。[Vector 写 P](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_noquant_mla.h#L408)、[Cube 使用 P/V](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h#L354)。

该尾部空间在 BMM1 阶段存 K_rope，在 BMM2 阶段复用为 P；槽起始的 K 内容作为共享 KV 供 BMM2 读取。[K 和 rope 写入](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h#L245)。

## 5. [P2] DAV_3510 实际调用 GetC2Position，原文引用了另一编译分支

原文位置：[第 4.2 节](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L434)、[对照表 BMM2 行](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L516)、[小结第 3 项](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L562)。

`flash_attention_noquant_block_cube.h` 在 `#if __NPU_ARCH__ == 3510` 下调用 `GetC2Position()`；`GetC2PositionForTrain()` 在对应的 `#else` 中。不能把后者作为本文 DAV_3510 的实际调用路径。[预处理分支](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L181)。

对于本文 FP16 D=DV=512，FAS 的 768 DV 模板不满足 `GetC2Position()` 的 UB 条件，且非 MLA 量化/可选 DN 特例，结果仍为 GM。FIA 普通 GQA 兜底的 512 模板同样返回 GM。[实际 GetC2Position](../../ops-transformer/attention/common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h#L228)。

**建议修正：** 保留“FAS BMM2 写 GM”结果，把源码链改成 DAV_3510 的 `GetC2Position → bmm2OutPos → bmm2Write2Ub=false`。

## 6. [P2] “FAS 占满 L0C，所以必须写 GM”的因果解释不符合实际计算

原文位置：[第 4.2 节容量解释](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L454)。

`128×512×4=256 KiB` 的算术成立，但实际 FAS BMM2 按 N=128 计算，每次 L0C 逻辑结果是 `128×128×4=64 KiB`，然后 Fixpipe 到 GM 中对应列偏移。代码没有同时在 L0C 保留完整 128×512 结果。[N 循环](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L691)、[MatmulFull](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L754)、[逐列块 Fixpipe](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L803)。L0C policy 为单槽 128 KiB 的双缓冲。[初始化](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L453)。

**建议修正：** 直接说明模板策略将此场景的 BMM2 放 GM、Vec2 按行分块处理；若要解释策略的性能动机，需要另有设计或性能证据。不能从“完整输出的理论体积”反推当前 L0C 实际占用。

## 7. [P2] MLA 的 512+64+512 是必要维度条件，不是完整能力门禁

原文位置：[第 3.1 节](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L325)。

除三个维度外，当前 MLA 重构 tiling 还要求非空、非量化、`mlaMode=ROPE_SPLIT_D512`；排除公共前缀、PSE/alibi、Q/KV 左 padding、后量化、sink、指定 D 不等长标志及 tensorlist 等特性；要求 sparse mode 为 0/3/4，mode=0 时不能传 mask；并要求 **N2=1、N1∈[1,128]**。[基本检查](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp#L50)、[特性检查](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp#L70)、[稀疏及 head 数检查](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp#L86)。

**建议修正：** 把小节标题改为“必要维度条件”，并补充上述 Host 门禁，避免把满足维度写成足以保证进入该 kernel。此处函数也没有以 `s1Size==1` 限定能力，单凭“Decode”名称不能推导仅支持单 token。

## 8. 需要补充的表述

1. **MLA 内存公式补字节数和 Q。** [原文 3.3](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L348) 的 432 KiB 结果正确，但表达式漏了 `sizeof(INPUT_T)=2`。完整预算为 `128×576×2×3=432 KiB`，加 `64×576×2=72 KiB` 的 Q，总计 **504 KiB**，在 512 KiB L1 内剩 8 KiB。共享槽中 V 为 128 KiB，P 为 16 KiB，恰好复用 144 KiB 槽的两部分。[共享槽分配](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_mla.h#L158)、[Q 分配](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h#L101)。建议统一写 KiB，避免字节和元素混用。
2. **一次 MatmulK 调用不等于一次 MMAD。** [原文 3.4](arch35-headdim-512-fia-fas.before-fix-20261105-140523.md#L365) 的“一次 MatmulK”成立；内部 `kLoops=ceil(576/128)=5`，分别处理 128/128/128/128/64，累加到 L0C。区别在于 Q/K+rope 已全宽在 L1，不是沿用普通路径的四轮 GM→L1 K 搬运。[MatmulK 循环](../../ops-transformer/attention/common/op_kernel/matmul.h#L684)、[MMAD](../../ops-transformer/attention/common/op_kernel/matmul.h#L751)。
3. **限定共同策略的 dtype/路径。** 原文总范围是“非量化”，但 128 宽 slab、S1=128、Vec2=16 行等主要按 FP16/BF16 描述。共用 Cube 对 FP32 使用不同的 `l1BaseD`，D 模板大于 128 时为 64；MLA 也采用全宽 576 的 L1 KV 槽。[l1BaseD 分档](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L196)。应限定为普通 FP16/BF16 D=DV=512，并单列 MLA；Q 本身仍全宽驻留 L1，不能把开头概括理解为 Q/K/V 都不全宽。
4. **说明训练前向范围。** 本文 FAS 证据是 `flash_attention_score` 前向，不覆盖 `flash_attention_score_grad`，应在范围中写清。
5. **补图和证据。** 原文第 0 节 Mermaid 的 `UB_or_GM` 与 `UB` 未连通，缺少 scores/BMM2 进入 UB 的路径；图中 `UB_248KB` 也未附源码或平台规格证据。当前 Host 是向平台查询 UB 容量，[查询位置](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp#L246)。该值不应与三项已核对的硬编码 Buffer 预算混为同一证据层级。

## 9. 已确认成立的内容

| 原文结论 | 校验结果与适用范围 | 代码依据 |
|---|---|---|
| L1=512 KiB、L0A/L0B=64 KiB、L0C=256 KiB | 引用的初始化值相符 | [L1](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h#L171)、[L0](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h#L169) |
| 普通 D=512 的 BMM1 四轮 K=128 | FP16/BF16 共用 Cube 的 L1 split K 路径成立；重构 GQA 中也有该代码，但 Host 不接受该普通 D=512 场景 | [共用 Cube 入口](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L656)、[K 循环](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h#L1835) |
| 重构 GQA BMM2 写 UB、Vec2 整行处理 | 对这个类的实现描述成立；不能据此声称普通 D=512 实际走它 | [Fixpipe config](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h#L77)、[Vec2](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_noquant_gqa.h#L411) |
| GQA PRELOAD_N=3、当前任务做 C1/Vec1，落后两任务做 C2/Vec2 | 对重构类实现成立 | [ExecuteTask](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h#L522) |
| MLA Cube M=64、S2=128，QK=576、DV=512 | 在当前 MLA Host 门禁内成立 | [唯一 Config](../../ops-transformer/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp#L458)、[Cube 约束](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h#L103) |
| MLA PRELOAD_N=2 | 成立，C2/Vec2 落后一个任务 | [常量](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_mla.h#L44)、[执行](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_mla.h#L493) |
| FAS D=512 对应 ALIGNED_768，FP16 S1=S2=128 | 成立；模板 768 不等于实际计算 D=768 | [D 分档](../../ops-transformer/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_basic.cpp#L53)、[S1/S2](../../ops-transformer/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_basic.cpp#L102) |
| FAS Vec2 每次 16 行，每个 FP32 数据块 32 KiB | Dv=512 时成立；运行时 dBasicBlock 是 Align64(Dv)，与编译模板 768 区分 | [运行时 Dv 对齐](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_kernel_base.h#L477)、[8192/Dv](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base.h#L1570) |
| FAS 四阶段流水及 GM 三缓冲 | 成立 | [流水](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h#L296)、[GM workspace](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_kernel_base.h#L278) |
| flash_attn 算子当前不接受 D=512 | 成立；公共参数检查还给出了比“Config 表没有”更直接的拒绝证据 | [支持组合白名单](../../ops-transformer/attention/flash_attn/op_host/checkers/common_checker.cpp#L240) |

FAS 的 `ProcessVec2DSplit` 在 UB 中按行块更新，但当一个 AIV 需要多个行块时，中间 O 还通过 `vec2ResGm` 保存/重载，不能把 32 KiB UB 块理解为全部 O 一直驻留 UB。[重载](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base.h#L1627)、[非末 S2 时写回](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base.h#L1700)。

## 10. 按当前代码修正后的主对照

以下限定 DAV_3510、FP16/BF16、普通非量化 D=DV=512、无 rope；MLA 列单列其 512+64 特例。输入须先满足各自公共检查和 Host 门禁。表中的 S1/M 是 Cube 模板容量；不足一块时实际有效行数更小，合轴场景还可能表示 G×S1，而非纯 query token 数。

| 项目 | FIA 普通 GQA D=512 当前路径 | FIA MLA 重构 | FAS 前向 D=512 |
|---|---|---|---|
| Host / 分发 | 兜底、isReconstructTemp=false，共用 infer kernel | MLA 特化、isReconstructTemp=true | FAS 基本 tiling、train 前向 kernel |
| Cube S1/M、S2 | 128、128 | 64、128 | 128、128 |
| QK/DV 编译模板 | 512 / 512 | 576 / 512 | 768 档 / 768 档 |
| 运行时 QK/DV | 512 / 512 | (512+64) / 512 | 512 / 512 |
| L1 K/V | 128 宽分块 | 576 宽 KV/P 共享槽 | 128 宽分块 |
| BMM1 | 4 轮 L1 K=128 | 一次 MatmulK 调用，内部 5 轮 L0 K 分块 | 4 轮 L1 K=128 |
| BMM2 | L1SplitN，每次 N=128，写 GM | MatmulN，写 UB | L1SplitN，每次 N=128，写 GM |
| Vec2 | 共用 ProcessVec2DSplit，Dv=512 时每次最多 16 行/AIV | 完整 Dv 的 UB 更新 | ProcessVec2DSplit，每次最多 16 行/AIV |

普通 FIA 的共用 Vector 从 `bmm2Write2Ub=false && splitD=true` 进入 `ProcessVec2DSplit`，[分发位置](../../ops-transformer/attention/flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base.h#L1728)；其 infer Vector 派生类继承这一实现，[继承位置](../../ops-transformer/attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_infer.h#L29)。因此原文“普通 FIA 用 UB、FAS 用 GM”的主对照，应替换为这三列实际路径。

FAS 的 DV 编译模板由 `CalcDVBasicBlock()` 确定；本场景不属于 192/128 的特例，`dVTemplateType=dTemplateType`，因此表中 QK/DV 都是 768 档。[DV 模板赋值](../../ops-transformer/attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.cpp#L2006)。
