#!/usr/bin/env python3
"""基于「26年新版课件参考模板」生成约 10 页 flash-attn-npu 技术分享 PPT。

内容不删减，按主题合并到更少页；字体/配色跟随模板（鸿蒙黑体、标题红 C00000）。
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "26年新版课件参考模板.pptx"
OUT = ROOT / "flash-attn-npu-tech-share.pptx"
OUT_FALLBACK = ROOT / "flash-attn-npu-tech-share-from-template.pptx"
MD_OUT = ROOT / "flash-attn-npu-tech-share.md"

FONT = "鸿蒙黑体"
RED = RGBColor(0xC0, 0x00, 0x00)
BLACK = RGBColor(0x00, 0x00, 0x00)
GRAY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF5, 0xF5, 0xF5)
LINE = RGBColor(0xDD, 0xDD, 0xDD)


def delete_slide(prs: Presentation, index: int) -> None:
    sldIdLst = prs.slides._sldIdLst
    sldId = sldIdLst[index]
    rId = sldId.get(qn("r:id"))
    prs.part.drop_rel(rId)
    sldIdLst.remove(sldId)


def duplicate_slide(prs: Presentation, index: int):
    """Duplicate a slide including picture relationships."""
    source = prs.slides[index]
    blank_layout = source.slide_layout
    dest = prs.slides.add_slide(blank_layout)

    # Remove default shapes from new slide (placeholders etc.)
    for shp in list(dest.shapes):
        sp = shp._element
        sp.getparent().remove(sp)

    # Copy relationships (images) from source
    image_rid_map = {}
    for rel in source.part.rels.values():
        if "image" in rel.reltype:
            new_rel = dest.part.relate_to(rel.target_part, rel.reltype)
            image_rid_map[rel.rId] = new_rel

    # Deep-copy shape tree children except extLst
    spTree = dest.shapes._spTree
    for child in source.shapes._spTree:
        tag = child.tag
        if tag.endswith("}nvGrpSpPr") or tag.endswith("}grpSpPr"):
            continue
        if tag.endswith("}extLst"):
            continue
        new_child = copy.deepcopy(child)
        # Remap image rIds in blip elements
        for blip in new_child.iter(qn("a:blip")):
            embed = blip.get(qn("r:embed"))
            if embed in image_rid_map:
                blip.set(qn("r:embed"), image_rid_map[embed])
        spTree.insert_element_before(new_child, "p:extLst")

    return dest


def set_runs_text(shape, text: str, *, size=None, bold=None, color=None, font=FONT):
    """Replace shape text keeping first paragraph run style as base."""
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.name = font
    if size is not None:
        run.font.size = size
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    return p


def set_multiline(shape, lines: list[str], *, size=Pt(14), bold=False, color=BLACK, font=FONT, space_after=Pt(6)):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = space_after
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = size
        run.font.bold = bold
        run.font.color.rgb = color


def find_shape(slide, name: str):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    return None


def find_shapes_by_prefix(slide, prefix: str):
    return [sh for sh in slide.shapes if sh.name.startswith(prefix)]


def clear_extra_shapes(slide, keep_names: set[str]):
    for sh in list(slide.shapes):
        if sh.name in keep_names:
            continue
        # keep slide number placeholders
        if "Placeholder" in sh.name or "占位符" in sh.name:
            continue
        if sh.has_text_frame and sh.text_frame.text.strip() == "https://gitcode.com/cann":
            continue
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and ("图片" in sh.name or "Picture" in sh.name or "logo" in sh.name.lower()):
            # keep footer logos roughly at bottom
            if sh.top > Emu(5_500_000):
                continue
        el = sh._element
        el.getparent().remove(el)


def add_textbox(slide, left, top, width, height, lines, *, size=Pt(14), bold=False, color=BLACK):
    box = slide.shapes.add_textbox(left, top, width, height)
    set_multiline(box, lines, size=size, bold=bold, color=color)
    return box


def add_table(slide, rows, left, top, width, height, font_size=Pt(11)):
    n_rows, n_cols = len(rows), len(rows[0])
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.name = FONT
            run.font.size = font_size
            run.font.bold = r == 0
            run.font.color.rgb = RED if r == 0 else BLACK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFD, 0xEC, 0xEC) if r == 0 else (WHITE if r % 2 else LIGHT_BG)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return table_shape


def set_toc_parts(slide, parts: list[str]):
    """Update TOC group rectangles Part 1..N."""
    for sh in slide.shapes:
        if sh.shape_type != MSO_SHAPE_TYPE.GROUP:
            continue
        rects = [sp for sp in sh.shapes if sp.has_text_frame]
        rects = sorted(rects, key=lambda x: x.top)
        for i, sp in enumerate(rects):
            if i < len(parts):
                set_runs_text(sp, parts[i], size=Pt(18), bold=True, color=WHITE)
            else:
                set_runs_text(sp, "", size=Pt(18), bold=True, color=WHITE)


def rebuild_clean():
    """Clean rebuild: copy template, keep cover/goals/toc/content/summary/thankyou, expand content."""
    import tempfile
    import os

    fd, work = tempfile.mkstemp(suffix=".pptx")
    os.close(fd)
    shutil.copy2(TEMPLATE, work)
    out_path = Path(work)
    prs = Presentation(str(out_path))

    # Delete style slide
    delete_slide(prs, 1)

    # Indices: 0 cover, 1 goals, 2 tocA, 3 contentA, 4 sumA, 5 tocB, 6 contentB, 7 sumB,
    #          8 tocC, 9 contentC, 10 sumC, 11 tocD, 12 contentD/总结?, 13 thank you
    # We want ~10 slides:
    # 0 cover
    # 1 goals -> 背景定位价值
    # 2 toc
    # 3 usage1 (from contentA)
    # 4 usage2 (dup content)
    # 5 architecture (dup)
    # 6 design (dup)
    # 7 capability (dup)
    # 8 migration (dup)
    # 9 summary (from sum or 总结 slide)
    # 10 thank you

    # Delete tocB.. before thank you, keep one content and one summary as masters, then dup.
    # After style delete: n=14
    # Keep 0,1,2,3,4 and last; delete 5..n-2
    while len(prs.slides) > 6:
        delete_slide(prs, 5)

    # Now: 0 cover, 1 goals, 2 toc, 3 content, 4 summary, 5 thankyou
    content_idx = 3
    # Duplicate content slide 5 times -> insert before thank you
    # python-pptx adds at end, so duplicate then reorder sldIdLst

    for _ in range(5):
        duplicate_slide(prs, content_idx)

    # Current order: 0c 1g 2toc 3content 4sum 5thanks 6..10 new contents
    # Reorder to: 0c 1g 2toc 3c0 4c1 5c2 6c3 7c4 8c5 9sum 10thanks
    # Move summary before thanks, contents after toc.

    def move_slide(prs, old_index, new_index):
        sldIdLst = list(prs.slides._sldIdLst)
        el = sldIdLst[old_index]
        prs.slides._sldIdLst.remove(el)
        prs.slides._sldIdLst.insert(new_index, el)

    # After dups: [0,1,2,3,4,5,6,7,8,9,10] where 5=thanks, 6-10=contents
    # Desired: [0,1,2, 3,6,7,8,9,10, 4, 5]
    # Easier: delete summary and thanks temporarily? Let's just assign by index after sorting.

    # Move thanks to end explicitly; move summary to before thanks
    # Current indices after 5 dups from content(3):
    # 0 cover, 1 goals, 2 toc, 3 content0, 4 summary, 5 thankyou, 6 c1, 7 c2, 8 c3, 9 c4, 10 c5

    # Move 6..10 to after 3 (positions 4..)
    # Moving carefully from back
    # Target: 0,1,2,3,6,7,8,9,10,4,5
    order = [0, 1, 2, 3, 6, 7, 8, 9, 10, 4, 5]
    sldIdLst = prs.slides._sldIdLst
    elems = [sldIdLst[i] for i in order]
    for el in list(sldIdLst):
        sldIdLst.remove(el)
    for el in elems:
        sldIdLst.append(el)

    # Final indices:
    # 0 cover, 1 goals, 2 toc, 3 usage1, 4 usage2, 5 arch, 6 design, 7 capability, 8 migration, 9 summary, 10 thankyou

    # --- Cover ---
    cover = prs.slides[0]
    if find_shape(cover, "文本框 1"):
        set_multiline(
            find_shape(cover, "文本框 1"),
            ["flash-attention-npu", "生态友好的昇腾注意力算子"],
            size=Pt(30),
            bold=True,
            color=BLACK,
            space_after=Pt(10),
        )
    if find_shape(cover, "文本框 2"):
        set_multiline(
            find_shape(cover, "文本框 2"),
            ["对内技术分享 · 可移植的用法与设计", "基于 CANN / CATLASS · Ascend 910 / 950"],
            size=Pt(14),
            color=GRAY,
            space_after=Pt(6),
        )

    # --- Goals / value ---
    goals = prs.slides[1]
    if find_shape(goals, "副标题 1"):
        set_runs_text(find_shape(goals, "副标题 1"), "背景、定位与价值", size=Pt(28), bold=True, color=RED)
    for name, text in [
        ("TextBox 4", "迁移痛点"),
        ("TextBox 19", "项目定位"),
        ("TextBox 34", "一句话价值"),
    ]:
        sh = find_shape(goals, name)
        if sh:
            set_runs_text(sh, text, size=Pt(16), bold=True, color=RED)
    for name, text in [
        ("TextBox 6", "业界栈围绕 flash_attn* 习惯组织代码"),
        ("TextBox 7", "换硬件时若接口重学，模型改动面大"),
        ("TextBox 9", "痛点：缺熟悉的调用表面"),
        ("TextBox 10", "不是「有没有注意力内核」"),
        ("TextBox 12", "把适配成本从「重写调用」"),
        ("TextBox 13", "降为「选型 + 验证」"),
        ("TextBox 15", "口径：不说「完全一样」"),
        ("TextBox 16", "讲共通设计语言与可移植"),
        ("TextBox 21", "参照主流 FlashAttention 核心设计"),
        ("TextBox 22", "基于 CATLASS / CANN 面向昇腾实现"),
        ("TextBox 24", "提供一致调用习惯，便于模型迁移"),
        ("TextBox 25", "并为后续 NPU 优化留演进空间"),
        ("TextBox 42", "生态友好：熟悉的 API 表面"),
        ("TextBox 43", "可移植：改 import + 选代际包"),
        ("TextBox 45", "可演进：910/950 分后端"),
        ("TextBox 46", "v2 / v3 / v4 分代覆盖同一套名字"),
    ]:
        sh = find_shape(goals, name)
        if sh:
            set_runs_text(sh, text, size=Pt(11), bold=False, color=BLACK)

    # --- TOC ---
    set_toc_parts(
        prs.slides[2],
        [
            "Part 1  用法共通",
            "Part 2  架构与设计共通",
            "Part 3  能力边界与审计",
            "Part 4  迁移实践与总结",
        ],
    )

    def prep_content(slide, title: str, subtitle: str):
        # Keep footer logo + gitcode + slide number; clear body placeholder text box 8; set title
        t = find_shape(slide, "副标题 1")
        if t:
            set_runs_text(t, title, size=Pt(28), bold=True, color=RED)
        sub = find_shape(slide, "文本框 8")
        if sub:
            set_runs_text(sub, subtitle, size=Pt(14), bold=False, color=GRAY)
        # Remove any previously added content shapes (not in original template set)
        original = {"副标题 1", "文本框 8", "文本框 16", "图片 7", "Slide Number Placeholder 1", "灯片编号占位符 5"}
        for sh in list(slide.shapes):
            if sh.name in original:
                continue
            if "Placeholder" in sh.name or "占位符" in sh.name:
                continue
            if sh.has_text_frame and "gitcode.com/cann" in sh.text_frame.text:
                continue
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and sh.top > Emu(5_500_000):
                continue
            # remove leftover tables/text from prior runs if regenerating
            if sh.name.startswith("Content_") or sh.name.startswith("Table"):
                sh._element.getparent().remove(sh._element)

    # helper to add named textbox
    def body_box(slide, name, left, top, width, height, lines, **kwargs):
        box = add_textbox(slide, left, top, width, height, lines, **kwargs)
        box.name = name
        return box

    EM = Emu
    L = EM(729306)
    W = EM(10720572)

    # --- 3 Usage 1: entries + migrate ---
    s = prs.slides[3]
    prep_content(s, "用法共通（一）：入口与最小迁移", "同一套名字，不同包名；迁移主成本在选型与验证")
    rows = [
        ["主流常见入口", "npu (v2)", "npu_3", "npu_4"],
        ["flash_attn_func", "✅", "✅", "✅ (910)"],
        ["flash_attn_varlen_func", "✅", "✅", "✅"],
        ["flash_attn_with_kvcache", "✅", "✅", "常经 varlen+page_table"],
        ["flash_attn_combine", "—", "✅", "✅"],
        ["get_scheduler_metadata", "✅", "✅", "✅ (910)"],
        ["packed 系列", "✅ 导出", "部分未导出", "—"],
    ]
    tbl = add_table(s, rows, L, EM(1550000), W, EM(2500000), font_size=Pt(11))
    tbl.name = "Content_table1"
    body_box(
        s,
        "Content_code",
        L,
        EM(4200000),
        W,
        EM(1600000),
        [
            "# 主流： from flash_attn import flash_attn_func",
            "# 昇腾： from flash_attn_npu import flash_attn_func   # 或 flash_attn_npu_3 / _4",
            "out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))",
            "安装：pip install flash-attn-npu --no-build-isolation   （包名不是 flash-attn）",
            "说明：调用点形状尽量不动；先锁定 GPU 侧代际，再选对应 flash_attn_npu*",
        ],
        size=Pt(12),
        color=BLACK,
    )

    # --- 4 Usage 2: params + generation ---
    s = prs.slides[4]
    prep_content(s, "用法共通（二）：参数语义与分代选型", "选代际，不选另一套 API 语言")
    rows = [
        ["类别", "共通习惯"],
        ["布局", "稠密 BSND (B,S,H,D)；变长 TND + cu_seqlens"],
        ["掩码", "causal 右下角对齐；window_size=(L,R)，-1=无限"],
        ["头结构", "MQA/GQA：nheads % nheads_k == 0"],
        ["训练旋钮", "dropout(偏v2)、deterministic、softcap、alibi(偏v2)"],
        ["推理旋钮", "kvcache；block_table(v2) / page_table(v3/v4)"],
        ["统计量", "softmax_lse；return_attn_probs / return_lse"],
    ]
    tbl = add_table(s, rows, L, EM(1500000), EM(5200000), EM(3200000), font_size=Pt(11))
    tbl.name = "Content_table_params"
    rows2 = [
        ["代际包", "贴近", "适合场景", "提示"],
        ["flash_attn_npu", "FA2", "训练、dropout/ALiBi、packed", "仅 910"],
        ["flash_attn_npu_3", "FA3", "metadata/combine/paged", "950 以前向为主"],
        ["flash_attn_npu_4", "FA4", "统一 varlen/dense、LSE", "部分 FA4 扩展未实现"],
    ]
    tbl2 = add_table(s, rows2, EM(6700000), EM(1500000), EM(4800000), EM(2500000), font_size=Pt(11))
    tbl2.name = "Content_table_gen"
    body_box(
        s,
        "Content_note",
        EM(6700000),
        EM(4200000),
        EM(4800000),
        EM(1200000),
        [
            "选型顺序：",
            "① 确认 GPU 侧 FA 代际",
            "② 选对应 flash_attn_npu*",
            "③ 确认 910 / 950 能力边界",
        ],
        size=Pt(12),
    )

    # --- 5 Architecture ---
    s = prs.slides[5]
    prep_content(s, "架构共通：三层栈与多后端分发", "用户只碰最上层；一个生态、多后端")
    body_box(
        s,
        "Content_stack",
        L,
        EM(1550000),
        W,
        EM(2000000),
        [
            "【三层栈】",
            "用户应用 / 框架层",
            "    ↓ import",
            "flash_attn_npu / _3 / _4   Python 包（torch.autograd / custom_op）",
            "    ↓",
            "C++ 扩展（bisheng / Ascend）",
            "    ↓",
            "CATLASS + CANN（AscendC 算子 + AICPU）",
        ],
        size=Pt(13),
    )
    body_box(
        s,
        "Content_dispatch",
        L,
        EM(3700000),
        W,
        EM(2000000),
        [
            "【包与设备分发】",
            "flash_attn_npu/     → 910 · FA2 表面",
            "flash_attn_npu_3/   → 运行时按设备名分发 910 / 950（interface + interface_950）",
            "flash_attn_npu_4/   → 同上",
            "构建：FLASH_ATTN_BUILD_VERSION=v2|v3|v4 ，FLASH_ATTN_BUILD_NPU=910|950",
            "运行：torch_npu.npu.get_device_name() 选择实现",
            "可移植不只 GPU→NPU，也包括 910↔950 —— 入口名保持稳定",
        ],
        size=Pt(12),
    )

    # --- 6 Design ---
    s = prs.slides[6]
    prep_content(s, "设计共通：算法抽象与内核流水线", "同一抽象层级：分块注意力，而非物化完整 S×S")
    body_box(
        s,
        "Content_algo",
        L,
        EM(1500000),
        EM(5200000),
        EM(3800000),
        [
            "【算法抽象】",
            "· 按块加载 K/V，在线累计 softmax 统计量",
            "· 前向产出 out 与 softmax_lse；反向消费 LSE",
            "· 数学目标不变：softmax(QKᵀ·scale)·V",
            "· 短 Q / 长 KV 可走 split-KV",
            "· 再 flash_attn_combine 合并 partial",
            "",
            "设计语言可复用：",
            "分块 · online softmax · LSE · combine",
        ],
        size=Pt(13),
    )
    body_box(
        s,
        "Content_pipe",
        EM(6700000),
        EM(1500000),
        EM(4800000),
        EM(3800000),
        [
            "【内核流水线】",
            "qk_matmul",
            "    → online_softmax(/quant)",
            "    → pv_matmul",
            "    → rescale_o",
            "",
            "工程组织习惯：",
            "· 模板化 tiling / autogen kernel 矩阵",
            "· 编译期开关组织因果/窗口路径",
            "· FlashDecode / num_splits + combine",
            "实现落在 Ascend 工具链，不是另一套公式",
        ],
        size=Pt(13),
    )

    # --- 7 Capability ---
    s = prs.slides[7]
    prep_content(s, "能力边界（诚实）与对齐审计", "可移植 ≠ 无差异；签名能传 ≠ 内核会用")
    rows = [
        ["特性", "v2", "v3", "v4", "备注"],
        ["Causal / SWA / GQA", "✅", "✅", "✅", "主路径"],
        ["varlen / paged KV", "✅", "✅", "✅", "页表命名有代际差异"],
        ["Backward", "✅", "✅", "✅910", "950：v3/v4 以前向为主"],
        ["Dropout / ALiBi", "✅", "—", "—", "训练偏 v2"],
        ["Softcap", "✅", "✅", "✅", "950 部分路径不支持"],
        ["metadata / FD / combine", "部分", "✅", "✅", "推理增强"],
        ["RoPE in kvcache", "—", "—", "—", "明确未支持"],
        ["FA4 score_mod 等", "—", "—", "未实现", "签名可接受"],
    ]
    tbl = add_table(s, rows, L, EM(1450000), W, EM(3000000), font_size=Pt(10))
    tbl.name = "Content_matrix"
    body_box(
        s,
        "Content_audit",
        L,
        EM(4600000),
        W,
        EM(1200000),
        [
            "PARAM_USAGE.md：对照主流实现指定 commit 做参数级审计",
            "标记：✅ 接受且生效 · ⚠️ 有限制 · ❌ 拒绝 · 👻 接受但无数值效应 · ➖ 仅包装",
            "价值：迁移评审可逐项核对期望，避免「接口能调通」掩盖「参数未生效」",
        ],
        size=Pt(12),
    )

    # --- 8 Migration ---
    s = prs.slides[8]
    prep_content(s, "迁移实践：生态钩子与 Checklist", "对主流框架集成路径友好；按标准动作落地")
    rows = [
        ["钩子", "意图"],
        ["torch.compile 测试", "自定义算子进入编译路径"],
        ["NPU Graph 测试", "推理图捕获场景"],
        ["非连续输入用例", "对齐社区边角行为"],
        ["精度 / 确定性反向", "训练侧可回归"],
    ]
    tbl = add_table(s, rows, L, EM(1500000), EM(5200000), EM(2500000), font_size=Pt(12))
    tbl.name = "Content_hooks"
    body_box(
        s,
        "Content_check",
        EM(6700000),
        EM(1500000),
        EM(4800000),
        EM(3800000),
        [
            "【迁移 Checklist】",
            "① 锁定 GPU 侧 FA 代际与关键入口",
            "② 选型 flash_attn_npu / _3 / _4，确认 910/950",
            "③ 对照 README 特性矩阵 + PARAM_USAGE.md",
            "④ 改写 import（必要时 block_table↔page_table）",
            "⑤ 回归精度、边角形状、图模式/compile",
            "",
            "有缺口（如 RoPE）提前拆到 host 侧",
        ],
        size=Pt(13),
    )

    # --- 9 Summary ---
    s = prs.slides[9]
    title = find_shape(s, "副标题 1")
    if title:
        set_runs_text(title, "总结与讨论", size=Pt(28), bold=True, color=RED)
    body = find_shape(s, "内容占位符 2")
    if body:
        set_multiline(
            body,
            [
                "用法：熟悉的入口与旋钮 → 迁移主成本下沉到选型与验证",
                "架构：分代包 + 三层栈 + 多后端分发",
                "设计：分块 / online softmax / LSE / combine",
                "边界：用特性矩阵与 PARAM_USAGE 诚实管理差异",
                "",
                "讨论①：优先保训练路径（v2）还是推理路径（v3/v4）？",
                "讨论②：950 训练 / FP8 等 roadmap 如何排期？",
                "讨论③：框架层是否需要统一适配层封装代际差异？",
                "",
                "结束语：生态友好，是让昇腾注意力说社区听得懂的语言；",
                "可移植，是把硬件差异关在后端与文档里。",
            ],
            size=Pt(14),
            color=BLACK,
            space_after=Pt(6),
        )

    # thank you page left as-is (index 10)

    import tempfile
    import os

    fd, tmp = tempfile.mkstemp(suffix=".pptx")
    os.close(fd)
    try:
        prs.save(tmp)
        for dest in (OUT, out_path, OUT_FALLBACK, ROOT / "flash-attn-npu-tech-share-final.pptx"):
            try:
                shutil.copy2(tmp, dest)
                print(f"Wrote {dest} ({len(prs.slides)} slides)")
                return len(prs.slides), dest
            except PermissionError:
                continue
        # last resort: leave in temp
        final = ROOT / "flash-attn-npu-tech-share-final.pptx"
        raise PermissionError(f"Saved temp only: {tmp}; could not copy into docs/")
    finally:
        pass


def write_md():
    md = """# flash-attention-npu：生态友好的昇腾注意力算子

> 对内技术分享 · **约 10 页**（模板：`26年新版课件参考模板.pptx`）  
> 配套 PPT：`flash-attn-npu-tech-share.pptx`  
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
| PPT | `docs/flash-attn-npu-tech-share.pptx` |
| 生成脚本 | `docs/generate_tech_share_pptx.py` |
"""
    MD_OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    n, path = rebuild_clean()
    write_md()
    print("done, slides=", n, "path=", path)
