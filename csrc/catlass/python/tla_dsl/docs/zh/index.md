# TLA DSL 文档

本目录是 TLA DSL（`python/tla_dsl`）的 MkDocs 文档源。阅读请优先使用 MkDocs 站点；仓库内 Markdown 为源文件。

文档按语言分目录：`docs/zh/` 为中文（默认），`docs/en/` 为英文。

## 文档索引

| 文档 | 范围 |
|-----|--------|
| [Kernel API 参考](kernel_api_reference.md) | Kernel 侧 Core API（`tla.copy`、`tla.mmad`、Vector 运算、同步等）。 |
| [Host Tensor 接入](framework_integration.md) | Host 侧 `from_dlpack` / `make_fake_tensor`，供 `tla.compile` / 启动使用。 |
| [静态与动态 Layout](dsl_dynamic_layout.md) | 静态 / 动态 layout；在 Host tensor 上标记动态；在 kernel 中编程。 |
| [DSL 语法约束](dsl_python_syntax_guide.md) | `@tla.kernel` 内允许的 Python 写法。 |
| [环境变量](environment_variables.md) | `CATLASS_DSL_*` 及相关环境变量。 |
| [构建 API 文档](dev_guide/02_api_docs.md) | 如何从英文 docstring 重新生成 `docs/en/kernel_api_reference.md`。 |

英文 Kernel API 由脚本生成（`python tools/generate_api_reference.py`）；中文 Kernel API 为手工维护，英文稿变更后请同步更新。

## 本地预览

在 `python/tla_dsl` 下执行：

```bash
python3 -m mkdocs serve
```

构建静态 HTML：

```bash
python3 -m mkdocs build
```

然后在浏览器中打开 `site/index.html`。
