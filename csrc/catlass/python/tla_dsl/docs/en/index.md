# TLA DSL Docs

This documentation set is a lightweight MkDocs site for the TLA DSL
(`python/tla_dsl`). Prefer the MkDocs site for reading; checked-in Markdown is
the source.

Chinese is the default locale. This English tree currently covers the pages
that have been translated.

## Document map

| Doc | Scope |
|-----|--------|
| [Kernel API Reference](kernel_api_reference.md) | Kernel-side Core APIs (`tla.copy`, `tla.mmad`, vector ops, sync, …). |
| [DSL Syntax Constraints](dsl_python_syntax_guide.md) | What Python is legal inside `@tla.kernel`. |

English Kernel API Markdown is **generated** (`python tools/generate_api_reference.py`);
the Chinese Kernel API page is **hand-maintained** and should be synced when the
English page changes.

## Local Preview

From `python/tla_dsl`:

```bash
python3 -m mkdocs serve
```

To build static HTML:

```bash
python3 -m mkdocs build
```

Then open `site/index.html` in a browser.
