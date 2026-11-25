"""Small MkDocs hooks used by the CATLASS documentation site."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from mkdocs.plugins import event_priority
from mkdocs.structure.files import File


_MARKDOWN_LINK = re.compile(
    r"(?P<prefix>\]\(\s*<?)"
    r"(?P<target>(?:\.\./)+[^)\s>]+)"
    r"(?P<suffix>>?(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\))"
)

_PAGE_NAV_TITLES = {
    "zh": {"README.md": "主页", "FAQ.md": "FAQ"},
    "en": {"README.md": "Home", "FAQ.md": "FAQ"},
}

_TOP_LEVEL_ORDER = {
    "README.md": 0,
    "1_Practice": 10,
    "2_Design": 20,
    "3_API": 30,
    "0x_new_versions": 40,
    "FAQ.md": 100,
}

_ASSETS_PATH = Path("assets")


def _asset_sources(assets_root: Path):
    """Yield centralized assets and their public paths."""
    for source in assets_root.rglob("*"):
        if source.is_file():
            yield _ASSETS_PATH / source.relative_to(assets_root), source


@event_priority(-200)
def on_files(files, config, **kwargs):
    """Expose centralized assets at the URL of the current language build."""
    assets_root = Path(config.docs_dir) / _ASSETS_PATH
    if not assets_root.is_dir():
        return files

    i18n = config.plugins["i18n"]
    locale = i18n.current_language
    default_locale = i18n.default_language

    # The i18n plugin filters root-level assets when document fallback is
    # disabled. Replace its partial view with virtual, locale-aware files.
    for file in list(files):
        if not file.abs_src_path:
            continue

        try:
            Path(file.abs_src_path).resolve().relative_to(assets_root.resolve())
        except ValueError:
            continue

        files.remove(file)

    output_prefix = Path() if locale == default_locale else Path(locale)
    for public_path, source in _asset_sources(assets_root):
        virtual_source = Path(locale) / public_path
        destination = output_prefix / public_path
        asset = File(
            virtual_source.as_posix(),
            src_dir=None,
            dest_dir=config.site_dir,
            use_directory_urls=config.use_directory_urls,
            dest_uri=destination.as_posix(),
        )
        asset.abs_src_path = str(source)
        asset.generated_by = "mkdocs_hooks"
        files.append(asset)

    return files


def _source_url(target: str, page, config) -> str:
    """Map a link that escapes docs_dir to the corresponding repository URL."""
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return target

    project_root = Path(config.config_file_path).parent.resolve()
    docs_root = Path(config.docs_dir).resolve()
    source_file = Path(page.file.abs_src_path)
    destination = (source_file.parent / unquote(parts.path)).resolve()

    try:
        destination.relative_to(docs_root)
        return target
    except ValueError:
        pass

    try:
        relative_path = destination.relative_to(project_root)
    except ValueError:
        return target

    if not destination.exists():
        return target

    branch = config.extra.get("source_branch", "master")
    view = "tree" if destination.is_dir() else "blob"
    repository_url = config.repo_url.rstrip("/")
    repository_path = quote(relative_path.as_posix(), safe="/")
    mapped_path = f"{repository_url}/{view}/{branch}/{repository_path}"
    return urlunsplit(("", "", mapped_path, parts.query, parts.fragment))


def on_page_markdown(markdown, page, config, **kwargs):
    """Rewrite repository-relative source links before MkDocs validates them."""

    def replace(match: re.Match[str]) -> str:
        target = _source_url(match.group("target"), page, config)
        return f"{match.group('prefix')}{target}{match.group('suffix')}"

    return _MARKDOWN_LINK.sub(replace, markdown)


def _nav_items(items):
    """Yield every item in a MkDocs navigation tree."""
    for item in items:
        yield item
        children = getattr(item, "children", None)
        if children:
            yield from _nav_items(children)


def _nav_source(item) -> str:
    """Return a navigation item's first source path, without its locale prefix."""
    current = item
    while getattr(current, "file", None) is None:
        children = getattr(current, "children", None)
        if not children:
            return ""
        current = children[0]

    parts = Path(current.file.src_uri).parts
    if parts and parts[0] in _PAGE_NAV_TITLES:
        parts = parts[1:]
    return "/".join(parts)


def _top_level_sort_key(item):
    source = _nav_source(item)
    first_part = source.split("/", 1)[0]
    return (_TOP_LEVEL_ORDER.get(first_part, 90), source)


def on_page_context(context, page, config, nav, **kwargs):
    """Polish top-level tabs while retaining fully automatic page discovery."""
    locale = getattr(page.file, "locale", "zh")
    titles = _PAGE_NAV_TITLES.get(locale, {})

    for item in _nav_items(nav.items):
        file = getattr(item, "file", None)
        if file and Path(file.src_uri).name in titles:
            item.title = titles[Path(file.src_uri).name]

    nav.items.sort(key=_top_level_sort_key)
    return context
