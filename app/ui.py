"""Shared look of the FlavorSwap web app: the brand banner and full-page HTML embeds."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

# Banner styling. Colours are translucent tints of the brand green so they work on the light and the
# dark theme alike; text inherits the theme's text colour.
STYLE = """
<style>
.fs-hero { padding: 30px 34px 28px; border-radius: 22px; margin: 0 0 6px;
  background: linear-gradient(135deg, rgba(15,123,85,0.16) 0%, rgba(15,123,85,0.05) 55%, rgba(15,123,85,0.02) 100%);
  border: 1px solid rgba(15,123,85,0.20); }
.fs-hero.compact { padding: 16px 22px; border-radius: 16px; }
.fs-brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; }
.fs-logo { width: 34px; height: 34px; border-radius: 10px; background: #0f7b55; color: #ffffff;
  display: inline-flex; align-items: center; justify-content: center; font-size: 1.15rem; font-weight: 700; }
.fs-tag { margin-left: auto; font-size: 0.8rem; font-weight: 600; padding: 4px 10px; border-radius: 999px;
  background: rgba(15,123,85,0.12); white-space: nowrap; }
.fs-title { font-size: clamp(1.9rem, 3.4vw, 2.8rem); line-height: 1.12; font-weight: 800;
  letter-spacing: -0.02em; margin: 16px 0 8px; }
.fs-sub { font-size: 1.1rem; opacity: 0.8; max-width: 780px; margin: 0; }
.fs-hero.compact .fs-sub { font-size: 0.95rem; margin: 0 0 0 44px; }
@media (max-width: 640px) { .fs-tag { display: none; } .fs-hero { padding: 22px 20px; }
  .fs-hero.compact .fs-sub { margin-left: 0; margin-top: 6px; } }
</style>
"""


def banner(compact: bool, subtitle: str = "Substitutes that cook like it, taste like it, or are better for you.") -> None:
    """Brand banner: large on the start page, a slim strip with a one-line subtitle elsewhere."""
    st.html(STYLE)
    brand = ('<div class="fs-brand"><span class="fs-logo">⇄</span><span>FlavorSwap</span>'
             '<span class="fs-tag">Computational Gastronomy · Group 27</span></div>')
    if compact:
        st.html(f'<div class="fs-hero compact">{brand}<p class="fs-sub">{html.escape(subtitle)}</p></div>')
    else:
        st.html(f'<div class="fs-hero">{brand}<div class="fs-title">Swap any ingredient. Keep the flavor.</div>'
                '<p class="fs-sub">Search an ingredient and get substitutes that cook like it, taste like it, '
                'or are better for you - ranked and explained with recipe data, flavor chemistry and '
                'nutrition.</p></div>')


@st.cache_data(show_spinner=False)
def _read_html(path: str, mtime: float) -> str:
    """File contents, cached until the file changes (mtime is part of the cache key)."""
    return Path(path).read_text(encoding="utf-8")


def embed_html(path: Path, title: str, rebuild_hint: str) -> None:
    """Show one of the project's self-contained HTML pages full width, auto-sized to its content.

    The page is given the app's current light/dark theme (both project pages honour a
    data-theme attribute on <html>); inside an iframe they would otherwise follow the OS setting.
    Only the project's own generated files are embedded, never user input."""
    if not path.exists():
        st.error(f"`{path.name}` has not been built yet. Run `{rebuild_hint}`.", icon=":material/error:")
        return
    page = _read_html(str(path), path.stat().st_mtime)
    theme = st.context.theme.type
    if theme in ("light", "dark"):
        page = page.replace("<head>", f'<head><script>document.documentElement.dataset.theme = "{theme}";</script>', 1)
    st.iframe(page, height="content", alt=title)
