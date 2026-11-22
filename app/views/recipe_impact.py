"""Deliverable 5 - recipe-level impact of healthier swaps across cuisines.

Embeds outputs/impact_dashboard.html, built by src/impact_analysis.py (or src/impact_dashboard.py).
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from ui import banner, embed_html  # noqa: E402

banner(compact=True, subtitle="Recipe impact - what healthier swaps do to 48,145 real recipes.")
st.caption("Deliverable 5 · FlavorSwap's healthier suggestions applied to validated Food.com recipes: change in "
           "calories, fat, saturated fat, sugars and sodium per serving and the share of aroma compounds kept, "
           "compared across cuisines. Use the scenario switch and hover any mark for details.")
embed_html(ROOT / "outputs" / "impact_dashboard.html", "Recipe impact dashboard", "python src/impact_dashboard.py")
