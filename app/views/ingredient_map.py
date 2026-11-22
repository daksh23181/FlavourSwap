"""Deliverable 2 - interactive ingredient map (Word2Vec embeddings projected with UMAP and t-SNE).

Embeds outputs/ingredient_map.html, built by src/make_embedding_map.py.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from ui import banner, embed_html  # noqa: E402

banner(compact=True, subtitle="Ingredient map - every ingredient placed by the recipes it appears in.")
st.caption("Deliverable 2 · Word2Vec embeddings of 3,584 ingredients learned from Food.com recipes, projected "
           "to 2-D with UMAP or t-SNE. Ingredients used in the same kinds of dishes sit close together. Search "
           "or click an ingredient to see its 10 nearest neighbours; switch the projection or highlight a food "
           "group at the top of the map.")
embed_html(ROOT / "outputs" / "ingredient_map.html", "Interactive ingredient map",
           "python src/make_embedding_map.py")
