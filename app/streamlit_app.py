"""FlavorSwap web app - entry point with the page menu.

Pages (app/views/):
  Find swaps      Deliverable 4: ranked, explained substitutes (start page; keeps ?ingredient=...&mode=... links)
  Ingredient map  Deliverable 2: interactive embedding map
  Recipe impact   Deliverable 5: impact of healthier swaps on real recipes, by cuisine

Run from the project root:   streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="FlavorSwap", page_icon=":material/swap_horiz:", layout="wide",
                   initial_sidebar_state="collapsed")

page = st.navigation([
    st.Page("views/find_swaps.py", title="Find swaps", icon=":material/swap_horiz:", default=True),
    st.Page("views/ingredient_map.py", title="Ingredient map", icon=":material/scatter_plot:", url_path="map"),
    st.Page("views/recipe_impact.py", title="Recipe impact", icon=":material/insights:", url_path="impact"),
], position="top")
page.run()
