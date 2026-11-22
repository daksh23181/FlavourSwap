"""Figures for Deliverable 5: per-cuisine change of calories, fat, saturated fat, sugars, sodium and
aroma retention after healthier swaps.

Form: small multiples (one panel per outcome), one series per panel (mean change per cuisine) drawn
as a dot with a 95% bootstrap CI whisker and a zero reference line. One colour (categorical slot 1),
recessive hairline grid, the same cuisine order in every panel, "All recipes" first.

  write_static_figures -> reports/figures/impact_by_cuisine_{single,all,reviewer}.png (matplotlib)

The interactive presentation dashboard is built by src/impact_dashboard.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANELS = [("Calories", "Calories %", "% per serving"), ("Total fat", "Total fat %", "% per serving"),
          ("Saturated fat", "Saturated fat %", "% per serving"), ("Sugars", "Sugars %", "% per serving"),
          ("Sodium", "Sodium %", "% per serving"), ("Aroma retained", "Aroma retained", "% of compounds kept")]
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "series": "#2a78d6"}


def _order(t: pd.DataFrame) -> list[str]:
    rest = t[t.cuisine != "All recipes"].sort_values("Calories %")
    return ["All recipes"] + rest.cuisine.tolist()


def _values(t: pd.DataFrame, key: str):
    if key == "Aroma retained":
        m = 100 * t[key].to_numpy()
        ci = np.array([[100 * a, 100 * b] for a, b in t[f"{key} CI"]])
    else:
        m = t[key].to_numpy()
        ci = np.array([list(x) for x in t[f"{key} CI"]])
    return m, ci


def write_static_figures(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
                         "axes.edgecolor": LIGHT["axis"], "axes.labelcolor": LIGHT["ink2"],
                         "xtick.color": LIGHT["ink2"], "ytick.color": LIGHT["ink"],
                         "text.color": LIGHT["ink"]})
    names = {"best single swap": "single", "all swaps": "all", "reviewer-reported swap": "reviewer"}
    for scen, t in tables.items():
        order = _order(t)
        t = t.set_index("cuisine").loc[order].reset_index()
        fig, axes = plt.subplots(2, 3, figsize=(13, 0.32 * len(order) * 2 + 2.2), sharey=True,
                                 facecolor=LIGHT["surface"])
        y = np.arange(len(order))[::-1]
        for ax, (title, key, unit) in zip(axes.ravel(), PANELS):
            m, ci = _values(t, key)
            ax.set_facecolor(LIGHT["surface"])
            ax.grid(axis="x", color=LIGHT["grid"], linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ref = 100 if key == "Aroma retained" else 0
            ax.axvline(ref, color=LIGHT["axis"], linewidth=1)
            ax.hlines(y, ci[:, 0], ci[:, 1], color=LIGHT["series"], linewidth=2)
            ax.scatter(m, y, s=36, color=LIGHT["series"], edgecolor=LIGHT["surface"], linewidth=1.5, zorder=3)
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=LIGHT["ink"])
            ax.set_xlabel(unit)
            ax.tick_params(axis="y", length=0)
        axes[0, 0].set_yticks(y)
        axes[0, 0].set_yticklabels(order)
        axes[1, 0].set_yticks(y)
        axes[1, 0].set_yticklabels(order)
        for ax in axes.ravel():
            for lab in ax.get_yticklabels():
                if lab.get_text() == "All recipes":
                    lab.set_fontweight("bold")
        fig.suptitle(f"Change per serving after healthier swaps ({scen}), mean and 95% CI",
                     x=0.01, ha="left", fontsize=13, fontweight="bold", color=LIGHT["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"impact_by_cuisine_{names[scen]}.png", dpi=160, facecolor=LIGHT["surface"])
        plt.close(fig)
