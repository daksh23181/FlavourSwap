"""End-to-end tests of the Streamlit app (Deliverable 4) with Streamlit's headless AppTest runner.

Run:  python tests/test_app.py      (exits non-zero on failure)
"""

import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
TUNED_FLAV = float(json.loads((ROOT / "models" / "hybrid_config.json").read_text())["weights"]["flav"])
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)
        print("FAIL", msg)


def fresh() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + \
            [s.value for s in at.subheader] + [h.value for h in at.header] + [t.value for t in at.title]
    return "\n".join(str(p) for p in parts)


def no_exception(at: AppTest, step: str) -> None:
    check(len(at.exception) == 0, f"{step}: exception {[e.value for e in at.exception]}")


def main() -> int:
    # 1. landing page
    at = fresh()
    no_exception(at, "landing")
    labels = {m.label for m in at.metric}
    check({"Ingredients", "Recipes learned from", "Aroma compounds", "Known swaps in the top 10"} <= labels,
          f"landing: stat tiles missing ({labels})")
    check(any("How it works" in str(h.value) for h in at.subheader), "landing: 'How it works' section missing")
    check(at.selectbox(key="ingredient").value is None, "landing: an ingredient is pre-selected")

    # 2. pick an ingredient -> source card + 10 suggestion cards (similar mode)
    at.selectbox(key="ingredient").set_value("butter").run()
    no_exception(at, "similar butter")
    body = texts(at)
    check("Similar swaps for butter" in body, "similar: results heading missing")
    go = [b for b in at.button if b.key and b.key.startswith("go-")]
    check(len(go) == 10, f"similar: expected 10 suggestion cards, got {len(go)}")
    check("Margarine" in body, "similar: margarine missing for butter")
    check(len(at.dataframe) == 1, "similar: compare table missing")
    check(body.count("Top pick") == 1, "similar: exactly one 'Top pick' badge expected")   # badges render as markdown
    check(", 4-trans" not in body or "2-trans, 4-trans-decadienal" in body, "similar: compound name split")

    # 3. healthier mode
    at.segmented_control(key="mode").set_value("Healthier").run()
    no_exception(at, "healthier butter")
    body = texts(at)
    check("Healthier swaps for butter" in body, "healthier: heading missing")
    check("Olive oil" in body, "healthier: olive oil missing for butter")

    # 4. navigate via a card button
    first = [b for b in at.button if b.key and b.key.startswith("go-")][0]
    target = first.key[3:]
    first.click().run()
    no_exception(at, "card navigation")
    check(at.selectbox(key="ingredient").value == target, f"card navigation: expected {target!r}")

    # 5-6. Free text ("Large Eggs") and unknown ingredients ("mozarella") cannot be entered through
    #      AppTest (it rejects selectbox values outside the option list); they are checked in a real
    #      browser against the running app instead (see README, "Web app").

    # 7. healthier mode for an ingredient without nutrition data -> clear explanation
    at = fresh()
    at.selectbox(key="ingredient").set_value("cheese").run()
    at.segmented_control(key="mode").set_value("Healthier").run()
    no_exception(at, "no nutrition data")
    check(any("no reliable USDA nutrition data" in i.value for i in at.info), "no-data: explanation missing")

    # 8. quick-pick chip
    at = fresh()
    at.pills(key="quick_pick").set_value("sour cream").run()
    no_exception(at, "quick pick")
    check(at.selectbox(key="ingredient").value == "sour cream", "quick pick: ingredient not selected")
    check(at.pills(key="quick_pick").value is None, "quick pick: chip not cleared")

    # 9. custom weights, number of results, reset
    at.sidebar.slider(key="w_flav").set_value(1.0).run()
    at.sidebar.slider(key="k").set_value(5).run()
    no_exception(at, "custom weights")
    go = [b for b in at.button if b.key and b.key.startswith("go-")]
    check(len(go) == 5, f"k=5: expected 5 cards, got {len(go)}")
    reset = [b for b in at.sidebar.button if "Reset" in b.label][0]
    reset.click().run()
    no_exception(at, "reset weights")
    check(abs(at.sidebar.slider(key="w_flav").value - TUNED_FLAV) < 1e-9, "reset: flavor weight not restored")

    # 10. every quick pick renders in both modes without errors
    for pick in ["butter", "sour cream", "sugar", "heavy cream", "ground beef", "mayonnaise",
                 "all purpose flour", "white rice"]:
        for mode in ["Similar", "Healthier"]:
            a = fresh()
            a.selectbox(key="ingredient").set_value(pick).run()
            a.segmented_control(key="mode").set_value(mode).run()
            no_exception(a, f"{pick}/{mode}")

    n = len(FAILURES)
    print("all app checks passed" if n == 0 else f"{n} failures")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
