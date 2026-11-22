"""Download the three raw data sources used by FlavorSwap into data/raw/.

Sources
-------
1. Food.com Recipes and Interactions (Kaggle: shuyangli94/food-com-recipes-and-user-interactions):
   RAW_recipes.csv (recipes) and RAW_interactions.csv (user reviews, mined for substitutions).
   Majumder et al., "Generating Personalized Recipes from Historical User Preferences", EMNLP 2019.
   Plus "Food.com Recipes with Search Terms and Tags" (Kaggle: shuyangli94/foodcom-recipes-with-
   search-terms-and-tags): the same Food.com recipes (same ids) with the raw ingredient lines incl.
   quantities and units, servings and serving size - used for recipe-level amounts (Deliverable 5).
2. Flavor network data (ingredient -> flavor compound), Ahn et al., "Flavor network and the
   principles of food pairing", Scientific Reports 1:196 (2011). Supplementary tables as mirrored
   in github.com/lingcheng99/Flavor-Network.
3. USDA FoodData Central, SR Legacy (April 2018) CSV release.

Run:  python src/download_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "shuyangli94/food-com-recipes-and-user-interactions"
)
FLAVOR_BASE = "https://raw.githubusercontent.com/lingcheng99/Flavor-Network/master/data"
FLAVOR_FILES = ["ingr_info.tsv", "comp_info.tsv", "ingr_comp.tsv"]
USDA_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"

# Only these members of the archives are needed.
FOODCOM_MEMBERS = ["RAW_recipes.csv", "RAW_interactions.csv"]
USDA_MEMBERS = ["food.csv", "nutrient.csv", "food_nutrient.csv", "food_category.csv",
                "food_portion.csv", "measure_unit.csv"]
KAGGLE_TERMS_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "shuyangli94/foodcom-recipes-with-search-terms-and-tags"
)
TERMS_MEMBERS = ["recipes_w_search_terms.csv"]


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.relative_to(ROOT)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (FlavorSwap course project)"})
    print(f"  downloading {url}")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def extract(zip_path: Path, members: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for m in members:
            if m not in names:
                sys.exit(f"ERROR: {m} not found in {zip_path.name}; archive has {sorted(names)}")
            target = out_dir / m
            if target.exists():
                continue
            with zf.open(names[m]) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            print(f"  extracted {target.relative_to(ROOT)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("[1/3] Food.com recipes")
    z = download(KAGGLE_URL, RAW / "_archives" / "foodcom.zip")
    extract(z, FOODCOM_MEMBERS, RAW / "foodcom")
    z = download(KAGGLE_TERMS_URL, RAW / "_archives" / "foodcom_search_terms.zip")
    extract(z, TERMS_MEMBERS, RAW / "foodcom")

    print("[2/3] Ahn et al. 2011 flavor network")
    for f in FLAVOR_FILES:
        download(f"{FLAVOR_BASE}/{f}", RAW / "flavor_network" / f)

    print("[3/3] USDA FoodData Central SR Legacy")
    z = download(USDA_URL, RAW / "_archives" / "usda_sr_legacy.zip")
    extract(z, USDA_MEMBERS, RAW / "usda_sr_legacy")

    manifest = RAW / "MANIFEST.txt"
    with open(manifest, "w", encoding="utf-8") as fh:
        for p in sorted(RAW.rglob("*")):
            if p.is_file() and "_archives" not in p.parts and p.name != "MANIFEST.txt":
                fh.write(f"{sha256(p)}  {p.stat().st_size:>12}  {p.relative_to(RAW).as_posix()}\n")
    print(f"Wrote {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
