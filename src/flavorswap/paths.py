"""Central file locations so every script agrees on where data lives."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
CURATION = ROOT / "data" / "curation"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

RAW_RECIPES = RAW / "foodcom" / "RAW_recipes.csv"
AHN_INGR = RAW / "flavor_network" / "ingr_info.tsv"
AHN_COMP = RAW / "flavor_network" / "comp_info.tsv"
AHN_INGR_COMP = RAW / "flavor_network" / "ingr_comp.tsv"
USDA_DIR = RAW / "usda_sr_legacy"
