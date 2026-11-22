"""A single food-group label per ingredient, used for colouring the embedding map, for group-level
analyses and for the substitution model's food-group transition prior.

Source priority: USDA food category (curated/verified link) > Ahn et al. category > name keywords.
The source is always recorded so keyword-derived labels can be told apart.

Two culinary refinements of the USDA categories matter for substitution:
  * eggs and dairy are separate groups (USDA files both under "Dairy and Egg Products", but eggs do
    not substitute for butter or cream cheese);
  * fresh aromatics and herbs (garlic, ginger, shallot, basil, parsley, cilantro, ...) and citrus
    zest count as Herbs & Spices, not Vegetables/Fruits (in cooking they are seasonings);
  * soy sauce, miso, ketchup and similar count as condiments (USDA files them under Legumes or
    Vegetables).
"""

from __future__ import annotations

import re

GROUPS = [
    "Dairy", "Eggs", "Meat & Poultry", "Fish & Seafood", "Vegetables", "Fruits",
    "Legumes, Nuts & Seeds", "Grains, Pasta & Bread", "Herbs & Spices", "Fats & Oils",
    "Sweets, Snacks & Baking", "Sauces, Condiments & Soups", "Beverages & Alcohol", "Other",
]

USDA_TO_GROUP = {
    "Dairy and Egg Products": "Dairy",
    "Spices and Herbs": "Herbs & Spices",
    "Fats and Oils": "Fats & Oils",
    "Poultry Products": "Meat & Poultry",
    "Pork Products": "Meat & Poultry",
    "Beef Products": "Meat & Poultry",
    "Lamb, Veal, and Game Products": "Meat & Poultry",
    "Sausages and Luncheon Meats": "Meat & Poultry",
    "Finfish and Shellfish Products": "Fish & Seafood",
    "Vegetables and Vegetable Products": "Vegetables",
    "Fruits and Fruit Juices": "Fruits",
    "Legumes and Legume Products": "Legumes, Nuts & Seeds",
    "Nut and Seed Products": "Legumes, Nuts & Seeds",
    "Cereal Grains and Pasta": "Grains, Pasta & Bread",
    "Breakfast Cereals": "Grains, Pasta & Bread",
    "Baked Products": "Grains, Pasta & Bread",
    "Sweets": "Sweets, Snacks & Baking",
    "Snacks": "Sweets, Snacks & Baking",
    "Soups, Sauces, and Gravies": "Sauces, Condiments & Soups",
    "Beverages": "Beverages & Alcohol",
    "Alcoholic Beverages": "Beverages & Alcohol",
}
USDA_EGG = re.compile(r"^Egg(,| substitute)")      # "Egg, whole, raw" / "Egg substitute" (not "Eggnog")

AHN_TO_GROUP = {
    "dairy": "Dairy", "meat": "Meat & Poultry", "fish/seafood": "Fish & Seafood",
    "vegetable": "Vegetables", "fruit": "Fruits", "nut/seed/pulse": "Legumes, Nuts & Seeds",
    "cereal/crop": "Grains, Pasta & Bread", "herb": "Herbs & Spices", "spice": "Herbs & Spices",
    "plant": "Herbs & Spices", "flower": "Herbs & Spices",
    "alcoholic beverage": "Beverages & Alcohol",
}

# Condiments that USDA files under Legumes (soy sauce, miso) or Vegetables (catsup).
CONDIMENTS = re.compile(r"\b(soy sauce|tamari|miso|ketchup|catsup|fish sauce|hoisin|teriyaki|oyster sauce)\b")
# Citrus zest/peel is a flavoring used by the gram (USDA files it under Fruits).
ZEST = re.compile(r"\b(zest|peel|rind)\b")
SEASONING_PLANTS = re.compile(
    r"\b(garlic|ginger|gingerroot|shallot|lemongrass|lemon grass|galangal|basil|parsley|cilantro|"
    r"coriander|dill|mint|chive|chives|thyme|rosemary|oregano|sage|tarragon|chervil|marjoram|"
    r"curry leaf|epazote)\b")

# Ordered: the first matching rule wins (e.g. "chicken broth" is a soup, not meat).
KEYWORD_RULES: list[tuple[str, str]] = [
    (r"\b(peanut|almond|cashew|sunflower|nut|hazelnut) butter\b", "Legumes, Nuts & Seeds"),
    (r"\b(textured vegetable protein|seitan|tempeh)\b", "Legumes, Nuts & Seeds"),
    (r"\b(sauce|dressing|soup|broth|stock|bouillon|gravy|salsa|ketchup|catsup|mustard|mayonnaise|"
     r"relish|vinegar|marinade|paste|pesto|chutney|dip|hummus)\b", "Sauces, Condiments & Soups"),
    (r"\b(wine|beer|vodka|rum|whiskey|whisky|bourbon|brandy|liqueur|schnapps|tequila|gin|cognac|"
     r"sherry|vermouth|juice|soda|cola|coffee|espresso|tea|lemonade|ale|champagne|kahlua|amaretto|"
     r"marnier|triple sec|cointreau|nectar|drink|water|beverage)\b", "Beverages & Alcohol"),
    (r"\b(oil|shortening|lard|margarine|cooking spray|spray)\b", "Fats & Oils"),
    (r"\b(sugar|syrup|honey|chocolate|cocoa|candy|candies|caramel|marshmallow|frosting|icing|"
     r"sprinkles|pudding|jam|jelly|preserve|marmalade|cookie|cake|brownie|extract|gelatin|jello|"
     r"pie filling|baking|yeast|pectin|sweetener|splenda|chips|chip|ice cream|sorbet|"
     r"whipped topping|cool whip|food coloring|popcorn|pretzel)\b", "Sweets, Snacks & Baking"),
    (r"\begg(s)?\b(?! (noodle|noodles|roll|wrapper|wrappers|bread|nog|pasta))", "Eggs"),
    (r"\b(cheese|milk|cream|yogurt|butter|buttermilk|ghee|half and half|kefir|ricotta|"
     r"mascarpone|paneer|creme fraiche)\b", "Dairy"),
    (r"\b(chicken|beef|pork|lamb|turkey|ham|bacon|sausage|steak|veal|venison|duck|chorizo|salami|"
     r"prosciutto|pancetta|pepperoni|meat|rib|ribs|brisket|roast|tenderloin|loin|hot dog|"
     r"frankfurter|kielbasa|bratwurst|goat|rabbit)\b", "Meat & Poultry"),
    (r"\b(fish|salmon|tuna|shrimp|crab|crabmeat|lobster|clam|mussel|oyster|scallop|cod|tilapia|"
     r"halibut|anchovy|sardine|prawn|squid|calamari|trout|catfish|snapper|haddock|mahi|swordfish|"
     r"seafood|crawfish|octopus)\b", "Fish & Seafood"),
    (r"\b(flour|bread|pasta|noodle|noodles|rice|tortilla|cracker|cereal|oat|oats|bun|roll|biscuit|"
     r"crust|pastry|dough|macaroni|spaghetti|penne|fettuccine|linguine|lasagna|couscous|quinoa|"
     r"barley|cornmeal|polenta|grits|bagel|muffin|pita|croissant|wrapper|crouton|granola|bran|"
     r"germ|breadcrumb|panko|stuffing|orzo|ziti|rotini|rigatoni|vermicelli|gnocchi|wheat)\b",
     "Grains, Pasta & Bread"),
    (r"\b(bean|lentil|lentils|chickpea|nut|nuts|almond|walnut|pecan|cashew|pistachio|hazelnut|"
     r"peanut|seed|tofu|tempeh|edamame|soy|soybean|macadamia)\b", "Legumes, Nuts & Seeds"),
    (r"\b(bell pepper|chili|jalapeno|poblano|serrano|habanero|chipotle|anaheim)\b", "Vegetables"),
    (r"\b(apple|banana|berry|cherry|grape|lemon|lime|orange|peach|pear|plum|mango|pineapple|melon|"
     r"kiwi|apricot|fig|date|raisin|currant|cranberry|coconut|papaya|pomegranate|grapefruit|"
     r"tangerine|clementine|nectarine|guava|passion fruit|rhubarb|zest|fruit|cantaloupe|"
     r"watermelon|strawberry|blueberry|raspberry|blackberry|prune)\b", "Fruits"),
    (r"\b(pepper|peppercorn|paprika|cumin|cinnamon|nutmeg|clove|ginger|turmeric|curry|oregano|"
     r"basil|thyme|rosemary|sage|parsley|cilantro|dill|mint|tarragon|bay leaf|herb|herbs|spice|"
     r"seasoning|masala|saffron|salt|cardamom|coriander|anise|allspice|mace|chive|marjoram|"
     r"vanilla)\b", "Herbs & Spices"),
    (r"\b(onion|garlic|tomato|potato|carrot|celery|lettuce|spinach|cabbage|broccoli|cauliflower|"
     r"squash|zucchini|cucumber|mushroom|corn|kale|chard|greens|leek|shallot|radish|beet|turnip|"
     r"parsnip|asparagus|artichoke|eggplant|okra|sprout|scallion|yam|olive|pickle|vegetable|"
     r"bok choy|arugula|watercress|fennel|jicama|tomatillo|avocado|pumpkin|pea|peas|slaw)\b",
     "Vegetables"),
]
_COMPILED = [(re.compile(p), g) for p, g in KEYWORD_RULES]


def _base_group(ingredient: str, usda_category: str | None, usda_description: str | None,
                ahn_category: str | None) -> tuple[str, str]:
    if usda_category:
        if usda_description and usda_description.startswith("Leavening agents"):
            return "Sweets, Snacks & Baking", "usda"
        if usda_category == "Dairy and Egg Products" and usda_description and USDA_EGG.match(usda_description):
            return "Eggs", "usda"
        if usda_category in USDA_TO_GROUP:
            return USDA_TO_GROUP[usda_category], "usda"
    if ahn_category and ahn_category in AHN_TO_GROUP:
        return AHN_TO_GROUP[ahn_category], "ahn"
    for pat, group in _COMPILED:
        if pat.search(ingredient):
            return group, "keyword"
    return "Other", "none"


def food_group(ingredient: str, usda_category: str | None, usda_description: str | None,
               ahn_category: str | None) -> tuple[str, str]:
    """Return (group, source)."""
    group, source = _base_group(ingredient, usda_category, usda_description, ahn_category)
    if CONDIMENTS.search(ingredient) and group in ("Legumes, Nuts & Seeds", "Vegetables", "Other"):
        return "Sauces, Condiments & Soups", source
    if group == "Fruits" and ZEST.search(ingredient):
        return "Herbs & Spices", source
    if group == "Vegetables" and SEASONING_PLANTS.search(ingredient):
        return "Herbs & Spices", source
    return group, source
