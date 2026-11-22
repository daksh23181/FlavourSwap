"""Ingredient-name normalisation for Food.com ingredient strings.

Food.com ingredients are already parsed out of the recipe text (no quantities), but the same
ingredient appears under many surface forms ("eggs", "large eggs", "egg"; "lemon, juice of",
"fresh lemon juice"). `canonicalize()` maps a raw string to one or more canonical names.

Design rule: merge only *surface variants and true synonyms*. Anything that changes nutrition or
is itself a plausible substitute (unsalted vs salted butter, skim vs whole milk, chicken stock vs
chicken broth, garlic powder vs garlic, ground vs fresh ginger) is deliberately kept apart,
because those are exactly the swaps the recommender must be able to suggest and be evaluated on.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------------------------
# 1. Whole-string splits (one raw string -> several ingredients)
# --------------------------------------------------------------------------------------------
_SALT_PEPPER = re.compile(
    r"^(?:(kosher|sea|coarse|table) )?salt (?:and|&) (?:freshly |fresh )?(?:cracked |coarse |coarsely )?"
    r"(?:ground )?(?:black )?pepper$"
)
# "lemon, juice of" / "limes, juice and zest of" / "orange, rind of" / "pomegranate, seeds of"
_PART_OF = re.compile(r"^(?P<fruit>[a-z ]+), (?P<parts>juice|zest|rind|peel|seeds|juice and zest|juice and rind)( of)?$")

# --------------------------------------------------------------------------------------------
# 2. Leading descriptors that never change what the ingredient is
# --------------------------------------------------------------------------------------------
_STRIP_LEADING = {
    "fresh", "freshly", "minced", "chopped", "finely", "coarsely", "roughly", "sliced", "thinly",
    "shredded", "grated", "peeled", "boneless", "skinless", "pure", "ripe", "cracked", "coarse",
    "large", "medium", "small", "jumbo", "extra-large", "packed", "liquid", "sifted", "unsifted",
}
# Do not strip a size word when it is part of a product name ("small curd cottage cheese").
_SIZE_WORDS = {"large", "medium", "small", "jumbo", "extra-large"}
_SIZE_KEEP_BEFORE = {"curd", "grain"}
_KEEP_WHOLE = {"shredded wheat", "shredded wheat cereal", "coarse salt", "coarse sea salt",
               "packed brown sugar", "liquid smoke", "cracked wheat", "fresh mozzarella",
               "fresh mozzarella cheese", "ripe olive", "ripe olives"}

# "ground X" -> "X" only for spices whose default (and USDA) form is already ground.
_GROUND_SPICES = {
    "cinnamon", "cumin", "nutmeg", "clove", "cloves", "allspice", "cardamom", "turmeric", "mace",
    "black pepper", "white pepper", "pepper", "cayenne pepper", "cayenne", "coriander", "paprika",
}

# --------------------------------------------------------------------------------------------
# 3. Singularisation of the final token
# --------------------------------------------------------------------------------------------
_NO_SINGULAR = {
    "asparagus", "couscous", "hummus", "molasses", "swiss", "grits", "oats", "bitters", "schnapps",
    "citrus", "watercress", "cress", "bass", "lemongrass", "octopus", "manis", "chablis", "cassis",
    "brussels", "jus", "verjus", "frais", "hollandaise", "series", "species", "lentils",
    "greens", "pois", "chips", "sprinkles", "noodles", "krispies", "flakes", "oreos", "nachos",
    "fries", "crisps", "cheerios", "chex", "doritos", "fritos", "cornflakes", "seasonings",
    "giblets", "tots", "pretzels", "bran flakes", "glass", "grass", "kiss", "plus", "patis",
}
_IRREGULAR = {
    "leaves": "leaf", "loaves": "loaf", "halves": "half", "knives": "knife",
    "cookies": "cookie", "brownies": "brownie", "pies": "pie", "veggies": "veggie",
    "smoothies": "smoothie", "goodies": "goodie", "calories": "calorie", "zucchinis": "zucchini",
    "tomatoes": "tomato", "potatoes": "potato", "mangoes": "mango", "avocadoes": "avocado",
    "chilies": "chili", "chilis": "chili", "chiles": "chile", "anchovies": "anchovy",
    "cloves": "clove", "olives": "olive", "chives": "chive", "endives": "endive",
    "dates": "date", "grapes": "grape", "limes": "lime", "pineapples": "pineapple",
}


def _singular(word: str) -> str:
    if word in _NO_SINGULAR or len(word) <= 3:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _singular_phrase(phrase: str) -> str:
    toks = phrase.split()
    if not toks:
        return phrase
    if phrase in _NO_SINGULAR:
        return phrase
    toks[-1] = _singular(toks[-1])
    return " ".join(toks)


# --------------------------------------------------------------------------------------------
# 4. Trailing form words that do not change the ingredient
# --------------------------------------------------------------------------------------------
_HERBS = {
    "basil", "mint", "parsley", "cilantro", "thyme", "oregano", "sage", "rosemary",
    "tarragon", "dill", "marjoram", "chervil", "lettuce", "spinach", "baby spinach",
    "romaine lettuce", "dried thyme", "dried oregano", "dried basil", "dried rosemary",
    "dried marjoram", "dried tarragon", "dried sage", "dried mint", "italian parsley",
    "flat leaf parsley", "flat-leaf parsley", "thai basil",
}
_HERB_FORM_SUFFIX = ("leaf", "leave", "sprig", "stem", "leaf and stem")

# --------------------------------------------------------------------------------------------
# 5. Explicit synonyms / surface variants -> canonical name (applied last)
# --------------------------------------------------------------------------------------------
ALIASES: dict[str, str] = {
    # salt
    "kosher salt": "salt", "sea salt": "salt", "table salt": "salt", "coarse salt": "salt",
    "coarse sea salt": "salt", "fine sea salt": "salt", "coarse kosher salt": "salt",
    "iodized salt": "salt", "fine salt": "salt", "flaky sea salt": "salt",
    # pepper (in Food.com an unqualified "pepper" is black pepper: "salt and pepper")
    "pepper": "black pepper", "ground pepper": "black pepper", "black peppercorns ground": "black pepper",
    "ground red pepper": "cayenne pepper", "cayenne": "cayenne pepper", "cayenne powder": "cayenne pepper",
    "red pepper": "red bell pepper", "green pepper": "green bell pepper",
    "yellow pepper": "yellow bell pepper", "orange pepper": "orange bell pepper",
    "crushed red pepper flake": "red pepper flakes", "crushed red pepper": "red pepper flakes",
    "red pepper flake": "red pepper flakes", "dried red pepper flake": "red pepper flakes",
    "dried red pepper flakes": "red pepper flakes", "crushed red pepper flakes": "red pepper flakes",
    "peppercorn": "black peppercorn", "whole black peppercorn": "black peppercorn",
    "whole peppercorn": "black peppercorn",
    # water
    "cold water": "water", "warm water": "water", "hot water": "water", "boiling water": "water",
    "ice water": "water", "lukewarm water": "water", "tap water": "water", "filtered water": "water",
    "cool water": "water", "very hot water": "water", "ice cold water": "water",
    # flour / sugar / starch
    "flour": "all purpose flour", "plain flour": "all purpose flour",
    "unbleached all purpose flour": "all purpose flour", "all purpose white flour": "all purpose flour",
    "white flour": "all purpose flour", "unbleached flour": "all purpose flour",
    "granulated sugar": "sugar", "white sugar": "sugar", "caster sugar": "sugar",
    "superfine sugar": "sugar", "granulated white sugar": "sugar", "fine sugar": "sugar",
    "confectioners sugar": "powdered sugar", "icing sugar": "powdered sugar",
    "confectioner sugar": "powdered sugar", "powdered confectioners sugar": "powdered sugar",
    "light brown sugar": "brown sugar", "dark brown sugar": "brown sugar",
    "packed brown sugar": "brown sugar", "golden brown sugar": "brown sugar",
    "firmly packed brown sugar": "brown sugar", "packed light brown sugar": "brown sugar",
    "corn starch": "cornstarch", "cornflour": "cornstarch",
    # eggs
    "egg": "egg", "extra large egg": "egg", "whole egg": "egg", "beaten egg": "egg",
    "egg yolks": "egg yolk", "egg whites": "egg white", "large egg white": "egg white",
    "large egg yolk": "egg yolk",
    # vanilla
    "vanilla": "vanilla extract", "vanilla essence": "vanilla extract",
    # alliums
    "garlic clove": "garlic", "clove garlic": "garlic", "garlic clove minced": "garlic",
    "minced garlic clove": "garlic", "minced garlic": "garlic", "chopped garlic": "garlic",
    "yellow onion": "onion", "brown onion": "onion", "spanish onion": "onion",
    "scallion": "green onion", "spring onion": "green onion", "green onion top": "green onion",
    "green scallion": "green onion",
    # herbs
    "fresh coriander": "cilantro", "coriander leaf": "cilantro", "green coriander": "cilantro",
    "chinese parsley": "cilantro", "coriander sprig": "cilantro", "fresh coriander leaf": "cilantro",
    "flat leaf parsley": "parsley", "italian parsley": "parsley", "curly parsley": "parsley",
    "flat leaf italian parsley": "parsley", "italian flat leaf parsley": "parsley",
    "parsley flake": "dried parsley", "dried parsley flake": "dried parsley", "parsley flakes": "dried parsley",
    "dried parsley flakes": "dried parsley",
    "gingerroot": "ginger", "ginger root": "ginger", "fresh ginger root": "ginger",
    "dry mustard": "dry mustard", "ground mustard": "dry mustard", "mustard powder": "dry mustard",
    "dry mustard powder": "dry mustard",
    "coriander powder": "coriander", "ground coriander seed": "coriander",
    "coriander seeds": "coriander seed", "cumin seeds": "cumin seed",
    "clove": "clove", "whole clove": "whole clove",
    # citrus
    "lemon rind": "lemon zest", "lemon peel": "lemon zest", "grated lemon peel": "lemon zest",
    "grated lemon rind": "lemon zest", "grated lemon zest": "lemon zest",
    "lime rind": "lime zest", "lime peel": "lime zest", "grated lime zest": "lime zest",
    "orange rind": "orange zest", "orange peel": "orange zest", "grated orange peel": "orange zest",
    "grated orange rind": "orange zest", "grated orange zest": "orange zest",
    "juice of lemon": "lemon juice", "juice of lime": "lime juice",
    "lemon juice fresh": "lemon juice", "lime juice fresh": "lime juice",
    "key lime": "lime", "key lime juice": "lime juice",
    # fats and oils
    "oil": "vegetable oil", "cooking oil": "vegetable oil",
    "extra virgin olive oil": "olive oil", "virgin olive oil": "olive oil",
    "extra light olive oil": "olive oil", "light olive oil": "olive oil",
    "sweet butter": "unsalted butter", "salted butter": "butter",
    "butter softened": "butter", "melted butter": "butter", "softened butter": "butter",
    "unsalted butter softened": "unsalted butter",
    # dairy
    "heavy whipping cream": "heavy cream", "whipping cream": "heavy cream", "double cream": "heavy cream",
    "heavy cream whipped": "heavy cream",
    "half and half cream": "half and half", "half & half": "half and half", "half and half": "half and half",
    "skim milk": "skim milk", "nonfat milk": "skim milk", "fat free milk": "skim milk",
    "non fat milk": "skim milk", "fat free skim milk": "skim milk",
    "lowfat milk": "low fat milk", "2% low fat milk": "2% milk", "2% reduced fat milk": "2% milk",
    "reduced fat milk": "2% milk", "1% low fat milk": "1% milk", "1% lowfat milk": "1% milk",
    "parmesan": "parmesan cheese", "parmigiano reggiano cheese": "parmesan cheese",
    "parmigiano reggiano": "parmesan cheese", "grated parmesan cheese": "parmesan cheese",
    "parmesan reggiano cheese": "parmesan cheese", "parmigiano": "parmesan cheese",
    "cheddar": "cheddar cheese", "sharp cheddar cheese": "cheddar cheese", "sharp cheddar": "cheddar cheese",
    "mild cheddar cheese": "cheddar cheese", "medium cheddar cheese": "cheddar cheese",
    "medium cheddar": "cheddar cheese", "extra sharp cheddar cheese": "cheddar cheese",
    "mild cheddar": "cheddar cheese", "extra sharp cheddar": "cheddar cheese",
    "mozzarella": "mozzarella cheese", "fresh mozzarella": "fresh mozzarella cheese",
    "fresh mozzarella cheese": "fresh mozzarella cheese",
    "monterey jack": "monterey jack cheese", "feta": "feta cheese", "ricotta": "ricotta cheese",
    "swiss": "swiss cheese", "gruyere": "gruyere cheese", "romano": "romano cheese",
    "pecorino romano": "romano cheese", "pecorino romano cheese": "romano cheese",
    "plain yogurt": "plain yogurt", "plain yoghurt": "plain yogurt", "natural yoghurt": "plain yogurt",
    "natural yogurt": "plain yogurt", "yoghurt": "yogurt", "greek yoghurt": "greek yogurt",
    "plain greek yogurt": "greek yogurt",
    # legumes / veg synonyms
    "garbanzo bean": "chickpea", "garbanzo": "chickpea", "chick pea": "chickpea",
    "garbanzo beans": "chickpea", "chick peas": "chickpea",
    "courgette": "zucchini", "aubergine": "eggplant", "capsicum": "bell pepper",
    "red capsicum": "red bell pepper", "green capsicum": "green bell pepper",
    "rocket": "arugula", "coriander root": "cilantro root",
    # meat
    "boneless chicken breast": "chicken breast", "skinless chicken breast": "chicken breast",
    "chicken breast half": "chicken breast", "boneless skinless chicken breast": "chicken breast",
    "chicken breast fillet": "chicken breast", "skinless chicken breast half": "chicken breast",
    "boneless chicken breast half": "chicken breast",
    "chicken thigh fillet": "chicken thigh", "boneless chicken thigh": "chicken thigh",
    "boneless skinless chicken thigh": "chicken thigh", "skinless chicken thigh": "chicken thigh",
    "lean ground beef": "lean ground beef", "extra lean ground beef": "lean ground beef",
    "ground chuck": "ground beef", "hamburger": "ground beef", "minced beef": "ground beef",
    "beef mince": "ground beef", "hamburger meat": "ground beef",
    "bacon slice": "bacon", "bacon strip": "bacon", "slice bacon": "bacon",
    # misc
    "ripe olive": "black olive", "jalapeno pepper": "jalapeno", "jalapeno chile": "jalapeno",
    "jalapeno chili": "jalapeno",
    "tomato": "tomato", "roma tomato": "plum tomato", "italian plum tomato": "plum tomato",
    "baking potato": "russet potato", "idaho potato": "russet potato",
    "liquid honey": "honey", "clear honey": "honey",
    "pecan half": "pecan", "walnut half": "walnut", "walnut piece": "walnut", "pecan piece": "pecan",
    "chopped walnut": "walnut", "chopped pecan": "pecan",
    "shredded coconut": "coconut", "flaked coconut": "coconut", "desiccated coconut": "coconut",
    "sweetened shredded coconut": "sweetened coconut", "sweetened flaked coconut": "sweetened coconut",
    "rolled oats": "rolled oats", "old fashioned oats": "rolled oats", "old fashioned rolled oats": "rolled oats",
    "oatmeal": "rolled oats", "old fashioned oatmeal": "rolled oats", "quick oats": "quick cooking oats",
    "quick cooking oatmeal": "quick cooking oats",
    "maple syrup": "maple syrup", "pure maple syrup": "maple syrup", "real maple syrup": "maple syrup",
    "semi sweet chocolate chip": "semisweet chocolate chips",
    "semisweet chocolate chip": "semisweet chocolate chips",
    "semi sweet chocolate chips": "semisweet chocolate chips",
    "semisweet chocolate morsel": "semisweet chocolate chips",
    "semi sweet chocolate morsel": "semisweet chocolate chips",
    "chocolate chip": "chocolate chips",
    "worcestershire": "worcestershire sauce",
    "tabasco": "tabasco sauce", "hot pepper sauce": "hot sauce",
    "dijon": "dijon mustard", "prepared mustard": "yellow mustard", "mustard": "mustard",
    "ketchup": "ketchup", "catsup": "ketchup", "tomato ketchup": "ketchup",
    "mayo": "mayonnaise", "real mayonnaise": "mayonnaise",
    "baking soda": "baking soda", "bicarbonate of soda": "baking soda", "bicarb soda": "baking soda",
    "soda": "baking soda",
    "chicken stock": "chicken stock", "chicken broth": "chicken broth",
    "beef stock": "beef stock", "beef broth": "beef broth",
    # cocoa: in North American recipes plain "cocoa" is unsweetened cocoa powder
    "cocoa": "cocoa powder", "unsweetened cocoa": "cocoa powder", "unsweetened cocoa powder": "cocoa powder",
    "baking cocoa": "cocoa powder", "unsweetened baking cocoa": "cocoa powder",
    # pasta shapes written with/without "pasta"
    "penne pasta": "penne", "fettuccine pasta": "fettuccine", "linguine pasta": "linguine",
    "spaghetti pasta": "spaghetti", "rigatoni pasta": "rigatoni", "rotini pasta": "rotini",
    "farfalle pasta": "bow tie pasta", "farfalle": "bow tie pasta", "orzo pasta": "orzo",
    "mini marshmallow": "miniature marshmallow",
    # spices written as "X powder"
    "turmeric powder": "turmeric", "cumin powder": "cumin", "cinnamon powder": "cinnamon",
    "nutmeg powder": "nutmeg", "cardamom powder": "cardamom",
    "chili flakes": "red pepper flakes", "dried chili flakes": "red pepper flakes",
    "salad oil": "vegetable oil",
    "condensed cream of mushroom soup": "cream of mushroom soup",
    "condensed cream of chicken soup": "cream of chicken soup",
    "condensed cream of celery soup": "cream of celery soup",
    "condensed tomato soup": "tomato soup",
    "dried dill weed": "dried dill", "dijon style mustard": "dijon mustard",
    "lemon pepper seasoning": "lemon pepper",
    "chili pepper": "chili", "red chili pepper": "red chili", "green chili pepper": "green chili",
    "serrano pepper": "serrano chili", "chipotle chilies in adobo": "chipotle chili in adobo",
    "chipotle chili in adobo sauce": "chipotle chili in adobo",
    "chipotle chilies in adobo sauce": "chipotle chili in adobo",
    "ziti pasta": "ziti", "marshmallow cream": "marshmallow creme",
    "instant coffee granule": "instant coffee", "instant coffee granules": "instant coffee",
    "unsweetened chocolate square": "unsweetened chocolate",
    "condensed milk": "sweetened condensed milk", "prepared yellow mustard": "yellow mustard",
    "smooth peanut butter": "creamy peanut butter", "chunky peanut butter": "crunchy peanut butter",
    "frozen green pea": "frozen pea", "green pea": "pea", "frozen corn kernel": "frozen corn",
    "sultana": "golden raisin", "white fish fillet": "fish fillet", "stewing beef": "beef stew meat",
    "ginger powder": "ground ginger", "kiwi fruit": "kiwi", "french baguette": "baguette",
    "dried breadcrumb": "dry breadcrumb", "fine dry breadcrumb": "dry breadcrumb",
    "unbleached white flour": "all purpose flour", "real butter": "butter",
    "sweet unsalted butter": "unsalted butter", "dry basil": "dried basil", "dry oregano": "dried oregano",
    "dry thyme": "dried thyme", "dry parsley": "dried parsley", "dry rosemary": "dried rosemary",
    "hellmanns mayonnaise": "mayonnaise", "best foods mayonnaise": "mayonnaise",
    "hellmanns real mayonnaise": "mayonnaise", "tamari soy sauce": "tamari",
}

# Applied to the cleaned full string *before* any stripping, for cases where stripping would
# destroy the meaning ("fresh coriander" is the leaf = cilantro, "ground coriander" is the seed).
_PRE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(fresh |dried )?(green )?coriander (leaf|leaves|sprig|sprigs|stem|stems|leaves and stems)$"),
     "cilantro"),
    (re.compile(r"^(fresh )+(green )?coriander$|^green coriander$"), "cilantro"),
    (re.compile(r"^fresh cayenne peppers?$"), "red chili"),
    (re.compile(r"^of fresh (.+)$"), r"fresh \1"),  # Food.com parsing artifact: "of fresh mint"
]

_WS = re.compile(r"\s+")
# Token-level spelling variants.
_HYPHEN_TOKENS = {
    "lowfat": "low fat", "nonfat": "non fat", "reducedfat": "reduced fat",
    "chile": "chili", "chiles": "chilies", "chilis": "chilies", "yoghurt": "yogurt", "soya": "soy",
}


def basic_clean(raw: str) -> str:
    s = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = s.replace("’", "").replace("'", "").replace("`", "").replace('"', "")
    s = s.replace("-", " ")
    s = _WS.sub(" ", s).strip(" ,.;:")
    toks = [_HYPHEN_TOKENS.get(t, t) for t in s.split()]
    return " ".join(toks)


def _strip_leading(s: str) -> str:
    if s in _KEEP_WHOLE:
        return s
    toks = s.split()
    changed = True
    while changed and len(toks) > 1:
        changed = False
        head = toks[0]
        rest = " ".join(toks[1:])
        if " ".join(toks) in _KEEP_WHOLE:
            break
        if head in ("ground",) and rest in _GROUND_SPICES:
            toks = toks[1:]
            changed = True
        elif head in ("extra",) and len(toks) > 2 and toks[1] in ("large",):
            toks = toks[2:]  # "extra large eggs"
            changed = True
        elif head in _STRIP_LEADING:
            if head in _SIZE_WORDS and toks[1] in _SIZE_KEEP_BEFORE:
                break
            toks = toks[1:]
            changed = True
    return " ".join(toks)


def _strip_herb_form(s: str) -> str:
    for suf in _HERB_FORM_SUFFIX:
        if s.endswith(" " + suf):
            base = s[: -len(suf) - 1]
            if base in _HERBS:
                return base
    return s


def _canonical_one(s: str) -> str:
    s = _strip_leading(s)
    s = _singular_phrase(s)
    s = _strip_herb_form(s)
    return ALIASES.get(s, s)


def canonicalize(raw: str) -> list[str]:
    """Return the canonical ingredient name(s) for a raw Food.com ingredient string."""
    s = basic_clean(raw)
    if not s:
        return []
    for pat, repl in _PRE_RULES:
        if pat.match(s):
            s = pat.sub(repl, s)
            break

    m = _SALT_PEPPER.match(s)
    if m:
        return ["salt", "black pepper"]

    m = _PART_OF.match(s)
    if m:
        fruit = _singular_phrase(_strip_leading(m.group("fruit").strip()))
        fruit = ALIASES.get(fruit, fruit)
        out = []
        for part in m.group("parts").split(" and "):
            part = {"rind": "zest", "peel": "zest", "seeds": "seed"}.get(part, part)
            out.append(_canonical_one(f"{fruit} {part}"))
        return out

    return [_canonical_one(s)]


# Kitchen supplies that Food.com users list as "ingredients"; they are dropped from recipes.
NON_FOOD = {
    "aluminum foil", "reynolds wrap foil", "toothpick", "skewer", "wooden skewer", "bamboo skewer",
    "metal skewer", "parchment paper", "wax paper", "plastic wrap", "cheesecloth", "paraffin wax",
    "paraffin wax block", "wooden popsicle stick", "popsicle stick", "string", "kitchen twine",
    "kitchen string", "ziploc bag", "jar", "paper", "paper towel", "oven cooking bag", "roasting bag",
    "brown paper bag", "paper cup", "paper baking cup", "cupcake liner", "ribbon", "plastic spoon",
}


def canonicalize_recipe(raw_ingredients: list[str]) -> list[str]:
    """Canonicalise a recipe's ingredient list, de-duplicating while keeping first-seen order."""
    seen: dict[str, None] = {}
    for raw in raw_ingredients:
        for c in canonicalize(raw):
            if c and c not in NON_FOOD:
                seen.setdefault(c, None)
    return list(seen)
