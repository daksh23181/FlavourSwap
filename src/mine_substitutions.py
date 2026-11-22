"""Deliverable 3 (ground truth) - mine known ingredient substitutions from Food.com user reviews.

Reviewers routinely report what they swapped ("used greek yogurt instead of sour cream",
"substituted honey for the sugar", "replaced the butter with applesauce"). This is the same source
used by published substitution datasets (Fatemi et al. 2023, Recipe1MSubs). The embeddings of
Deliverable 2 were trained on recipe ingredient lists only, never on review text, so these pairs are
an independent evaluation set.

Precision safeguards (a mention is kept only if all hold):
  * the cue is one of a fixed set of unambiguous substitution phrasings, with the direction
    (old -> new) fixed by the phrasing;
  * the NEW ingredient is a vocabulary ingredient that sits directly against the cue phrase
    (no free-floating words in between);
  * the OLD ingredient is an ingredient of the reviewed recipe itself (exact canonical name, or the
    unique recipe ingredient whose name ends with it: "sugar" -> the recipe's "brown sugar");
  * no negation right before the cue ("would NOT substitute ..."), and no coordination that makes
    the target ambiguous ("butter and sugar instead of ...").

Outputs:
    data/processed/substitution_mentions.csv   one row per accepted mention (with text snippet)
    data/processed/substitution_pairs.csv      aggregated (source -> target) pairs with support
    data/processed/substitution_counts.csv     the same pairs, counts only (no review text; used by the app)

Run:  python src/mine_substitutions.py
"""

from __future__ import annotations

import html
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402
from flavorswap.normalize import NON_FOOD, canonicalize  # noqa: E402

MAX_NGRAM = 5
NEGATIONS = {"not", "never", "no", "dont", "didnt", "wouldnt", "wont", "cant", "cannot", "shouldnt",
             "couldnt", "doesnt", "isnt", "wasnt", "nor", "without"}
COORD = {"and", "or", "&", "/", "plus", "+", "nor"}
COORD_BEFORE = COORD | {","}   # "butter, sugar instead of ..." -> ambiguous NEW
# Words that may precede the ingredient inside a span (skipped when looking for its start).
LEAD_SKIP = {"the", "a", "an", "some", "all", "regular", "my", "our", "your", "of", "just", "only",
             "extra", "more", "less", "same", "amount", "equal", "real", "using", "use",
             "used", "with", "cup", "cups", "c", "tbsp", "tbs", "tablespoon", "tablespoons", "tsp",
             "teaspoon", "teaspoons", "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "can",
             "cans", "jar", "package", "pkg", "bag", "stick", "sticks", "bunch", "handful",
             "about", "approx", "approximately", "little", "bit", "few", "dash", "pinch", "box", "g",
             "grams", "ml", "quart", "pint", "container", "carton", "bottle", "envelope", "packet",
             "lots", "lot", "plenty", "tons", "splash", "drizzle", "touch", "squeeze"}
VERBS = r"(?:used|use|using|added|add|tried|subbed|substituted|went with|opted for|put in|put)"
# Words that may sit directly before the NEW ingredient. Anything else ("DRIED minced onions",
# "COMTE cheese") means the reviewer's ingredient is longer than what we matched -> rejected.
BEFORE_OK = LEAD_SKIP | {
    "used", "use", "using", "added", "add", "adding", "tried", "try", "trying", "subbed", "sub",
    "substituted", "substitute", "put", "putting", "made", "make", "making", "went", "opted", "chose",
    "choose", "bought", "had", "have", "prefer", "preferred", "like", "liked", "love", "loved", "i",
    "we", "you", "also", "then", "would", "will", "could", "might", "may", "should", "can", "it",
    "them", "with", "in", "did", "do", "does", "decided", "actually", "instead", "both", "but",
    "just", "only", "and", "or", "for"}
# NEW preceded by these (after determiners) is the object of a location phrase, not the
# substitute: "added mushrooms TO THE gravy instead of onions".
LOCATIVE = {"to", "into", "onto", "over", "on", "at", "from", "under", "inside"}
DETERMINERS = {"the", "a", "an", "my", "our", "your", "some", "this", "that"}
# "used milk to mix WITH eggs instead of water": the substitute is not the thing mixed in.
MIX_VERBS = {"mix", "mixed", "mixing", "combine", "combined", "combining", "toss", "tossed",
             "top", "topped", "blend", "blended", "along", "together"}
ALTERNATIVE = {"or", "/", "nor"}   # "coconut OR soy milk": the substitute is not one ingredient

TOKEN = re.compile(r"[a-z0-9%]+(?:[-'][a-z0-9]+)*|&|/|\+|,")


def tokens(text: str) -> list[str]:
    return [t.replace("'", "") for t in TOKEN.findall(text)]


def negated(prefix: list[str]) -> bool:
    """Negation in the few words right before the substitution ("would not substitute ...")."""
    return any(t in NEGATIONS for t in prefix[-4:])


@lru_cache(maxsize=None)
def canon(phrase: str) -> str | None:
    out = canonicalize(phrase)
    if len(out) != 1 or out[0] in NON_FOOD:
        return None
    return out[0]


def _is_number(t: str) -> bool:
    return bool(re.fullmatch(r"[0-9][0-9/.%]*", t))


class Miner:
    def __init__(self, vocab: set[str], recipe_ings: dict[int, frozenset[str]]):
        self.vocab = vocab
        self.recipe_ings = recipe_ings
        # head nouns of vocabulary ingredients ("yogurt", "cheese", "sauce", ...)
        self.heads = {v.split()[-1] for v in vocab} - BEFORE_OK

    def new_ok_before(self, before: list[str]) -> bool:
        """Context to the left of a NEW ingredient must be a clean boundary: walking left over
        articles / quantities / units ("1/2 cup of", "some of the"), the first real word must be a
        verb, pronoun or non-locative preposition. Rejects "cream OF celery", "soak THE onions",
        "DRIED minced onions", "added mushrooms TO THE gravy", "mixed WITH eggs"."""
        i = len(before) - 1
        while i >= 0 and (before[i] in LEAD_SKIP or before[i] in DETERMINERS or _is_number(before[i])
                          or before[i] == "/"):   # "1/2 cup of" is tokenised as 1 / 2 cup of
            i -= 1
        if i < 0:
            return True
        t = before[i]
        if t in LOCATIVE or t not in BEFORE_OK:
            return False
        if t == "with" and i >= 1 and before[i - 1] in MIX_VERBS:
            return False
        return True

    def new_ok_after(self, after: list[str]) -> bool:
        """A NEW ingredient matched forwards must not be cut short ("coconut milk | yogurt")."""
        return not after or (after[0] not in self.heads and after[0] not in ALTERNATIVE)

    # ------------------------------------------------------------------ span matching
    def match_forward(self, toks: list[str]) -> tuple[str, int, int] | None:
        """Longest vocabulary ingredient starting near the beginning of `toks`
        (after skippable lead words). Returns (canonical, start, end)."""
        start = 0
        while start < len(toks) and start < 6 and toks[start] in LEAD_SKIP:
            start += 1
        if start >= len(toks) or not toks[start][0].isalpha():
            return None
        for n in range(min(MAX_NGRAM, len(toks) - start), 0, -1):
            if "," in toks[start:start + n]:
                continue
            c = canon(" ".join(toks[start:start + n]))
            if c and c in self.vocab:
                return c, start, start + n
        return None

    def match_backward(self, toks: list[str]) -> tuple[str, int, int] | None:
        """Longest vocabulary ingredient ENDING at the last token of `toks`."""
        end = len(toks)
        if end == 0:
            return None
        for n in range(min(MAX_NGRAM, end), 0, -1):
            phrase = toks[end - n:end]
            if (phrase[0] in LEAD_SKIP and n > 1) or "," in phrase or not phrase[0][0].isalpha():
                continue
            c = canon(" ".join(phrase))
            if c and c in self.vocab:
                return c, end - n, end
        return None

    def resolve_old(self, old: str, recipe_id: int) -> str | None:
        """The replaced ingredient must belong to the reviewed recipe."""
        ings = self.recipe_ings.get(recipe_id)
        if not ings:
            return None
        if old in ings:
            return old
        hits = [i for i in ings if i.endswith(" " + old)]
        return hits[0] if len(hits) == 1 else None

    # ------------------------------------------------------------------ patterns
    def mine_sentence(self, s: str, recipe_id: int) -> list[tuple[str, str, str, str]]:
        """Return list of (old, new, pattern, snippet)."""
        out = []

        def emit(old_c, new_c, pat, span):
            if old_c is None or new_c is None:
                return
            old_r = self.resolve_old(old_c, recipe_id)
            if old_r is None or old_r not in self.vocab or new_c == old_r or new_c not in self.vocab:
                return
            out.append((old_r, new_c, pat, s[max(0, span[0] - 60):span[1] + 60].strip()))

        def another_ingredient_follows(after: list[str]) -> bool:
            """True for "... butter and sugar": a coordinated second ingredient makes the pair
            ambiguous. "... butter and it was great" is fine."""
            if not after or after[0] not in COORD:
                return False
            nxt = self.match_forward(after[1:MAX_NGRAM + 7])
            return nxt is not None

        # P1: NEW (instead of | in place of | rather than) OLD
        for m in re.finditer(r"\b(instead of|in place of|rather than)\b", s):
            left, right = tokens(s[:m.start()]), tokens(s[m.end():])
            if not left or not right or left[-1] in COORD:
                continue
            old = self.match_forward(right[:MAX_NGRAM + 6])
            if not old:
                continue
            nl = left[-MAX_NGRAM - 1:]
            new = self.match_backward(nl)
            # Shared head noun: "olive instead of peanut OIL" -> new = "olive oil". Only when the
            # old ingredient is itself modifier + head (so "half and half instead of milk" is safe).
            head = old[0].split()[-1]
            for n in range(min(MAX_NGRAM - 1, len(nl)) if len(old[0].split()) >= 2 else 0, 0, -1):
                if "," in nl[-n:] or not nl[-n][0].isalpha():
                    continue
                c = canon(" ".join(nl[-n:] + [head]))
                if c and c in self.vocab and c != old[0] and (new is None or n >= new[2] - new[1]):
                    new = (c, len(nl) - n, len(nl))
                    break
            if not new:
                continue
            before_new = left[:len(left) - len(nl) + new[1]]
            if before_new and before_new[-1] in COORD_BEFORE:
                continue
            if not self.new_ok_before(before_new):
                continue
            if negated(before_new) or another_ingredient_follows(right[old[2]:]):
                continue
            emit(old[0], new[0], "X instead of Y", m.span())

        # P2: instead of OLD, (i|we) VERB NEW
        for m in re.finditer(rf"\binstead of ([a-z0-9 %'\-]+?),\s*(?:i|we)\s+{VERBS}\s+", s):
            old = self.match_forward(tokens(m.group(1)))
            rest = tokens(s[m.end():])
            new = self.match_forward(rest[:MAX_NGRAM + 4])
            if not old or not new:
                continue
            if old[2] != len(tokens(m.group(1))):
                continue  # old span must be exactly the ingredient (plus lead words)
            if another_ingredient_follows(rest[new[2]:]) or not self.new_ok_after(rest[new[2]:]):
                continue
            emit(old[0], new[0], "instead of Y, used X", m.span())

        # P3: (substituted|subbed|swapped|replaced|...) A (for|with) B
        for m in re.finditer(r"\b(substitut(?:ed|e|ing)|subb?(?:ed|ing)?|swapped(?: out)?|swap(?: out)?|"
                             r"replaced|replace|replacing)\s+", s):
            verb = m.group(1)
            if negated(tokens(s[:m.start()])):
                continue
            rest = tokens(s[m.end():])
            first = self.match_forward(rest[:MAX_NGRAM + 6])
            if not first:
                continue
            nxt = rest[first[2]:]
            if not nxt or nxt[0] not in ("for", "with"):
                continue
            second = self.match_forward(nxt[1:MAX_NGRAM + 7])
            if not second or second[1] != _lead_len(nxt[1:]):
                continue
            if another_ingredient_follows(nxt[1 + second[2]:]):
                continue
            prep = nxt[0]
            if verb.startswith("replac") and prep == "for":
                continue  # "replaced A for B": direction genuinely ambiguous
            if verb.startswith("sub") and prep == "for":
                # "substituted A for B" is often misused in reverse; when both are recipe
                # ingredients the direction cannot be recovered, so the mention is skipped.
                if self.resolve_old(first[0], recipe_id) and self.resolve_old(second[0], recipe_id):
                    continue
            new_is_second = verb.startswith(("replac", "swap")) or prep == "with"
            if new_is_second and not self.new_ok_after(nxt[1 + second[2]:]):
                continue
            if verb.startswith(("replac", "swap")):
                # replaced OLD with NEW ; swapped (out) OLD for/with NEW
                old_c, new_c = first[0], second[0]
            elif prep == "for":
                # substituted / subbed NEW for OLD
                old_c, new_c = second[0], first[0]
            else:
                # substituted / subbed OLD with NEW
                old_c, new_c = first[0], second[0]
            emit(old_c, new_c, f"{verb.split()[0]} A {prep} B", m.span())

        # P4: NEW as a (substitute|sub|replacement) for OLD
        for m in re.finditer(r"\bas a (?:substitute|sub|replacement) for\b", s):
            left, right = tokens(s[:m.start()]), tokens(s[m.end():])
            if not left or not right:
                continue
            new = self.match_backward(left[-MAX_NGRAM - 1:])
            old = self.match_forward(right[:MAX_NGRAM + 6])
            if not new or not old:
                continue
            nl = left[-MAX_NGRAM - 1:]
            before_new = left[:len(left) - len(nl) + new[1]]
            if (before_new and before_new[-1] in COORD_BEFORE) or negated(before_new):
                continue
            if not self.new_ok_before(before_new):
                continue
            if another_ingredient_follows(right[old[2]:]):
                continue
            emit(old[0], new[0], "X as a substitute for Y", m.span())
        return out


def _lead_len(toks: list[str]) -> int:
    n = 0
    while n < len(toks) and n < 6 and toks[n] in LEAD_SKIP:
        n += 1
    return n


SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+|\s*[!?]+\s*|\.\s|\(|\)")
CUES = re.compile(r"instead of|in place of|rather than|substitut|subb|swap|replac|as a sub")


def main() -> None:
    ing = pd.read_csv(paths.PROCESSED / "ingredients.csv")
    vocab = set(ing.ingredient)
    rec = pd.read_parquet(paths.PROCESSED / "recipes.parquet", columns=["recipe_id", "ingredients"])
    recipe_ings = {int(r): frozenset(i) for r, i in zip(rec.recipe_id, rec.ingredients)}
    miner = Miner(vocab, recipe_ings)

    rev = pd.read_csv(paths.RAW / "foodcom" / "RAW_interactions.csv")
    rev = rev[rev.review.notna()].reset_index(drop=True)
    rev["review_id"] = rev.index
    text = rev.review.map(lambda t: html.unescape(t).lower())
    has_cue = text.str.contains(CUES)
    print(f"{len(rev):,} reviews, {has_cue.sum():,} contain a substitution cue")

    rows = []
    for rid, uid, rcp, t in zip(rev.review_id[has_cue], rev.user_id[has_cue], rev.recipe_id[has_cue],
                               text[has_cue]):
        seen = set()
        for sent in SENT_SPLIT.split(t):
            if not sent or not CUES.search(sent):
                continue
            for old, new, pat, snip in miner.mine_sentence(sent, int(rcp)):
                if (old, new) in seen:
                    continue
                seen.add((old, new))
                rows.append((rid, uid, rcp, old, new, pat, snip))
    men = pd.DataFrame(rows, columns=["review_id", "user_id", "recipe_id", "source", "target",
                                      "pattern", "snippet"])
    men.to_csv(paths.PROCESSED / "substitution_mentions.csv", index=False)

    pairs = (men.groupby(["source", "target"])
             .agg(n_reviews=("review_id", "nunique"), n_users=("user_id", "nunique"),
                  n_recipes=("recipe_id", "nunique"), example=("snippet", "first"))
             .reset_index().sort_values(["n_reviews", "source"], ascending=[False, True]))
    pairs.to_csv(paths.PROCESSED / "substitution_pairs.csv", index=False)
    pairs.drop(columns="example").to_csv(paths.PROCESSED / "substitution_counts.csv", index=False)
    print(f"{len(men):,} accepted mentions -> {len(pairs):,} distinct pairs; "
          f"{(pairs.n_users >= 2).sum():,} pairs reported by >= 2 users")
    print(men.pattern.value_counts().to_string())


if __name__ == "__main__":
    main()
