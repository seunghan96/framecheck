"""Usage norms: the range of change sizes on which 10-K writers use each magnitude word.

A word is flagged as
  inflated  if |change| < q10 of the sizes writers use it for (intense and moderate words),
  deflated  if |change| > q90 of the sizes writers use it for (mild and moderate words).
An intense word on a very large change, or a mild word on a tiny change, is not a misfit.
Lemmas with fewer than 30 writer uses fall back to the norms of their class (mild / moderate / intense), and an
intense word is never licensed below the lower bound of the intense class.
"""
from pathlib import Path

import pandas as pd

from .lexicon import find_words, word_class

RESOURCES = Path(__file__).resolve().parent / "resources"


class UsageNorms:
    def __init__(self, norms=RESOURCES / "norms.tsv", class_norms=RESOURCES / "norms_class.tsv", lo="0.1", hi="0.9"):
        n = pd.read_csv(norms, sep="\t", index_col=0)
        c = pd.read_csv(class_norms, sep="\t", index_col=0)
        self.norm = {w: (r[lo], r["0.5"], r[hi]) for w, r in n.iterrows()}
        self.cnorm = {w: (r[lo], r["0.5"], r[hi]) for w, r in c.iterrows()}
        # an intense word is never licensed below the floor of the intense class, whatever its own (noisier) floor
        for w, (a, m, b) in list(self.norm.items()):
            if word_class(w) == "intense":
                self.norm[w] = (max(a, self.cnorm["intense"][0]), m, b)

    def range_of(self, lemma):
        if lemma in self.norm:
            return self.norm[lemma]
        return self.cnorm.get(word_class(lemma))

    def flag(self, lemma, size):
        """'inflated', 'deflated' or None for one magnitude word on a change of |size| %."""
        cls = word_class(lemma)
        rng = self.range_of(lemma) if cls != "evaluative" else None
        if rng is None:
            return None
        if size < rng[0] and cls in ("intense", "moderate"):
            return "inflated"
        if size > rng[2] and cls in ("mild", "moderate"):
            return "deflated"
        return None

    def check(self, sentence, size):
        """size: |relative change| in %. Returns a list of dicts, one per magnitude word found."""
        return [{"lemma": w, "cls": word_class(w), "flag": self.flag(w, size), "range": self.range_of(w)}
                for w in find_words(sentence)]

    def advice(self, lemma):
        lo, mid, hi = self.range_of(lemma)
        word = lemma.replace("v:", "")
        return (f'In 10-K filings, "{word}" is typically used for changes between {lo:g}% and {hi:g}% '
                f"(median {mid:g}%).")

    def fitting(self, size, words=("marginal", "slight", "modest", "moderate", "significant", "substantial",
                                   "sharp", "dramatic")):
        """Common words whose writer range covers this size (mild words have no lower bound, intense no upper)."""
        out = []
        for w in words:
            lo, _, hi = self.range_of(w)
            cls = word_class(w)
            if (cls == "mild" and size <= hi) or (cls == "intense" and size >= lo) or (lo <= size <= hi):
                out.append(w)
        return out

