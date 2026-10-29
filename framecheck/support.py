"""Step 2 for causes: is a cause clause supported by the input?

A cause clause starts with a causal connective (see edit.CAUSE_CUT). When the input gives causes (e.g., analyst
notes), the clause is kept if most of its stemmed content words occur in the input: the share must reach TAU,
which was fit on drafts of the reference firms. Without such an input, every cause clause is unsupported.
"""
import re

TAU = 0.58

WORD = re.compile(r"\w+(?:[.,'’]\w+)*")
STOP = set("""a an the and or of to in on for by at as with from that which this these those its their our his her it is are was
were be been being has have had due driven primarily mainly largely partly partially mostly principally attributable result
results resulting reflecting reflects reflected owing because caused led thanks fueled attributed indicating highlighting
underscoring signaling suggesting increase increased increases decrease decreased decreases higher lower compared prior year
years fiscal period change changes company s we us also well other such than more less over during while""".split())


def tokens(text):
    return [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(text)]


CONN = re.compile(r"^\s*(?:,|;|\sand)?\s*(?:(?:which|that)\s+(?:(?:can|could|may|might)\s+be|was|were|is|are)\s+)?", re.I)


def cause_text(text, start):
    return CONN.sub("", text[start:]).strip().rstrip(".")


def stem(w):
    w = w.lower()
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)][:6]
    return w[:6]


def lex_support(span_text, premise):
    """Share of the cause's content words (stemmed) that occur in the input."""
    cw = [stem(t) for _, _, t in tokens(span_text) if t.lower() not in STOP and not t[0].isdigit()]
    if not cw:
        return 1.0
    ps = {stem(t) for _, _, t in tokens(premise)}
    return sum(c in ps for c in cw) / len(cw)
