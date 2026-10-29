"""Metrics of the paper, scored per text (verbatim from the evaluation code).

  graded (Judged)  a mild, moderate, or intense magnitude word (evaluative words are not counted)
  misfit           a word flagged as inflated or deflated against its usage norm
  cause            a causal connective (the drafting input gives no cause)
  placeholder      a slot such as "[reason]" or "$X"
  invented         an amount the input does not give (see `invented`)
  number           the size of the change as given, up to rounding
Each metric is a share over texts: average per model first, then over models.
"""
import re

from .lexicon import find_words, word_class

UP = re.compile(r"\b(increas|rose|rise|rising|grew|grow|higher|up\b|improv|gain|climb|jump|surg|soar|expan|rall|"
                r"spik|skyrocket|edged up|inched up|ticked up|advanc|boost)", re.I)
DOWN = re.compile(r"\b(decreas|declin|fell|fall|lower|down\b|drop|reduc|plung|plummet|slip|dip|contract|shr[ai]nk|"
                  r"tumbl|slump|eas(ed|ing)|sank|sink|slid|deteriorat|weaken|loss)", re.I)
NUMS = re.compile(r"\d+(?:\.\d+)?")
ANYNUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def any_number(t):
    """True if the text states any number other than a year."""
    for x in ANYNUM.findall(t):
        x = x.replace(",", "").rstrip(".")
        if not (len(x) == 4 and x.isdigit() and 1900 <= int(x) <= 2100):
            return True
    return False

CAUSE = re.compile(r"\b(due to|driven by|primarily|attributable to|as a result of|reflect(?:s|ing|ed)?|"
                   r"result(?:ed|ing)? from|owing to|because|caused by|led by|thanks to|fueled by)\b", re.I)
AMOUNT = re.compile(r"\$\s?\d|\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)\b|\b\d{1,3}(?:,\d{3})+\b", re.I)

def numbers(t):
    """Non-year numbers in a text, as floats."""
    out = set()
    for x in ANYNUM.findall(t):
        x = x.replace(",", "").rstrip(".")
        if len(x) == 4 and x.isdigit() and 1900 <= int(x) <= 2100:
            continue
        try:
            out.add(float(x))
        except ValueError:
            pass
    return out


SCALED = re.compile(r"(?<![QqHh])(\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand|bn|mm|m|k|b)?\b", re.I)
UNIT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mm": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}


def scaled_numbers(t):
    out = []
    for x, u in SCALED.findall(t):
        x = x.replace(",", "").rstrip(".")
        if not u and len(x) == 4 and x.isdigit() and 1900 <= int(x) <= 2100:
            continue
        try:
            out.append(float(x) * UNIT.get(u.lower(), 1.0))
        except ValueError:
            pass
    return out


def invented_number(text, source, size):
    """True if the text states a number that matches no number of the source (up to units and rounding)."""
    base = scaled_numbers(source)
    # numbers derived from the source (differences of two amounts) are not invented
    src = base + [abs(a - b) for a in base for b in base if a != b] + [size]
    for v in scaled_numbers(text):
        if v == 0:
            continue
        ok = any(u and abs(v - u * f) <= 0.012 * abs(u * f) + 0.051 for u in src for f in (1, 1e3, 1e-3, 1e6, 1e-6))
        if not ok:
            return True
    return False


PLACEHOLDER = re.compile(r"\[[^\]]*\]|\$X\b|\bX million", re.I)


def score_draft(mc, d, a, direction):
    """Score text d about a change of |a| % in `direction` ("up"/"down"); mc is a UsageNorms."""
    words = find_words(d)
    cls = [word_class(w) for w in words]
    flags = [f["flag"] for f in mc.check(d, a)]
    nums = [float(x) for x in NUMS.findall(d.replace(",", ""))]
    good_dir = (UP if direction == "up" else DOWN).search(d) is not None
    number = any(abs(x - a) <= max(0.051, 0.006 * a) or round(x) == round(a) and abs(x - a) < 1 for x in nums)
    graded = any(c in ("mild", "moderate", "intense") for c in cls)
    return {"graded": graded, "word_only": graded and not any_number(d),
            "misfit": ("inflated" in flags) or ("deflated" in flags),
            "mild": "mild" in cls, "intense": "intense" in cls, "evalw": "evaluative" in cls,
            "inflated": "inflated" in flags, "deflated": "deflated" in flags,
            "number": number, "cause": CAUSE.search(d) is not None, "placeholder": PLACEHOLDER.search(d) is not None,
            "dir_ok": good_dir, "words": words}




def invented(text, source=None, size=None):
    """Drafting: the input gives only a percentage, so any amount is invented.
    Summaries (or inputs with notes): a number that matches no number of the source is invented."""
    if source is None:
        return AMOUNT.search(text) is not None
    return invented_number(text, source, size)


def gap(judged, cause, placeholder, writers=1.5):
    """Overall gap (in %-points): distance to the writers' judgment rate plus the shares with a cause or a placeholder."""
    return abs(judged - writers) + cause + placeholder
