"""Lexicon of magnitude words and the finder used throughout FrameCheck.

A magnitude word judges the size of a change: adverbs ("slightly"), adjectives on a change noun
("a modest increase"), and manner verbs ("surged"). Each surface form maps to a lemma, and each lemma
belongs to a class: mild, moderate, intense, or evaluative (words such as "strong" that carry no size norm).
"""
import re

ADV = {"slightly": "slight", "marginally": "marginal", "modestly": "modest", "moderately": "moderate",
       "somewhat": "somewhat", "significantly": "significant", "substantially": "substantial",
       "considerably": "considerable", "sharply": "sharp", "dramatically": "dramatic", "steeply": "steep",
       "minimally": "minimal", "materially": "material", "meaningfully": "meaningful", "markedly": "marked",
       "noticeably": "noticeable", "strongly": "strong", "massively": "massive", "tremendously": "tremendous",
       "remarkably": "remarkable", "sizably": "sizable"}
ADJ = {"slight": "slight", "marginal": "marginal", "modest": "modest", "moderate": "moderate",
       "significant": "significant", "substantial": "substantial", "considerable": "considerable",
       "sharp": "sharp", "dramatic": "dramatic", "steep": "steep", "small": "small", "minor": "minor",
       "large": "large", "minimal": "minimal", "material": "material", "meaningful": "meaningful",
       "marked": "marked", "noticeable": "noticeable", "massive": "massive", "tremendous": "tremendous",
       "remarkable": "remarkable", "sizable": "sizable", "sizeable": "sizable", "robust": "robust",
       "strong": "strong", "huge": "huge", "notable": "notable"}
VERB = {"surged": "v:surge", "soared": "v:soar", "jumped": "v:jump", "climbed": "v:climb", "rallied": "v:rally",
        "spiked": "v:spike", "skyrocketed": "v:skyrocket", "rocketed": "v:rocket", "leapt": "v:leap",
        "leaped": "v:leap", "plunged": "v:plunge", "plummeted": "v:plummet", "tumbled": "v:tumble",
        "slumped": "v:slump", "sank": "v:sink", "dived": "v:dive", "dove": "v:dive", "dipped": "v:dip",
        "slipped": "v:slip", "eased": "v:ease", "slid": "v:slide", "crashed": "v:crash", "tanked": "v:tank",
        "edged": "v:edge", "ticked": "v:tick", "inched": "v:inch", "crept": "v:creep", "ballooned": "v:balloon",
        "shrank": "v:shrink", "nudged": "v:nudge"}
MILD = {"slight", "marginal", "modest", "minimal", "small", "minor", "somewhat", "v:edge", "v:tick", "v:inch",
        "v:creep", "v:dip", "v:slip", "v:ease", "v:nudge"}
MODERATE = {"moderate", "noticeable", "v:climb", "v:slide"}
INTENSE = {"significant", "substantial", "considerable", "sharp", "dramatic", "steep", "large", "material",
           "meaningful", "marked", "massive", "tremendous", "remarkable", "sizable", "huge", "v:surge", "v:soar",
           "v:jump", "v:rally", "v:spike", "v:skyrocket", "v:rocket", "v:leap", "v:plunge", "v:plummet",
           "v:tumble", "v:slump", "v:sink", "v:dive", "v:crash", "v:tank", "v:balloon", "v:shrink"}
# evaluative words with no size norm in 10-K usage; counted but never flagged
EVALUATIVE = {"robust", "strong", "notable"}
LEX = {**ADV, **ADJ, **VERB}
TOKEN = re.compile(r"[A-Za-z]+")
SKIP = {"year", "over", "period", "quarter", "on", "and", "sequential", "annual", "overall"}
CHANGE_NOUNS = {"increase", "increases", "decrease", "decreases", "decline", "declines", "drop", "drops", "rise",
                "rises", "growth", "gain", "gains", "improvement", "improvements", "reduction", "reductions", "jump",
                "change", "changes", "fall", "uptick", "downturn", "deterioration", "expansion", "contraction",
                "surge", "upswing", "boost", "loss", "losses", "shift", "swing", "climb", "dip", "slowdown",
                "acceleration", "rebound", "recovery", "erosion", "contribution", "advance", "upturn", "slide"}
# "significant portion of", "material adverse", "small business" etc. do not grade a change
NOT_GRADING_NEXT = {"portion", "part", "amount", "number", "majority", "share", "adverse", "business", "businesses",
                    "customers", "customer", "weakness", "weaknesses", "than", "below", "above", "factor", "factors",
                    "risk", "risks", "judgment", "estimates", "accounting", "competition", "uncertainty"}


def word_class(lemma):
    if lemma in MILD:
        return "mild"
    if lemma in MODERATE:
        return "moderate"
    if lemma in INTENSE:
        return "intense"
    if lemma in EVALUATIVE:
        return "evaluative"
    return "other"


def find_words(sentence):
    spans = list(TOKEN.finditer(sentence))
    toks = [m.group(0) for m in spans]
    out = []
    for i, t in enumerate(toks):
        lemma = LEX.get(t.lower())
        if not lemma:
            continue
        if i and toks[i - 1].lower() in {"no", "not", "without", "nor", "any"}:
            continue  # "no significant change" grades nothing
        # a negation up to three words earlier in the same clause ("did not increase significantly")
        clause = re.split(r"[,.;:!?]", sentence[:spans[i].start()])[-1].lower().split()
        if any(w in {"no", "not", "without", "nor", "never"} or w.endswith("n't") for w in clause[-3:]):
            continue
        if t.lower() in ADJ:
            # the adjective must grade a change noun right after it, allowing one modifier and time phrases
            # ("a slight increase", "a modest year-over-year increase", "a significant revenue increase")
            nxt = [x.lower() for x in toks[i + 1:i + 8] if x.lower() not in SKIP]
            prev = toks[i - 1].lower() if i else ""
            if prev in {"raw", "than"} or not nxt or nxt[0] in NOT_GRADING_NEXT or \
                    not any(x in CHANGE_NOUNS for x in nxt[:2]):
                continue
        out.append(lemma)
    return out


