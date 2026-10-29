"""Extract (magnitude expression, actual change) pairs from 10-K MD&A text.

For each change word ("increased", "a decline", ...) we record the modifier that
grades it ("slightly", "a significant", or none) and recover the size of the
change from the clause that follows:
  pct    - stated percentage change      "decreased slightly by $0.5 million, or 3%"
  amount - two amounts, change computed  "increased modestly to $16.2 million from $15.8 million"
  delta  - change amount plus new level  "increased slightly by $0.4 million to $16.2 million"
  pp     - ratio moving between levels   "increased slightly to 53% ... compared to 52%"
For amounts we also keep the interval implied by rounding of the reported figures
("$0.5 million to $0.6 million" is anything from +8% to +38%).
Pairs whose sign disagrees with the change word are dropped.

usage: python extract_pairs.py <out.jsonl> [--glob 'data/raw/*/*.jsonl'] [--section section_7]
"""
import argparse
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ADVERBS = {
    "slightly": "slight", "marginally": "marginal", "modestly": "modest",
    "moderately": "moderate", "somewhat": "somewhat", "significantly": "significant",
    "substantially": "substantial", "considerably": "considerable",
    "sharply": "sharp", "dramatically": "dramatic", "steeply": "steep",
    "minimally": "minimal", "materially": "material", "meaningfully": "meaningful",
}
ADJECTIVES = {
    "slight": "slight", "marginal": "marginal", "modest": "modest",
    "moderate": "moderate", "significant": "significant", "substantial": "substantial",
    "considerable": "considerable", "sharp": "sharp", "dramatic": "dramatic",
    "steep": "steep", "small": "small", "minor": "minor", "large": "large",
    "minimal": "minimal", "material": "material", "meaningful": "meaningful",
}
LEXICON = set(ADVERBS) | set(ADJECTIVES)
# words right after the modifier/change that mean the modifier is not grading this change
NOT_MAGNITUDE_NEXT = {"offset", "than", "below", "above", "less", "more", "lower", "higher"}
UP_VERBS = {"increased", "increases", "grew", "rose", "improved", "gained"}
DOWN_VERBS = {"decreased", "decreases", "declined", "declines", "fell", "dropped", "reduced"}
UP_NOUNS = {"increase", "growth", "rise", "improvement", "gain"}
DOWN_NOUNS = {"decrease", "decline", "drop", "reduction"}
UP = UP_VERBS | UP_NOUNS
DOWN = DOWN_VERBS | DOWN_NOUNS
CHANGE = UP | DOWN
# manner-of-change verbs carry magnitude themselves ("edged up", "surged", "plunged")
MANNER = {"surged": "surge", "soared": "soar", "jumped": "jump", "climbed": "climb", "rallied": "rally",
          "spiked": "spike", "skyrocketed": "skyrocket", "rocketed": "rocket", "leapt": "leap", "leaped": "leap",
          "plunged": "plunge", "plummeted": "plummet", "tumbled": "tumble", "slumped": "slump", "sank": "sink",
          "dived": "dive", "dove": "dive", "dipped": "dip", "slipped": "slip", "eased": "ease", "slid": "slide",
          "crashed": "crash", "tanked": "tank", "edged": "edge", "ticked": "tick", "inched": "inch", "crept": "creep"}
MANNER_UP = {"surge", "soar", "jump", "climb", "rally", "spike", "skyrocket", "rocket", "leap"}
PARTICLE_VERBS = {"edge", "tick", "inch", "creep"}  # direction comes from up/higher/down/lower

WORD = re.compile(r"[A-Za-z]+")
# thousands separators must group by three, so "3,7%" (decimal comma) is not read as 37
NUM = r"\(?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?"
UNIT_PAT = r"(?:\s*(?:thousand|million|billion|[mMbBkK])\b)?"
MONEY_PAT = r"\$\s?" + NUM + UNIT_PAT
MONEY = re.compile(r"\$\s?(" + NUM + r")(?:\s*(thousand|million|billion|[mMbBkK])\b)?")
SCALE = {"": 1, "thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9}
SENT_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])|\n+")
RATIO_CONTEXT = re.compile(r"as a percent|percentage of|percent of|as a % of|\bmargin\b|\brate\b|\bratio\b", re.I)
TAIL_CHARS = 220   # how far after the change word to look for numbers
SUBJECT_WORDS = 10


def num(s):
    neg = s.startswith("(")
    v = float(s.strip("()").replace(",", ""))
    return -v if neg else v


def money(text):
    """First $ amount in text -> (value, rounding half-width) or None."""
    m = MONEY.search(text)
    if not m:
        return None
    raw = m.group(1).strip("()")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    scale = SCALE[(m.group(2) or "").lower()]
    return num(m.group(1)) * scale, 0.5 * 10 ** -decimals * scale, bool(m.group(2))


def rel_change(a, b):
    """% change from a to b plus the [lo, hi] range consistent with rounding."""
    (av, ae, au), (bv, be, bu) = a, b
    if av - ae <= 0:
        return None
    if au != bu and (bv if au else av) < 1e3:  # "$8.4" vs "$8.9 million": a dropped unit
        return None
    lo = 100 * ((bv - be) - (av + ae)) / (av + ae)
    hi = 100 * ((bv + be) - (av - ae)) / (av - ae)
    return 100 * (bv - av) / av, lo, hi


def read_change(tail, direction, lookahead=""):
    """Return (kind, value, lo, hi) for the first recoverable change in `tail`, else None."""
    sign = 1 if direction == "up" else -1
    lt = tail.lower()
    # pp: ratio levels "to 53% ... from/compared to 52%", "(26.6% in 2017 vs. 27.1% in 2016)"
    m = (re.search(r"\bto\s+(" + NUM + r")\s?%.{0,120}?\b(?:from|compared (?:to|with)|versus|vs\.?)\s+(" + NUM + r")\s?%", lt)
         or re.search(r"^[^%$]{0,25}?\(\s*(" + NUM + r")\s?%[^%]{0,60}?\b(?:versus|vs\.?|compared (?:to|with))\s+(" + NUM + r")\s?%", lt)
         or re.search(r"^[^%$]{0,40}?\(\s*(" + NUM + r")\s?%\s*\)[^%]{0,40}?\bcompared (?:to|with)[^%(]{0,30}\(\s*(" + NUM + r")\s?%", lt))
    if m:
        v = num(m.group(1)) - num(m.group(2))
        return "pp", v, v, v
    # pct: "by 3%", "or 3%", "an increase of 3%", "(3%)", "increased 3%"
    m = re.search(r"(\bby|\bor|\bof|\(|^\s*)\s*(" + NUM + r")\s?(?:%|percent\b|pct\b)", lt)
    if m:
        after = lt[m.end():] + lookahead.lower()
        # "30% of revenues", "(37.9% for 2001 versus 40.3%)" are levels; "rose 5% in 2019" is a change
        share = re.match(r"\s*\)?\s*(?:of\b|versus|vs|\)?\s*as compared)", after) or \
            (m.group(1) == "(" and re.match(r"\s*\)?\s*(?:in|for)\b", after))
        other = re.match(r"\s*\)?,?\s*(?:increase|decrease|reduction|decline|growth|rise|drop|improvement)s?\s+in\b", after)
        level = re.search(r"\b(?:to|from|at)\s*$", lt[:m.start()])
        # "of 3%" only counts right after the change noun ("an increase of 3%")
        loose_of = m.group(1) == "of" and lt[:m.start()].strip() != ""
        if not (share or other or level or loose_of):
            v = sign * abs(num(m.group(2)))
            return "pct", v, v, v
    # amount: "from $A to $B"
    m = re.search(r"\bfrom\s+(" + MONEY_PAT + r").{0,40}?\bto\s+(" + MONEY_PAT + ")", tail)
    if m:
        a, b = money(m.group(1)), money(m.group(2))
        if a and b and (r := rel_change(a, b)):
            return ("amount",) + r
    # amount: "to $B ... compared to / from $A"
    m = re.search(r"\bto\s+(" + MONEY_PAT + r").{0,120}?\b(?:compared (?:to|with)|from)\s+(" + MONEY_PAT + ")", tail)
    if m:
        b, a = money(m.group(1)), money(m.group(2))
        if a and b and (r := rel_change(a, b)):
            return ("amount",) + r
    # delta: "by $D ... to $B"
    m = re.search(r"\bby\s+(" + MONEY_PAT + r").{0,40}?(?<!compared )\bto\s+(" + MONEY_PAT + ")", tail)
    if m:
        d, b = money(m.group(1)), money(m.group(2))
        if d and b:
            a = (b[0] - sign * abs(d[0]), b[1] + d[1], b[2])
            if (r := rel_change(a, b)):
                return ("delta",) + r
    return None


def tail_end(sent, words, i, e):
    """Stop the tail at the next change word so numbers belong to this event."""
    end = min(len(sent), e + TAIL_CHARS)
    for k in range(i + 1, len(words)):
        s2, w2 = words[k][0], words[k][2]
        if s2 >= end:
            break
        new_clause = w2 in {"higher", "lower"} and words[k - 1][2] in {"was", "were", "is", "are"}
        if (w2 in CHANGE or w2 in MANNER or new_clause) and s2 - e > 15:
            return s2
    return end


def manner_pair(sent, words, i):
    s, e, w = words[i]
    lemma = MANNER[w]
    if lemma in PARTICLE_VERBS:
        nxt = words[i + 1][2] if i + 1 < len(words) else ""
        if nxt not in {"up", "higher", "down", "lower"}:
            return []
        direction = "up" if nxt in {"up", "higher"} else "down"
        e = words[i + 1][1]
    else:
        direction = "up" if lemma in MANNER_UP else "down"
    end = tail_end(sent, words, i, e)
    got = read_change(sent[e:end], direction, sent[end:end + 40])
    if got is None:
        return []
    kind, val, lo, hi = got
    if val == 0 or (val > 0) != (direction == "up") or abs(val) > (100 if kind == "pp" else 1000):
        return []
    subj_start = words[max(0, i - SUBJECT_WORDS)][0]
    return [{"modifier": "v:" + lemma, "surface": w, "change_word": w, "direction": direction, "kind": kind,
             "value": round(val, 4), "lo": round(lo, 4), "hi": round(hi, 4),
             "subject": sent[subj_start:s].strip(), "span": [s, end]}]


def pairs_from_sentence(sent):
    words = [(m.start(), m.end(), m.group().lower()) for m in WORD.finditer(sent)]
    out = []
    for i, (s, e, w) in enumerate(words):
        if w in MANNER:
            out += manner_pair(sent, words, i)
            continue
        if w not in CHANGE:
            continue
        # adverb right before or after a verb ("slightly increased", "increased slightly"),
        # adjective right before a noun ("a slight increase")
        mod, mj = None, i
        if w in UP_VERBS | DOWN_VERBS:
            for j in (i - 1, i + 1):
                if 0 <= j < len(words) and words[j][2] in ADVERBS and \
                        sent[min(words[j][1], e):max(words[j][0], s)].strip() == "":
                    mod, mj = words[j][2], j
                    break
        elif i > 0 and words[i - 1][2] in ADJECTIVES:
            mod, mj = words[i - 1][2], i - 1
        if mod is None:
            # unmodified change: no lexicon word anywhere near, so it cannot be a parse miss
            near = {words[j][2] for j in range(max(0, i - 3), min(len(words), i + 4))}
            if near & LEXICON:
                continue
        nxt = words[mj + 1][2] if mj + 1 < len(words) else ""
        nxt2 = words[i + 1][2] if i + 1 < len(words) else ""
        if nxt in NOT_MAGNITUDE_NEXT or nxt2 in NOT_MAGNITUDE_NEXT:
            continue
        # stop the tail at the next change word so numbers belong to this event
        end = min(len(sent), e + TAIL_CHARS)
        for k in range(i + 1, len(words)):
            s2, w2 = words[k][0], words[k][2]
            if s2 >= end:
                break
            new_clause = w2 in {"higher", "lower"} and words[k - 1][2] in {"was", "were", "is", "are"}
            if (w2 in CHANGE or new_clause) and s2 - e > 15:
                end = s2
                break
        direction = "up" if w in UP else "down"
        got = read_change(sent[e:end], direction, sent[end:end + 40])
        if got is None:
            continue
        kind, val, lo, hi = got
        if kind == "pct" and RATIO_CONTEXT.search(sent[max(0, s - 80):end]):
            kind = "pp"  # "decreased slightly (0.42%) as a percentage of sales" is a ratio change
        if kind != "pp" and (val == 0 or (val > 0) != (direction == "up") or abs(val) > 1000):
            continue  # sign mismatch or garbage
        if kind == "pp" and (val == 0 or (val > 0) != (direction == "up") or abs(val) > 100):
            continue
        subj_start = words[max(0, min(i, mj) - SUBJECT_WORDS)][0]
        out.append({
            "modifier": "none" if mod is None else (ADVERBS.get(mod) or ADJECTIVES[mod]),
            "surface": mod or "", "change_word": w, "direction": direction, "kind": kind,
            "value": round(val, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "subject": sent[subj_start:words[min(i, mj)][0]].strip(),
            "span": [words[min(i, mj)][0], end],
        })
    return out


def process_file(args):
    path, section = args
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get(section) or ""
            for sent in SENT_SPLIT.split(text):
                sent = " ".join(sent.split())
                if not 20 < len(sent) < 800:
                    continue
                for p in pairs_from_sentence(sent):
                    p.update(cik=d.get("cik"), year=d.get("year"), doc=d.get("filename"), sentence=sent)
                    rows.append(p)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--glob", default="data/raw/*/*.jsonl")
    ap.add_argument("--section", default="section_7")
    args = ap.parse_args()
    files = sorted(Path(".").glob(args.glob))
    with Pool(min(96, len(files))) as pool:
        results = pool.map(process_file, [(f, args.section) for f in files])
    rows = [r for rs in results for r in rs]
    seen, n = set(), 0
    with open(args.out, "w") as f:
        for r in rows:
            key = (r["doc"], r["sentence"], r["span"][0])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"{n} pairs from {len(files)} files -> {args.out}")
    kept = [r for r in rows if r["modifier"] != "none"]
    print("modified:", len(kept), Counter(r["kind"] for r in kept))
    for k, v in Counter(r["modifier"] for r in rows).most_common():
        print(k, v)


if __name__ == "__main__":
    main()
