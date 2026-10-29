"""Score drafts and print the main results (shares of drafts in %, mean over models).

Rows
  Plain prompt            the main instruction alone
  + Norm card             the instruction with the usage norms of eight common words (prompt baseline)
  + Style instruction     the instruction with the style sentence (prompt baseline)
  + Faithful instruction  the instruction with "add nothing beyond the input" (prompt baseline)
  Template                a fixed sentence built from the input fields, no model
  FrameCheck              size-matched retrieval (step 1), then the cause cut and the rule edit (steps 2-3)
  FrameCheck (style)      the style sentence instead of retrieval, then steps 2-3

usage: python scripts/evaluate.py [--drafts data/drafts/drafts_main.jsonl.gz] [--items data/draft_items.jsonl]
Drafts are JSON lines {"id", "model", "cond", "drafts": [...]}, as written by scripts/draft.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framecheck import UsageNorms, cut_cause, edit  # noqa: E402
from framecheck.metrics import gap, invented, score_draft  # noqa: E402

ROWS = [("Plain prompt", "p0", False), ("+ Norm card", "norms", False), ("+ Style instruction", "style", False),
        ("+ Faithful instruction", "faith", False), ("Template", "template", False),
        ("FrameCheck", "retrieve_nat", True), ("FrameCheck (style)", "style", True)]
COLS = {"graded": "Judged", "misfit": "Misfit", "cause": "Cause", "placeholder": "Placeholder",
        "invented": "Invented", "number": "Number"}
WRITERS_JUDGED = 1.5  # judgment rate of 10-K writers on 28K test-firm sentences with a stated change


def framecheck_edit(norms, d, size):
    """Steps 2-3 for a single drafted sentence: the input gives no cause, so a cause clause is cut, and then each
    misfit magnitude word is deleted or replaced."""
    cut = cut_cause(d)
    return edit(cut, [c["lemma"] for c in norms.check(cut, size) if c["flag"]])


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--drafts", default=root / "data/drafts/drafts_main.jsonl.gz")
    ap.add_argument("--items", default=root / "data/draft_items.jsonl")
    args = ap.parse_args()
    norms = UsageNorms()
    items = pd.read_json(args.items, lines=True).set_index("id")
    drafts = pd.read_json(args.drafts, lines=True)

    rows = []
    for i, r in items.iterrows():  # template baseline: no model
        t = f"{r['subject']} {'increased' if r['value'] > 0 else 'decreased'} {r['abs']:.1f}% in fiscal {r['year']} compared with fiscal {r['year'] - 1}."
        sc = score_draft(norms, t, r["abs"], r["direction"]); sc["invented"] = invented(t)
        rows.append(dict(sc, row="Template", model="template"))
    for name, cond, ours in ROWS:
        for x in drafts[drafts.cond == cond].itertuples():
            r = items.loc[x.id]
            for d in x.drafts:
                if ours:
                    d = framecheck_edit(norms, d, r["abs"])
                sc = score_draft(norms, d, r["abs"], r["direction"])
                sc["invented"] = invented(d)
                rows.append(dict(sc, row=name, model=x.model))
    df = pd.DataFrame(rows)
    # mean per model first, then over models
    t = df.groupby(["row", "model"])[list(COLS)].mean().groupby("row").mean() * 100
    t = t.reindex([name for name, _, _ in ROWS]).rename(columns=COLS)
    t["Gap"] = [gap(j, c, p, WRITERS_JUDGED) for j, c, p in zip(t.Judged, t.Cause, t.Placeholder)]
    t["Models"] = df.groupby("row").model.nunique()
    pd.set_option("display.width", 200)
    print(f"10-K writers: Judged {WRITERS_JUDGED}")
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()
