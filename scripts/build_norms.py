"""Build the usage norms from 10-K filings (reference firms only).

1. Download the MD&A sections of EDGAR-CORPUS (Loukas et al., 2021), e.g. from the Hugging Face dataset
   `eloukas/edgar-corpus`, as JSON lines with the fields filename, cik, year, section_7, under data/raw/<year>/.
2. Extract (magnitude word, size of change) pairs:
       python scripts/extract_pairs.py data/pairs.jsonl --glob 'data/raw/*/*.jsonl'
3. Split firms into reference (70%) and test (30%) firms, and compute the norms on the reference firms:
       python scripts/build_norms.py data/pairs.jsonl --out framecheck/resources
   writes norms.tsv (per lemma, quantiles of |relative change| in %, lemmas with at least 30 uses),
   norms_class.tsv (per class: mild / moderate / intense), and test_firms.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framecheck.lexicon import word_class  # noqa: E402

MAIN = ["marginal", "slight", "modest", "moderate", "significant", "substantial", "sharp", "dramatic"]
Q = [.05, .1, .25, .5, .75, .9, .95]


def test_firms(df, seed=0):
    """30% of the firms, drawn once with a fixed seed; all evaluation items come from these firms."""
    rng = np.random.default_rng(seed)
    d = df[df.modifier.isin(MAIN + ["none"])].copy()
    d["abs"] = d.value.abs()
    reliable = (d.kind != "amount") | ((d.hi - d.lo).abs() < 0.5 * d["abs"])
    d = d[reliable & (d.kind != "delta")]
    firms = d.cik.astype(str).unique()
    return set(rng.choice(firms, size=int(0.3 * len(firms)), replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--out", default="framecheck/resources")
    args = ap.parse_args()
    df = pd.read_json(args.pairs, lines=True)
    tf = test_firms(df)
    # relative changes from a stated percentage or two amounts precise enough to compute it
    df = df[df.kind.isin(["pct", "amount"])].copy()
    df["abs"] = df.value.abs()
    df = df[(df.kind != "amount") | ((df.hi - df.lo).abs() < 0.5 * df["abs"])]
    df = df[(df["abs"] > 0) & (df["abs"] < 1000)]
    df["test"] = df.cik.astype(str).isin(tf)

    pool = df[~df.test & (df.modifier != "none")].drop_duplicates("sentence")
    q = pool.groupby("modifier")["abs"].quantile(Q).unstack()
    q["n"] = pool.groupby("modifier").size()
    q = q[q.n >= 30].round(2)
    # class-level norms for lemmas too rare to have their own
    pool = pool.assign(cls=pool.modifier.map(word_class))
    c = pool.groupby("cls")["abs"].quantile(Q).unstack()
    c["n"] = pool.groupby("cls").size()
    out = Path(args.out)
    q.to_csv(out / "norms.tsv", sep="\t")
    c.round(2).to_csv(out / "norms_class.tsv", sep="\t")
    (out / "test_firms.txt").write_text("\n".join(sorted(tf)) + "\n")
    print(q)
    print(c.round(2))


if __name__ == "__main__":
    main()
