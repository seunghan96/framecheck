"""Draft one MD&A sentence per test change with an open LLM (vLLM), for the main conditions.

conditions
  p0            plain: the main instruction alone
  norms         prompt baseline: the norm card of eight common words before the input
  style         the style sentence after the instruction (FrameCheck (style) before its edit)
  retrieve_nat  FrameCheck step 1: six size-matched 10-K sentences of other firms before the input
  summ          summarize the writer's own sentence for an investor update (at most 15 words)

Steps 2-3 of FrameCheck need no model call and are applied in scripts/evaluate.py.

usage: python scripts/draft.py <hf_model> <tag> [--conds p0,norms,style,retrieve_nat,summ] [--tp N]
writes data/drafts/draft_<tag>.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from framecheck import UsageNorms  # noqa: E402
from framecheck.prompts import ASK, STYLE, SUMM, SYSTEM, fact, norm_card, retrieval_block  # noqa: E402
from framecheck.retrieval import SizeMatchedRetriever  # noqa: E402

N = 4  # drafts per item
EXTRA_TOK = int(os.environ.get("EXTRA_TOK", 0))  # extra budget for reasoning models (gpt-oss)


def clean(t):
    """First sentence-like line of the reply, without a preface line or surrounding quotes."""
    t = t.split("assistantfinal")[-1] if "assistantfinal" in t else t  # gpt-oss: drop the analysis channel
    lines = [x.strip() for x in t.strip().split("\n") if x.strip()]
    if len(lines) > 1 and lines[0].endswith(":"):
        lines = lines[1:]
    return lines[0].strip('"“”* ') if lines else ""


def main():
    from vllm import LLM, SamplingParams
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tag")
    ap.add_argument("--conds", default="p0,norms,style,retrieve_nat,summ")
    ap.add_argument("--items", default=ROOT / "data/draft_items.jsonl")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    conds = args.conds.split(",")
    items = pd.read_json(args.items, lines=True)
    if args.limit:
        items = items.sample(args.limit, random_state=0)
    card = norm_card(UsageNorms())
    retriever = SizeMatchedRetriever() if "retrieve_nat" in conds else None

    def prompt(r, c):
        f = fact(r.subject, r.year, r.value)
        return {"p0": lambda: f + ASK,
                "norms": lambda: card + f + ASK,
                "style": lambda: f + ASK + STYLE,
                "retrieve_nat": lambda: retrieval_block(retriever.retrieve(r.value)) + f + ASK,
                "summ": lambda: SUMM.format(sentence=r.sentence)}[c]()

    llm = LLM(args.model, max_model_len=2048, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_util,
              trust_remote_code=True, seed=0)
    tok = llm.get_tokenizer()

    def chat(q):
        name = args.model.lower()
        kw = {"enable_thinking": False} if "qwen3" in name else ({"reasoning_effort": "low"} if "gpt-oss" in name else {})
        try:
            return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True, **kw)
        except Exception:  # models without a system role
            return tok.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + q}], tokenize=False,
                                           add_generation_prompt=True, **kw)

    sample = SamplingParams(n=N, temperature=0.7, top_p=0.95, max_tokens=160 + EXTRA_TOK, seed=0)
    jobs = [(r, c) for r in items.itertuples() for c in conds]
    outs = llm.generate([chat(prompt(r, c)) for r, c in jobs], sample)
    out = ROOT / f"data/drafts/draft_{args.tag}.jsonl"
    with open(out, "w") as f:
        for (r, c), o in zip(jobs, outs):
            f.write(json.dumps({"id": int(r.id), "model": args.tag, "cond": c,
                                "drafts": [clean(x.text) for x in o.outputs]}) + "\n")
    print("wrote", out, len(jobs))


if __name__ == "__main__":
    main()
