"""FrameCheck: keep an LLM-drafted financial text within its input.

  Step 1 (before drafting)  add size-matched 10-K sentences of other firms to the prompt
  Step 2 (after drafting)   check each judgment word against its usage norm, and each cause against the input
  Step 3 (after drafting)   edit only the flagged spans by rule, keeping the number; no model call
"""
from .edit import CAUSE_CUT, cut_cause, edit
from .norms import UsageNorms
from .prompts import ASK, NOTES, STYLE, fact, retrieval_block
from .support import TAU, cause_text, lex_support


class FrameCheck:
    """FrameCheck(style=False) uses retrieval; FrameCheck(style=True) is the lightweight variant that replaces
    retrieval with one style sentence (for pipelines that cannot hold retrieved sentences)."""

    def __init__(self, style=False, norms=None, retriever=None, tau=TAU):
        self.style = style
        self.norms = norms or UsageNorms()
        self.tau = tau
        self._retriever = retriever

    @property
    def retriever(self):
        if self._retriever is None:  # load the pool only when needed
            from .retrieval import SizeMatchedRetriever
            self._retriever = SizeMatchedRetriever()
        return self._retriever

    # ---- step 1
    def build_prompt(self, subject, year, value, notes=None):
        """User message for drafting one MD&A sentence about a change of `value` % (signed)."""
        notes = NOTES.format(notes=notes) if notes else ""
        if self.style:
            return notes + fact(subject, year, value) + ASK + STYLE
        return retrieval_block(self.retriever.retrieve(value)) + notes + fact(subject, year, value) + ASK

    # ---- step 2
    def check(self, text, size, context=None):
        """Flags for a text about a change of |size| %. `context` is any input that gives causes (e.g., notes);
        without it, every cause is unsupported, as the input gives only the number."""
        words = [dict(w, range=tuple(float(x) for x in w["range"]) if w["range"] else None)
                 for w in self.norms.check(text, abs(size))]
        causes = []
        m = CAUSE_CUT.search(text)
        if m:
            span = cause_text(text, m.start())
            support = lex_support(span, context) if context else 0.0
            causes.append({"span": span, "support": round(support, 3), "supported": support >= self.tau})
        return {"words": words, "causes": causes}

    # ---- step 3
    def revise(self, text, size, context=None):
        """Cut an unsupported cause clause, then delete or replace each misfit magnitude word."""
        flags = self.check(text, size, context)
        if any(not c["supported"] for c in flags["causes"]):
            text = cut_cause(text)
        lemmas = [w["lemma"] for w in self.norms.check(text, abs(size)) if w["flag"]]
        return edit(text, lemmas)

    def __call__(self, text, size, context=None):
        return self.revise(text, size, context)
