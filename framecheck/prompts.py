"""Prompts used in the experiments (verbatim)."""

SYSTEM = "You are an experienced financial reporting specialist at a US public company."
FACT = "Line item: {subject}\nPeriod: fiscal {year} compared with fiscal {prev}\nChange: {sign}{abs:.1f}%\n\n"
ASK = ("Write one sentence describing this change for the Management's Discussion and Analysis (MD&A) section "
       "of the company's annual report on Form 10-K. Reply with the sentence only.")
SUMM = ('Sentence from a company\'s 10-K MD&A section:\n"{sentence}"\n\nSummarize this sentence for a short investor '
        "update in at most 15 words. Reply with the summary only.")
# FrameCheck (style): one sentence that replaces retrieval
STYLE = (" Follow the style of 10-K filings, which usually report a change with its number alone and rarely add "
         "words such as slightly or significantly.")
NOTES = "Notes from the finance team: {notes}\n\n"
RETRIEVAL_HEADER = "Sentences from other companies' 10-K filings that describe changes of a similar size:\n"
NORM_WORDS = ["marginal", "slight", "modest", "moderate", "significant", "substantial", "sharp", "dramatic"]


def fact(subject, year, value):
    """The drafting input: a line item, a fiscal year, and a signed relative change in %."""
    return FACT.format(subject=subject, year=year, prev=year - 1, sign="+" if value > 0 else "-", abs=abs(value))


def retrieval_block(sentences):
    return RETRIEVAL_HEADER + "\n".join(f'- "{s}"' for s in sentences) + "\n\n"


def norm_card(norms):
    """Baseline: a card that states the usage norm of eight common words."""
    lines = ["How often-used magnitude words are used in real 10-K filings (typical range of the change):"]
    for w in NORM_WORDS:
        lo, mid, hi = norms.range_of(w)
        lines.append(f"- {w}: {lo:g}% to {hi:g}% (median {mid:g}%)")
    return "\n".join(lines) + "\n\n"
