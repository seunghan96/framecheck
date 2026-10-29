"""Step 3 rules: delete or replace flagged magnitude words, and cut unsupported cause clauses. No model call."""
import re

from .lexicon import ADV, LEX

UP_VERBS = {"v:surge", "v:soar", "v:jump", "v:climb", "v:rally", "v:spike", "v:skyrocket", "v:rocket", "v:leap",
            "v:balloon", "v:edge", "v:tick", "v:inch", "v:creep", "v:nudge"}


def edit(sentence, lemmas):
    """Rule-based fix: delete flagged adverbs and adjectives, and replace flagged manner verbs by a plain verb."""
    out = sentence
    for lemma in lemmas:
        forms = [f for f, l in LEX.items() if l == lemma]
        for f in sorted(forms, key=len, reverse=True):
            if lemma.startswith("v:"):
                plain = "increased" if lemma in UP_VERBS else "decreased"
                new = re.sub(r"\b" + f + r"\b(\s+(?:up|down|higher|lower))?", plain, out, count=1, flags=re.I)
            elif f in ADV:
                new = re.sub(r"\s*\b" + f + r"\b", "", out, count=1, flags=re.I)
            else:
                def drop(m):
                    art = m.group(1)
                    if art:  # a/an agreement with the next word ("a significant increase" -> "an increase")
                        art = (art[0] + ("n" if m.group(2)[0].lower() in "aeiou" else "")) + " "
                    return (art or "") + m.group(2)
                new = re.sub(r"\b(?:(an?)\s+)?" + f + r"\s+(\w)", drop, out, count=1, flags=re.I)
            if new != out:
                out = new
                break
    return re.sub(r"\s{2,}", " ", out).replace(" ,", ",").replace(" .", ".")


CAUSE_CUT = re.compile(r"\s*(?:,|;|\sand)?\s*(?:(?:which|that)\s+(?:(?:can|could|may|might)\s+be|was|were|is|are)\s+)?"
                       r"(?:(?:primarily|mainly|largely|partly|partially|mostly|principally)\s+)?"
                       r"(?:due to|driven by|attributable to|as a result of|reflecting|which reflects|resulting from|owing to|"
                       r"because of|caused by|led by|thanks to|fueled by|attributed to|indicating|highlighting|underscoring|"
                       r"signaling|suggesting)\b.*$",
                       re.I | re.S)


CUT_VERB = re.compile(r"\b(?:increased|decreased|rose|fell|grew|declined|dropped|climbed|gained|improved|reduced|"
                      r"surged|jumped|soared|plunged|edged|dipped|totaled|"
                      r"remained|went|up|down|higher|lower)\b", re.I)
CUT_BAD_END = re.compile(r"\b(?:was|were|is|are|be|been|by|of|to|in|from|the|a|an|and|with|for|at)$", re.I)


def cut_cause(sentence):
    """Rule-based fix for a cause that the input does not give: cut the clause from its connective to the end."""
    out = CAUSE_CUT.sub("", sentence).rstrip(" ,;")
    if out == sentence.rstrip(" ,;"):
        return sentence
    # keep the sentence when the cut would drop its numbers or leave a fragment (e.g., a sentence that starts with the cause)
    nums = re.findall(r"\d+(?:\.\d+)?", sentence)
    kept = re.findall(r"\d+(?:\.\d+)?", out)
    # keep the sentence unless what remains still stands: it keeps the number, has a change verb, and does not
    # end on a dangling word ("Revenue growth was", "The increase in revenue")
    if nums and not kept:
        return sentence
    if len(out.split()) < 5 and (not CUT_VERB.search(out) or CUT_BAD_END.search(out)):
        return sentence
    return out if out.endswith(".") else out + "."


