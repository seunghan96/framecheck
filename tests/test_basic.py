import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framecheck import FrameCheck, UsageNorms, find_words, word_class  # noqa: E402
from framecheck.metrics import score_draft  # noqa: E402

fc = FrameCheck()
norms = UsageNorms()


def test_lexicon():
    assert find_words("Sales increased slightly by 2%.") == ["slight"]
    assert find_words("A modest increase of 2%.") == ["modest"]
    assert find_words("Revenue surged 40%.") == ["v:surge"]
    assert find_words("Large customers accounted for 20% of sales.") == []  # not on a change noun
    assert find_words("There was no significant change.") == []  # negated
    assert word_class("strong") == "evaluative"


def test_norms_flags():
    assert norms.flag("significant", 2.0) == "inflated"
    assert norms.flag("significant", 40.0) is None
    assert norms.flag("slight", 30.0) == "deflated"
    assert norms.flag("slight", 1.0) is None
    assert norms.flag("strong", 1.0) is None  # evaluative words have no size norm


def test_revise_paper_example():
    draft = "Revenue surged 3% on higher prices, due to strong demand."
    assert fc.revise(draft, 3.0, context="Revenue: +3%. Higher prices.") == "Revenue increased 3% on higher prices."


def test_fitting_word_and_supported_cause_are_kept():
    assert fc.revise("Revenue increased significantly by 45%.", 45.0) == "Revenue increased significantly by 45%."
    s = "Revenue increased 3%, driven by higher prices."
    assert fc.revise(s, 3.0, context="Revenue: +3%. Higher prices.") == s
    assert fc.revise(s, 3.0) == "Revenue increased 3%."  # no input gives the cause


def test_retrieval_matches_size_and_direction():
    sents = fc.retriever.retrieve(-12.0)
    assert len(sents) == 6


def test_metrics():
    sc = score_draft(norms, "Revenue increased significantly by 3%, driven by strong demand.", 3.0, "up")
    assert sc["graded"] and sc["misfit"] and sc["cause"] and sc["number"] and not sc["placeholder"]
