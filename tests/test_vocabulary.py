"""No always-wrong phrase from GLOSSARY.md's avoided terms in the docs, docstrings or examples (#56, #73)."""

import glob
import os
import re
from typing import NamedTuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Avoided(NamedTuple):
    use_instead: str  #: the glossary term
    near_miss: str  #: correct text close to the phrase, which the scan must not flag


#: Phrases that are always wrong in this repo. A phrase that is right in some contexts (for example "labels
#: only" for an input of labels) belongs to review, not here. Each near-miss is correct text close to its phrase.
AVOIDED = {
    "domain rule": _Avoided("override (domain knowledge that forces a label)",
                            "In the time domain, rules of thumb fail."),
    "under limits": _Avoided("under constraints (a limit is a criterion's threshold)", "a statistic under its limit"),
    "within a tolerance": _Avoided("within a limit", "equal within numerical tolerance"),
    "sample period": _Avoided("sample spacing (a period is the optimiser's counting span)",
                              "counts of segments per period, sampled every second"),
    "last decided state": _Avoided("last decided label (a sample has a label, a segment a state)",
                                   "a segment's state, from the last decided labels"),
    "segment's label": _Avoided("segment's state (a sample has a label, a segment a state)",
                                "each segment's labelled samples"),
    "a passing state": _Avoided("a passing label (a sample has a label, a segment a state)",
                                "by passing state between calls"),
}

#: Prose the users read. GLOSSARY.md (which lists the avoided terms) and CHANGELOG.md (history) are not scanned.
SCANNED = ["README.md", "src/**/*.py", "docs/source/**/*.md", "docs/source/**/*.rst", "docs/source/**/*.py",
           "examples/**/*.py"]


#: Between two words of a phrase: spaces, or one line break with the next line's indent and comment marker.
_GAP = r"(?:[ \t]+|[ \t]*\r?\n[ \t]*(?:#[ \t]*)?)"


def _phrase_pattern(phrase):
    """``phrase`` as whole words (a plural too), wrapped onto at most two lines (text or comments), either apostrophe."""
    words = (re.escape(w).replace("'", "['’]") for w in phrase.split())
    return r"\b" + _GAP.join(words) + r"s?\b"


_PHRASES = list(AVOIDED)
_PATTERN = re.compile("|".join(f"(?P<p{i}>{_phrase_pattern(p)})" for i, p in enumerate(_PHRASES)), re.IGNORECASE)


def _avoided_phrases(text):
    """``(line number, phrase as written, use instead)`` for each avoided phrase in ``text``."""
    return [(text.count("\n", 0, m.start()) + 1, m.group(0), AVOIDED[_PHRASES[int(m.lastgroup[1:])]].use_instead)
            for m in _PATTERN.finditer(text)]


def test_avoided_phrases_are_found_with_their_replacement():
    text = "A step of 1.5 sample periods.\nOverrides are Domain Rules.\nA segment's state."
    assert _avoided_phrases(text) == [(1, "sample periods", AVOIDED["sample period"].use_instead),
                                      (2, "Domain Rules", AVOIDED["domain rule"].use_instead)]


def test_avoided_phrases_wrapped_once_in_text_or_a_comment_or_with_a_curly_apostrophe_are_found():
    text = 'Steps of one sample\n    period apart.\nThe segment’s label.\n# Best-fit segments under\n# limits.'
    assert [(n, use) for n, _, use in _avoided_phrases(text)] == [
        (1, AVOIDED["sample period"].use_instead), (3, AVOIDED["segment's label"].use_instead),
        (4, AVOIDED["under limits"].use_instead)]


def test_words_apart_or_inside_other_words_are_not_avoided_phrases():
    text = ("one sample\n\nperiod of\nin the domain\n\n# Rules\nresample periodically\n"
            "a sample # period\nunder limitsless skies")
    assert _avoided_phrases(text) == []


def test_each_avoided_phrase_is_flagged_and_its_near_miss_is_not():
    def words(s):
        return set(re.findall(r"[\w']+", s.lower()))
    for phrase, a in AVOIDED.items():
        assert [found for _, found, _ in _avoided_phrases(phrase)] == [phrase], phrase
        assert words(a.near_miss) & words(phrase), f"{phrase!r}: its near-miss shares no word with it"
        assert _avoided_phrases(a.near_miss) == [], f"{phrase!r} flags its own near-miss"


def test_case_folded_letters_are_reported_not_raised():
    assert [(n, use) for n, _, use in _avoided_phrases("x\n\u017fample period")] == [
        (2, AVOIDED["sample period"].use_instead)]


def test_docs_and_docstrings_avoid_always_wrong_phrases():
    files = sorted({p for pattern in SCANNED for p in glob.glob(os.path.join(ROOT, pattern), recursive=True)
                    if "_build" not in p.split(os.sep)})
    assert files, f"nothing to scan under {ROOT}"
    hits = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            hits += [f"{os.path.relpath(path, ROOT)}:{n}: {phrase!r} -> {use}"
                     for n, phrase, use in _avoided_phrases(f.read())]
    assert not hits, "phrases GLOSSARY.md avoids:\n" + "\n".join(hits)
