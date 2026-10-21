"""No always-wrong phrase from GLOSSARY.md's avoided terms in the docs, docstrings or examples (#56, #73)."""

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Phrases that are always wrong in this repo, with the glossary term to use instead. A phrase that is
#: right in some contexts (for example "labels only" for an input of labels) belongs to review, not here.
AVOIDED = {
    "domain rule": "override (domain knowledge that forces a label)",
    "under limits": "under constraints (a limit is a criterion's threshold)",
    "within a tolerance": "within a limit",
    "sample period": "sample spacing (a period is the optimiser's counting span)",
    "last decided state": "last decided label (a sample has a label, a segment a state)",
    "segment's label": "segment's state (a sample has a label, a segment a state)",
    "a passing state": "a passing label (a sample has a label, a segment a state)",
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
    return [(text.count("\n", 0, m.start()) + 1, m.group(0), AVOIDED[_PHRASES[int(m.lastgroup[1:])]])
            for m in _PATTERN.finditer(text)]


def test_avoided_phrases_are_found_with_their_replacement():
    text = "A step of 1.5 sample periods.\nOverrides are Domain Rules.\nA segment's state."
    assert _avoided_phrases(text) == [(1, "sample periods", AVOIDED["sample period"]),
                                      (2, "Domain Rules", AVOIDED["domain rule"])]


def test_avoided_phrases_wrapped_once_in_text_or_a_comment_or_with_a_curly_apostrophe_are_found():
    text = 'Steps of one sample\n    period apart.\nThe segment’s label.\n# Best-fit segments under\n# limits.'
    assert [(n, use) for n, _, use in _avoided_phrases(text)] == [
        (1, AVOIDED["sample period"]), (3, AVOIDED["segment's label"]), (4, AVOIDED["under limits"])]


def test_words_apart_or_inside_other_words_are_not_avoided_phrases():
    text = ("one sample\n\nperiod of\nin the domain\n\n# Rules\nresample periodically\n"
            "a sample # period\nunder limitsless skies")
    assert _avoided_phrases(text) == []


def test_case_folded_letters_are_reported_not_raised():
    assert [(n, use) for n, _, use in _avoided_phrases("x\n\u017fample period")] == [(2, AVOIDED["sample period"])]


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
