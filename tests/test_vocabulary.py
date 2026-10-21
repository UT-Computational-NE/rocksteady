"""The docs, docstrings and examples avoid phrases GLOSSARY.md avoids that are always wrong (#56, #73)."""

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
}

#: Prose the users read. GLOSSARY.md (which lists the avoided terms) and CHANGELOG.md (history) are not scanned.
SCANNED = ["README.md", "src/**/*.py", "docs/source/**/*.md", "docs/source/**/*.rst", "docs/source/**/*.py",
           "examples/**/*.py"]


def _phrase_pattern(phrase):
    """``phrase`` with any line break between its words (and a comment marker after one), and either apostrophe."""
    words = (re.escape(w).replace("'", "['’]") for w in phrase.split())
    return r"\s+(?:#\s*)?".join(words)


_PATTERN = re.compile("|".join(_phrase_pattern(p) for p in AVOIDED), re.IGNORECASE)


def _avoided_phrases(text):
    """``(line number, phrase as written, use instead)`` for each avoided phrase in ``text``."""
    hits = []
    for m in _PATTERN.finditer(text):
        key = " ".join(re.sub(r"#", " ", m.group(0)).split()).lower().replace("’", "'")
        hits.append((text.count("\n", 0, m.start()) + 1, m.group(0), AVOIDED[key]))
    return hits


def test_avoided_phrases_are_found_with_their_replacement():
    text = "A step of 1.5 sample periods.\nOverrides are Domain Rules.\nA segment's state."
    assert _avoided_phrases(text) == [(1, "sample period", AVOIDED["sample period"]),
                                      (2, "Domain Rule", AVOIDED["domain rule"])]


def test_avoided_phrases_wrapped_across_lines_or_with_a_curly_apostrophe_are_found():
    text = 'Steps of one sample\n    period apart.\nThe segment’s label.\n# Best-fit segments under\n# limits.'
    assert [(n, use) for n, _, use in _avoided_phrases(text)] == [
        (1, AVOIDED["sample period"]), (3, AVOIDED["segment's label"]), (4, AVOIDED["under limits"])]


def test_docs_and_docstrings_avoid_always_wrong_phrases():
    files = sorted({p for pattern in SCANNED for p in glob.glob(os.path.join(ROOT, pattern), recursive=True)
                    if "_build" not in p.split(os.sep)})
    hits = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            hits += [f"{os.path.relpath(path, ROOT)}:{n}: {phrase!r} -> {use}"
                     for n, phrase, use in _avoided_phrases(f.read())]
    assert files, f"nothing to scan under {ROOT}"
    assert not hits, "phrases GLOSSARY.md avoids:\n" + "\n".join(hits)
