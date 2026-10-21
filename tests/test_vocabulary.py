"""The docs, docstrings and examples avoid the phrases GLOSSARY.md retires (#56, #73)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

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
SCANNED = ["README.md", "src/**/*.py", "docs/source/**/*.md", "docs/source/**/*.py", "examples/**/*.py"]

_PATTERN = re.compile("|".join(re.escape(p) for p in AVOIDED), re.IGNORECASE)


def avoided_phrases(text):
    """``(line number, phrase, use instead)`` for each avoided phrase in ``text``."""
    return [(n, m.group(0), AVOIDED[m.group(0).lower()])
            for n, line in enumerate(text.splitlines(), 1) for m in _PATTERN.finditer(line)]


def test_avoided_phrases_are_found_with_their_replacement():
    text = "A step of 1.5 sample periods.\nOverrides are Domain Rules.\nA segment's state."
    assert avoided_phrases(text) == [(1, "sample period", AVOIDED["sample period"]),
                                     (2, "Domain Rule", AVOIDED["domain rule"])]


def _files():
    files = sorted({p for pattern in SCANNED for p in ROOT.glob(pattern) if "_build" not in p.parts})
    if not files:
        pytest.skip("no source tree next to the tests (installed package)")
    return files


def test_docs_and_docstrings_avoid_retired_phrases():
    hits = [f"{p.relative_to(ROOT)}:{n}: {phrase!r} -> {use}"
            for p in _files() for n, phrase, use in avoided_phrases(p.read_text(encoding="utf-8"))]
    assert not hits, "phrases GLOSSARY.md avoids:\n" + "\n".join(hits)
