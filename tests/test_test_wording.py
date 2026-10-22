"""Tests say what they pin down, not the review or pull request that found it (#92, #106)."""

import glob
import io
import os
import re
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A review or a pull request by number: "review 1 of PR #83", "Review 2 for #61", "(PR #46)". The history of a
#: test is in its PR; the test says what it checks. An issue number alone, such as "(#45)", is not flagged.
_PATTERN = re.compile(r"\breview\s+\d+\s+(?:of|for)\s+(?:PR\s*)?#\d+|\bPR\s*#\d+", re.IGNORECASE)

#: Correct text close to the pattern, which the scan must not flag.
NEAR_MISS = "A review of the record: segments in [p, q) (#45), as the PR body's evidence shows."


def _named(source):
    """``(line number, text as written)`` for each review or PR named in a comment or string of ``source``."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            out += [(tok.start[0] + tok.string.count("\n", 0, m.start()), m.group(0))
                    for m in _PATTERN.finditer(tok.string)]
    return out


def test_a_review_or_pr_by_number_is_found_and_its_near_miss_is_not():
    source = ('"""Review 1 of PR #83: a tie."""\n'
              "# -- review regressions (PR #46) --\n"
              "x = 1  # review 2 for #61\n"
              f"y = {NEAR_MISS!r}\n")
    assert _named(source) == [(1, "Review 1 of PR #83"), (2, "PR #46"), (3, "review 2 for #61")]


def test_no_test_names_a_review_or_a_pr():
    here = os.path.abspath(__file__)
    found = []
    for path in sorted(glob.glob(os.path.join(ROOT, "tests", "**", "*.py"), recursive=True)):
        if os.path.abspath(path) == here:  # this file names them as examples
            continue
        with open(path, encoding="utf-8") as f:
            found += [f"{os.path.relpath(path, ROOT)}:{line}: {text!r}" for line, text in _named(f.read())]
    assert not found, ("Say what the test or section pins down, not the review or PR that found it:\n"
                       + "\n".join(found))
