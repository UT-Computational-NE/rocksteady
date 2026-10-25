"""A test's comments and docstrings do not name the review or pull request that found it (#92, #106)."""

import glob
import io
import os
import re
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A review or a pull request by number: "review 1 of PR #83", "Review 2 for #61", "review 1 on #82", "(PR #46)",
#: "pull request #102".
#: The history of a test is in its PR; the test says what it checks. An issue number alone, "(#45)", is not flagged.
_PATTERN = re.compile(r"\breview\s+\d+\s+(?:of|for|on|in)\s+(?:PR\s*)?#\d+"
                      r"|\b(?:PR|pull\s+request)\s*#\d+", re.IGNORECASE)

#: Comments and string text; from Python 3.12 an f-string's text is FSTRING_MIDDLE, not STRING.
_TEXT = {tokenize.COMMENT, tokenize.STRING} | {getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}

#: Correct text close to the pattern, which the scan must not flag.
NEAR_MISS = ("A review of the record: segments in [p, q) (#45), as the PR body's evidence shows; APR #3 and a "
             "preview 2 of #4 are neither.")


def _named(source):
    """``(line number, text as written)`` for each review or PR named in a comment or string of ``source``."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _TEXT:
            out += [(tok.start[0] + tok.string.count("\n", 0, m.start()), m.group(0))
                    for m in _PATTERN.finditer(tok.string)]
    return out


def test_a_review_or_pr_by_number_is_found_and_its_near_miss_is_not():
    source = ('"""Review 1 of PR #83: a tie."""\n'
              "# -- review regressions (PR #46) --\n"
              "x = 1  # review 2 for #61\n"
              "# The probe of review 1 on #82\n"
              'z = f"in review 3 in #9 {x}"\n'
              f"y = {NEAR_MISS!r}\n"
              '"""A docstring\n\nwhose third line names Review 1 of PR #83."""\n'
              "# as pull request #102 did\n")
    assert _named(source) == [(1, "Review 1 of PR #83"), (2, "PR #46"), (3, "review 2 for #61"), (4, "review 1 on #82"),
                              (5, "review 3 in #9"), (9, "Review 1 of PR #83"), (10, "pull request #102")]


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
