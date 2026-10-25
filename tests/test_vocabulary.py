"""No always-wrong phrase from GLOSSARY.md's avoided terms in the docs, docstrings or examples (#56, #73, #81, #84,
#89)."""

import glob
import os
import re
from typing import NamedTuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Avoided(NamedTuple):
    use_instead: str  #: the glossary term
    near_misses: tuple  #: correct texts close to the phrase, which the scan must not flag
    after: tuple = ()  #: the phrase counts only after one of these words
    not_before: tuple = ()  #: the phrase does not count before one of these words


#: Words that make the phrase after them a noun: "the passing state" names a sample's label a state, while "by passing
#: state" is a verb. A form without one, such as "keeps passing state", is left to review.
_DETERMINERS = ("a", "the", "its", "their", "each")

#: Phrases that are always wrong in this repo. A phrase that is right in some contexts (for example "labels
#: only" for an input of labels) belongs to review, not here. Each near-miss is correct text on a boundary of its
#: phrase's pattern (see ``BOUNDARIES``).
AVOIDED = {
    "domain rule": _Avoided("override (domain knowledge that forces a label)",
                            ("In the time domain, rules of thumb fail.", "Time-domain rules of thumb.",
                             "The frequency domain rules out aliasing."), not_before=("out",)),
    "under limits": _Avoided("under constraints (a limit is a criterion's threshold)",
                             ("the criterion a statistic falls under; limits are its thresholds",)),
    "within a tolerance": _Avoided("within a limit", ("Within a tolerance band of the sensor.",), not_before=("band",)),
    "sample period": _Avoided("sample spacing (a period is the optimiser's counting span)",
                              ("the sample periodogram",)),
    "last decided state": _Avoided("last decided label (a sample has a label, a segment a state)",
                                   ("Whichever label was last decided, state it.",)),
    "segment's label": _Avoided("segment's state (a sample has a label, a segment a state)",
                                ("each segment's labelled samples",)),
    "passing state": _Avoided("… passing label (a sample has a label, a segment a state)",
                              ("by passing state between calls",), after=_DETERMINERS),
}

#: Correct text that an entry flags by choice, because narrowing the entry would let its wrong uses through. Reword
#: such text: "all statistics under their limits" passes.
FLAGGED_BY_CHOICE = {
    "all statistics under limits": "under limits",
}

#: Prose the users read. GLOSSARY.md (which lists the avoided terms) and CHANGELOG.md (history) are not scanned.
SCANNED = ["README.md", "src/**/*.py", "docs/source/**/*.md", "docs/source/**/*.rst", "docs/source/**/*.py",
           "examples/**/*.py"]


#: Between two words of a phrase: spaces, or one line break with the next line's indent and comment marker.
_GAP = r"(?:[ \t]+|[ \t]*\r?\n[ \t]*(?:#[ \t]*)?)"


#: The boundaries of a phrase's pattern; a near-miss is close to its phrase when relaxing one of them flags it.
BOUNDARIES = ("start", "gap", "end", "after", "not_before")


def _phrase_pattern(phrase, avoided, relaxed=None):
    """``phrase`` as whole words (a plural too), wrapped onto at most two lines (text or comments), either apostrophe.

    ``relaxed`` names a boundary in ``BOUNDARIES`` to drop: the pattern then also matches text close to the phrase.
    """
    words = [re.escape(w).replace("'", "['’]") for w in phrase.split()]
    if avoided.after and relaxed != "after":
        words.insert(0, "(?:" + "|".join(avoided.after) + ")")
    start = "" if relaxed == "start" else r"(?<![\w-])"
    end = "" if relaxed == "end" else r"\b"
    if avoided.not_before and relaxed != "not_before":
        end += f"(?!{_GAP}(?:{'|'.join(avoided.not_before)})\\b)"
    return start + (r"\W+" if relaxed == "gap" else _GAP).join(words) + "s?" + end


_PHRASES = list(AVOIDED)
_PATTERN = re.compile("|".join(f"(?P<p{i}>{_phrase_pattern(p, AVOIDED[p])})" for i, p in enumerate(_PHRASES)),
                      re.IGNORECASE)


def _avoided_phrases(text):
    """``(line number, phrase as written, use instead)`` for each avoided phrase in ``text``."""
    return [(text.count("\n", 0, m.start()) + 1, m.group(0), AVOIDED[_PHRASES[int(m.lastgroup[1:])]].use_instead)
            for m in _PATTERN.finditer(text)]


#: Words a near-miss may share with its phrase without being close to it.
_ARTICLES = {"a", "an", "the"}


def _words(text):
    return set(re.findall(r"[\w']+", text.lower())) - _ARTICLES


def _guarded(phrase, avoided, near_miss):
    """The boundaries of ``phrase``'s pattern that keep ``near_miss`` from being flagged."""
    return {b for b in BOUNDARIES if re.search(_phrase_pattern(phrase, avoided, b), near_miss, re.IGNORECASE)}


def _near_miss_problems(phrase, avoided):
    """The problems with ``avoided``'s near-misses for ``phrase``, each as a sentence; none when all is fine."""
    problems = []
    for near_miss in avoided.near_misses:
        problems += _one_near_miss_problems(phrase, avoided, near_miss)
    on = set().union(*(_guarded(phrase, avoided, n) for n in avoided.near_misses))
    problems += [f"{phrase!r}: no near-miss sits on its {narrowing}"
                 for narrowing in ("after", "not_before") if getattr(avoided, narrowing) and narrowing not in on]
    return problems


def _one_near_miss_problems(phrase, avoided, near_miss):
    problems = []
    if not _words(near_miss) & _words(phrase):
        problems.append(f"{phrase!r}: its near-miss {near_miss!r} shares no word but an article with it")
    elif _avoided_phrases(near_miss):
        problems.append(f"{phrase!r} flags its own near-miss {near_miss!r}")
    elif not _guarded(phrase, avoided, near_miss):
        problems.append(f"{phrase!r}: its near-miss {near_miss!r} sits on no boundary of the pattern")
    return problems


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


def test_a_word_after_or_a_hyphen_before_that_makes_an_avoided_phrase_correct_text_is_not_flagged():
    text = ("The frequency domain rules out aliasing.\nTime-domain rules of thumb.\n"
            "Within a tolerance band of the sensor, or within a tolerance\n    # band.\n"
            "Domain rules outweigh guesses. Within a tolerance of 0.1.")
    assert [(n, use) for n, _, use in _avoided_phrases(text)] == [
        (5, AVOIDED["domain rule"].use_instead), (5, AVOIDED["within a tolerance"].use_instead)]


def test_correct_text_flagged_by_choice_is_flagged_as_its_phrase():
    assert FLAGGED_BY_CHOICE
    for text, phrase in FLAGGED_BY_CHOICE.items():
        assert [use for _, _, use in _avoided_phrases(text)] == [AVOIDED[phrase].use_instead], text


def test_passing_state_after_a_determiner_is_found_and_passing_state_as_a_verb_is_not():
    text = ("A sample keeps the passing state.\nIts passing\n    state, or their passing states.\n"
            "Done by passing state between calls.\npassing state between calls")
    found = _avoided_phrases(text)
    assert [(n, phrase) for n, phrase, _ in found] == [(1, "the passing state"), (2, "Its passing\n    state"),
                                                       (3, "their passing states")]
    assert all(use.startswith("… passing label") for _, _, use in found)


def test_each_avoided_phrase_is_flagged_and_its_near_misses_pass_the_checks():
    for phrase, avoided in AVOIDED.items():
        example = f"{avoided.after[0]} {phrase}" if avoided.after else phrase
        assert [found for _, found, _ in _avoided_phrases(example)] == [example], phrase
        assert _near_miss_problems(phrase, avoided) == []


def test_a_near_miss_sharing_only_an_article_with_its_phrase_is_a_problem():
    assert _near_miss_problems("a passing state", _Avoided("a passing label", ("a dog sat down",))) == [
        "'a passing state': its near-miss 'a dog sat down' shares no word but an article with it"]


def test_a_narrowing_that_no_near_miss_sits_on_is_a_problem():
    avoided = _Avoided("override", ("The domain rulebook.",), after=("the",), not_before=("out",))
    assert _near_miss_problems("domain rule", avoided) == [
        "'domain rule': no near-miss sits on its after", "'domain rule': no near-miss sits on its not_before"]


def test_each_boundary_of_the_patterns_has_a_near_miss_on_it():
    on = set().union(*(_guarded(phrase, avoided, near_miss)
                       for phrase, avoided in AVOIDED.items() for near_miss in avoided.near_misses))
    assert on == set(BOUNDARIES)


def test_a_near_miss_the_scan_flags_is_a_problem_after_one_that_passes():
    avoided = _Avoided("sample spacing", ("the sample periodogram", "one sample period"))
    assert _near_miss_problems("sample period", avoided) == ["'sample period' flags its own near-miss 'one sample period'"]


def test_a_near_miss_on_no_boundary_of_its_pattern_is_a_problem():
    far = "counts of segments per period, sampled every second"
    assert _near_miss_problems("sample period", _Avoided("sample spacing", (far,))) == [
        f"'sample period': its near-miss {far!r} sits on no boundary of the pattern"]


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
