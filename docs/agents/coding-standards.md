# Coding standards

The judgement calls a reviewer checks in every diff. A rule a tool can decide lives in a check instead:

- `tools/lint.sh`: the lint rules this repo enforces.
- mypy: the types in `src/rocksteady`.
- `tests/test_vocabulary.py`: the phrases `GLOSSARY.md` avoids that are always wrong. A phrase wrong only in some contexts is a review call.
- `tests/test_wording_of_tests.py`: tests say what they pin down, not the review or PR that found them.

Add a rule here when reviews raise the same call twice, or when the user admits one sooner (recorded on #92). Move it into a check, and delete it here, once a tool can decide it.

## Rules

- **Pin the exact value.** When code converts a value (coerces a type, counts, normalises a form), a test asserts the exact result. Its input must be one on which a weaker version of the code gives a different answer:
  - `type(x) is bool`, not `x == True`, which `np.True_` also passes;
  - a count of two, not one, so that `.any()` cannot stand in for `.sum()`.

  In PR #104, a dropped `bool()` and a count that the only test held at 1 both survived the tests.
- **A widened input brings its degenerate cases.** When a change makes an argument accept a new form, test that form's degenerate cases in the same change: a repeated item, an empty collection, a different order.

  In PR #104, accepting `(name, bool)` pairs as well as a mapping let one mask be named twice. Which pair won then depended on the order a set iterated in.
- **A restatement keeps the code's qualifiers.** When prose restates what the code does (a guide page, the README, the CHANGELOG, a docstring, an example's comment), it carries over each condition its source states: "on its own", "when … name states only", "for a call both accept", "once the model is fitted in full", "except with default bounds". A restatement that drops one claims more than the code does. `docs/agents/review.md`'s *Probing restatements* finds such a sentence once a rule changes; this rule keeps the qualifiers in from the first draft.

  The reviews of PR #131 (#67) found seven restatements that dropped one, among them:
  - a catch-all "takes only the segments no earlier subtype can take", without the module docstring's "on its own";
  - results "equal 0.5.0's", true only for calls both versions accept;
  - "pruning stays exact", without the exponential models' caveat on default bounds;
  - an example's comment saying `classify_segments` returns the optimiser's subtypes, without "when the caps, durations and penalties name states only".
