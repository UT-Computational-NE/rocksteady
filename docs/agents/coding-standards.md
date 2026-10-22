# Coding standards

The judgement calls a reviewer checks in every diff. A rule a tool can decide lives in a check instead:

- `tools/lint.sh`: the lint rules this repo enforces.
- mypy: the types in `src/rocksteady`.
- `tests/test_vocabulary.py`: the phrases `GLOSSARY.md` avoids that are always wrong. A phrase wrong only in some contexts is a review call.
- `tests/test_wording_of_tests.py`: tests say what they pin down, not the review that found them.

Add a rule here when reviews raise the same call twice. Move it into a check, and delete it here, once a tool can decide it.

## Rules

- **Pin the exact value.** When code converts a value (coerces a type, counts, normalises a form), a test asserts the exact result. Its input must be one on which a weaker version of the code gives a different answer:
  - `type(x) is bool`, not `x == True`, which `np.True_` also passes;
  - a count of two, not one, so that `.any()` cannot stand in for `.sum()`.

  In PR #104, a dropped `bool()` and a count that the only test held at 1 both survived the tests.
- **A widened input brings its degenerate cases.** When a change makes an argument accept a new form, test that form's degenerate cases in the same change: a repeated item, an empty collection, a different order.

  In PR #104, accepting `(name, bool)` pairs as well as a mapping let one mask be named twice. Which pair won then depended on the order a set iterated in.
