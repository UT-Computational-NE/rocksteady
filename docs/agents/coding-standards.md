# Coding standards

The judgement calls a reviewer checks in every diff. A rule a tool can decide lives in a check instead:

- `tools/lint.sh`: the lint rules this repo enforces.
- mypy: the types in `src/rocksteady`.
- `tests/test_vocabulary.py`: the phrases `GLOSSARY.md` avoids that are always wrong. A phrase wrong only in some contexts is a review call.
- `tests/test_wording_of_tests.py`: a test does not name the review or PR that found it.

Add a rule here when reviews raise the same call twice, or when the user admits one sooner (recorded on #92). Move it into a check, and delete it here, once a tool can decide it.

## Rules

- **Pin the exact value.** When code converts a value (coerces a type, counts, normalises a form), checks the items of a collection, or applies an offset, an origin, a phase or a zone, a test asserts the exact result. Its input must be one on which a weaker version of the code gives a different answer:
  - `type(x) is bool`, not `x == True`, which `np.True_` also passes;
  - a count of two, not one, so that `.any()` cannot stand in for `.sum()`;
  - for `any(...)` or `all(...)` over options, states, segments or masks, the deciding item (one that matches for `any`, one that fails for `all`) neither first nor last, beside items that decide the other way, or else two cases beside such items, one with it first and one with it last, so that neither the other check, `items[0]` nor `items[-1]` can stand in;
  - for an offset, an origin, a phase or a zone, a value that the code's fallback cannot reach by accident: not a whole number of periods from the default (an origin 3600 s from 0 with 7200 s periods, UTC-6 with 4 h periods), and not already in the zone a conversion targets (New York, not UTC, when the code may convert to UTC).

  In PR #104, a dropped `bool()` and a count that the only test held at 1 both survived the tests. In PR #139, `all(...)` and then the first option's subtype each survived in place of `any(...)` over the options: every case had subtypes on every state, and then only on the first state. The tests then needed a case at each end, subtypes on the first state only and on the last state only: with only the last-state case, the last option's subtype would have survived. In PR #136, a UTC-6 origin with 2 h periods was a whole number of periods from UTC, and an origin of 3e8 s with 300 s periods a whole number from 0 s, so a `tz_convert` mutant and then every mutant that ignored the origin survived. In PR #152, a UTC origin beside a naive index gave the same periods whether or not the code converted it to UTC.
- **A widened input brings its degenerate cases.** When a change makes an argument accept a new form, test that form's degenerate cases in the same change: a repeated item, an empty collection, a different order.

  In PR #104, accepting `(name, bool)` pairs as well as a mapping let one mask be named twice. Which pair won then depended on the order a set iterated in.
- **A test claims no more than it checks.** A test's name, docstring and comments say only what its assertions pin down. A word such as "every", "only" or "and nothing else", or a mechanism the test never exercises, is a claim its assertions must check; otherwise narrow the words, or add the check.

  In PR #126 (#118), a guard's docstring said "Every test marked `checkout_only` runs in a checkout …" while the guard checked a fixed list, and "Only the checkout-only tests check the conftest's PYTHONPATH under another install, so a checkout must run each of them", though only one of them does. In PR #120 (#109), a test named "…matches missing text and nothing else" checked three differing cases (fixed in #122).
- **A restatement keeps the code's qualifiers.** When prose (a guide page, the README, the CHANGELOG, a docstring, an example's comment) restates what the code does, or what a docstring or another page says it does, it carries over each qualifier the source states: "on its own", "for any call that both accept", "once its model is fitted in full". A restatement that drops one claims more than the code does. It is the prose counterpart of *A test claims no more than it checks*. `docs/agents/review.md`'s *Probing restatements* finds the sentences a change has made untrue; this rule asks for the qualifiers in the first draft.

  In PR #131 (#67), the own review and review 1 found four sentences that had dropped a qualifier:
  - "a catch-all declared last … takes only the segments that no earlier subtype can take within the constraints", without the module docstring's "on its own" (with the least-sum tie-break, a catch-all can take a segment to let a neighbour take an earlier subtype);
  - "returns the segments, objective and counts that 0.5.0 returned", without "for any call that both accept"; the first fix, "for any call that 0.5.0 also accepted", was still too wide;
  - "a longer segment never costs less", without "once its model is fitted in full", and "Pruning stays exact", without the exponential models' caveat on default bounds;
  - "On the optimiser's own segments, they give back the optimiser's subtypes", an example's comment, without the guide's "when the call's caps, durations and penalties name states only" (the docstring's "when those constraints name states only").
