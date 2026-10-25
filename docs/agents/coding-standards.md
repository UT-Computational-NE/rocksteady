# Coding standards

The judgement calls a reviewer checks in every diff. A rule a tool can decide lives in a check instead:

- `tools/lint.sh`: the lint rules this repo enforces.
- mypy: the types in `src/rocksteady`.
- `tests/test_vocabulary.py`: the phrases `GLOSSARY.md` avoids that are always wrong. A phrase wrong only in some contexts is a review call.
- `tests/test_wording_of_tests.py`: tests say what they pin down, not the review or PR that found them.

Add a rule here when reviews raise the same call twice, or when the user admits one sooner (recorded on #92). Move it into a check, and delete it here, once a tool can decide it.

## Rules

- **Pin the exact value.** When code converts a value (coerces a type, counts, normalises a form) or checks the items of a collection, a test asserts the exact result. Its input must be one on which a weaker version of the code gives a different answer:
  - `type(x) is bool`, not `x == True`, which `np.True_` also passes;
  - a count of two, not one, so that `.any()` cannot stand in for `.sum()`;
  - for `any(...)` or `all(...)` over options, states, segments or masks, the deciding item (one that matches for `any`, one that fails for `all`) after the first, beside items that decide the other way, not first or alone, so that neither the other check nor `items[0]` can stand in.

  In PR #104, a dropped `bool()` and a count that the only test held at 1 both survived the tests. In PR #139, `all(...)` and then the first option's subtype each survived in place of `any(...)` over the options: every case had subtypes on every state, and then only on the first state.
- **A widened input brings its degenerate cases.** When a change makes an argument accept a new form, test that form's degenerate cases in the same change: a repeated item, an empty collection, a different order.

  In PR #104, accepting `(name, bool)` pairs as well as a mapping let one mask be named twice. Which pair won then depended on the order a set iterated in.
- **A restatement keeps the code's qualifiers.** When prose (a guide page, the README, the CHANGELOG, a docstring, an example's comment) restates what the code does, or what a docstring or another page says it does, it carries over each qualifier the source states: "on its own", "for any call that both accept", "once its model is fitted in full". A restatement that drops one claims more than the code does. It is the prose counterpart of #128's rule for tests. `docs/agents/review.md`'s *Probing restatements* finds the sentences a change has made untrue; this rule asks for the qualifiers in the first draft.

  In PR #131 (#67), the own review and review 1 found four sentences that had dropped a qualifier:
  - "a catch-all declared last … takes only the segments that no earlier subtype can take within the constraints", without the module docstring's "on its own" (with the least-sum tie-break, a catch-all can take a segment to let a neighbour take an earlier subtype);
  - "returns the segments, objective and counts that 0.5.0 returned", without "for any call that both accept"; the first fix, "for any call that 0.5.0 also accepted", was still too wide;
  - "a longer segment never costs less", without "once its model is fitted in full", and "Pruning stays exact", without the exponential models' caveat on default bounds;
  - "On the optimiser's own segments, they give back the optimiser's subtypes", an example's comment, without the guide's "when the call's caps, durations and penalties name states only" (the docstring's "when those constraints name states only").
