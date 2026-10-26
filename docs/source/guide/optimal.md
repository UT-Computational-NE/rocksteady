# Best-fit segments under constraints

```{include} ../../../README.md
:start-after: "## Best-fit segments under constraints"
:end-before: "## Extending rocksteady"
```

## Subtypes

A cap on states, such as `{"steady": 3, "transient": 2}`, says how many
segments a day may have, but not which subtypes. An analyst thinks of a flow
loop's day as two cold plateaus, one hot plateau, a heat-up and a cool-down.
For a reactor it is zero-power and at-power plateaus, a startup and a
shutdown. A cap on states alone lets the optimiser spend both transients on
two small moves, or show a heat-up as a jump between two steady fits.

A **subtype** is a caller-named refinement of one state: a cold or a hot
steady segment, a heat-up transient. You declare subtypes with `rs.Subtype`
and pass them as `subtypes={name: Subtype}`. Every constraint can then name
them, and each segment of the result carries its subtype. Names, thresholds,
event masks and allowed sequences all come from you (or your adapter);
rocksteady supplies none. A `Subtype` is plain, immutable data, so an
adapter can build its scheme from its own configuration and keep its
thresholds there.

### Conditions

A subtype may have **conditions**: tests on the segment alone that it must
meet to take the subtype. A segment must meet all of its subtype's
conditions.

| Condition | A segment meets it when |
|---|---|
| `level=(lo, hi)` | its mean lies in `[lo, hi)`. `None` leaves an end open, so `(None, 40)` and `(40, None)` split every level at 40, with no gap and no overlap. |
| `peak=(lo, hi)` | its maximum lies in `[lo, hi)`, either end open: a startup must reach operating level, say. There is no condition on a minimum. |
| `max_rise_s`, `rise_fraction` | it rises within `max_rise_s` seconds of its first sample to `y0 + rise_fraction * (top - y0)`, where `y0` is its first finite sample and `top` its maximum. `rise_fraction` defaults to 0.95. Because the rise is measured from the segment's own start, it works for a baseline that is not zero (a thermocouple at 20 °C), and it tells a pulse from an ordinary rise. |
| `events={mask: True}` or `{mask: False}` | it contains, or does not contain, a flagged sample of that **event mask** in `[start, stop)`. |

Level, peak and rise are computed from the segment's finite samples, the
same values you would compute to classify a segment after the fact; a
segment with no finite sample meets none of them. An event condition reads
the mask alone, whether or not the signal is finite there. Conditions
change which segments the optimiser may choose, not how it fits them; a
subtype's own model (below) changes that.

An event mask is a boolean array with one value per sample. It flags the
samples at which an event occurred: the heater switched, a rod moved. Pass
masks by name, `events={"heater": mask}`, and name them in the conditions,
so that a driven transient is told apart from an undriven one, actuator by
actuator. A mask has no tolerance. If an event can land just before a
segment's start, widen the mask yourself, as the example below does; the
tolerance is then yours to see and to set.

### Every segment of a subdivided state takes a subtype

Once you declare any subtype of a state, every segment of that state takes
one of its subtypes. No segment is left untyped, so no cap can be dodged.
To leave some segments in a group of their own, declare a **catch-all**: a
subtype with no conditions, such as `"other": rs.Subtype("transient")`. You
can cap a catch-all, restrict it in `transitions` or give it a penalty, like
any other subtype.

````{admonition} Why there is no untyped segment (ADR 0001)
```{include} ../../adr/0001-subtypes-are-exhaustive.md
:start-line: 2
```
````

States for which you declare no subtypes behave as before, and their
segments have `subtype=None`. A call with no `subtypes`, `events`,
`initial`, `final` or `overflow_penalty` returns the segments, objective
and counts that 0.5.0 returned, for any call that both accept (*Changed* in the CHANGELOG lists
the calls that 0.5.0 accepted and that are now refused).

### How constraints combine on a state and its subtypes

| Constraint | With a state and its subtypes |
|---|---|
| `max_counts` | Both caps bind. A state's cap counts all of its segments, whatever their subtype, so `{"steady": 3, "hot": 1}` allows at most three steady segments a period, and at most one of them hot. A segment counts against every period it touches, by subtype as by state. `overflow_penalty` makes a cap soft (see *Soft caps*). |
| `transitions` | A state's name stands for all of its subtypes. The allowed pairs are the union over all entries, so `"drift": ["steady"]` lets a drift be followed by any steady subtype. |
| `initial`, `final` | As for `transitions`: a state's name stands for all of its subtypes. |
| `min_duration_s`, `max_duration_s` | Both bind: a segment's duration must lie within its state's bounds and its subtype's. A subtype that no duration can satisfy is refused. |
| `penalty` | A subtype's penalty replaces its state's. When the mapping does not name it, a subtype takes its state's. With `"bic"`, a subtype takes its state's, unless it has a model of its own: then it takes that model's (more parameters, a higher penalty). A higher penalty makes a subtype rarer. |

Subtypes of one state may overlap: two of them can accept the same segment.
The optimiser chooses by total cost. Two subtypes that accept a segment,
carry the same penalty and fit it with the same model cost exactly the
same, so declaration order breaks the tie. Precisely, among segmentations of equal objective the optimiser
takes the least sum, over segments, of each subtype's position among its
state's subtypes. So no segment could, on its own, take an earlier-declared
subtype that accepts it, at the same penalty, without breaking a
constraint. A catch-all declared last, at the same penalty, takes no segment
that could switch alone to an earlier subtype; it can still take one when
that lets a neighbour take an earlier subtype instead. That sum can also
decide an exact tie between states or boundaries, so declaring
subtypes can change which of two equally good segmentations comes back.

Overlap is how a transient's subtype can follow from its neighbours. A
heat-up and a cool-down can have the same conditions (the heater switched)
and be told apart by `transitions` alone: a heat-up goes from cold to hot, a
cool-down from hot to cold.

### A subtype's own model

A subtype fits its segments with its state's model unless you give it one:
`rs.Subtype("transient", model=2, max_rise_s=60)` fits a pulse with a
parabola among transients fitted with a line, say. The spec is one model, as
in {py:mod}`rocksteady.fit`, not keyed by state. The optimiser then costs
each of the subtype's segments with that model, so a segment can take the
subtype because its model fits it better, even over a subtype declared
first. A model is not a condition: `classify_segments` ignores it.

The reports fit segments by their subtype in the same way. A model spec
keyed by state may list subtypes under `"subtypes"`:

```python
{"steady": 0, "transient": 1, "subtypes": {"pulse": 2}}
```

A segment whose `subtype` is listed there is fitted with that model, and any
other with its state's. `res.models` is that spec for the optimiser's
signal, so `fit_report(values, res.segments, {"power": res.models})` fits
each segment as the optimiser did. State keys are still checked, so a
mistyped state name is refused, and a name under `"subtypes"` that is a
state's is refused too.

### The first and last segments

`initial` and `final` restrict which states or subtypes the record's first
and last segments may take. Both default to any, so existing calls keep
their results. Use them when a segment cut off by the record's edge cannot
be judged. A transient at the start of the record has no cold side, so you
cannot tell whether it is a heat-up. `initial={"steady"}` makes the record
start on a plateau. `initial={"steady", "other"}` sends a cut-off transient
to a catch-all instead. A record of one segment must satisfy both. A data
gap is not an edge.

### Example: a flow loop

A heated loop over two days: cold and hot plateaus, a heat-up when the
heater switches on and a cool-down when it switches off. On the second
evening the loop drifts up 4 K with the heater untouched. The whole example
is `docs/source/examples/subtypes.py`, and the test suite runs it.

The scheme names five subtypes and the order they may come in:

```{literalinclude} ../examples/subtypes.py
:language: python
:start-after: "# [scheme]"
:end-before: "# [scheme]"
```

`heat-up` and `cool-down` have the same condition; only `transitions` tells
them apart. `drift` needs the heater not to have switched, so no segment that
contains a switch is a drift.

The heater log gives the minute of each switch, but the loop temperature
answers about half an hour later. The best boundary therefore falls after
the flagged sample, and a heat-up starting there would not contain it.
Widen the mask to cover the lag:

```{literalinclude} ../examples/subtypes.py
:language: python
:start-after: "# [events]"
:end-before: "# [events]"
```

Without the widening, the first day's cool-down starts too late to contain
its switch, and the optimiser calls it a drift.

Then cap the subtypes per day and segment:

```{literalinclude} ../examples/subtypes.py
:language: python
:start-after: "# [optimise]"
:end-before: "# [optimise]"
```

The segments it prints:

```text
Mon 00:00 to Mon 07:59  steady     cold
Mon 08:00 to Mon 10:59  transient  heat-up
Mon 11:00 to Mon 15:59  steady     hot
Mon 16:00 to Mon 18:59  transient  cool-down
Mon 19:00 to Tue 06:59  steady     cold
Tue 07:00 to Tue 09:29  transient  heat-up
Tue 09:30 to Tue 12:59  steady     hot
Tue 13:00 to Tue 15:29  transient  cool-down
Tue 15:30 to Tue 18:59  steady     cold
Tue 19:00 to Tue 20:59  transient  drift
Tue 21:00 to Tue 23:59  steady     cold
```

The night's cold plateau runs from Monday evening into Tuesday, so it
counts against both days. `min_duration_s` keeps every plateau at least an
hour long. `max_duration_s` on the transients is the main lever on run time
with the exponential model.

### Reading the result

Each segment's `state` stays the plain state, so
{py:func}`~rocksteady.report.fit_report`,
{py:func}`~rocksteady.metrics.stats_by_state` and
{py:func}`~rocksteady.optimal.compare_segments` take the result as they
always have. The subtype is in `Segment.subtype`, `None` for a state
without subtypes. `res.counts` has a column per declared subtype after the
states' columns, so each cap can be checked at a glance:

```text
            steady  transient  cold  hot  heat-up  cool-down  drift
period
2026-01-05       3          2     2    1        1          1      0
2026-01-06       4          3     3    1        1          1      1
```

`print(res)` lists the counts by subtype, `res.subtypes` holds the declared
subtypes, and `res.info["events"]` holds the masks.

The reports can pool segments by subtype too. With `by="subtype"`,
{py:func}`~rocksteady.report.fit_report` and
{py:func}`~rocksteady.metrics.stats_by_state` follow each state's row with
one row per subtype its segments carry, sorted by name;
{py:func}`~rocksteady.optimal.count_by_period` follows each state's column
with its subtypes' columns, counting as `res.counts` does; and
{py:func}`~rocksteady.optimal.compare_segments` labels each sample with its
segment's subtype, so two subtyped segmentations (the optimiser's and
`classify_segments`' on the detector's) can be compared subtype by subtype.
A state whose segments carry no subtype keeps a single row, column or
label, as before. These work on segments from any source; a subtype named
like a state (or like `"all"` in a table, `"none"` in a comparison), or
carried by segments of two states, is refused. A report's table of segments
has a `subtype` column whether or not `by="subtype"`.

```{literalinclude} ../examples/subtypes.py
:language: python
:start-after: "# [report]"
:end-before: "# [report]"
```

### Subtypes for segments you already have

`rs.classify_segments` gives subtypes to segments from anywhere else (the
detector's, a log's, another run's) by the same definitions. Your
subtypes and your constraints then share one definition.

```{literalinclude} ../examples/subtypes.py
:language: python
:start-after: "# [classify]"
:end-before: "# [classify]"
```

It runs the optimiser's dynamic program with the breakpoints and states
fixed and no count caps, durations, penalties or subtypes' own models:
those constrain or cost the optimiser's choice of segments, not what a
subtype means. It respects `transitions`, `initial` and `final`, so a
transient between cold and hot is typed a heat-up here too, and ties go by
the same least sum of positions as in the optimiser. On the optimiser's own
segments, given the same `transitions`, `initial` and `final`, it returns
the optimiser's subtypes when the call's caps, durations and penalties name
states only and no subtype has a model of its own. A cap on a subtype can
make the optimiser put a segment in a catch-all that `classify_segments`,
without caps, gives to the capped subtype; a subtype's own model can win a
segment in the optimiser that `classify_segments` gives to a subtype
declared before it. It returns new segments with
`subtype` set; boundaries and states are unchanged.

### Mistakes refused up front

Definitions are checked before the search starts, so a mistake surfaces
before a long run. A `ValueError` names each of these:
- a subtype of an unknown state, or of a state with no model in `models`;
- a subtype named like a state;
- a name in a constraint that is neither a state with a model nor a
  declared subtype;
- a condition naming a mask that `events` does not include, a mask that is
  not boolean or not one value per sample, or a mask named twice in one
  subtype;
- a range that is not two numbers in order (`lo < hi` when both are given);
- a negative `max_rise_s`, or a `rise_fraction` outside (0, 1] or without
  `max_rise_s`;
- a subtype whose duration bounds, with its state's, leave no duration;
- a subtype's `model` that is keyed by state, names no registered model or
  has options its model does not take;
- an `overflow_penalty` naming what `max_counts` does not cap, or a cost
  that is not a finite number of at least 0.

### Methods, pruning and infeasible calls

All three methods take subtypes. `refine` keeps each segment's subtype
while it moves boundaries, and rejects any move after which a segment fails
its subtype's conditions or the constraints fail. It then types its final
segments again by the same program, so ties go the same way for every
method.
Pruning is as exact with subtypes as without: it is decided on a
segment's model cost before any condition, for each model in use (a
state's, and each subtype's own), so whatever subtype could take a segment
changes nothing about it. Without subtypes, it is exact for least
squares once a model is fitted in full, except that the exponential models'
default time-constant bounds scale with the span; give explicit bounds, or
`prune=False`, for a strict guarantee (*Segment costs* in {doc}`methods`).

When the conditions and the hard constraints leave no
segmentation, the `ValueError` names the earliest time after which none can
continue, the period containing it, and the caps and transitions in force
(and `initial` and `final`, when given). In a long batch run, that points to the unusual day. See *Infeasible
constraints* in {doc}`methods`, which also says what such a call costs.
When `classify_segments` finds no choice of subtypes, its `ValueError`
names the earliest segment boundary after which none can continue, and the
`transitions`, `initial` and `final` in force.

### Soft caps

Every cap is hard by default. On an unusual day whose data cannot fit them,
a hard cap distorts the segmentation (two heat-ups merged into one long
segment) or leaves none, which stops a batch run over months of records.
`overflow_penalty={name: cost}` makes the caps of those names soft: each
segment beyond the cap in a period adds `cost` to the objective, counted as
the cap counts (once against every period a segment touches). The other
caps stay hard. The overflow per period is not capped; only the cost keeps
it in check, so choose it against the SSE a segment saves. `transitions`, `initial`,
`final` and durations stay hard.

```{literalinclude} ../examples/subtypes.py
:language: python
:start-after: "# [soft]"
:end-before: "# [soft]"
```

`res.overflow` is indexed as `res.counts` is, with a column per
soft-capped name holding the segments beyond its cap in each period, and
`res.info["overflow_cost"]` is what they added to the objective
(`objective = SSE + penalties + overflow cost`). `print(res)` names the
caps that overflowed, so a batch run can flag the days to look at.

The search stays exact: past the cap every further segment costs the same,
so the program counts a soft cap only up to the cap and needs no more count
layers than a hard one. A cheap overflow can make the cheap constant fits
that bound the search choose segments a slower model cannot fit, and the
search then runs without pruning, exactly but no faster.

### A sketch of a reactor's scheme

The same mechanism covers a reactor. Power is in percent of full power. The
thresholds and the masks (one per actuator) are the adapter's to choose:

```{literalinclude} ../examples/subtypes.py
:language: python
:start-after: "# [reactor]"
:end-before: "# [reactor]"
```

- **Zero power and at power** are told apart by level, as cold and hot are.
- **Startup** must reach operating power (`peak`), so a bump that never gets
  there is not a startup; from zero power it can only be `other`.
- **Pulse** must reach a high peak within `PULSE_RISE_S` (`max_rise_s`),
  which tells it from an ordinary increase in power. It returns to zero
  power.
- **Shutdown** has no conditions. `transitions` alone makes it the
  transient from at power to zero power.
- **Perturbation** and **maneuver** both go from at power to at power. They
  are told apart by the actuator that drove them, each with its own mask.
- **Other** is the catch-all, declared last. Cap it, or give it a higher
  `penalty`, to make the optimiser look harder for a known subtype.

Durations can be set per subtype too: `max_duration_s={"pulse": 60}` caps
a pulse's length without changing other transients.

The example file checks this scheme with `classify_segments` on a record
segmented by hand. Four of its transients are `other`: a rise to low power
(no startup), an excursion at power that no rod drove (neither a maneuver
nor a perturbation), a slow excursion to a pulse's height and a fast, low
bump (neither of them a pulse).
