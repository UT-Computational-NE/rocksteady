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
the mask alone, whether or not the signal is finite there. A subtype has no
model of its own: its segments are fitted with its state's model. So a
subtype changes which segments the optimiser may choose, not how it fits
them.

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
`initial` or `final` returns the segments, objective and counts that 0.5.0
returned, for any call that 0.5.0 also accepted.

### How constraints combine on a state and its subtypes

| Constraint | With a state and its subtypes |
|---|---|
| `max_counts` | Both caps bind. A state's cap counts all of its segments, whatever their subtype, so `{"steady": 3, "hot": 1}` allows at most three steady segments a period, and at most one of them hot. A segment counts against every period it touches, by subtype as by state. |
| `transitions` | A state's name stands for all of its subtypes. The allowed pairs are the union over all entries, so `"drift": ["steady"]` lets a drift be followed by any steady subtype. |
| `initial`, `final` | As for `transitions`: a state's name stands for all of its subtypes. |
| `min_duration_s`, `max_duration_s` | Both bind: a segment's duration must lie within its state's bounds and its subtype's. A subtype that no duration can satisfy is refused. |
| `penalty` | A subtype's penalty replaces its state's. With `"bic"`, or when the mapping does not name it, a subtype takes its state's. A higher penalty makes a subtype rarer. |

Subtypes of one state may overlap: two of them can accept the same segment.
The optimiser chooses by total cost. Two subtypes that accept a segment and
carry the same penalty cost exactly the same, and the tie goes to the one
declared first. A catch-all declared last, with the same penalty as the
others, therefore takes only the segments that no earlier subtype can take
within the constraints. Precisely, among segmentations of equal objective
the optimiser takes the least sum of the subtypes' positions. That sum can
also decide an exact tie between states or boundaries, so declaring
subtypes can change which of two equally good segmentations comes back.

Overlap is how a transient's subtype can follow from its neighbours. A
heat-up and a cool-down can have the same conditions (the heater switched)
and be told apart by `transitions` alone: a heat-up goes from cold to hot, a
cool-down from hot to cold.

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
fixed and no count caps, durations or penalties: those constrain the
optimiser's choice of segments, not what a subtype means. It respects
`transitions`, `initial` and `final`, so a transient between cold and hot is
typed a heat-up here too, and ties go to the subtype declared first, as in
the optimiser. On the optimiser's own segments, given the same
`transitions`, `initial` and `final`, it returns the optimiser's subtypes
when the call's caps, durations and penalties name states only. A cap on a
subtype can make the optimiser put a segment in a catch-all that `classify_segments`,
without caps, gives to the capped subtype. It returns new segments with
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
- a subtype whose duration bounds, with its state's, leave no duration.

### Methods, pruning and infeasible calls

All three methods take subtypes. `refine` keeps each segment's subtype
while it moves boundaries, and rejects any move after which a segment fails
its subtype's conditions or the constraints fail. It then types its final
segments again by the same program, so ties go the same way for every
method.
Pruning stays exact: it is decided on a segment's model cost before any
condition, and once its model is fitted in full a longer segment never
costs less, whatever subtype could take it.

Constraints stay hard. When the conditions and constraints leave no
segmentation, the `ValueError` names the earliest time after which none can
continue, the period containing it, and the caps and transitions in force.
In a long batch run, that points to the unusual day. See *Infeasible
constraints* in {doc}`methods`, which also says what such a call costs.
When `classify_segments` finds no choice of subtypes, its `ValueError`
names the earliest segment boundary after which none can continue, and the
`transitions`, `initial` and `final` in force.

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
segmented by hand, including a run to low power (no startup, so `other`)
and a slow excursion to a pulse's height (no pulse, so `other`).
