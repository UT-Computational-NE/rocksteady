# Glossary

**Block** — A group of consecutive samples reduced to a block mean. Criteria
evaluate windows of block means rather than raw samples.

**Condition** — A caller-defined check of a segment's samples used to decide
whether it matches a subtype. Conditions describe the subtype; optimizer
constraints describe which segmentations are allowed.

**Constraint** — An optimization requirement or scoring setting that guides
segment selection. These include segment-count caps per period, allowed state
or subtype transitions, duration bounds, and per-segment penalties in the
objective.

**Criterion** — A statistical test applied to a signal over a trailing window
of block means. A criterion returns a statistic, its uncertainty, a limit, and
an uncertainty multiplier.

**Limit** — A criterion's threshold for its statistic, in the statistic's
units. `CriterionResult.limit` is this threshold; it is distinct from the
constraints on an optimized segmentation.

**Override** — Caller-supplied domain knowledge that may force a detector
label on a sample. An override can defer to the detector by returning no label.

**Period** — A span used by `optimal_segments` to count how many segments of a
state or subtype are used. A segment is counted in every period it touches.

**Sample spacing** — The time between consecutive samples on the sampling
grid. This is the preferred prose term; the public `Timing.sample_period_s`
field retains its existing name.

**Segment** — A contiguous run of samples assigned to one state. An optimized
segmentation fits each segment with the model for its state. Established API
names such as `batch.accepted_intervals` and `Report.by_interval` keep their
existing names.

**State** — One of the segment labels `steady`, `transient`, or
`indeterminate`. Detector verdicts can be more specific (`disturbed`,
`settling`, `holding`) and batch segmentation groups those verdicts into
states.

**Subtype** — A caller-defined name for a kind of segment within a state, such
as a cold plateau or a startup. A subtype can be defined by conditions on the
segment and by its allowed neighbors.
