# rocksteady

Steady-state detection for any measured system: a causal detector that certifies when signals are steady, and tools that split a finished record into segments, fit them and report on them. Applications supply the signals, thresholds and domain knowledge; nothing here is specific to one of them.

## Detection

**Label**:
The detector's verdict for one sample: disturbed, settling, indeterminate, holding or steady.
_Avoid_: state (for a sample), status, flag

**Criterion**:
A test of one signal over the trailing window that the detector must pass to certify a sample as steady.
_Avoid_: rule, check, condition (that is a subtype's)

**Disturbance**:
An external action, found in the signals or supplied from a log, that resets the detector's clocks.
_Avoid_: upset, perturbation (for the detector's concept)

**Event**:
An externally logged control, configuration or alarm action, with a start and a duration.
_Avoid_: trigger, incident

**Limit**:
The threshold a criterion's statistic is compared against.
_Avoid_: tolerance (for the criterion's threshold), bound

**Override**:
Domain knowledge that forces a label regardless of the criteria, such as a system below its operating floor being steady.
_Avoid_: domain rule, forcing rule

## Segments

**Segment**:
A contiguous run of samples in one state; a record's segments cover it in order without overlap.
_Avoid_: interval, block (a block is a slice of the detector's window), run

**State**:
What a segment is: steady, transient or indeterminate.
_Avoid_: base state, state type, type, label (for a segment), regime, mode

**Subtype**:
A caller-named refinement of one state, such as a cold or a hot steady segment, or a startup transient.
_Avoid_: kind, class, category, label, regime, mode

**Condition**:
A test on a segment alone (its level, peak, rise time, or whether it contains an event) that it must meet to take a subtype.
_Avoid_: rule, criterion, filter

**Event mask**:
A named boolean series with one value per sample, flagging the samples at which an event occurred; a subtype's event condition reads it.
_Avoid_: event (for the mask), flags, trigger

**Best-fit segment**:
A segment chosen by the optimiser to describe a finished record as well as possible within its constraints; it is not certified steady by the detector.
_Avoid_: optimal state, detected segment

**Constraint**:
A restriction on the optimiser's whole segmentation: counts per period, which state or subtype may follow which, which may take the record's first and last segments, durations, and penalties.
_Avoid_: limit (that is a criterion's threshold), rule

**Period**:
The span of time (a day, a week) that per-period counts of segments apply to.
_Avoid_: window (the detector's), sample period (say sample spacing)
