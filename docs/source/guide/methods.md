# Methods

This page states what rocksteady computes, in the notation of the code.
Every formula links to the function that implements it. Citations are to
the {doc}`../references`.

## Notation

Samples arrive at times $t_i = t_0 + i\,\Delta$, where the sample spacing
$\Delta$ is `Timing.sample_period_s`. A **signal** $s(x_i)$ is a scalar
computed from the raw sample $x_i$: a column, or a builder such as the mean
or spread of several columns (`rocksteady.signals`).

## Blocks and the window

The detector keeps the last $W$ samples, the **window**
(`Timing.window_s` rounded to whole samples). It evaluates the last $n$
whole **blocks** of $B$ samples each (`Timing.block_s` rounded), where
$n = \lfloor W / B \rfloor$. Each criterion sees the block means of one
signal,

$$
\bar y_m = \frac{1}{|V_m|} \sum_{i \in V_m} s(x_i), \qquad m = 1, \dots, n,
$$

where $V_m$ holds the valid samples of block $m$. All $B$ samples are used
when none is missing. With tolerated gaps (`Timing.max_gap_s`), a window is
evaluated only if every block has $|V_m| \ge c_{\min} B$
(`Timing.min_block_coverage`); otherwise it is undecided. The block times
are centred on the window,

$$
\tau_m = \left(m - 1 - \tfrac{n - 1}{2}\right) B\,\Delta ,
$$

so slopes are per second, and the intercept is the window's mid-level.
These are the times of complete blocks. A block with tolerated missing
samples is placed at the mean time of its valid samples, which is where its
mean was measured, so criteria never treat a partial block as if it were
centred.
Block means average out sample-to-sample noise and much of its short-range
correlation.

## The three-way test and the margin ratio

A criterion returns a statistic $S \ge 0$, its standard uncertainty $u$, a
limit $L > 0$, and a coverage factor $k$
({py:class}`~rocksteady.core.CriterionResult`). The true value of the
statistic is taken to lie in $S \pm k u$. Comparing that interval with the
limit gives one of three outcomes:

$$
\text{outcome} =
\begin{cases}
\text{PASS} & S + k u \le L \\
\text{FAIL} & S - k u > L \\
\text{UNDECIDED} & \text{otherwise.}
\end{cases}
$$

This is an **equivalence test** (Wellek 2010; Schuirmann 1987): a window
passes only when the data *demonstrate* that the quantity is within its
limit, not merely when they fail to show that it is outside. The **margin
ratio** is

$$
\eta = \frac{S + k u}{L} ,
$$

so $\eta \le 1$ passes. A window passes when every criterion passes, and the
verdict reports the worst ratio, $\eta = \max_c \eta_c$
(`Verdict.eta`, `Verdict.limiting`). Any non-finite value, a negative $u$ or
$k$, or a non-positive $L$ makes a result invalid, and an invalid result is
undecided, never a pass.

## Criteria

### Deviation from a model

{py:class}`~rocksteady.criteria.DeviationCriterion` bounds the largest
deviation of the block means from a fitted level or line:

$$
S = \max_m \left| \bar y_m - \hat y_m \right|, \qquad
\hat y_m =
\begin{cases}
\bar{\bar y} & \text{(constant)} \\
a + b\,\tau_m & \text{(linear: ordinary least squares)}
\end{cases}
$$

With `relative=True` the statistic becomes $S / \max(|\bar{\bar y}|, f)$,
where $f$ is `scale_floor`. The uncertainty $u = \sigma(\bar{\bar y})$ comes
from the noise model: the standard deviation of a block mean due to
measurement noise, possibly a function of level, in the units of the
statistic (relative when `relative=True`). The default coverage is $k = 3$.

This is a calibrated heuristic, not a confidence statement about the true
maximum deviation. The maximum absolute residual of pure noise over $n$
blocks is itself about 1.5 to 2 $\sigma$ for $n = 10$, so calibrate $L$ and
$\sigma$ together on data known to be steady.

### Drift: an equivalence test on the slope

{py:class}`~rocksteady.criteria.DriftCriterion` certifies that the drift is
smaller than a limit $b_{\text{crit}}$. With the ordinary least-squares slope

$$
\hat b = \frac{\sum_m (\tau_m - \bar\tau)(\bar y_m - \bar{\bar y})}{S_{\tau\tau}},
\qquad S_{\tau\tau} = \sum_m (\tau_m - \bar\tau)^2 ,
$$

the test is $|\hat b| + k\,u_b \le b_{\text{crit}}$. Block means of real
signals are autocorrelated, and an uncertainty that ignores this certifies
limits the data do not support. The uncertainty method is pluggable
(`method=`, `registry.slope_methods`).

#### `"ar1"` (default)

{py:func}`~rocksteady.stats.slope_ar1` models the residuals
$e_m = \bar y_m - \hat a - \hat b \tau_m$ as an AR(1) process. Their lag-1
autocorrelation is bias-corrected for a regression on a trend,
analogously to Marriott and Pope (1954), and clipped:

$$
r_1 = \frac{\sum_m e_m e_{m+1}}{\sum_m e_m^2}, \qquad
\hat\phi = \operatorname{clip}\!\left(r_1 + \frac{2 + 4 r_1}{n},\; 0,\; 0.98\right).
$$

With $R_{ij} = \hat\phi^{|i-j|}$, the design matrix $X = [\mathbf 1, \tau - \bar\tau]$,
and the residual-maker $M = I - X (X^\top X)^{-1} X^\top$, the error
variance is estimated without bias given $\hat\phi$, and the slope variance
is exact for AR(1) errors:

$$
\hat\sigma^2 = \frac{e^\top e}{\operatorname{tr}(M R)}, \quad
\operatorname{tr}(M R) = n - \operatorname{tr}\!\left[(X^\top X)^{-1} X^\top R X\right], \quad
u_b^2 = \hat\sigma^2 \, \frac{(\tau - \bar\tau)^\top R\,(\tau - \bar\tau)}{S_{\tau\tau}^2}.
$$

$R$ is never formed: $R x$ is computed in $O(n)$ by a forward and a
backward recursion ({py:func}`rocksteady.fit.ar1_apply`). The coverage
factor is the one-sided Student-$t$ quantile at the effective number of
degrees of freedom:

$$
k = t_{0.975,\,\nu}, \qquad \nu = \max\!\left(\left\lfloor n \tfrac{1 - \hat\phi}{1 + \hat\phi} \right\rfloor - 2,\; 1\right).
$$

#### `"nw"`

{py:func}`~rocksteady.stats.slope_nw` is a Newey and West (1987)
heteroskedasticity- and autocorrelation-consistent standard error with a
Bartlett kernel. With $g_m = (\tau_m - \bar\tau)\, e_m$ and a bandwidth
$\ell$ equal to the number of leading positive residual autocorrelations,
capped at $\lfloor n/4 \rfloor$:

$$
u_b^2 = \frac{1}{S_{\tau\tau}^2}\left[ \sum_m g_m^2 + 2 \sum_{j=1}^{\ell} \left(1 - \frac{j}{\ell + 1}\right) \sum_m g_m g_{m+j} \right],
\qquad k = 2 .
$$

#### Calibration

Monte Carlo false-pass rates at the limit (true slope equal to
$b_{\text{crit}}$, AR(1) block-mean noise, 3000 replications,
`tools/coverage_mc.py`). The nominal rate is 2.5 % for `"ar1"` and 2.3 % for
`"nw"`:

| blocks $n$ | 10 | | | | 40 | | | | 90 | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $\phi$ | 0 | .5 | .8 | .9 | 0 | .5 | .8 | .9 | 0 | .5 | .8 | .9 |
| `"ar1"` | 1.7 | 5.1 | 7.7 | 9.3 | 1.9 | 3.9 | 4.6 | 4.8 | 2.4 | 3.0 | 3.3 | 4.1 |
| `"nw"` | 7.2 | 16.0 | 24.6 | 27.9 | 3.6 | 10.6 | 17.7 | 23.5 | 2.8 | 7.3 | 12.9 | 18.2 |

### Band

{py:class}`~rocksteady.criteria.BoundCriterion` tests that the window mean
lies in $[\ell, h]$:

$$
S = \left|\bar{\bar y} - \tfrac{\ell + h}{2}\right|, \quad L = \tfrac{h - \ell}{2}, \quad
u = \frac{s_{\bar y}}{\sqrt{n / \tau_{\text{int}}}}, \quad k = 2,
$$

where $s_{\bar y}$ is the standard deviation of the block means and
$\tau_{\text{int}}$ (in blocks) is the integrated autocorrelation time of
the linearly detrended block means. With fewer than about 10 blocks,
$\tau_{\text{int}}$, and so $u$, is unreliable.

### Asymptote: the remaining approach

{py:class}`~rocksteady.criteria.AsymptoteCriterion` assumes a first-order
approach, $y(s) = y_\infty + A\,e^{-s/\tau}$ with $s = t - t_0$ from the
first block time, and bounds the **remaining approach** at the current
sample. That is $(B - 1)\Delta/2$ after the nominal centre of the last block,
for $B$ samples per block and sample spacing $\Delta$; the detector supplies
them through `Criterion.bind_timing`. The current sample is always a valid
one, so this holds even when tolerated gaps have moved the block times:

$$
d(\tau) = A\,e^{-s_e/\tau} = -\tau\, y'(s_e).
$$

For a fixed $\tau$ the model is linear in $(y_\infty, A)$. Each block mean
averages the curve over that block's valid samples $V_m$, so the basis is
averaged over the same samples. This is exact for complete blocks and for
blocks with tolerated gaps, whatever the pattern of missing samples. (A
block's centroid time alone is exact only for a line.)

$$
g_m(\tau) = \frac{1}{|V_m|} \sum_{i \in V_m} e^{-s_i/\tau} ,
$$

which the detector supplies through the block masks
(`Criterion.uses_block_masks`). With $g$ centred, $g_c = g - \bar g$:

$$
\hat A(\tau) = \frac{g_c^\top (y - \bar y)}{g_c^\top g_c}, \qquad
\operatorname{SSE}(\tau), \qquad
u\big(\hat d(\tau)\big) = e^{-s_e/\tau}\, \frac{\hat\sigma_\tau \sqrt{g_c^\top R\, g_c}}{g_c^\top g_c},
\qquad \hat\sigma_\tau^2 = \frac{\operatorname{SSE}(\tau)}{\operatorname{tr}(M_\tau R)} ,
$$

with $R$ the AR(1) correlation, as in the drift test. Its $\hat\phi$ is taken
from the residuals at the best $\hat\tau$ (refined between grid points) and
bias-corrected for three parameters: $r_1 + (3 + 5 r_1)/n$, clipped to
$[0, 0.98]$.

A test that trusts $\hat\tau$ passes too often: noise that makes the curve
look sharper makes $\hat\tau$ shorter and $\hat d$ smaller. So the test uses
every $\tau$ the data allow. The profile-likelihood confidence set is

$$
\mathcal T = \left\{ \tau \in [\tau_{\min}, \tau_{\max}] :\; \operatorname{SSE}(\tau) \le \operatorname{SSE}(\hat\tau) \left(1 + \frac{t^2_{0.975,\nu}}{\nu}\right) \right\},
\qquad \nu = \max\!\left(\left\lfloor n \tfrac{1 - \hat\phi}{1 + \hat\phi}\right\rfloor - 3,\; 1\right),
$$

on a log-spaced grid, widened by one grid point at each end. With
$k = t_{0.975,\nu}$, the interval for $|d|$ is

$$
\Big[\, \min_{\tau \in \mathcal T} \big(|\hat d(\tau)| - k\,u(\tau)\big)^+ ,\;\; \max_{\tau \in \mathcal T} \big(|\hat d(\tau)| + k\,u(\tau)\big) \Big],
$$

and the three-way test compares it with the limit.

$\tau_{\max}$ is the caller's bound on the process's time constant. A
first-order approach is monotone and slows down, so $|y'(s_e)|$ is at most
the window's mean slope $|b|$ and $|d| \le \tau_{\max} |b|$. When the window
cannot resolve the curve (a flat window, or one much shorter than $\tau$),
$\mathcal T$ reaches $\tau_{\max}$. There $g$ is nearly linear in $s$, and
$\hat d(\tau_{\max}) \approx -\tau_{\max} b$: the bound applies by itself,
with no separate mode and no data-dependent switch.

Monte Carlo false-pass rates at the limit (noise $0.1\,L$,
$\tau_{\max} = 2\tau$, `tools/coverage_mc.py --asymptote`, 2000
replications; nominal 2.5 %):

| blocks, $\phi$ | span $0.5\tau$ | $1\tau$ | $2\tau$ | $4\tau$ | $8\tau$ |
|---|---|---|---|---|---|
| 20, 0 / 0.5 / 0.8 | 2.0 / 3.2 / 5.1 | 1.4 / 3.0 / 4.7 | 0.8 / 1.9 / 3.9 | 0.0 / 0.1 / 0.1 | 0 |
| 40, 0 / 0.5 / 0.8 | 1.5 / 2.4 / 2.5 | 1.1 / 2.1 / 2.5 | 0.5 / 1.2 / 1.4 | 0 | 0 |
| 90, 0 / 0.5 / 0.8 | 1.6 / 1.9 / 2.0 | 1.0 / 1.5 / 1.6 | 0.5 / 0.5 / 1.0 | 0 | 0 |

With $\tau_{\max}$ equal to the true $\tau$, the true value sits at the edge
of the search, and the rates reach at most 6.0 % at 40 blocks and 5.1 % at
90. Give $\tau_{\max}$ a margin. One evaluation costs about 1 ms (1.5 ms with 300-sample blocks).

## Integrated autocorrelation time and the uncertainty of a mean

For a stationary series with autocorrelation $\rho(j)$, the variance of the
mean of $N$ samples is exactly

$$
\operatorname{Var}(\bar r) = \frac{\sigma^2}{N}\left[1 + 2\sum_{j=1}^{N-1}\left(1 - \frac{j}{N}\right)\rho(j)\right] ,
$$

with no terms beyond lag $N - 1$, where there are no sample pairs
(Zhang 2006; Sokal 1997). rocksteady estimates the correlations and
truncates the sum at an estimated cutoff $J \le N - 1$:

$$
\hat\rho(j) = \frac{\overline{r_i r_{i+j}}}{\overline{r_i^2}}, \qquad
\tau_{\text{int}} = \Delta \cdot \max\!\left(1,\; 1 + 2 \sum_{j=1}^{J} \hat\rho(j)\right),
$$

on residuals $r$: about a fit for a signal, de-meaned for an error. The sum
stops before the first non-positive $\hat\rho$, or at $J$, a quarter of the
longest piece. This lag-normalised estimator with a first-zero cutoff was within
3 % of the true uncertainty for $\phi \le 0.9$ in Monte Carlo checks. The
significance-based cutoff of Zhang (2006) underestimated by about 25 %
(`stats.tau_int`).

Over several pieces (the segments of one state, split at gaps and
non-finite samples), lag products are pooled within pieces only
({py:func}`~rocksteady.stats.acf_tau`, `pair_weighted=True`). Each lag is
weighted by $C_j / N$, the fraction of samples that have a lag-$j$ partner
inside their piece:

$$
\operatorname{Var}(\bar r) = \frac{s^2}{N}\left[1 + 2 \sum_j \frac{C_j}{N}\, \hat\rho(j)\right]
\;\Rightarrow\;
N_{\text{eff}} = \frac{N \Delta}{\tau_{\text{int}}}, \qquad u(\text{mean}) = \frac{s}{\sqrt{N_{\text{eff}}}} .
$$

For one piece the weight is $(N - j)/N$, the exact finite-sample factor.
`N_eff < 10` is flagged: a normal coverage factor is not appropriate there.
A Type B uncertainty $u_B$ combines in quadrature,
$u_c = \sqrt{u(\text{mean})^2 + u_B^2}$ (Zhang 2006, eq. 29; JCGM 100:2008).

## Disturbances, settling and holding

**Disturbances** mark samples where the system is acted on:
- {py:class}`~rocksteady.disturbances.StepChange` is active when the means
  of the last $m$ and the previous $m$ samples of a signal differ by more
  than a threshold.
- {py:class}`~rocksteady.disturbances.EventLog` is active on every sample
  whose interval $(t_{i-1}, t_i]$ overlaps a logged event
  $[t_s, t_s + d]$.

**The sequence** ({py:class}`~rocksteady.core.Detector`) runs for each
sample:
1. A disturbance labels the sample DISTURBED and resets the clocks.
2. An override may force a label.
3. Until $W$ samples have arrived, the label is INDETERMINATE.
4. Until $\max\!\left(W, \operatorname{round}\big((T_{\text{window}} + T_{\text{settle}})/\Delta\big)\right)$
   quiet samples have passed since the last disturbance, the label is
   SETTLING (INDETERMINATE at the start of a record). Here
   $T_{\text{window}}$ and $T_{\text{settle}}$ are `window_s` and `settle_s`
   as given, not rounded first (`Timing.quiet_samples`).
5. Then the criteria are evaluated. FAIL gives SETTLING. PASS gives HOLDING
   until the run of passing windows has lasted `hold_s`, then STEADY.
   UNDECIDED gives INDETERMINATE, or keeps the last decided label with
   `hold_when_undecided`.

Every verdict carries a {py:class}`~rocksteady.core.Reason`. The detector
is causal, so batch evaluation of a record reproduces the online labels
exactly.

## Fits by segment

A record is split into **segments by state**
({py:func}`~rocksteady.states.segments_from`):
- a steady run is backfilled over the window that certified it;
- transient and indeterminate runs fill the rest.

Each segment is fitted with the model for its state
({py:mod}`rocksteady.fit`):

| Model | Form | Parameters |
|---|---|---|
| `poly` (order $p$) | $\sum_{j=0}^{p} c_j x^j$, with $x$ the time scaled to $[-1, 1]$ | derivatives at the centre |
| `first_order` | $y_\infty + (y_0 - y_\infty)\, e^{-(t - t_0)/\tau}$ | $y_0, y_\infty, \tau, \lambda = 1/\tau$ |
| `exp` | $c + A\, e^{(t - t_0)/T}$ | $c, A, T$ (growth $T > 0$) |

The exponential models are fitted by **variable projection** (Golub and
Pereyra 1973, 2003). For a fixed time constant the model is linear in the
other parameters, so the sum of squares is minimised over $\log|\tau|$
alone: a log-spaced grid, then golden-section refinement, with a linear
least-squares solve at each point.

**Parameter covariance.** With $J$ the Jacobian at the solution (in the
linear parameters and $\log\tau$) and AR(1) residual correlation $R$, the
covariance is a sandwich (White 1980), generalising the drift test above:

$$
\operatorname{Cov}(\hat\theta) = (J^\top J)^{-1} J^\top (\hat\sigma^2 R)\, J (J^\top J)^{-1}, \qquad
\hat\sigma^2 = \frac{e^\top e}{\operatorname{tr}(M R)},
$$

with $\hat\phi = \operatorname{clip}\!\big(r_1 + (m + (m + 2) r_1)/n,\, 0,\, 0.995\big)$
for $m$ fitted parameters. For a mean this reduces to Marriott and Pope's
$(1 + 3 r_1)/n$. The columns of $J$ are scaled to unit norm and factorised by
SVD, so the result does not depend on units. Parameters along a direction
the data cannot determine get infinite variance. Derived parameters ($\tau$,
$\lambda$, $A$ at the start) follow by the delta method. `tau_unidentifiable`
flags a time constant the segment cannot resolve: span $< \tau/2$,
$\tau < 5\Delta$, or $u(\tau) > \tau/2$.

**Per segment** ({py:func}`~rocksteady.metrics.signal_stats`): the mean;
the residual standard deviation with the fit's degrees of freedom removed;
$\tau_{\text{int}}$ and $N_{\text{eff}}$ of the residuals; and
$u(\text{mean}) = s / \sqrt{N_{\text{eff}}}$.

## Comparison statistics

For $e = a - b$ (a fit against data, or one series against another;
{py:func}`~rocksteady.metrics.error_stats`):

$$
\text{bias} = \bar e, \quad
\text{RMSE} = \sqrt{\overline{e^2}}, \quad
\text{NRMSE} = \frac{\text{RMSE}}{\text{norm}(b)}, \quad
u(\text{bias}) = \frac{s_e}{\sqrt{N_{\text{eff}}}} ,
$$

with $\text{norm}(b) = |\bar b|$ by default (or the range or standard
deviation of $b$), and $r$ Pearson's correlation. $N_{\text{eff}}$ comes
from the **de-meaned, not detrended** error, because slow wander in an
offset is real uncertainty in the bias.

## Best-fit segments under constraints

{py:func}`~rocksteady.optimal.optimal_segments` chooses breakpoints
$0 = b_0 < b_1 < \dots < b_K = N$ and a state $s_k$ for each segment, to
minimise

$$
\sum_{k=1}^{K} \Big[ C_{s_k}(b_{k-1}, b_k) + \lambda_{s_k} \Big],
\qquad C_s(p, q) = \min_\theta \sum_{i=p}^{q-1} \big(y_i - f_s(t_i; \theta)\big)^2 ,
$$

subject to the constraints:
- at most $N_s$ segments of each state per period, a segment counting
  against every period it touches;
- the allowed state transitions;
- per-state minimum and maximum durations.

The default penalty is BIC-like (Schwarz 1978):
$\lambda_s = \hat\sigma^2 (p_s + 1) \ln N$, with $p_s$ the model's
parameters and $\hat\sigma = 1.4826\,\operatorname{MAD}(\Delta y)/\sqrt 2$
estimated robustly from first differences.

This is optimal partitioning by **dynamic programming** (Bellman 1961;
Jackson et al. 2005; reviewed by Truong et al. 2020). It is extended to
track the last segment's state and the counts used in the current
period. With $F(k, s, c)$ the best cost of a segmentation ending at
candidate breakpoint $k$ with last state $s$ and counts $c$:

$$
F(k, s, c) = \min_{j < k,\; s',\; c'} \Big[ F(j, s', c') + C_s(B_j, B_k) + \lambda_s \Big],
$$

where $c$ follows from $c'$ by adding the new segment in each period it
touches; the transition $s' \to s$ must be allowed and the duration in
range. For $m$ candidate breakpoints, $S$ states and $L$ count layers, this
costs $O(m^2 S^2 L)$ segment evaluations.

**Segment costs.** Constant and linear costs come from range sums of
$y$, $y^2$, $x$, $x^2$ and $xy$. These are kept in blocks with compensated
(Neumaier 1974) block totals, and carry a rounding-error bound; a segment
whose residual is not resolved above that bound is fitted directly. Other
models are fitted with `fit.fit_model`. A candidate segment is dropped once
its own cost exceeds the total cost of a known feasible solution. For least
squares that is exact: once its model is fitted in full, a segment's cost
never decreases as it grows. The exponential models' default search bounds
scale with the span; give explicit bounds, or `prune=False`, for a strict
guarantee.

Optimal breakpoints for $K - 1$ segments are not, in general, a subset of
those for $K$. Fitting $y = t$ with constants splits at $1/2$ for two
segments and at $1/3, 2/3$ for three. That is why `"subset"` (choosing among
given candidates) is an approximation, `"refine"` searches locally around
them, and `"exact"` searches every sample or a grid.
