"""
Monte Carlo calibration of the drift test's slope-uncertainty methods, and
of the asymptote criterion.

Drift: for each registered slope method, window length (in blocks) and AR(1)
autocorrelation of the block means, estimate the false-pass rate of the
equivalence test ``|b| + k u_b <= limit`` when the true slope equals the
limit. With the limit far above the noise this is P((b - limit)/u_b < -k),
estimated here from both tails of zero-slope noise.

Asymptote (``--asymptote``): a first-order approach whose true remaining
approach at the window end equals the limit, with AR(1) block-mean noise,
for each window length, autocorrelation and window span / tau; the
criterion's ``tau_max`` is ``--tau-factor`` times the true tau.

Usage: PYTHONPATH=src python tools/coverage_mc.py [--reps 3000] [--seed 0] [--asymptote]
"""

import argparse

import numpy as np

from rocksteady import registry


def ar1_noise(rng: np.random.Generator, n: int, phi: float, burn: int = 300) -> np.ndarray:
    e = rng.normal(size=n + burn)
    x = np.empty_like(e)
    x[0] = e[0] / np.sqrt(1 - phi * phi)
    for i in range(1, len(e)):
        x[i] = phi * x[i - 1] + e[i]
    return x[burn:]


def false_pass_rate(method: str, n_blocks: int, phi: float, reps: int, seed: int = 0, block_s: float = 30.0,
                    **options) -> float:
    """One-sided false-pass rate (fraction) of ``method`` at the limit."""
    rng = np.random.default_rng(seed)
    fn = registry.slope_methods.get(method)
    t = (np.arange(n_blocks) - (n_blocks - 1) / 2.0) * block_s
    hits = 0
    for _ in range(reps):
        b, u, k = fn(t, ar1_noise(rng, n_blocks, phi), **options)
        hits += abs(b) > k * u
    return hits / (2.0 * reps)


def asymptote_pass_rate(n_blocks: int, phi: float, span_over_tau: float, reps: int, seed: int = 0,
                        noise_over_limit: float = 0.1, remaining_over_limit: float = 1.0, tau_factor: float = 2.0,
                        block_s: float = 30.0) -> float:
    """
    Pass rate (fraction) of ``AsymptoteCriterion`` (limit 1) on
    ``y = -A exp(-s / tau) + noise``, with the remaining approach at the
    window end equal to ``remaining_over_limit``. At 1 this is the false-pass
    rate; below 1 it is the power.
    """
    from rocksteady import AsymptoteCriterion, Outcome, Timing

    rng = np.random.default_rng(seed)
    timing = Timing(sample_period_s=1.0, block_s=block_s, window_s=n_blocks * block_s)  # 1 s samples
    b, dt = timing.block_samples, timing.sample_period_s
    t = (np.arange(n_blocks) - (n_blocks - 1) / 2.0) * timing.effective_block_s  # block centres, as a Detector
    samples = (t - 0.5 * (b - 1) * dt)[:, None] + np.arange(b)[None, :] * dt   # every sample's time
    s_end = (t[-1] + 0.5 * (b - 1) * dt) - samples[0, 0]                        # the current sample
    tau = s_end / span_over_tau
    amp = remaining_over_limit * np.exp(s_end / tau)
    curve = (-amp * np.exp(-(samples - samples[0, 0]) / tau)).mean(axis=1)      # block means of the curve
    crit = AsymptoteCriterion("x", 1.0, tau_max=tau_factor * tau)
    crit.bind_timing(timing)  # as a Detector does
    scale = noise_over_limit * np.sqrt(1 - phi * phi)  # block-mean noise with standard deviation noise_over_limit
    passes = sum(crit.evaluate(t, curve + scale * ar1_noise(rng, n_blocks, phi)).outcome == Outcome.PASS
                 for _ in range(reps))
    return passes / reps


def main_asymptote(args):
    ratios = args.span_over_tau
    print(f"asymptote: pass rate (%), noise/limit {args.noise}, remaining/limit {args.remaining}, "
          f"tau_max = {args.tau_factor} tau; rows (blocks, phi), columns window span / tau")
    print(f"{'':14s}" + "".join(f"{r:>8g}" for r in ratios))
    for n in args.blocks:
        for p in args.phi:
            rates = [100 * asymptote_pass_rate(n, p, r, args.reps, args.seed, args.noise, args.remaining,
                                               args.tau_factor) for r in ratios]
            print(f"{f'{n}, {p:g}':14s}" + "".join(f"{x:8.1f}" for x in rates))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, nargs="+", default=[10, 40, 90])
    ap.add_argument("--phi", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9])
    ap.add_argument("--asymptote", action="store_true", help="calibrate AsymptoteCriterion instead")
    ap.add_argument("--span-over-tau", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    ap.add_argument("--noise", type=float, default=0.1, help="block-mean noise / limit (asymptote)")
    ap.add_argument("--remaining", type=float, default=1.0, help="true remaining approach / limit (asymptote)")
    ap.add_argument("--tau-factor", type=float, default=2.0, help="tau_max / true tau (asymptote)")
    args = ap.parse_args()
    if args.asymptote:
        return main_asymptote(args)
    cells = [(n, p) for n in args.blocks for p in args.phi]
    print("false-pass rate (%) at the limit; columns are (blocks, phi)")
    print(f"{'method':8s}" + "".join(f"{f'{n},{p:g}':>9s}" for n, p in cells))
    for method in registry.slope_methods.names():
        rates = [100 * false_pass_rate(method, n, p, args.reps, args.seed) for n, p in cells]
        print(f"{method:8s}" + "".join(f"{r:9.1f}" for r in rates))


if __name__ == "__main__":
    main()
