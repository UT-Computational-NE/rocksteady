"""
Monte Carlo calibration of the drift test's slope-uncertainty methods.

For each registered slope method, window length (in blocks) and AR(1)
autocorrelation of the block means, estimate the false-pass rate of the
equivalence test ``|b| + k u_b <= limit`` when the true slope equals the
limit. With the limit far above the noise this is P((b - limit)/u_b < -k),
estimated here from both tails of zero-slope noise.

Usage: python tools/coverage_mc.py [--reps 3000] [--seed 0]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, nargs="+", default=[10, 40, 90])
    ap.add_argument("--phi", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9])
    args = ap.parse_args()
    cells = [(n, p) for n in args.blocks for p in args.phi]
    print("false-pass rate (%) at the limit; columns are (blocks, phi)")
    print(f"{'method':8s}" + "".join(f"{f'{n},{p:g}':>9s}" for n, p in cells))
    for method in registry.slope_methods.names():
        rates = [100 * false_pass_rate(method, n, p, args.reps, args.seed) for n, p in cells]
        print(f"{method:8s}" + "".join(f"{r:9.1f}" for r in rates))


if __name__ == "__main__":
    main()
