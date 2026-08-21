"""Statistical helpers — cluster bootstrap and McNemar's test (§5.1)."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


def cluster_bootstrap_ci(
    cluster_values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lo, hi) using cluster bootstrap.

    `cluster_values` is one scalar per cluster (e.g., one rate per scenario).
    """
    arr = np.asarray(cluster_values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    n = arr.size
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = stat(arr[idx])
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(stat(arr)), float(lo), float(hi)


def upper_95_one_sided(successes: int, trials: int) -> float:
    """One-sided 95% upper bound.

    Uses Clopper-Pearson when SciPy is available.  The artifact also runs on a
    minimal Python installation; in that case it uses the exact zero-event
    bound and the one-sided Wilson upper bound for nonzero counts, recording
    the distinction in the output methodology.
    """
    if trials == 0:
        return 1.0
    if successes == trials:
        return 1.0
    if successes == 0:
        return float(1.0 - 0.05 ** (1.0 / trials))
    try:
        from scipy.stats import beta
        return float(beta.ppf(0.95, successes + 1, trials - successes))
    except ImportError:
        z = 1.6448536269514722
        p = successes / trials
        denom = 1.0 + z * z / trials
        centre = p + z * z / (2.0 * trials)
        radius = z * ((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5)
        return float(min(1.0, (centre + radius) / denom))


def wilson_95_interval(successes: int, trials: int) -> Tuple[float, float]:
    """Two-sided 95% Wilson score interval for a nonzero proportion."""
    if trials <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    radius = z * ((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** .5) / denom
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def mcnemar(b: int, c: int) -> Tuple[float, float]:
    """Standard McNemar's χ² with continuity correction.

    `b` = pairs where method A succeeds and B fails.
    `c` = pairs where method B succeeds and A fails.
    Returns (chi2, two-sided p-value).
    """
    from scipy.stats import chi2 as chi2dist

    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1.0 - chi2dist.cdf(chi2, df=1)
    return float(chi2), float(p)


def percentile(values: Sequence[float], p: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def mcnemar_exact(b: int, c: int) -> float:
    """Return the exact two-sided McNemar p-value for paired outcomes."""
    import math

    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = min(b, c)
    numerator = 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1))
    probability = math.ldexp(numerator, -discordant)
    return float(min(1.0, probability))
