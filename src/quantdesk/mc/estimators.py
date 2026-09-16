"""Sample accumulation and standard errors.

The one thing this module exists to get right is *what counts as a sample*.

Under antithetic sampling you draw `Z`, evaluate the payoff at `+Z` and `-Z`,
and average the two. That average is **one** observation, not two::

    Y_i = (f(Z_i) + f(-Z_i)) / 2
    estimate = mean(Y)
    standard error = std(Y, ddof=1) / sqrt(N)      # N pairs, not 2N paths

Computing the standard error over the 2N individual paths instead treats
negatively correlated draws as independent, understates the variance of the
estimator, and inflates the apparent benefit of the technique — the first trap
in CLAUDE.md §5, and the reason `Accumulator.n_samples` counts pairs and says so.

Variance is accumulated with Chan's parallel update rather than by summing
squares. Sum-of-squares is the obvious implementation and it is catastrophic
here: a discounted option price has a mean of order 10 and a standard deviation
of order 10, but a deep out-of-the-money payoff can have a mean of 0.01 against
values of 100, and `E[X^2] - (E[X])^2` then cancels away most of the significant
digits. Chan's form never forms that difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Accumulator",
    "PairedAccumulator",
    "antithetic_average",
    "relative_standard_error",
    "control_variate_samples",
]


@dataclass
class Accumulator:
    """Running count, mean and variance over samples arriving in chunks.

    Chunked simulation (M2 simulates in blocks so that 200,000 x 252 float64 is
    never allocated at once) means statistics must combine across blocks
    without keeping the paths. This holds `n`, the mean, and `M2` — the sum of
    squared deviations from the running mean — and merges each new block in
    with Chan, Golub and LeVeque's update, which is stable regardless of the
    order or size of the blocks.
    """

    n_samples: int = 0
    mean: float = 0.0
    m2: float = 0.0
    #: Total simulated paths behind those samples. Equals n_samples for crude
    #: Monte Carlo and 2 * n_samples under antithetic pairing. Recorded so a
    #: report can state compute cost and statistical sample size separately —
    #: confusing them is how the antithetic trap gets written.
    n_paths: int = 0
    #: Sample-count history per chunk, for reproducibility checks.
    chunk_sizes: list[int] = field(default_factory=list)

    def update(self, samples: np.ndarray, *, paths_used: int | None = None) -> "Accumulator":
        """Merge a block of samples. Returns self so calls can be chained."""
        samples = np.asarray(samples, dtype=np.float64).ravel()
        n_b = samples.size
        if n_b == 0:
            return self
        if not np.all(np.isfinite(samples)):
            raise ValueError("non-finite sample in accumulator update")

        mean_b = float(samples.mean())
        # Two-pass within the block: accurate, and the block is already in memory.
        m2_b = float(np.sum(np.square(samples - mean_b)))

        if self.n_samples == 0:
            self.n_samples, self.mean, self.m2 = n_b, mean_b, m2_b
        else:
            n_a, mean_a, m2_a = self.n_samples, self.mean, self.m2
            n = n_a + n_b
            delta = mean_b - mean_a
            self.mean = mean_a + delta * (n_b / n)
            self.m2 = m2_a + m2_b + delta * delta * (n_a * n_b / n)
            self.n_samples = n

        self.n_paths += n_b if paths_used is None else paths_used
        self.chunk_sizes.append(n_b)
        return self

    @property
    def variance(self) -> float:
        """Sample variance with ddof=1. NaN for a single sample, which is honest."""
        if self.n_samples < 2:
            return float("nan")
        return self.m2 / (self.n_samples - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def standard_error(self) -> float:
        """std / sqrt(n_samples).

        Note the denominator: samples, not paths. Under antithetic pairing
        those differ by a factor of two, and using paths here is the trap.
        """
        if self.n_samples < 2:
            return float("nan")
        return float(np.sqrt(self.variance / self.n_samples))

    def as_dict(self) -> dict[str, float | int]:
        """Result-JSON shaped summary, with the estimate and its error together."""
        return {
            "estimate": self.mean,
            "standard_error": self.standard_error,
            "sample_std": self.std,
            "n_samples": self.n_samples,
            "n_paths": self.n_paths,
            "n_chunks": len(self.chunk_sizes),
        }


def antithetic_average(payoff_plus: np.ndarray, payoff_minus: np.ndarray) -> np.ndarray:
    """Pair a payoff evaluated at ``+Z`` and at ``-Z`` into one sample per pair."""
    plus = np.asarray(payoff_plus, dtype=np.float64)
    minus = np.asarray(payoff_minus, dtype=np.float64)
    if plus.shape != minus.shape:
        raise ValueError(f"antithetic arms must match in shape, got {plus.shape} vs {minus.shape}")
    return 0.5 * (plus + minus)


def relative_standard_error(acc: Accumulator) -> float:
    """Standard error as a fraction of the estimate. Undefined at a zero estimate."""
    if acc.mean == 0.0:
        return float("nan")
    return acc.standard_error / abs(acc.mean)


@dataclass
class PairedAccumulator:
    """Running moments of two jointly-sampled quantities, plus their co-moment.

    Needed for the control variate, which requires Cov(X, Y) and hence a
    covariance that survives chunking. Extends Chan's update to the cross term::

        C = C_a + C_b + (mean_x_b - mean_x_a) * (mean_y_b - mean_y_a) * n_a * n_b / n

    and is stable for the same reason the variance update is: it never forms
    ``E[XY] - E[X]E[Y]``, which cancels catastrophically when the means are
    large relative to the spread.
    """

    n_samples: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    c_xy: float = 0.0
    n_paths: int = 0
    chunk_sizes: list[int] = field(default_factory=list)

    def update(
        self, x: np.ndarray, y: np.ndarray, *, paths_used: int | None = None
    ) -> "PairedAccumulator":
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError(f"paired samples must match in shape, got {x.shape} vs {y.shape}")
        n_b = x.size
        if n_b == 0:
            return self
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("non-finite sample in paired accumulator update")

        mx_b, my_b = float(x.mean()), float(y.mean())
        dx, dy = x - mx_b, y - my_b
        m2x_b, m2y_b = float(np.sum(dx * dx)), float(np.sum(dy * dy))
        cxy_b = float(np.sum(dx * dy))

        if self.n_samples == 0:
            self.n_samples, self.mean_x, self.mean_y = n_b, mx_b, my_b
            self.m2_x, self.m2_y, self.c_xy = m2x_b, m2y_b, cxy_b
        else:
            n_a = self.n_samples
            n = n_a + n_b
            delta_x, delta_y = mx_b - self.mean_x, my_b - self.mean_y
            weight = n_a * n_b / n
            self.m2_x += m2x_b + delta_x * delta_x * weight
            self.m2_y += m2y_b + delta_y * delta_y * weight
            self.c_xy += cxy_b + delta_x * delta_y * weight
            self.mean_x += delta_x * (n_b / n)
            self.mean_y += delta_y * (n_b / n)
            self.n_samples = n

        self.n_paths += n_b if paths_used is None else paths_used
        self.chunk_sizes.append(n_b)
        return self

    @property
    def var_x(self) -> float:
        return self.m2_x / (self.n_samples - 1) if self.n_samples > 1 else float("nan")

    @property
    def var_y(self) -> float:
        return self.m2_y / (self.n_samples - 1) if self.n_samples > 1 else float("nan")

    @property
    def covariance(self) -> float:
        return self.c_xy / (self.n_samples - 1) if self.n_samples > 1 else float("nan")

    @property
    def correlation(self) -> float:
        denom = np.sqrt(self.var_x * self.var_y)
        return float(self.covariance / denom) if denom > 0 else float("nan")

    @property
    def beta_star(self) -> float:
        """Cov(X, Y) / Var(Y) — the variance-minimising control coefficient.

        Emphatically not 1. Beta is 1 only when X and Y move one-for-one; using
        1 otherwise weakens the reduction, and if the true beta is below 0.5 it
        makes the variance *worse* than no control variate at all.
        """
        return float(self.covariance / self.var_y) if self.var_y > 0 else float("nan")

    @property
    def standard_error_x(self) -> float:
        if self.n_samples < 2:
            return float("nan")
        return float(np.sqrt(self.var_x / self.n_samples))

    @property
    def theoretical_variance_ratio(self) -> float:
        """1 − ρ², the best a single control variate can do."""
        rho = self.correlation
        return float(1.0 - rho * rho)


def control_variate_samples(
    target: np.ndarray, control: np.ndarray, expected_control: float, beta: float
) -> np.ndarray:
    """``X - beta * (Y - E[Y])``, one controlled sample per input sample.

    Unbiased for any fixed ``beta``, because ``E[Y - E[Y]] = 0`` — the control
    variate shifts variance around without moving the mean. That is only true
    while ``beta`` is fixed independently of this sample; a ``beta`` fitted to
    the same draws introduces an O(1/N) bias, which is why it is estimated on a
    separate pilot.
    """
    target = np.asarray(target, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if target.shape != control.shape:
        raise ValueError(f"shapes must match, got {target.shape} vs {control.shape}")
    return target - beta * (control - expected_control)
