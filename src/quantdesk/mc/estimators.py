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

__all__ = ["Accumulator", "antithetic_average", "relative_standard_error"]


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
