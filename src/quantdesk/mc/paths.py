"""Geometric Brownian motion path generation, and the chunked driver over it.

The scheme, exact in law at every step because GBM has a closed-form transition::

    S_{t+dt} = S_t * exp((r - q - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z)

Accumulated in logs and exponentiated once per point, rather than multiplied
step by step: it costs the same and avoids compounding rounding through 252
multiplications.

Three deliberate separations, each of which pays for itself later:

**Drawing normals is separate from turning normals into prices.** Every
function here takes `Z` as an argument. That is what makes antithetic sampling
a matter of passing `-Z`, common random numbers in M6 a matter of passing the
same `Z` to two revaluations, and the numba-versus-NumPy comparison a
comparison of arithmetic rather than of random streams.

**The NumPy reference is permanent.** It is not scaffolding to be deleted once
the kernel works. It is the thing the kernel is tested against, and the two are
written to perform their arithmetic in the same order so that the comparison is
meaningful down to the last bit. Debugging a numba kernel without a known-good
reference is a trap; CLAUDE.md §4 M2 says not to walk into it.

**Terminal-only simulation is a distinct path from stepped simulation.** A
European payoff needs only `S_T`, whose distribution is known exactly, so
simulating it in one step introduces **no discretisation error at all**. M5
depends on that: the gap it measures between Monte Carlo and closed form must
be pure sampling noise, and it can only be pure sampling noise if no time
stepping happened.

**Memory.** 200,000 paths x 252 steps of float64 is 403 MB in a single
allocation, and there is no reason to hold it: every product here needs only a
running statistic per path. `simulate` therefore works in chunks — 10,000 paths
at a time by default, so 20 chunks — reducing each chunk to per-path payoffs
and discarding the paths. Peak path memory is about 20 MB regardless of the
total path count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from quantdesk.config import Market
from quantdesk.mc.estimators import (
    Accumulator,
    PairedAccumulator,
    antithetic_average,
    control_variate_samples,
)
from quantdesk.rng import spawn

__all__ = [
    "Engine",
    "DEFAULT_CHUNK_PATHS",
    "NUMBA_AVAILABLE",
    "PairedRun",
    "terminal_prices",
    "price_paths",
    "chunk_sizes",
    "simulate",
    "simulate_paired",
]

Engine = Literal["numpy", "numba"]

#: Paths held in memory at once. 10,000 x 252 float64 is about 20 MB.
DEFAULT_CHUNK_PATHS = 10_000

try:  # numba is a declared dependency; degrade loudly rather than silently
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on a broken install
    NUMBA_AVAILABLE = False


def _drift_and_diffusion(market: Market, dt: float) -> tuple[float, float]:
    """Per-step log drift and log diffusion scale."""
    drift = (market.rate - market.div_yield - 0.5 * market.vol**2) * dt
    diffusion = market.vol * np.sqrt(dt)
    return float(drift), float(diffusion)


def _validate(market: Market, maturity: float) -> None:
    if not isinstance(market, Market):
        raise TypeError(f"market must be a Market, got {type(market).__name__}")
    if maturity < 0:
        raise ValueError(f"maturity must be non-negative, got {maturity}")


# --------------------------------------------------------------------------
# Terminal prices: one exact step, no discretisation error
# --------------------------------------------------------------------------
def terminal_prices(z: np.ndarray, market: Market, maturity: float) -> np.ndarray:
    """``S_T`` given standard normal draws, in a single exact step."""
    _validate(market, maturity)
    z = np.asarray(z, dtype=np.float64)
    drift, diffusion = _drift_and_diffusion(market, maturity)
    return market.spot * np.exp(drift + diffusion * z)


# --------------------------------------------------------------------------
# Stepped paths: NumPy reference and numba kernel
# --------------------------------------------------------------------------
def _price_paths_numpy(z: np.ndarray, spot: float, drift: float, diffusion: float) -> np.ndarray:
    """Reference implementation. The kernel below must reproduce this exactly.

    The log-drift is placed in column 0 and the increments in columns 1..m
    before a single cumulative sum, so the accumulation order is
    ``((log_s0 + inc_0) + inc_1) + ...`` — identical to the sequential loop in
    the kernel. Adding ``log_s0`` to a cumsum of the increments instead would
    associate the additions differently and cost bit-for-bit agreement.
    """
    n_paths, n_steps = z.shape
    buf = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    buf[:, 0] = np.log(spot)
    np.multiply(z, diffusion, out=buf[:, 1:])
    np.add(buf[:, 1:], drift, out=buf[:, 1:])
    np.cumsum(buf, axis=1, out=buf)

    out = np.empty_like(buf)
    out[:, 0] = spot  # exactly the input spot, not exp(log(spot))
    np.exp(buf[:, 1:], out=out[:, 1:])
    return out


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True, fastmath=False)
    def _paths_kernel(z, log_spot, drift, diffusion, out):  # pragma: no cover - compiled
        n_paths, n_steps = z.shape
        for i in prange(n_paths):
            running = log_spot
            for j in range(n_steps):
                running += drift + diffusion * z[i, j]
                out[i, j + 1] = np.exp(running)

    def _price_paths_numba(z: np.ndarray, spot: float, drift: float, diffusion: float) -> np.ndarray:
        out = np.empty((z.shape[0], z.shape[1] + 1), dtype=np.float64)
        out[:, 0] = spot
        _paths_kernel(z, float(np.log(spot)), drift, diffusion, out)
        return out


def price_paths(
    z: np.ndarray,
    market: Market,
    maturity: float,
    engine: Engine = "numpy",
) -> np.ndarray:
    """Full price paths from normal draws.

    Parameters
    ----------
    z:
        Shape ``(n_paths, n_steps)`` standard normal draws.
    engine:
        ``"numpy"`` for the reference, ``"numba"`` for the compiled kernel.

    Returns
    -------
    Array of shape ``(n_paths, n_steps + 1)``. Column 0 is the spot, exactly,
    for every path.
    """
    _validate(market, maturity)
    z = np.ascontiguousarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z must be 2-D (n_paths, n_steps), got shape {z.shape}")
    if z.shape[1] == 0:
        raise ValueError("n_steps must be at least 1")

    dt = maturity / z.shape[1]
    drift, diffusion = _drift_and_diffusion(market, dt)

    if engine == "numpy":
        return _price_paths_numpy(z, market.spot, drift, diffusion)
    if engine == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError("numba is not installed; engine='numba' is unavailable")
        return _price_paths_numba(z, market.spot, drift, diffusion)
    raise ValueError(f"engine must be 'numpy' or 'numba', got {engine!r}")


# --------------------------------------------------------------------------
# Chunked driver
# --------------------------------------------------------------------------
def chunk_sizes(n_units: int, chunk: int) -> list[int]:
    """Split ``n_units`` into blocks of at most ``chunk``, largest-first, exact."""
    if n_units < 1:
        raise ValueError(f"n_units must be >= 1, got {n_units}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    full, remainder = divmod(n_units, chunk)
    sizes = [chunk] * full + ([remainder] if remainder else [])
    assert sum(sizes) == n_units
    return sizes


def simulate(
    *,
    seed: int,
    market: Market,
    maturity: float,
    n_paths: int,
    payoff: Callable[[np.ndarray], np.ndarray],
    n_steps: int | None = None,
    chunk_paths: int = DEFAULT_CHUNK_PATHS,
    antithetic: bool = False,
    engine: Engine = "numpy",
) -> Accumulator:
    """Run a chunked Monte Carlo and return the accumulated estimate.

    Parameters
    ----------
    n_paths:
        Total simulated paths. Under ``antithetic=True`` this must be even, and
        the statistical sample size is ``n_paths // 2`` — the accumulator
        reports both, separately, precisely so they cannot be confused.
    payoff:
        Called once per chunk. Receives ``S_T`` of shape ``(n,)`` when
        ``n_steps is None``, or paths of shape ``(n, n_steps + 1)`` otherwise,
        and returns one value per path. Discounting is the payoff's job, since
        only it knows the payment date.
    chunk_paths:
        Paths held in memory at once, counted as paths rather than as samples,
        because it is a memory budget.

    Notes
    -----
    Each chunk draws from its own generator, derived via
    ``SeedSequence.spawn``. Chunk count therefore changes the numbers, which is
    why ``chunk_paths`` is recorded as a benchmark input rather than treated as
    a free tuning knob.
    """
    _validate(market, maturity)
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"antithetic requires an even n_paths, got {n_paths}")
    if n_steps is not None and n_steps < 1:
        raise ValueError(f"n_steps must be >= 1 when given, got {n_steps}")

    n_units = n_paths // 2 if antithetic else n_paths
    chunk_units = max(1, chunk_paths // 2 if antithetic else chunk_paths)
    sizes = chunk_sizes(n_units, chunk_units)
    generators = spawn(seed, len(sizes))

    def prices(draws: np.ndarray) -> np.ndarray:
        if n_steps is None:
            return terminal_prices(draws, market, maturity)
        return price_paths(draws, market, maturity, engine=engine)

    acc = Accumulator()
    for size, generator in zip(sizes, generators, strict=True):
        shape = (size,) if n_steps is None else (size, n_steps)
        z = generator.standard_normal(shape)

        if antithetic:
            samples = antithetic_average(payoff(prices(z)), payoff(prices(-z)))
            acc.update(samples, paths_used=2 * size)
        else:
            acc.update(payoff(prices(z)), paths_used=size)

    return acc


@dataclass
class PairedRun:
    """One simulation carrying a target payoff and its control side by side.

    `joint` holds the marginal moments of both and their covariance — enough
    for the realised correlation, and so for the `1 - rho^2` prediction. When a
    `beta` is supplied, `controlled` additionally accumulates
    `X - beta*(Y - E[Y])`, so the uncontrolled and controlled arms come out of
    the same draws and differ only by the control adjustment rather than by
    sampling noise.
    """

    joint: PairedAccumulator
    controlled: Accumulator | None
    beta: float | None
    expected_control: float | None
    antithetic: bool

    @property
    def uncontrolled_estimate(self) -> float:
        return self.joint.mean_x

    @property
    def uncontrolled_standard_error(self) -> float:
        return self.joint.standard_error_x


def simulate_paired(
    *,
    seed: int,
    market: Market,
    maturity: float,
    n_paths: int,
    target: Callable[[np.ndarray], np.ndarray],
    control: Callable[[np.ndarray], np.ndarray],
    expected_control: float | None = None,
    beta: float | None = None,
    n_steps: int | None = None,
    chunk_paths: int = DEFAULT_CHUNK_PATHS,
    antithetic: bool = False,
    engine: Engine = "numpy",
) -> PairedRun:
    """Simulate a target payoff and a control payoff on the same paths.

    The control's exact expectation must be known — here it comes from M1's
    closed form — which is what lets `Y - E[Y]` be subtracted without biasing
    the estimate.

    Under `antithetic=True` **both** legs are pair-averaged before anything
    else happens, and the control variate is then applied to the pair averages.
    That ordering matters: the optimal beta for pair averages is not the
    optimal beta for individual paths, because pairing changes the joint
    distribution of the two payoffs. The caller must supply a beta estimated
    under the same pairing convention, and `bench/variance_reduction.py` runs a
    separate pilot for each arm precisely for this reason.
    """
    _validate(market, maturity)
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"antithetic requires an even n_paths, got {n_paths}")
    if (beta is None) != (expected_control is None):
        raise ValueError("beta and expected_control must be supplied together, or neither")

    n_units = n_paths // 2 if antithetic else n_paths
    chunk_units = max(1, chunk_paths // 2 if antithetic else chunk_paths)
    sizes = chunk_sizes(n_units, chunk_units)
    generators = spawn(seed, len(sizes))

    def prices(draws: np.ndarray) -> np.ndarray:
        if n_steps is None:
            return terminal_prices(draws, market, maturity)
        return price_paths(draws, market, maturity, engine=engine)

    joint = PairedAccumulator()
    controlled = Accumulator() if beta is not None else None

    for size, generator in zip(sizes, generators, strict=True):
        shape = (size,) if n_steps is None else (size, n_steps)
        z = generator.standard_normal(shape)

        if antithetic:
            plus, minus = prices(z), prices(-z)
            x = antithetic_average(target(plus), target(minus))
            y = antithetic_average(control(plus), control(minus))
            paths_used = 2 * size
        else:
            simulated = prices(z)
            x, y = target(simulated), control(simulated)
            paths_used = size

        joint.update(x, y, paths_used=paths_used)
        if controlled is not None:
            controlled.update(
                control_variate_samples(x, y, expected_control, beta), paths_used=paths_used
            )

    return PairedRun(
        joint=joint,
        controlled=controlled,
        beta=beta,
        expected_control=expected_control,
        antithetic=antithetic,
    )
