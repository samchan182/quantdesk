"""Brownian bridge barrier crossing.

**The idea in one line.** You observe the price once per step, but it moves
continuously in between; it can cross a barrier and return unseen.

**Where the formula comes from, and why it is a product of two logarithms.**
Conditional on its two endpoints, a Brownian motion over one step is a Brownian
bridge, and the probability that a bridge starting at `a` and ending at `b`
(both above zero) ever touches zero has the closed form `exp(-2ab/(variance))`.
Working in logs, the distances from the barrier at the two endpoints are
`ln(S_t/B)` and `ln(S_{t+dt}/B)`, and the log-price has variance `sigma^2*dt`
over the step. Substituting gives

    lower barrier:  p = exp(-2 * ln(S_t/B) * ln(S_{t+dt}/B) / (sigma^2 * dt))
    upper barrier:  p = exp(-2 * ln(B/S_t) * ln(B/S_{t+dt}) / (sigma^2 * dt))

The product of two logarithms is therefore the product of the two endpoint
distances — the bridge has to come back from *both* ends, so the probability of
touching falls off with each. It is one formula, not two: the upper case is the
lower case applied to the reflected process.

**The direction of the bias, which decides the sign of every number in M7.**
Discrete monitoring misses the excursions between observations, so it *understates*
the crossing probability, leaves too many paths alive, and therefore
**overprices a knock-out**. Applying the correction lowers the price.

**When not to use this.** If the contract itself is monitored on daily closing
prices and the simulation steps daily, the simulation is already exact and this
correction must not be applied — it would price a contract nobody wrote. That
judgement is M7's subject and is recorded in DEFENSE.md.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["Direction", "crossing_probability", "bridge_survival"]

Direction = Literal["down", "up"]


def crossing_probability(
    s_start: np.ndarray,
    s_end: np.ndarray,
    barrier: float,
    vol: float,
    dt: float,
    direction: Direction = "down",
) -> np.ndarray:
    """Probability the path touched ``barrier`` between two observed endpoints.

    Returns 1.0 wherever an endpoint is already on the far side of the barrier:
    the crossing is then observed directly and there is nothing to infer.
    """
    if vol <= 0 or dt <= 0:
        raise ValueError("crossing probability needs positive volatility and step size")
    if barrier <= 0:
        raise ValueError(f"barrier must be positive, got {barrier}")

    s_start = np.asarray(s_start, dtype=np.float64)
    s_end = np.asarray(s_end, dtype=np.float64)
    variance = vol * vol * dt

    if direction == "down":
        already = (s_start <= barrier) | (s_end <= barrier)
        d_start = np.log(np.maximum(s_start, barrier) / barrier)
        d_end = np.log(np.maximum(s_end, barrier) / barrier)
    elif direction == "up":
        already = (s_start >= barrier) | (s_end >= barrier)
        d_start = np.log(barrier / np.minimum(s_start, barrier))
        d_end = np.log(barrier / np.minimum(s_end, barrier))
    else:
        raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")

    probability = np.exp(-2.0 * d_start * d_end / variance)
    return np.where(already, 1.0, probability)


def bridge_survival(
    paths: np.ndarray,
    barrier: float,
    vol: float,
    dt: float,
    generator: np.random.Generator,
    direction: Direction = "down",
) -> np.ndarray:
    """Boolean mask of paths that never touched the barrier, bridge-corrected.

    At each step a uniform is drawn and the path is knocked out with the
    bridge probability for that step. Paths whose observed endpoints already
    breach the barrier are knocked out with certainty, since
    `crossing_probability` returns 1 for them.

    The uniforms come from the caller's generator, so the result is
    reproducible from the caller's seed like everything else (R5). Note that
    this consumes randomness *in addition to* the path draws, so a bridge-
    corrected run and an uncorrected one on the same seed share their paths but
    not their subsequent stream.
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError(f"expected paths of shape (n, steps+1), got {paths.shape}")

    probability = crossing_probability(
        paths[:, :-1], paths[:, 1:], barrier, vol, dt, direction=direction
    )
    uniforms = generator.random(probability.shape)
    return ~np.any(uniforms < probability, axis=1)


def observed_survival(
    paths: np.ndarray, barrier: float, direction: Direction = "down"
) -> np.ndarray:
    """Survival under *discrete* monitoring: only the observed closes count.

    The uncorrected comparison arm, and — when the contract really is monitored
    on those closes — the correct answer rather than an approximation to one.
    """
    paths = np.asarray(paths, dtype=np.float64)
    observations = paths[:, 1:]
    if direction == "down":
        return ~np.any(observations <= barrier, axis=1)
    if direction == "up":
        return ~np.any(observations >= barrier, axis=1)
    raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
