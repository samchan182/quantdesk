"""quantdesk — derivatives pricing, execution and pre-trade risk, built from scratch.

Every headline number in REPORT.md is produced by code in this package, from a
registered seed, and is traceable to a result JSON in results/.
"""

from quantdesk.rng import rng, spawn

__all__ = ["rng", "spawn"]
