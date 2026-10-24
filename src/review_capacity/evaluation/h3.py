"""The rule that selects the H3 configuration (DR-010; protocol Hypotheses; Issue #159).

H3 is confirmatory for one configuration, chosen on V2 only, before the Test period is unsealed.
Among the eight configurations (DR-011 order: LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b,
FTT-b) it is the one with the most V2 frauds inside the daily top-K of R3 at the primary capacity
(2 %, K = 56). Counts are compared as integers. Only when counts are exactly equal, the tie-break
steps apply in order:

1. the higher V2 Average Precision, compared exactly;
2. the fixed order above, declared only to make the rule deterministic.

The rule uses V2 labels, which is within V2's role (they also select the R1 threshold and fit the
R4 calibrators), and no Test information.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from review_capacity.development.records import CONFIGURATION_NAMES
from review_capacity.errors import DataValidationError

PRIMARY_CAPACITY = 0.02
RULE_VERSION = "1"  # DR-010, selection rule


@dataclass(frozen=True)
class H3Candidate:
    """One configuration's V2 result under R3 at the primary capacity."""

    configuration: str
    frauds_captured: int
    frauds: int
    average_precision: float


@dataclass(frozen=True)
class H3Choice:
    """The selected configuration and the tie-break steps that decided it, in order."""

    selected: str
    tie_break_steps: tuple[str, ...]


def select_h3(candidates: Sequence[H3Candidate]) -> H3Choice:
    """Apply the selection rule to the eight configurations (in DR-011 order)."""
    names = [c.configuration for c in candidates]
    if names != list(CONFIGURATION_NAMES):
        raise DataValidationError(f"The H3 selection needs the eight configurations, not {names}")
    for c in candidates:
        counts = (c.frauds_captured, c.frauds)
        if any(isinstance(n, bool) or not isinstance(n, int) for n in counts):
            raise DataValidationError(f"{c.configuration}: the fraud counts must be integers")
        precision = c.average_precision
        if isinstance(precision, bool) or not isinstance(precision, int | float):
            raise DataValidationError(f"{c.configuration}: Average Precision must be a number")
        if not 0.0 <= precision <= 1.0:  # False for NaN too
            raise DataValidationError(
                f"{c.configuration}: Average Precision must lie within [0, 1], not {precision!r}"
            )
    if len({c.frauds for c in candidates}) != 1 or candidates[0].frauds < 1:
        raise DataValidationError("Every configuration must count the same, non-zero V2 frauds")
    if any(not 0 <= c.frauds_captured <= c.frauds for c in candidates):
        raise DataValidationError("The frauds captured lie between 0 and the V2 frauds")
    best = max(c.frauds_captured for c in candidates)
    tied = [c for c in candidates if c.frauds_captured == best]
    steps: list[str] = []
    if len(tied) > 1:
        steps.append("average_precision")
        best_ap = max(c.average_precision for c in tied)
        tied = [c for c in tied if c.average_precision == best_ap]
        if len(tied) > 1:
            steps.append("fixed_order")
    return H3Choice(tied[0].configuration, tuple(steps))
