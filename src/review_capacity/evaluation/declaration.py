"""The evaluation declaration (``config/evaluation.yaml``): the machine-readable form of the
scientific settings of the confirmatory evaluation (Issue #187).

The file is read once; the same bytes are parsed, hashed and recorded (``load_declaration``), so a
run's records always show the declaration in use. The code fixes which keys exist and their types:
a missing, repeated or mistyped entry raises ``SpecificationError``, and no value falls back to a
default. The sources of the values are the protocol and DR-002, DR-003, DR-005, DR-006, DR-007,
DR-010 and DR-011; unit tests compare the file with those documents and with the development
constants it repeats (so the development record contracts are unchanged). The data-derived
false-negative slice boundaries are frozen in this file by a later issue.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import Any

from review_capacity.config import SEED_LIMIT, _read_yaml
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.models.run_record import SettingsSnapshot
from review_capacity.paths import ProjectPaths

# The child streams of DR-006 (Resampling conventions), in their fixed order.
STREAMS: tuple[str, ...] = (
    "pigeonhole_h1_h2",
    "naive_s1",
    "day_block_s2",
    "day_block_h3",
    "day_block_rq3",
    "pigeonhole_rq3",
    "pigeonhole_false_negative",
    "day_block_secondary",
)
BIT_GENERATOR = "PCG64"
HOLM_FAMILY: tuple[str, ...] = ("H1", "H2", "H3")
REFERENCE = "logistic_regression"  # DR-011 §5; outside the eight configurations
WEIGHTINGS = ("u", "b")  # unweighted, balanced
DELTA_TOLERANCE = 1e-12  # h3_delta equals h3_target * h3_delta_relative up to this


@dataclass(frozen=True)
class Resampling:
    """B, the root seed, the child streams and the day-block lengths (DR-006, DR-003)."""

    replicates: int
    root_seed: int
    bit_generator: str
    streams: tuple[str, ...]
    block_length: int
    block_length_sensitivity: tuple[int, ...]
    circular: bool


@dataclass(frozen=True)
class Levels:
    """Interval levels: differences and secondary outcomes; H1 interpretation, H2 and H3."""

    difference: float
    interpretation: float


@dataclass(frozen=True)
class Inference:
    """SESOI, the H3 target and tolerance, the Holm level and family (DR-003, DR-005)."""

    sesoi: float
    sesoi_sensitivity: tuple[float, ...]
    h3_target: float
    h3_delta_relative: float
    h3_delta: float
    h3_tolerance_sensitivity: tuple[float, ...]
    alpha: float
    holm_family: tuple[str, ...]


@dataclass(frozen=True)
class Capacity:
    """The review-capacity shares and the budgets K derived from V2 (DR-002)."""

    shares: tuple[Fraction, ...]
    primary: Fraction
    budgets: tuple[int, ...]


@dataclass(frozen=True)
class Configurations:
    """The eight configurations in DR-011 order, their families and the reference."""

    order: tuple[str, ...]
    tree: tuple[str, ...]
    neural: tuple[str, ...]
    reference: str


@dataclass(frozen=True)
class CalibrationReporting:
    """Reliability bins and the metric clipping constant (protocol Calibration; DR-007 §8)."""

    reliability_bins: int
    epsilon: float


@dataclass(frozen=True)
class Rq3:
    """The cost-ratio grid, the upper end of the continuous range and the friction values."""

    ratio_grid: tuple[int, ...]
    ratio_max: int
    friction: tuple[int, ...]


@dataclass(frozen=True)
class FalseNegative:
    """The rule parameters of the false-negative slices (protocol False-Negative Analysis)."""

    minimum_frauds: int
    amount_quantiles: int
    score_band_multiples: tuple[int, ...]
    interval_level: float
    test_weeks: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EvaluationDeclaration:
    """Everything the evaluation declaration fixes, as data."""

    resampling: Resampling
    levels: Levels
    inference: Inference
    capacity: Capacity
    configurations: Configurations
    contrasts: tuple[tuple[str, str], ...]
    calibration_reporting: CalibrationReporting
    rq3: Rq3
    false_negative: FalseNegative


def _mapping(raw: object, what: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SpecificationError(f"{what} must be a mapping, not {raw!r}")
    missing = [key for key in keys if key not in raw]
    unknown = [key for key in raw if key not in keys]
    if missing or unknown:
        raise SpecificationError(f"{what}: missing {missing}, unknown {unknown}")
    return raw


def _integer(value: object, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpecificationError(f"{what} must be an integer of at least {minimum}, not {value!r}")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SpecificationError(f"{what} must be a finite number, not {value!r}")
    return float(value)


def _unit(value: object, what: str) -> float:
    """A number strictly between 0 and 1."""
    number = _number(value, what)
    if not 0 < number < 1:
        raise SpecificationError(f"{what} must lie strictly between 0 and 1, not {value!r}")
    return number


def _flag(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise SpecificationError(f"{what} must be true or false, not {value!r}")
    return value


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecificationError(f"{what} must be a non-empty string, not {value!r}")
    return value


def _items(value: object, what: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise SpecificationError(f"{what} must be a non-empty list, not {value!r}")
    return value


def _unique[T](items: Sequence[T], what: str) -> tuple[T, ...]:
    if len(set(items)) != len(items):
        raise SpecificationError(f"{what} must not repeat a value: {list(items)}")
    return tuple(items)


def _integers(value: object, what: str, *, minimum: int = 0) -> tuple[int, ...]:
    return _unique(
        [_integer(item, f"{what} entry", minimum=minimum) for item in _items(value, what)], what
    )


def _increasing[T: (int, float, Fraction)](items: tuple[T, ...], what: str) -> tuple[T, ...]:
    if list(items) != sorted(items):
        raise SpecificationError(f"{what} must be in increasing order: {list(items)}")
    return items


def _positive_numbers(value: object, what: str) -> tuple[float, ...]:
    numbers = _unique([_number(item, f"{what} entry") for item in _items(value, what)], what)
    if any(number <= 0 for number in numbers):
        raise SpecificationError(f"{what} must be positive: {list(numbers)}")
    return numbers


def _names(value: object, what: str) -> tuple[str, ...]:
    return _unique([_text(item, f"{what} entry") for item in _items(value, what)], what)


def _resampling(raw: object) -> Resampling:
    entry = _mapping(
        raw,
        "resampling",
        (
            "replicates",
            "root_seed",
            "bit_generator",
            "streams",
            "block_length",
            "block_length_sensitivity",
            "circular",
        ),
    )
    root_seed = _integer(entry["root_seed"], "resampling.root_seed")
    if root_seed >= SEED_LIMIT:
        raise SpecificationError(
            f"resampling.root_seed must be below {SEED_LIMIT}, not {root_seed}"
        )
    if entry["bit_generator"] != BIT_GENERATOR:
        raise SpecificationError(f"resampling.bit_generator must be {BIT_GENERATOR}")
    if entry["streams"] != list(STREAMS):
        raise SpecificationError(f"resampling.streams must be {list(STREAMS)} (DR-006 order)")
    block = _integer(entry["block_length"], "resampling.block_length", minimum=1)
    sensitivity = _integers(
        entry["block_length_sensitivity"], "resampling.block_length_sensitivity", minimum=1
    )
    if block in sensitivity:
        raise SpecificationError("resampling.block_length_sensitivity must not repeat the primary")
    if _flag(entry["circular"], "resampling.circular"):
        raise SpecificationError("resampling.circular must be false (DR-003, DR-006)")
    return Resampling(
        _integer(entry["replicates"], "resampling.replicates", minimum=1),
        root_seed,
        BIT_GENERATOR,
        STREAMS,
        block,
        sensitivity,
        False,
    )


def _levels(raw: object) -> Levels:
    entry = _mapping(raw, "levels", ("difference", "interpretation"))
    return Levels(
        _unit(entry["difference"], "levels.difference"),
        _unit(entry["interpretation"], "levels.interpretation"),
    )


def _inference(raw: object) -> Inference:
    entry = _mapping(
        raw,
        "inference",
        (
            "sesoi",
            "sesoi_sensitivity",
            "h3_target",
            "h3_delta_relative",
            "h3_delta",
            "h3_tolerance_sensitivity",
            "alpha",
            "holm_family",
        ),
    )
    target = _unit(entry["h3_target"], "inference.h3_target")
    relative = _unit(entry["h3_delta_relative"], "inference.h3_delta_relative")
    delta = _unit(entry["h3_delta"], "inference.h3_delta")
    if abs(target * relative - delta) > DELTA_TOLERANCE:
        raise SpecificationError("inference.h3_delta must equal h3_target * h3_delta_relative")
    if entry["holm_family"] != list(HOLM_FAMILY):
        raise SpecificationError(f"inference.holm_family must be {list(HOLM_FAMILY)}")
    sesoi = _unit(entry["sesoi"], "inference.sesoi")
    sesoi_sensitivity = _positive_numbers(entry["sesoi_sensitivity"], "inference.sesoi_sensitivity")
    if sesoi in sesoi_sensitivity:
        raise SpecificationError("inference.sesoi_sensitivity must not repeat the SESOI")
    return Inference(
        sesoi,
        sesoi_sensitivity,
        target,
        relative,
        delta,
        _positive_numbers(entry["h3_tolerance_sensitivity"], "inference.h3_tolerance_sensitivity"),
        _unit(entry["alpha"], "inference.alpha"),
        HOLM_FAMILY,
    )


def _share(value: object, what: str) -> Fraction:
    """The decimal share as the exact fraction it writes (0.02 is 1/50)."""
    return Fraction(str(_unit(value, what)))


def _capacity(raw: object) -> Capacity:
    entry = _mapping(raw, "capacity", ("shares", "primary", "budgets"))
    shares = _increasing(
        _unique(
            [_share(item, "capacity.shares entry") for item in _items(entry["shares"], "shares")],
            "capacity.shares",
        ),
        "capacity.shares",
    )
    primary = _share(entry["primary"], "capacity.primary")
    if primary not in shares:
        raise SpecificationError("capacity.primary must be one of capacity.shares")
    budgets = _increasing(_integers(entry["budgets"], "capacity.budgets", minimum=1), "budgets")
    if len(budgets) != len(shares):
        raise SpecificationError("capacity.budgets needs one budget per share")
    return Capacity(shares, primary, budgets)


def _configurations(raw: object) -> Configurations:
    entry = _mapping(raw, "configurations", ("order", "tree", "neural", "reference"))
    tree, neural = (
        _names(entry["tree"], "configurations.tree"),
        _names(entry["neural"], "configurations.neural"),
    )
    models = (*tree, *neural)
    expected = [f"{model}-{weighting}" for weighting in WEIGHTINGS for model in models]
    if entry["order"] != expected:
        raise SpecificationError(
            f"configurations.order must be every model unweighted, then balanced: {expected}"
        )
    if entry["reference"] != REFERENCE:
        raise SpecificationError(f"configurations.reference must be {REFERENCE}")
    return Configurations(tuple(expected), tree, neural, REFERENCE)


def _contrasts(raw: object, known: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in _items(raw, "contrasts"):
        if not isinstance(item, list) or len(item) != 2:
            raise SpecificationError(f"a contrast must be a pair [a, b], not {item!r}")
        first, second = _text(item[0], "contrast"), _text(item[1], "contrast")
        if first == second or first not in known or second not in known:
            raise SpecificationError(f"contrast {item!r} must name two different configurations")
        pairs.append((first, second))
    return _unique(pairs, "contrasts")


def _calibration_reporting(raw: object) -> CalibrationReporting:
    entry = _mapping(raw, "calibration_reporting", ("reliability_bins", "epsilon"))
    epsilon = _number(entry["epsilon"], "calibration_reporting.epsilon")
    if not 0 < epsilon < 0.5:
        raise SpecificationError("calibration_reporting.epsilon must lie in (0, 0.5)")
    return CalibrationReporting(
        _integer(entry["reliability_bins"], "calibration_reporting.reliability_bins", minimum=2),
        epsilon,
    )


def _rq3(raw: object) -> Rq3:
    entry = _mapping(raw, "rq3", ("ratio_grid", "ratio_max", "friction"))
    grid = _increasing(
        _integers(entry["ratio_grid"], "rq3.ratio_grid", minimum=2), "rq3.ratio_grid"
    )
    maximum = _integer(entry["ratio_max"], "rq3.ratio_max", minimum=2)
    if maximum < grid[-1]:
        raise SpecificationError("rq3.ratio_max must not be below the largest grid ratio")
    return Rq3(grid, maximum, _integers(entry["friction"], "rq3.friction", minimum=1))


def _week(item: object) -> tuple[int, int]:
    if not isinstance(item, list) or len(item) != 2:
        raise SpecificationError(f"a Test week must be a pair [first, last], not {item!r}")
    first = _integer(item[0], "test_weeks first day", minimum=1)
    last = _integer(item[1], "test_weeks last day", minimum=1)
    if first > last:
        raise SpecificationError(f"a Test week must not end before it starts: {item!r}")
    return first, last


def _false_negative(raw: object) -> FalseNegative:
    entry = _mapping(
        raw,
        "false_negative",
        (
            "minimum_frauds",
            "amount_quantiles",
            "score_band_multiples",
            "interval_level",
            "test_weeks",
        ),
    )
    multiples = _increasing(
        _integers(entry["score_band_multiples"], "false_negative.score_band_multiples", minimum=1),
        "false_negative.score_band_multiples",
    )
    if multiples[0] != 1:
        raise SpecificationError("false_negative.score_band_multiples must start at 1")
    weeks = tuple(_week(item) for item in _items(entry["test_weeks"], "false_negative.test_weeks"))
    for (_, previous_last), (next_first, _) in pairwise(weeks):
        if next_first != previous_last + 1:
            raise SpecificationError("false_negative.test_weeks must be consecutive days")
    return FalseNegative(
        _integer(entry["minimum_frauds"], "false_negative.minimum_frauds", minimum=1),
        _integer(entry["amount_quantiles"], "false_negative.amount_quantiles", minimum=2),
        multiples,
        _unit(entry["interval_level"], "false_negative.interval_level"),
        weeks,
    )


def parse_declaration(raw: dict[str, Any]) -> EvaluationDeclaration:
    """The declaration in a parsed ``config/evaluation.yaml``; ``SpecificationError`` if it is
    malformed or inconsistent."""
    top = _mapping(
        raw,
        "evaluation.yaml",
        (
            "resampling",
            "levels",
            "inference",
            "capacity",
            "configurations",
            "contrasts",
            "calibration_reporting",
            "rq3",
            "false_negative",
        ),
    )
    configurations = _configurations(top["configurations"])
    return EvaluationDeclaration(
        resampling=_resampling(top["resampling"]),
        levels=_levels(top["levels"]),
        inference=_inference(top["inference"]),
        capacity=_capacity(top["capacity"]),
        configurations=configurations,
        contrasts=_contrasts(top["contrasts"], configurations.order),
        calibration_reporting=_calibration_reporting(top["calibration_reporting"]),
        rq3=_rq3(top["rq3"]),
        false_negative=_false_negative(top["false_negative"]),
    )


def load_declaration(paths: ProjectPaths) -> tuple[EvaluationDeclaration, SettingsSnapshot]:
    """Load ``config/evaluation.yaml`` once, with a snapshot of the bytes it was parsed from: the
    same bytes are parsed, hashed and recorded."""
    path = paths.evaluation_spec
    try:
        content = path.read_bytes()
    except OSError as error:
        raise FileAccessError(f"{path}: cannot be read ({error.strerror})") from error
    raw = _read_yaml(path, content=content)
    try:
        declaration = parse_declaration(raw)
    except SpecificationError as error:
        raise SpecificationError(f"{path}: {error}") from error
    snapshot = SettingsSnapshot(
        path=path.relative_to(paths.root).as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content.decode("utf-8"),
    )
    return declaration, snapshot
