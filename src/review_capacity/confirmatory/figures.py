"""The cost figures of the Test evaluation (Issue #209; DR-007 §2 and §9).

Two Scalable Vector Graphics (SVG) figures drawn from the exact curve segments of the ``test_rq3``
records of the Platt (primary) calibrator, with no resampling and no new analysis:

- the **cost-difference curve**: relative Delta J against log rho, one panel per capacity, one
  line per configuration;
- the **utilisation curve**: R4 utilisation against log rho.

On a segment of ratios rho that R4 declines ``u`` queue members, ``f`` of them frauds,

$$
\\Delta J(\\rho) = \\rho f - u, \\qquad
\\frac{\\Delta J}{J_{\\mathrm{R3}}} =
\\frac{\\rho f - u}{N + \\rho \\, \\mathrm{FN}_{\\mathrm{R3}}}, \\qquad
\\text{utilisation} = \\frac{N - u}{K \\, D}
$$

where `N` is R3's reviews, `FN_R3` its false negatives, `K` the daily budget and `D` the Test days.
Every segment is drawn between its own two ends, so the segment boundaries and the jumps between
segments are exact (a jump is a vertical line); inside a wide cost segment the smooth curve is
sampled at ``SAMPLES_PER_DECADE`` points per decade. Each figure marks, on every panel, the
reference ratio 1 / prev_V2 of DR-007 §2 (a dashed line) and, for each configuration, the
saturation ratio (a triangle on the lower edge) and, in the cost figure, the break-even ratio (a
dot on the curve, at the value just below it).

The files are reproducible byte for byte **in the pinned rendering environment** (the locked
matplotlib version with its bundled DejaVu Sans font): every setting that affects the output is
set here from the library defaults, a fixed hash salt is used, text stays text, and no date is
written. Path simplification is the library's, at its default sub-pixel threshold.
"""

from __future__ import annotations

import io
import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

FIGURES_DIR = "figures"
COST_FIGURE = "cost_difference.svg"
UTILISATION_FIGURE = "utilisation.svg"
CALIBRATOR = "platt"
SAMPLES_PER_DECADE = 40
# DR-007 section 2 [D]: V2 has 77,822 transactions and 2,637 frauds (3.39 %); the ratio at which
# p* = 1 / rho equals the V2 fraud prevalence is marked on every figure.
REFERENCE_RATIO = 77_822 / 2_637
RATIO_MAX = 1000.0
TITLE = {
    COST_FIGURE: "Relative cost difference Delta J / J_R3 of R4 against R3 (Platt, Test)",
    UTILISATION_FIGURE: "R4 utilisation (Platt, Test)",
}
PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#9D755D",
    "#BAB0AC",
)


def _curve(entry: Mapping[str, Any], cost: bool) -> tuple[Any, Any]:
    """The points of one configuration's curve at one capacity: the two ends of every exact
    segment (and the interior samples of a wide cost segment), in increasing ratio."""
    queue, k = entry["queue"], entry["k"]
    reviews, misses = queue["reviews"], queue["frauds"] - queue["captured"]
    days = reviews / k  # R3 reviews K a day on every Test day
    xs: list[npt.NDArray[np.float64]] = []
    ys: list[npt.NDArray[np.float64]] = []
    for segment in entry["curve"]:
        low, high = float(segment["start"]), float(segment["end"])
        u, f = segment["unused"], segment["frauds_forgone"]
        if cost:
            width = math.log10(high / low) if high > low else 0.0
            ratios = np.geomspace(low, high, max(2, math.ceil(width * SAMPLES_PER_DECADE) + 1))
            values = (ratios * f - u) / (reviews + ratios * misses)
        else:
            ratios = np.array([low, high])
            values = np.full(2, (reviews - u) / (k * days))
        xs.append(ratios)
        ys.append(values)
    return np.concatenate(xs), np.concatenate(ys)


def _value_below(entry: Mapping[str, Any], ratio: float) -> float:
    """Relative Delta J at the left limit of ``ratio`` (the value the curve has just below it)."""
    queue, curve = entry["queue"], entry["curve"]
    starts = [segment["start"] for segment in curve]
    segment = curve[max(bisect_left(starts, ratio) - 1, 0)]
    misses = queue["frauds"] - queue["captured"]
    u, f = segment["unused"], segment["frauds_forgone"]
    return float((ratio * f - u) / (queue["reviews"] + ratio * misses))


def _style() -> dict[str, Any]:
    """The library defaults with the settings of the figures: nothing ambient is inherited."""
    import matplotlib

    defaults: dict[str, Any] = {
        str(k): v for k, v in matplotlib.rcParamsDefault.items() if k != "backend"
    }
    return defaults | {
        "svg.hashsalt": "review-capacity",
        "svg.fonttype": "none",
        "font.family": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "path.simplify": True,
        "path.simplify_threshold": 0.111,
        "figure.dpi": 100,
        "savefig.dpi": 100,
    }


def render_figures(records: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    """The two figures (file name -> SVG bytes) from the ``test_rq3`` records in ``records``."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    analyses = {
        record["configuration"]: record
        for record in records
        if record["record"] == "test_rq3" and record["calibrator"] == CALIBRATOR
    }
    if not analyses:
        return {}
    names = sorted(analyses, key=lambda name: list(analyses).index(name))
    capacities = [entry["capacity"] for entry in analyses[names[0]]["capacities"]]
    figures: dict[str, bytes] = {}
    with matplotlib.rc_context(cast(Any, _style())):
        for file, cost in ((COST_FIGURE, True), (UTILISATION_FIGURE, False)):
            figure, axes = pyplot.subplots(
                1, len(capacities), figsize=(4 * len(capacities), 3.6), sharey=True, squeeze=False
            )
            for column, capacity in enumerate(capacities):
                axis = axes[0][column]
                for number, name in enumerate(names):
                    entry = analyses[name]["capacities"][column]
                    colour = PALETTE[number % len(PALETTE)]
                    xs, ys = _curve(entry, cost)
                    axis.plot(xs, ys, color=colour, label=name, lw=1.1)
                    saturation = entry["saturation"]["ratio"]
                    if saturation is not None and saturation <= RATIO_MAX:
                        axis.plot(
                            [saturation], [0.0], "^", color=colour, ms=4,
                            transform=axis.get_xaxis_transform(), clip_on=False,
                        )  # fmt: skip
                    ratio = entry["break_even"]["ratio"]
                    if cost and ratio is not None:
                        axis.plot([ratio], [_value_below(entry, ratio)], "o", color=colour, ms=4)
                axis.axvline(REFERENCE_RATIO, color="#444444", ls="--", lw=0.7)
                axis.set_xscale("log")
                axis.set_xlim(1.0, RATIO_MAX)
                axis.set_title(f"capacity {capacity:g}", fontsize=9)
                axis.set_xlabel("cost ratio rho (log scale)", fontsize=8)
                if cost:
                    axis.axhline(0.0, color="#444444", lw=0.6)
                axis.grid(True, lw=0.3, color="#DDDDDD")
            axes[0][0].set_ylabel("relative Delta J" if cost else "utilisation", fontsize=8)
            axes[0][-1].legend(fontsize=6, loc="best", frameon=False)
            figure.suptitle(TITLE[file], fontsize=10)
            figure.text(
                0.5, 0.005,
                "dashed line: 1 / prev_V2; triangle: saturation ratio; dot: break-even ratio",
                ha="center", fontsize=6,
            )  # fmt: skip
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            buffer = io.BytesIO()
            figure.savefig(
                buffer, format="svg", metadata={"Date": None, "Creator": "review-capacity"}
            )
            pyplot.close(figure)
            figures[f"{FIGURES_DIR}/{file}"] = buffer.getvalue()
    return figures
