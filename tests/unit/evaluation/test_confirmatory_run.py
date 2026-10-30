"""The confirmatory analysis end to end on a Test-sized synthetic input (no data is read): the
resampling engines feed H1, H2, H3, Holm, S1, S2, the SESOI views and McNemar at the declared
B = 10,000, to show that the pieces fit together and run in unit-test time."""

from __future__ import annotations

import numpy as np
import pytest

from review_capacity.evaluation.bootstrap import entity_indices
from review_capacity.evaluation.contrasts import estimands
from review_capacity.evaluation.declaration import load_declaration
from review_capacity.evaluation.hypotheses import analyse_h1, analyse_h2, analyse_h3, decide
from review_capacity.evaluation.mcnemar import cell_mcnemar
from review_capacity.evaluation.resampling import (
    Replicates,
    day_block_indices,
    day_ratios,
    h3_day_block_sets,
    naive_ratios,
    pigeonhole_ratios,
    stream_generator,
)
from review_capacity.evaluation.sensitivity import analyse_s1, analyse_s2, sesoi_views
from support.synthetic import REPO_PATHS

FRAUDS, DAYS, ENTITIES = 2660, 28, 1350  # the projected Test size (protocol)
CAPTURE_PROBABILITY = np.array([0.40, 0.39, 0.34, 0.33, 0.40, 0.39, 0.34, 0.33])
ALERT_RATE = np.array([0.0195, 0.0205, 0.0135, 0.0260, 0.0200, 0.0190, 0.0210, 0.0200])


def test_the_whole_confirmatory_analysis_runs_at_the_declared_replicate_count() -> None:
    declaration = load_declaration(REPO_PATHS)[0]
    table = estimands(declaration)
    replicates = declaration.resampling.replicates
    rng = np.random.default_rng(2026)

    day_index = rng.integers(0, DAYS, FRAUDS).astype(np.intp)
    entity_index = entity_indices([str(e) for e in rng.integers(0, ENTITIES, FRAUDS)])
    captures = rng.random((8, FRAUDS)) < CAPTURE_PROBABILITY[:, np.newaxis]
    recalls = captures.mean(axis=1)

    pigeonhole = pigeonhole_ratios(
        stream_generator(declaration, "pigeonhole_h1_h2"),
        captures,
        day_index,
        DAYS,
        entity_index,
        replicates=replicates,
    )
    naive = naive_ratios(stream_generator(declaration, "naive_s1"), captures, replicates=replicates)
    assert pigeonhole.ratios.shape == (replicates, 8)
    assert pigeonhole.undefined_count == 0

    # per-day sufficient statistics for the day-block analyses
    captured_by_day = np.vstack(
        [np.bincount(day_index, weights=row, minlength=DAYS) for row in captures]
    )
    frauds_by_day = np.bincount(day_index, minlength=DAYS).astype(np.float64)
    volume = rng.integers(2500, 3000, DAYS).astype(np.float64)
    alerts_by_day = np.vstack([rng.binomial(volume.astype(int), rate) for rate in ALERT_RATE])
    observed_rates = alerts_by_day.sum(axis=1) / volume.sum()

    s2_indices = day_block_indices(
        stream_generator(declaration, "day_block_s2"),
        DAYS,
        declaration.resampling.block_length,
        replicates=replicates,
    )
    s2 = day_ratios(captured_by_day, frauds_by_day, s2_indices)
    h3_sets = {
        block_set.block_length: Replicates(
            day_ratios(alerts_by_day, volume, block_set.indices).ratios
        )
        for block_set in h3_day_block_sets(declaration, DAYS)
    }

    h1 = analyse_h1(declaration, table, recalls, pigeonhole)
    h2 = analyse_h2(declaration, table, recalls, pigeonhole)
    h3 = analyse_h3(declaration, "LGB-b", observed_rates, h3_sets)
    primary = decide(declaration, h1, h2, h3)

    for p in (h1.p.value, h2.p.value, h3.primary.p.value):
        assert 0.0 < p <= 1.0
    assert all(
        adjusted >= raw
        for adjusted, raw in zip(
            primary.holm.adjusted, (h1.p.value, h2.p.value, h3.primary.p.value), strict=True
        )
    )
    assert h1.interval.lower <= h1.estimate <= h1.interval.upper
    assert len(h3.others) == 7
    assert [v.block_length for v in h3.sensitivity] == [1, 7]

    s2_outcome = analyse_s2(declaration, table, recalls, s2, primary)
    s1_outcome = analyse_s1(declaration, table, recalls, pigeonhole, naive, primary)
    assert 0.0 < s2_outcome.comparison.p_h1 <= 1.0
    assert len(s1_outcome.quantities) == 5  # estimable or not, depending on the clustering
    assert len(sesoi_views(declaration, primary)) == 3
    assert len(cell_mcnemar(declaration, captures)) == 8
    assert recalls == pytest.approx(CAPTURE_PROBABILITY, abs=0.04)
