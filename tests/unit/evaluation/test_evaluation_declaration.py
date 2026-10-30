"""The evaluation declaration (``config/evaluation.yaml``): loading, validation, and agreement with
the protocol, the Decision Records and the development constants it repeats (the documents are read
as text, so a change to either side fails the test)."""

from __future__ import annotations

import copy
import hashlib
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import yaml

from review_capacity.config import load_dataset_spec
from review_capacity.development.records import RQ3_FRICTION, RQ3_RATIO_GRID, RQ3_RATIO_MAX
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.evaluation.bootstrap import LEVEL, REPLICATES, SEED
from review_capacity.evaluation.calibration import BINS, EPSILON
from review_capacity.evaluation.declaration import (
    STREAMS,
    EvaluationDeclaration,
    load_declaration,
    parse_declaration,
)
from review_capacity.evaluation.ranking import CAPACITY_SHARES, capacity_budgets
from review_capacity.models.registry import CONFIGURATIONS, MODELS, Family
from review_capacity.paths import ProjectPaths
from support.synthetic import REPO_PATHS

DOCS = REPO_PATHS.root / "docs"
PROTOCOL = (DOCS / "protocol.md").read_text(encoding="utf-8")
DR003 = (DOCS / "decisions" / "DR-003-rq2-h3-policy-transfer-fidelity.md").read_text("utf-8")
DR005 = (DOCS / "decisions" / "DR-005-sesoi-and-h1-h2-inference.md").read_text("utf-8")
DR006 = (DOCS / "decisions" / "DR-006-bootstrap-inference-strategy.md").read_text("utf-8")
DR007 = (DOCS / "decisions" / "DR-007-rq3-cost-ratio-and-break-even.md").read_text("utf-8")
DR010 = (DOCS / "decisions" / "DR-010-h3-confirmatory-configuration.md").read_text("utf-8")


@pytest.fixture(scope="module")
def declaration() -> EvaluationDeclaration:
    return load_declaration(REPO_PATHS)[0]


def raw() -> dict[str, Any]:
    loaded = yaml.safe_load(REPO_PATHS.evaluation_spec.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def rejects(change: Any, fragment: str) -> None:
    document = raw()
    change(document)
    with pytest.raises(SpecificationError, match=fragment):
        parse_declaration(document)


def search(pattern: str, text: str) -> re.Match[str]:
    match = re.search(pattern, text)
    assert match is not None, pattern
    return match


def percent(text: str) -> float:
    return int(text) / 100


def numbers(text: str) -> list[int]:
    return [int(item) for item in re.findall(r"\d+", text)]


# --- loading -------------------------------------------------------------------------------------


def test_the_real_declaration_loads_with_a_snapshot_of_its_bytes() -> None:
    declaration, snapshot = load_declaration(REPO_PATHS)
    content = REPO_PATHS.evaluation_spec.read_bytes()
    assert snapshot.path == "config/evaluation.yaml"
    assert snapshot.sha256 == hashlib.sha256(content).hexdigest()
    assert snapshot.content == content.decode("utf-8")
    assert declaration.resampling.streams == STREAMS


def test_a_missing_file_is_a_file_access_error(tmp_path: Path) -> None:
    with pytest.raises(FileAccessError, match="cannot be read"):
        load_declaration(ProjectPaths(tmp_path))


def test_a_malformed_file_names_its_path(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "evaluation.yaml").write_text("resampling: [1]\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match=r"evaluation\.yaml"):
        load_declaration(ProjectPaths(tmp_path))


def test_a_repeated_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    text = REPO_PATHS.evaluation_spec.read_text(encoding="utf-8")
    (tmp_path / "config" / "evaluation.yaml").write_text(
        text.replace("  root_seed: 42\n", "  root_seed: 42\n  root_seed: 43\n", 1), encoding="utf-8"
    )
    with pytest.raises(SpecificationError, match="cannot be read as YAML") as error:
        load_declaration(ProjectPaths(tmp_path))
    assert "repeated key" in str(error.value.__cause__)


# --- validation ----------------------------------------------------------------------------------


def test_unknown_and_missing_sections_are_rejected() -> None:
    rejects(lambda d: d.update(extra=1), "unknown \\['extra'\\]")
    rejects(lambda d: d.pop("levels"), "missing \\['levels'\\]")
    rejects(lambda d: d["rq3"].pop("ratio_max"), "rq3: missing \\['ratio_max'\\]")
    rejects(lambda d: d.update(inference=[1]), "inference must be a mapping")


def test_numbers_must_have_the_declared_types() -> None:
    rejects(lambda d: d["resampling"].update(replicates=True), "replicates must be an integer")
    rejects(lambda d: d["resampling"].update(replicates=1.5), "replicates must be an integer")
    rejects(lambda d: d["resampling"].update(root_seed=-1), "root_seed must be an integer")
    rejects(lambda d: d["levels"].update(difference="0.95"), "must be a finite number")
    rejects(lambda d: d["levels"].update(difference=1.0), "strictly between 0 and 1")
    rejects(lambda d: d["inference"].update(alpha=0), "strictly between 0 and 1")
    rejects(lambda d: d["calibration_reporting"].update(epsilon=0.6), "must lie in \\(0, 0.5\\)")
    rejects(lambda d: d["calibration_reporting"].update(reliability_bins=1), "at least 2")


def test_the_resampling_rules_are_fixed() -> None:
    rejects(lambda d: d["resampling"].update(root_seed=2**32), "must be below")
    rejects(lambda d: d["resampling"].update(bit_generator="MT19937"), "bit_generator must be")
    rejects(lambda d: d["resampling"]["streams"].reverse(), "streams must be")
    rejects(lambda d: d["resampling"].update(circular=True), "must be false")
    rejects(lambda d: d["resampling"].update(circular="no"), "true or false")
    rejects(
        lambda d: d["resampling"].update(block_length_sensitivity=[3, 7]), "must not repeat the pri"
    )
    rejects(
        lambda d: d["resampling"].update(block_length_sensitivity=[1, 1]), "must not repeat a value"
    )


def test_the_inference_values_must_be_consistent() -> None:
    rejects(lambda d: d["inference"].update(h3_delta=0.005), "must equal h3_target")
    rejects(lambda d: d["inference"].update(holm_family=["H1", "H2"]), "holm_family must be")
    rejects(lambda d: d["inference"].update(sesoi_sensitivity=[0.02, 0.05]), "must not repeat")
    rejects(
        lambda d: d["inference"].update(h3_tolerance_sensitivity=[0.1, -0.2]), "must be positive"
    )


def test_the_capacity_rows_must_match_one_another() -> None:
    rejects(lambda d: d["capacity"].update(primary=0.03), "must be one of capacity.shares")
    rejects(lambda d: d["capacity"].update(budgets=[28, 56, 139]), "one budget per share")
    rejects(lambda d: d["capacity"].update(shares=[0.02, 0.01, 0.05, 0.1]), "increasing order")
    rejects(lambda d: d["capacity"].update(budgets=[56, 28, 139, 278]), "increasing order")


def test_the_configurations_follow_the_model_order() -> None:
    rejects(lambda d: d["configurations"]["order"].reverse(), "every model unweighted, then")
    rejects(lambda d: d["configurations"].update(reference="lasso"), "reference must be")
    rejects(lambda d: d["configurations"].update(tree=[]), "non-empty list")


def test_contrasts_name_two_different_known_configurations() -> None:
    rejects(lambda d: d["contrasts"].append(["LGB-b", "LGB-b"]), "two different configurations")
    rejects(lambda d: d["contrasts"].append(["LGB-b", "XGB-u"]), "two different configurations")
    rejects(lambda d: d["contrasts"].append(["LGB-b"]), "must be a pair")
    rejects(lambda d: d["contrasts"].append(d["contrasts"][0]), "must not repeat a value")


def test_the_rq3_and_false_negative_rules_are_checked() -> None:
    rejects(lambda d: d["rq3"].update(ratio_max=500), "must not be below the largest")
    rejects(lambda d: d["rq3"].update(ratio_grid=[5, 2]), "increasing order")
    rejects(lambda d: d["false_negative"].update(score_band_multiples=[2, 5]), "start at 1")
    rejects(lambda d: d["false_negative"].update(minimum_frauds=0), "at least 1")
    rejects(lambda d: d["false_negative"]["test_weeks"].pop(1), "consecutive days")
    rejects(lambda d: d["false_negative"]["test_weeks"].append([190, 183]), "must not end before")
    rejects(lambda d: d["false_negative"]["test_weeks"].append(5), "must be a pair")


# --- agreement with the development constants ----------------------------------------------------


def test_the_declaration_repeats_the_development_constants(
    declaration: EvaluationDeclaration,
) -> None:
    assert declaration.resampling.replicates == REPLICATES
    assert declaration.resampling.root_seed == SEED
    assert declaration.levels.difference == LEVEL
    assert declaration.capacity.shares == CAPACITY_SHARES
    assert declaration.rq3.ratio_grid == RQ3_RATIO_GRID
    assert declaration.rq3.ratio_max == RQ3_RATIO_MAX
    assert declaration.rq3.friction == RQ3_FRICTION
    assert declaration.calibration_reporting.reliability_bins == BINS
    assert declaration.calibration_reporting.epsilon == EPSILON


def test_the_configurations_are_those_of_the_registry(declaration: EvaluationDeclaration) -> None:
    configurations = declaration.configurations
    assert configurations.order == tuple(c.name for c in CONFIGURATIONS)
    assert configurations.tree == tuple(m.label for m in MODELS if m.family is Family.TREE)
    assert configurations.neural == tuple(m.label for m in MODELS if m.family is Family.NEURAL)


def test_the_budgets_follow_the_v2_rule_of_dr_002(declaration: EvaluationDeclaration) -> None:
    v2 = load_dataset_spec(REPO_PATHS).partition("v2")
    derived = capacity_budgets(v2)
    assert declaration.capacity.shares == tuple(c.share for c in derived)
    assert declaration.capacity.budgets == tuple(c.k for c in derived)


def test_the_test_weeks_tile_the_test_partition(declaration: EvaluationDeclaration) -> None:
    test = load_dataset_spec(REPO_PATHS).partition("test")
    weeks = declaration.false_negative.test_weeks
    assert weeks[0][0] == test.first_day
    assert weeks[-1][1] == test.last_day
    assert all(last - first == 6 for first, last in weeks)


# --- agreement with the protocol and the Decision Records ----------------------------------------


def test_the_resampling_block_matches_dr_006_and_dr_003(declaration: EvaluationDeclaration) -> None:
    resampling = declaration.resampling
    assert f"**B = {resampling.replicates:,}**" in DR006
    assert f"one root seed, **{resampling.root_seed}**" in DR006
    assert f"{resampling.bit_generator} bit generator" in DR006
    primary = f"**ℓ (block length) = {resampling.block_length} days, primary.**"  # noqa: RUF001
    assert primary in DR003
    sensitivity = search(r"\*\*Sensitivity:\*\* ℓ = (\d+) \(.*?\) and ℓ = (\d+)", DR003)  # noqa: RUF001
    assert resampling.block_length_sensitivity == tuple(int(g) for g in sensitivity.groups())
    assert "Circular wrap is not used" in DR006


def test_the_streams_follow_the_order_of_dr_006() -> None:
    section = DR006[
        DR006.index("- **Seeds and streams.**") : DR006.index("- **Replicates with zero")
    ]
    listed = re.findall(r"^  (\d+)\. (.+?)[;.]$", section, re.MULTILINE)
    assert [int(index) for index, _ in listed] == list(range(1, len(STREAMS) + 1))
    markers = (
        "H1/H2 two-way pigeonhole",
        "S1 fraud-level iid",
        "S2 day-level moving-block",
        "H3 day-level moving-block",
        "RQ3 day-level moving-block",
        "RQ3 two-way pigeonhole",
        "false-negative analysis",
        "secondary outcomes recomputed",
    )
    assert len(listed) == len(markers)
    for (_, text), marker in zip(listed, markers, strict=True):
        assert marker in text


def test_the_levels_match_dr_006(declaration: EvaluationDeclaration) -> None:
    assert declaration.levels.difference == percent(
        search(r"(\d+) % for difference estimates", DR006).group(1)
    )
    assert declaration.levels.interpretation == percent(
        search(r"(\d+) % for H1's practical interpretation", DR006).group(1)
    )
    assert declaration.false_negative.interval_level == percent(
        search(r"the (\d+) % percentile interval of the slice miss rate", PROTOCOL).group(1)
    )


def test_the_inference_block_matches_the_protocol(declaration: EvaluationDeclaration) -> None:
    inference = declaration.inference
    assert inference.sesoi == percent(
        search(r"\(SESOI\) = (\d+) percentage points of Recall@2%", PROTOCOL).group(1)
    )
    sensitivity = search(r"SESOIs of (\d+), (\d+) and (\d+) pp", PROTOCOL)
    assert inference.sesoi_sensitivity == tuple(percent(g) for g in sensitivity.groups())
    relative = search(r"relative tolerance of (\d+) % of the (\d+) % target", PROTOCOL)
    assert inference.h3_delta_relative == percent(relative.group(1))
    assert inference.h3_target == percent(relative.group(2))
    assert inference.h3_delta == float(
        search(r"Δ = (0\.\d+) \(±20 % relative\)", PROTOCOL).group(1)
    )
    tolerances = search(r"±(\d+) %, ±(\d+) % and ±(\d+) % bounds", PROTOCOL)
    assert inference.h3_tolerance_sensitivity == tuple(percent(g) for g in tolerances.groups())
    assert inference.alpha == float(search(r"family-wise α = (0\.\d+)", PROTOCOL).group(1))  # noqa: RUF001
    assert "confirmatory family {H1, H2, H3}" in PROTOCOL
    assert inference.holm_family == ("H1", "H2", "H3")


def test_the_capacity_block_matches_the_protocol(declaration: EvaluationDeclaration) -> None:
    line = search(
        r"K = (\d+) / \*\*(\d+)\*\* / (\d+) / (\d+) reviews per relative day"
        r" for c = (\d+) / (\d+) / (\d+) / (\d+) %",
        PROTOCOL,
    )
    values = [int(group) for group in line.groups()]
    assert declaration.capacity.budgets == tuple(values[:4])
    assert declaration.capacity.shares == tuple(Fraction(share, 100) for share in values[4:])
    assert declaration.capacity.primary == Fraction(values[5], 100)
    assert declaration.capacity.budgets[1] == values[1]


def test_the_configuration_order_matches_dr_010(declaration: EvaluationDeclaration) -> None:
    assert ", ".join(declaration.configurations.order) in DR010
    assert "LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b" in PROTOCOL


def test_the_contrasts_match_dr_005(declaration: EvaluationDeclaration) -> None:
    row = search(r"per weighting: (.+?) \(eight contrasts\)", DR005).group(1)
    pairs = [tuple(item.split(" − ")) for item in re.split(r"[,;] ", row)]
    assert declaration.contrasts == tuple(pairs)
    assert len(declaration.contrasts) == 8


def test_the_calibration_reporting_matches_the_protocol(declaration: EvaluationDeclaration) -> None:
    bins = search(r"with (\d+) equal-count bins", PROTOCOL)
    assert declaration.calibration_reporting.reliability_bins == int(bins.group(1))
    assert "with ε = 10⁻¹⁵" in PROTOCOL
    assert declaration.calibration_reporting.epsilon == 1e-15


def test_the_rq3_block_matches_dr_007(declaration: EvaluationDeclaration) -> None:
    grid = search(r"ρ ∈ G = \{([\d, ]+)\}", DR007).group(1)  # noqa: RUF001
    assert declaration.rq3.ratio_grid == tuple(numbers(grid))
    friction = search(r"φ = C_FP / C_R ∈ \{([\d, ]+)\}", DR007).group(1)
    assert declaration.rq3.friction == tuple(numbers(friction))
    assert f"ρ ∈ (1, {declaration.rq3.ratio_max}]" in DR007  # noqa: RUF001


def test_the_false_negative_block_matches_the_protocol(declaration: EvaluationDeclaration) -> None:
    rules = declaration.false_negative
    assert rules.minimum_frauds == int(search(r"fewer than (\d+) frauds", PROTOCOL).group(1))
    assert rules.amount_quantiles == 5
    assert "Quintiles of `TransactionAmt` over all Train transactions" in PROTOCOL
    weeks = search(r"Test days (\d+)–(\d+), (\d+)–(\d+), (\d+)–(\d+) and (\d+)–(\d+)", PROTOCOL)
    days = [int(group) for group in weeks.groups()]
    assert rules.test_weeks == tuple(zip(days[::2], days[1::2], strict=True))
    assert "ranks K + 1 … 2K (near miss), 2K + 1 … 5K, and above 5K" in PROTOCOL
    assert rules.score_band_multiples == (1, 2, 5)
