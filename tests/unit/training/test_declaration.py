"""The training declaration (``config/training.yaml``): loading, validation, and agreement with
DR-015 (the Decision Record is read as text, so a change to either one fails the test)."""

from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from review_capacity.config import load_dataset_spec
from review_capacity.errors import FileAccessError, SpecificationError
from review_capacity.paths import ProjectPaths
from review_capacity.training.declaration import (
    MODEL_NAMES,
    TrainingDeclaration,
    load_declaration,
    parse_declaration,
)
from support.synthetic import REPO_PATHS

DR015 = REPO_PATHS.root / "docs" / "decisions" / "DR-015-common-training-rules.md"
TEXT = DR015.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declaration() -> TrainingDeclaration:
    return load_declaration(REPO_PATHS)[0]


def raw() -> dict[str, Any]:
    loaded = yaml.safe_load(REPO_PATHS.training_spec.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def rejects(change: Any, fragment: str) -> None:
    document = raw()
    change(document)
    with pytest.raises(SpecificationError, match=fragment):
        parse_declaration(document)


def test_the_real_declaration_loads_with_a_snapshot_of_its_bytes() -> None:
    declaration, snapshot = load_declaration(REPO_PATHS)
    content = REPO_PATHS.training_spec.read_bytes()
    assert snapshot.path == "config/training.yaml"
    assert snapshot.sha256 == hashlib.sha256(content).hexdigest()
    assert snapshot.content == content.decode("utf-8")
    assert tuple(declaration.models) == MODEL_NAMES
    assert declaration.streams == MODEL_NAMES


def test_a_missing_file_is_a_file_access_error(tmp_path: Path) -> None:
    with pytest.raises(FileAccessError, match="cannot be read"):
        load_declaration(ProjectPaths(tmp_path))


def test_a_malformed_file_names_its_path(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "training.yaml").write_text("search: [1]\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match=r"training\.yaml"):
        load_declaration(ProjectPaths(tmp_path))


# --- validation ----------------------------------------------------------------------------------


def test_unknown_and_missing_sections_are_rejected() -> None:
    rejects(lambda d: d.update(extra=1), "unknown \\['extra'\\]")
    rejects(lambda d: d.pop("seeds"), "missing \\['seeds'\\]")
    rejects(lambda d: d["search"].pop("trials"), "search: missing \\['trials'\\]")


def test_streams_follow_the_model_order_then_the_reference() -> None:
    rejects(lambda d: d["search"]["streams"].reverse(), "search.streams must be")
    rejects(lambda d: d["models"].pop("logistic_regression"), "missing \\['logistic_regression'\\]")


def test_numbers_are_typed_and_bounded() -> None:
    rejects(lambda d: d["search"].update(trials=1), "at least 2")
    rejects(lambda d: d["search"].update(trials=True), "search.trials")
    rejects(lambda d: d["search"].update(sampler_seed=2**32), "below 4294967296")
    rejects(lambda d: d["seeds"].update(additional=[43, 43]), "must not repeat")
    rejects(lambda d: d["seeds"].update(additional=[]), "non-empty list")
    rejects(lambda d: d["v1_objective"].update(daily_k=0), "daily_k")
    rejects(lambda d: d["early_stopping"]["tree"].update(patience=2000), "below the maximum")


def test_parameters_are_checked() -> None:
    def parameter(d: dict[str, Any], model: str = "lightgbm", index: int = 0) -> dict[str, Any]:
        entry: dict[str, Any] = d["models"][model]["parameters"][index]
        return entry

    rejects(lambda d: parameter(d).update(distribution="normal"), "unknown distribution")
    rejects(lambda d: parameter(d).update(low=0.5, high=0.5), "low must be below high")
    rejects(lambda d: parameter(d).update(low=0.0), "positive low bound")
    rejects(lambda d: parameter(d).update(reference=0.9), "outside the declared bounds")
    rejects(lambda d: parameter(d, index=1).update(reference=31.5), "integer value")
    rejects(lambda d: parameter(d, index=1).update(low=0.5), "low")
    rejects(lambda d: parameter(d).pop("reference"), "missing \\['reference'\\]")
    rejects(lambda d: parameter(d).update(factor=8), "unknown \\['factor'\\]")
    rejects(
        lambda d: d["models"]["lightgbm"]["parameters"].append(dict(parameter(d))),
        "repeated parameter names",
    )
    rejects(lambda d: d["models"]["lightgbm"]["parameters"].clear(), "non-empty list")


def test_scaled_parameters_need_a_factor_and_a_multiple_as_reference() -> None:
    def token(d: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = d["models"]["ft_transformer"]["parameters"][1]
        return entry

    assert token(raw())["name"] == "d_token"
    rejects(lambda d: token(d).pop("factor"), "missing \\['factor'\\]")
    rejects(lambda d: token(d).update(reference=190), "integer value")
    rejects(lambda d: token(d).update(reference=520), "outside the declared bounds")


def test_a_mixture_reference_may_be_zero_but_a_plain_one_must_lie_in_bounds() -> None:
    def lambda_l2(d: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = d["models"]["lightgbm"]["parameters"][5]
        return entry

    assert lambda_l2(raw())["reference"] == 0.0
    rejects(lambda d: lambda_l2(d).update(reference=20.0), "outside the declared bounds")
    rejects(lambda d: lambda_l2(d).update(distribution="log_uniform"), "outside the declared")


def test_tree_stopping_settings_agree_with_the_early_stopping_rule() -> None:
    rejects(
        lambda d: d["models"]["lightgbm"]["fixed"].update(num_iterations=1000), "num_iterations"
    )
    rejects(lambda d: d["models"]["catboost"]["fixed"].update(od_wait=10), "od_wait")


def test_fixed_settings_must_be_exactly_the_declared_ones_with_their_types() -> None:
    rejects(lambda d: d["models"]["lightgbm"]["fixed"].pop("boost_from_average"), "missing")
    rejects(lambda d: d["models"]["lightgbm"]["fixed"].update(extra=1), "unknown \\['extra'\\]")
    rejects(lambda d: d["models"]["lightgbm"]["fixed"].update(bagging_freq="1"), "kind int")
    rejects(lambda d: d["models"]["lightgbm"]["fixed"].update(extra_trees="false"), "kind bool")
    rejects(lambda d: d["models"]["catboost"]["fixed"].update(model_size_reg=None), "kind number")
    rejects(lambda d: d["models"]["ft_transformer"]["fixed"].update(no_weight_decay=[]), "strings")
    rejects(
        lambda d: d["models"]["logistic_regression"]["fixed"].update(class_weight="balanced"),
        "kind null",
    )
    rejects(lambda d: d["models"]["resnet"]["fixed"].clear(), "missing")


def test_optimiser_settings_are_exact_and_typed() -> None:
    rejects(lambda d: d["neural_training"]["optimiser"].pop("eps"), "missing")
    rejects(lambda d: d["neural_training"]["optimiser"].update(epsilon=1.0e-8), "unknown")
    rejects(lambda d: d["neural_training"]["optimiser"].update(amsgrad="false"), "kind bool")
    rejects(lambda d: d["neural_training"]["optimiser"].update(betas=[0.9]), "kind pair")
    rejects(lambda d: d["neural_training"]["optimiser"].update(eps=float("nan")), "kind number")


def test_searched_and_fixed_names_do_not_overlap() -> None:
    def rename(d: dict[str, Any]) -> None:
        d["models"]["lightgbm"]["parameters"][0]["name"] = "boosting"

    rejects(rename, "searched and fixed settings overlap: \\['boosting'\\]")


def test_neural_training_values_are_checked() -> None:
    rejects(
        lambda d: d["neural_training"].update(micro_batch_candidates=[512, 1024]),
        "largest first, starting at the batch",
    )
    rejects(lambda d: d["neural_training"]["evaluation_batch"].pop("resnet"), "missing")
    rejects(lambda d: d["neural_training"].update(effective_batch=0), "effective_batch")


def test_a_repeated_yaml_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    text = REPO_PATHS.training_spec.read_text(encoding="utf-8")
    (tmp_path / "config" / "training.yaml").write_text(text + "\nseeds: {}\n", encoding="utf-8")
    with pytest.raises(SpecificationError, match="cannot be read as YAML"):
        load_declaration(ProjectPaths(tmp_path))


# --- agreement with DR-015 -----------------------------------------------------------------------


def number(text: str) -> float:
    return float(text.replace(",", ""))


def table_after(heading: str) -> dict[str, tuple[str, str]]:
    """The parameter rows of the first table after ``heading`` in DR-015: name to (search
    distribution, reference)."""
    lines = TEXT[TEXT.index(heading) :].splitlines()[1:]
    rows: dict[str, tuple[str, str]] = {}
    started = False
    for line in lines:
        if not line.startswith("|"):
            if started:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        started = True
        if cells[0].startswith("`"):
            rows[cells[0].strip("`")] = (cells[1], cells[2])
    return rows


DISTRIBUTION = re.compile(
    r"(?P<mix>0\.5 · \{0\} \+ 0\.5 · )?(?:(?P<factor>\d+) · )?"
    r"(?P<kind>Uniform|LogUniform|UniformInt|LogUniformInt)\[(?P<low>[^,\]]+), (?P<high>[^\]]+)\]"
)


def parse_distribution(text: str) -> tuple[str, float, float, int | None]:
    """DR-015's notation as (declaration distribution, low, high, factor)."""
    match = DISTRIBUTION.fullmatch(text)
    assert match is not None, text
    kind = {
        "Uniform": "uniform",
        "LogUniform": "log_uniform",
        "UniformInt": "uniform_int",
        "LogUniformInt": "log_uniform_int",
    }[match["kind"]]
    if match["mix"]:
        kind = f"mixture_{kind}"
    if match["factor"]:
        kind = "scaled_uniform_int"
    factor = int(match["factor"]) if match["factor"] else None
    return kind, number(match["low"]), number(match["high"]), factor


TABLES = {
    "lightgbm": "**LightGBM** (central processing unit",
    "catboost": "**CatBoost** (graphics processing unit",
    "resnet": "**ResNet-style MLP** (GPU):",
    "ft_transformer": "**FT-Transformer** (GPU):",
}


@pytest.mark.parametrize("model", TABLES)
def test_search_spaces_and_references_match_dr015(
    declaration: TrainingDeclaration, model: str
) -> None:
    rows = table_after(TABLES[model])
    space = declaration.space(model)
    assert [p.name for p in space.parameters] == list(rows)  # same parameters, same draw order
    for parameter in space.parameters:
        distribution, reference = rows[parameter.name]
        kind, low, high, factor = parse_distribution(distribution)
        assert (parameter.distribution, parameter.low, parameter.high) == (kind, low, high)
        assert parameter.factor == factor, parameter.name
        match = re.match(r"[-+]?[0-9.]+(?:e[-+]?\d+)?", reference)
        assert match is not None, reference
        assert parameter.reference == number(match.group(0)), parameter.name


def test_the_logistic_regression_space_matches_dr015(declaration: TrainingDeclaration) -> None:
    (parameter,) = declaration.space("logistic_regression").parameters
    assert (parameter.name, parameter.distribution) == ("C", "log_uniform")
    assert (parameter.low, parameter.high, parameter.reference) == (1e-4, 1e4, 1.0)
    assert "trials 2–10 draw C from LogUniform[10⁻⁴, 10⁴]" in TEXT
    assert "trial 1 uses C (inverse regularisation strength) = 1" in TEXT


def test_search_budget_streams_and_seeds_match_dr015(declaration: TrainingDeclaration) -> None:
    assert "**Budget: 10 trials per configuration.**" in TEXT
    assert declaration.trials == 10
    assert "sampler seed 42" in TEXT
    assert declaration.sampler_seed == 42
    assert "fifth child stream of the sampler of §1" in TEXT
    assert declaration.streams[4] == "logistic_regression"
    assert "**Additional seeds: 43 and 44**" in TEXT
    assert declaration.additional_seeds == (43, 44)
    assert "**Neural preprocessing seed: 0.**" in TEXT
    assert declaration.preprocessing_seed == 0


def test_early_stopping_matches_dr015(declaration: TrainingDeclaration) -> None:
    rows = {
        line.split("|")[1].strip(): [c.strip() for c in line.strip().strip("|").split("|")]
        for line in TEXT.splitlines()
        if line.startswith(("| Tree-based (LightGBM", "| Neural (ResNet"))
    }
    tree = next(v for k, v in rows.items() if k.startswith("Tree-based"))
    neural = next(v for k, v in rows.items() if k.startswith("Neural"))

    def first_integer(cell: str) -> int:
        return int(number(re.search(r"[\d,]+", cell).group(0)))  # type: ignore[union-attr]

    assert (first_integer(tree[1]), first_integer(tree[2])) == (50, 2000)
    assert (first_integer(neural[1]), first_integer(neural[2])) == (5, 15)
    assert (declaration.tree_stopping.patience, declaration.tree_stopping.maximum) == (50, 2000)
    assert (declaration.neural_stopping.patience, declaration.neural_stopping.maximum) == (5, 15)


def test_the_v1_daily_budget_is_the_derived_value(declaration: TrainingDeclaration) -> None:
    v1 = load_dataset_spec(REPO_PATHS).partition("v1")
    days = v1.last_day - v1.first_day + 1
    assert round(0.02 * v1.rows / days) == declaration.v1_daily_k == 62
    assert "&= 62" in TEXT


def test_neural_training_values_match_dr015(declaration: TrainingDeclaration) -> None:
    neural = declaration.neural
    assert "**effective batch of 1,024 Train rows**" in TEXT
    assert neural.effective_batch == 1024
    assert "micro-batches of 1,024, 512, 256, 128 and 64 rows" in TEXT
    assert neural.micro_batch_candidates == (1024, 512, 256, 128, 64)
    assert "at least 1,024 MiB" in TEXT
    assert neural.preflight_free_margin_mib == 1024
    assert "4,096 rows for the ResNet-style MLP and 16 for FT-Transformer" in TEXT
    assert dict(neural.evaluation_batch) == {"resnet": 4096, "ft_transformer": 16}
    assert "`betas` (0.9, 0.999)" in TEXT
    assert "`eps` 1e-8" in TEXT
    assert "(`foreach=False`, `fused=False`)" in TEXT
    optimiser = neural.optimiser
    assert optimiser["name"] == "AdamW"
    assert (optimiser["betas"], optimiser["eps"]) == ([0.9, 0.999], 1e-8)
    assert (optimiser["amsgrad"], optimiser["foreach"], optimiser["fused"]) == (False,) * 3


def settings(paragraph_start: str, paragraph_end: str) -> dict[str, Any]:
    """The ``name=value`` settings in the DR-015 text between two markers."""
    start = TEXT.index(paragraph_start)
    section = TEXT[start : TEXT.index(paragraph_end, start)]
    found: dict[str, Any] = {}
    for name, value in re.findall(r"`(\w+)=(\"?[\w./-]+\"?)`", section):
        found[name] = convert(value)
    return found


def convert(value: str) -> Any:
    if value.startswith('"'):
        return value.strip('"')
    lowered = {"true": True, "false": False, "none": None}
    if value.lower() in lowered:
        return lowered[value.lower()]
    try:
        return int(value)
    except ValueError:
        return float(value)


def test_lightgbm_fixed_settings_match_dr015(declaration: TrainingDeclaration) -> None:
    documented = settings("Fixed LightGBM settings, set explicitly:", "All are LightGBM 4.7.0")
    assert len(documented) == 21
    assert dict(declaration.space("lightgbm").fixed) == documented


def test_catboost_fixed_settings_match_dr015(declaration: TrainingDeclaration) -> None:
    documented = settings("Fixed CatBoost settings of", "`random_strength` is not searched")
    assert set(documented) == {
        "random_strength",
        "model_size_reg",
        "permutation_count",
        "fold_permutation_block",
        "boost_from_average",
        "metric_period",
        "iterations",
        "od_wait",
    }
    fixed = dict(declaration.space("catboost").fixed)
    assert fixed.pop("eval_metric") == "v1_objective"
    assert fixed == documented


def test_logistic_regression_fixed_settings_match_dr015(declaration: TrainingDeclaration) -> None:
    documented = settings("- **Fitting.** Each candidate is fitted", "- **Selection.** The highest")
    fixed = dict(declaration.space("logistic_regression").fixed)
    assert fixed.pop("solver") == "lbfgs"
    for name in ("fit_intercept", "class_weight", "warm_start", "l1_ratio", "tol", "max_iter"):
        assert fixed[name] == documented[name], name
    assert '`solver="lbfgs"`' in TEXT


def test_fixed_neural_settings_match_dr015(declaration: TrainingDeclaration) -> None:
    fixed = dict(declaration.space("ft_transformer").fixed)
    assert (
        "`residual_dropout=0`, `d_ffn_factor=4/3`, 8 attention heads, the ReGLU "
        "(rectified gated linear unit) activation, pre-normalisation and Kaiming initialisation"
    ) in TEXT
    assert fixed["residual_dropout"] == 0.0
    assert fixed["d_ffn_factor"] == 4 / 3
    assert (fixed["n_heads"], fixed["activation"]) == (8, "reglu")
    assert (fixed["normalization"], fixed["initialization"]) == ("pre", "kaiming")
    assert fixed["no_weight_decay"] == ["feature_tokenizer", "layer_norm", "bias"]
    assert "Weight decay applies to every parameter, as in REF-103" in TEXT
    assert dict(declaration.space("resnet").fixed) == {"weight_decay_scope": "all"}


def test_the_catboost_reference_learning_rate_is_the_documented_rule(
    declaration: TrainingDeclaration,
) -> None:
    """DR-015 §6: CatBoost's automatic rule for this setting, evaluated once and recorded."""
    train_rows = load_dataset_spec(REPO_PATHS).partition("train").rows
    maximum = declaration.tree_stopping.maximum
    coefficient_n, coefficient_0, coefficient_i = -0.085, -2.055, -0.414
    rate = min(
        math.exp(coefficient_n * math.log(train_rows) + coefficient_0)
        * (maximum / 1000) ** coefficient_i,
        0.5,
    )
    (learning_rate,) = [
        p for p in declaration.space("catboost").parameters if p.name == "learning_rate"
    ]
    assert learning_rate.reference == round(rate, 6) == 0.03234


# --- the computational-cost procedure (DR-015 §9) ------------------------------------------------


def test_timing_settings_match_dr015(declaration: TrainingDeclaration) -> None:
    timing = declaration.timing
    assert "**3 timed full passes**" in TEXT
    assert timing.scoring_passes == 3
    assert "**one full prediction per boundary** for the tree-based models" in TEXT
    assert timing.warmup_tree == 1
    assert "**three batch calls**" in TEXT
    assert timing.warmup_neural == 3
    assert "**10,000 distinct V1 rows**" in TEXT
    assert timing.latency_rows == 10_000
    assert "**100 untimed warm-up calls**" in TEXT
    assert timing.latency_warmup == 100
    assert "**3 rounds**" in TEXT
    assert timing.latency_rounds == 3
    assert "`numpy.percentile` using the `linear` method" in TEXT
    assert timing.percentile_method == "linear"
    assert "**60-second settling interval**" in TEXT
    assert timing.settling_seconds == 60
    assert "**every 100 ms**" in TEXT
    assert timing.host_sample_interval_ms == 100


def test_the_timing_batches_are_the_evaluation_batches_of_the_neural_models(
    declaration: TrainingDeclaration,
) -> None:
    assert "4,096 rows for the ResNet-style MLP, 16 for FT-Transformer" in TEXT
    assert dict(declaration.neural.evaluation_batch) == {"resnet": 4096, "ft_transformer": 16}


def test_the_latency_sample_fits_in_v1() -> None:
    """Feasibility only: that the rows come from V1 is the runner's rule (C11a)."""
    v1 = load_dataset_spec(REPO_PATHS).partition("v1")
    assert load_declaration(REPO_PATHS)[0].timing.latency_rows <= v1.rows


def test_the_timing_block_is_checked() -> None:
    rejects(lambda d: d.pop("timing"), "missing")
    rejects(lambda d: d["timing"].update({"extra": 1}), "unknown")
    rejects(lambda d: d["timing"]["scoring"].update({"passes": 0}), "passes")
    rejects(lambda d: d["timing"]["scoring"].update({"warmup_tree": True}), "warmup_tree")
    rejects(lambda d: d["timing"]["latency"].update({"rounds": 1.5}), "rounds")
    rejects(
        lambda d: d["timing"]["latency"].update({"percentile_method": "lower"}),
        "percentile_method",
    )
    rejects(lambda d: d["timing"]["session"].update({"settling_seconds": -1}), "settling_seconds")
    rejects(lambda d: d["timing"]["memory"].pop("host_sample_interval_ms"), "missing")
    rejects(lambda d: d["timing"]["memory"].update({"host_sample_interval_ms": 0}), "interval_ms")
    rejects(lambda d: d["timing"]["latency"].update({"sample_rows": 0}), "sample_rows")
    rejects(lambda d: d["timing"]["latency"].update({"warmup_calls": 0}), "warmup_calls")
    rejects(lambda d: d["timing"]["scoring"].update({"warmup_neural": 0}), "warmup_neural")
    rejects(lambda d: d["timing"]["scoring"].update({"warmup_tree": "1"}), "warmup_tree")
