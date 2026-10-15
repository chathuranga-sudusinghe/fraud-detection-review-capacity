"""``config/models.yaml`` encodes DR-013 and malformed settings are rejected (no data read)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from review_capacity.config import TRAINED_MODELS, load_model_settings
from review_capacity.errors import SpecificationError
from review_capacity.models.registry import MODELS
from review_capacity.paths import ProjectPaths
from support.synthetic import REPO_PATHS


def test_lightgbm_settings_match_dr013() -> None:
    settings = load_model_settings(REPO_PATHS)
    cpu = settings.lightgbm.params("cpu")
    construction = {
        "max_bin": 255,
        "min_data_in_bin": 3,
        "bin_construct_sample_cnt": 200_000,
        "enable_bundle": False,
        "is_enable_sparse": False,
        "force_row_wise": False,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "use_missing": True,
        "zero_as_missing": False,
        "use_quantized_grad": False,
        "linear_tree": False,
        "forcedsplits_filename": "",
        "forcedbins_filename": "",
        "histogram_pool_size": -1,
        "objective": "binary",
    }
    assert construction.items() <= cpu.items()
    assert cpu["device_type"] == "cpu"
    assert cpu["deterministic"] is True
    assert cpu["num_threads"] == 24
    assert "gpu_use_dp" not in cpu
    assert "seed" not in cpu  # seeds come from the run, never from the device settings
    # DR-013 §2: LightGBM's CUDA learner fails on categorical features above 256 categories
    assert settings.lightgbm.devices == ("cpu",)
    with pytest.raises(SpecificationError, match="no gpu settings"):
        settings.lightgbm.params("gpu")


def test_training_devices_match_dr013() -> None:
    settings = load_model_settings(REPO_PATHS)
    assert settings.training_devices == {
        "lightgbm": "cpu",
        "catboost": "gpu",
        "resnet": "gpu",
        "ft_transformer": "gpu",
    }
    assert {model.name for model in MODELS} == set(TRAINED_MODELS)
    assert settings.training_device("lightgbm") == "cpu"
    assert settings.catboost.devices == ("gpu", "cpu")
    with pytest.raises(ValueError, match="Unknown model"):
        settings.training_device("xgboost")


def test_catboost_settings_match_dr013() -> None:
    settings = load_model_settings(REPO_PATHS)
    gpu = settings.catboost.params("gpu")
    cpu = settings.catboost.params("cpu")
    shared = {
        "border_count": 254,
        "bootstrap_type": "Bayesian",
        "boosting_type": "Plain",
        "has_time": True,
        "counter_calc_method": "SkipTest",
        "loss_function": "Logloss",
        "feature_border_type": "GreedyLogSum",
        "nan_mode": "Min",
        "max_ctr_complexity": 4,
        "one_hot_max_size": 2,
        "grow_policy": "SymmetricTree",
        "score_function": "Cosine",
        "random_score_type": "NormalWithModelSizeDecrease",
        "leaf_estimation_method": "Newton",
        "leaf_estimation_backtracking": "AnyImprovement",
        "sampling_unit": "Object",
    }
    for params in (gpu, cpu):
        assert shared.items() <= params.items()
        assert "subsample" not in params  # rejected with Bayesian bootstrapping
    assert gpu["task_type"] == "GPU"
    assert gpu["devices"] == "0"
    assert cpu["thread_count"] == 24
    assert gpu["simple_ctr"] == gpu["combinations_ctr"] == ["Borders", "FeatureFreq"]
    assert cpu["task_type"] == "CPU"
    assert cpu["simple_ctr"] == cpu["combinations_ctr"] == ["Borders", "Counter"]


def test_neural_and_seed_settings_match_dr013() -> None:
    settings = load_model_settings(REPO_PATHS)
    assert settings.main_seed == 42
    assert settings.neural.matmul_fp32_precision == "ieee"
    assert settings.neural.allow_bf16_reduced_precision_reduction is False
    assert settings.neural.deterministic_algorithms is True
    assert settings.neural.cudnn_benchmark is False
    assert (settings.neural.training_dtype, settings.neural.prediction_dtype) == (
        "bfloat16",
        "float32",
    )
    assert settings.neural.gpu_attention_backend == "efficient"
    assert settings.lightgbm_build.version == "4.7.0"
    assert len(settings.lightgbm_build.library_sha256) == 64


def test_parameters_for_an_unknown_device_are_a_defect() -> None:
    settings = load_model_settings(REPO_PATHS)
    with pytest.raises(ValueError, match="Unsupported device"):
        settings.lightgbm.params("tpu")  # type: ignore[arg-type]


def _write_models(root: Path, edit: Callable[[dict[str, Any]], None]) -> ProjectPaths:
    content = yaml.safe_load(REPO_PATHS.models_spec.read_text(encoding="utf-8"))
    edit(content)
    (root / "config").mkdir()
    (root / "config" / "models.yaml").write_text(yaml.safe_dump(content), encoding="utf-8")
    return ProjectPaths(root)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c["lightgbm"]["cpu"].update(max_bin=63), r"unexpected keys \['max_bin'\]"),
        (lambda c: c["lightgbm"]["cpu"].pop("deterministic"), r"missing keys \['deterministic'\]"),
        (lambda c: c["lightgbm"].update(gpu={"device_type": "cuda"}), r"unexpected keys \['gpu'\]"),
        (lambda c: c["training_devices"].update(lightgbm="gpu"), "settings for"),
        (lambda c: c["training_devices"].update(resnet="tpu"), "must be one of"),
        (lambda c: c["training_devices"].pop("catboost"), r"missing keys \['catboost'\]"),
        (lambda c: c["lightgbm"]["shared"].update(max_bin="255"), "must be an integer"),
        (lambda c: c["lightgbm"]["shared"].update(max_bin=1), "greater than 1"),
        (lambda c: c["lightgbm"]["shared"].update(enable_bundle=0), "true or false"),
        (lambda c: c["lightgbm"]["shared"].update(histogram_pool_size=float("nan")), "finite"),
        (lambda c: c["catboost"]["gpu"].update(devices=0), "non-empty string"),
        (lambda c: c["catboost"]["gpu"].update(simple_ctr="Borders"), "must be a list"),
        (lambda c: c["catboost"]["gpu"].update(simple_ctr=[]), "non-empty list"),
        (lambda c: c["catboost"]["cpu"].update(simple_ctr=["Borders", "Borders"]), "Repeated"),
        (lambda c: c["neural"].update(training_dtype="banana"), "must be one of"),
        (lambda c: c["neural"].update(gpu_attention_backend="typo"), "must be one of"),
        (lambda c: c["neural"].update(extra=1), r"unexpected keys \['extra'\]"),
        (lambda c: c["reproducibility"].update(main_seed=-1), "between 0 and"),
        (lambda c: c["reproducibility"].update(main_seed=2**32), "between 0 and"),
        (lambda c: c["lightgbm"]["build"].update(commit="x"), "unexpected keys"),
        (lambda c: c["lightgbm"]["cpu"].update(device_type="cuda"), "device_type='cpu'"),
        (lambda c: c["catboost"]["cpu"].update(task_type="GPU"), "task_type='CPU'"),
        (lambda c: c["lightgbm"]["build"].update(library_sha256="abc"), "64 hex"),
        (lambda c: c["reproducibility"].update(main_seed=True), "must be an integer"),
        (lambda c: c["neural"].update(cudnn_benchmark="no"), "true or false"),
        (lambda c: c["catboost"]["shared"].update(border_count={"a": 1}), "must be an integer"),
        (lambda c: c.pop("neural"), r"missing keys \['neural'\]"),
    ],
)
def test_malformed_model_settings_rejected(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    paths = _write_models(tmp_path, edit)
    with pytest.raises(SpecificationError, match=message):
        load_model_settings(paths)


@pytest.mark.parametrize(
    "content",
    [
        "? [invalid, key]\n: value\n",  # a sequence as a mapping key: TypeError in PyYAML
        "reproducibility:\n  main_seed: !!int invalid\n",  # ValueError in PyYAML
    ],
)
def test_unreadable_yaml_is_a_specification_error(tmp_path: Path, content: str) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(SpecificationError, match="cannot be read as YAML") as caught:
        load_model_settings(ProjectPaths(tmp_path))
    assert isinstance(caught.value.__cause__, TypeError | ValueError)
