"""The FT-Transformer (REF-087, REF-103) on the shared neural loop, on small synthetic tensors on
the CPU (no project data; the GPU path, the memory-efficient attention backend and the 758-token
sequence are exercised by the GPU probe, Issue #77). Global PyTorch and random-generator state
changed by a fit is restored."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.nn.attention import SDPBackend

from review_capacity.config import ModelSettings, load_model_settings
from review_capacity.errors import ModelError, SpecificationError
from review_capacity.models import ft_transformer_model
from review_capacity.models.base import class_weights, sample_weights
from review_capacity.models.ft_transformer_model import (
    FTTransformer,
    FTTransformerModel,
    MultiheadAttention,
    TransformerBlock,
    attention_backend,
)
from review_capacity.models.neural_training import MANIFEST_FILE, NeuralData, predict_logits
from review_capacity.models.registry import Weighting, configuration
from review_capacity.models.run_record import (
    jsonable,
    load_settings_with_snapshot,
    model_run_record,
)
from support.neural import (
    FT_HYPERPARAMETERS,
    FT_TRANSFORMER,
    LAYOUT,
    neural_data,
    neural_inputs,
    preserved_torch_state,
)
from support.synthetic import REPO_PATHS


@pytest.fixture(autouse=True)
def restore_torch_state() -> Iterator[None]:
    with preserved_torch_state():
        yield


@pytest.fixture(scope="module")
def settings() -> ModelSettings:
    return load_model_settings(REPO_PATHS)


def _balanced(data: NeuralData, train: NeuralData) -> NeuralData:
    weights = sample_weights(data.labels, class_weights(train.labels, Weighting.BALANCED))
    return replace(data, weights=weights)


@pytest.fixture(scope="module")
def train() -> NeuralData:
    data = neural_data(640, seed=1)
    return _balanced(data, data)


@pytest.fixture(scope="module")
def valid(train: NeuralData) -> NeuralData:
    return _balanced(neural_data(320, seed=2), train)


def _model(settings: ModelSettings, seed: int = 42, name: str = "FTT-b") -> FTTransformerModel:
    return FTTransformerModel(
        configuration(name), seed, settings=settings, layout=LAYOUT, device="cpu"
    )


def _fit(settings: ModelSettings, train: NeuralData, valid: NeuralData) -> FTTransformerModel:
    model = _model(settings)
    model.fit(train, valid, FT_HYPERPARAMETERS)
    return model


@pytest.fixture(scope="module")
def fitted(settings: ModelSettings, train: NeuralData, valid: NeuralData) -> FTTransformerModel:
    with preserved_torch_state():
        return _fit(settings, train, valid)


def _network(**changes: Any) -> FTTransformer:
    arguments: dict[str, Any] = {
        "n_numeric": 4,
        "cardinalities": (5, 3),
        "d_token": 24,
        "n_layers": 3,
        "d_ffn_factor": 2.0,
        "attention_dropout": 0.2,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.1,
        "gpu_backend": SDPBackend.EFFICIENT_ATTENTION,
    }
    return FTTransformer(**{**arguments, **changes})


def _inputs(rows: int, seed: int) -> tuple[Tensor, Tensor]:
    inputs = neural_inputs(rows, seed)
    return torch.from_numpy(inputs.numeric), torch.from_numpy(inputs.categorical)


# --- Architecture ---------------------------------------------------------------------------------


def test_network_follows_the_reference_architecture() -> None:
    network = _network()
    tokenizer = network.tokenizer
    assert tuple(tokenizer.weight.shape) == (1 + 4, 24)  # the [CLS] row and the numeric inputs
    assert tuple(tokenizer.bias.shape) == (4 + 2, 24)  # one bias per input, none for [CLS]
    embeddings = tokenizer.category_embeddings
    assert embeddings is not None
    assert tuple(embeddings.weight.shape) == (5 + 3, 24)
    assert tokenizer.category_offsets.tolist() == [0, 5]
    assert len(network.blocks) == 3
    for index, block in enumerate(network.blocks):
        assert isinstance(block, TransformerBlock)
        assert (block.norm0 is None) == (index == 0)  # no first attention normalisation
        assert (block.linear0.out_features, block.linear1.in_features) == (2 * 48, 48)  # ReGLU
        attention = block.attention
        assert (attention.n_heads, attention.d_head, attention.d_padded) == (8, 3, 8)
        for projection in (attention.query, attention.key, attention.value, attention.output):
            assert bool((projection.bias == 0).all())
    assert network.head.out_features == 1


def test_tokens_are_the_reference_feature_tokenizer() -> None:
    network = _network().eval()
    tokenizer = network.tokenizer
    numeric, categorical = _inputs(5, seed=0)
    with torch.no_grad():
        tokens = tokenizer(numeric, categorical)
    assert tuple(tokens.shape) == (5, 1 + 4 + 2, 24)
    weight, bias = tokenizer.weight.detach(), tokenizer.bias.detach()
    embeddings = tokenizer.category_embeddings
    assert embeddings is not None
    table = embeddings.weight.detach()
    torch.testing.assert_close(tokens[:, 0], weight[0].expand(5, -1))  # [CLS]: no bias
    for j in range(4):
        expected = bias[j] + numeric[:, j : j + 1] * weight[1 + j]
        torch.testing.assert_close(tokens[:, 1 + j], expected)
    for j, offset in enumerate((0, 5)):
        expected = bias[4 + j] + table[offset + categorical[:, j]]
        torch.testing.assert_close(tokens[:, 5 + j], expected)


def _reference_attention(
    module: MultiheadAttention, x_query: Tensor, x_key_value: Tensor
) -> Tensor:
    """The attention of REF-103 (``bin/ft_transformer.py``), written out without dropout."""

    def heads(x: Tensor) -> Tensor:
        batch, tokens, d = x.shape
        d_head = d // module.n_heads
        return x.reshape(batch, tokens, module.n_heads, d_head).transpose(1, 2)

    q, k = heads(module.query(x_query)), heads(module.key(x_key_value))
    v = heads(module.value(x_key_value))
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
    merged = (weights @ v).transpose(1, 2).reshape(x_query.shape[0], x_query.shape[1], -1)
    output: Tensor = module.output(merged)
    return output


@pytest.mark.parametrize("d_token", [24, 64])  # heads of 3 (padded to 8) and of 8 (no padding)
def test_attention_is_the_reference_attention(d_token: int) -> None:
    torch.manual_seed(0)
    module = MultiheadAttention(
        d_token, dropout=0.3, gpu_backend=SDPBackend.EFFICIENT_ATTENTION
    ).eval()
    x = torch.randn(4, 7, d_token)
    with torch.no_grad():
        full = module(x, x)
        cls_only = module(x[:, :1], x)
        expected = _reference_attention(module, x, x)
    torch.testing.assert_close(full, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cls_only, expected[:, :1], rtol=1e-5, atol=1e-6)
    with torch.no_grad():  # attention dropout is 0 outside training
        assert torch.equal(module(x, x), full)


@pytest.mark.parametrize("d_token", [24, 64])
@pytest.mark.parametrize("cls_only", [False, True])
def test_attention_gradients_are_the_reference_gradients(d_token: int, cls_only: bool) -> None:
    """With dropout off, the padded attention has the gradients of the REF-103 formula, for the
    inputs and every projection."""
    torch.manual_seed(1)
    module = MultiheadAttention(d_token, dropout=0.0, gpu_backend=SDPBackend.EFFICIENT_ATTENTION)
    x = torch.randn(3, 6, d_token, requires_grad=True)
    cotangent = torch.randn(3, 1 if cls_only else 6, d_token)
    inputs = [x, *module.parameters()]

    def gradients(output: Tensor) -> tuple[Tensor, ...]:
        return torch.autograd.grad((output * cotangent).sum(), inputs)

    query = x[:, :1] if cls_only else x
    actual = gradients(module(query, x))
    expected = gradients(_reference_attention(module, query, x))
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_the_backend_is_restricted_on_cuda_only(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[SDPBackend] = []

    def restricted(backend: SDPBackend) -> nullcontext[None]:
        requested.append(backend)
        return nullcontext()

    monkeypatch.setattr(ft_transformer_model, "sdpa_kernel", restricted)
    attention_backend("cuda", SDPBackend.EFFICIENT_ATTENTION)
    assert requested == [SDPBackend.EFFICIENT_ATTENTION]
    assert isinstance(attention_backend("cpu", SDPBackend.EFFICIENT_ATTENTION), nullcontext)
    assert requested == [SDPBackend.EFFICIENT_ATTENTION]  # the CPU keeps PyTorch's selection


def test_attention_passes_its_backend_dropout_and_true_head_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends: list[tuple[str, SDPBackend]] = []
    calls: list[dict[str, Any]] = []
    original = torch.nn.functional.scaled_dot_product_attention

    def backend_spy(device_type: str, gpu_backend: SDPBackend) -> nullcontext[None]:
        backends.append((device_type, gpu_backend))
        return nullcontext()

    def attention_spy(query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tensor:
        calls.append({"head": query.shape[-1], **kwargs})
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(ft_transformer_model, "attention_backend", backend_spy)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", attention_spy)
    module = MultiheadAttention(24, dropout=0.3, gpu_backend=SDPBackend.EFFICIENT_ATTENTION)
    x = torch.randn(2, 5, 24)
    module.train()(x, x)
    module.eval()(x, x)
    assert backends == [("cpu", SDPBackend.EFFICIENT_ATTENTION)] * 2
    assert [call["dropout_p"] for call in calls] == [0.3, 0.0]  # 0 outside training
    assert {call["scale"] for call in calls} == {1 / math.sqrt(3)}  # heads of 3, not padded 8
    assert {call["head"] for call in calls} == {8}


def test_the_last_block_keeps_the_cls_output() -> None:
    network = _network().eval()
    numeric, categorical = _inputs(6, seed=1)
    with torch.no_grad():
        tokens = network.tokenizer(numeric, categorical)
        block = network.blocks[-1]
        full = block(tokens, cls_only=False)
        cls_only = block(tokens, cls_only=True)
    assert tuple(cls_only.shape) == (6, 1, 24)
    torch.testing.assert_close(cls_only[:, 0], full[:, 0], rtol=1e-5, atol=1e-6)


def test_forward_and_backward_passes() -> None:
    torch.manual_seed(0)
    network = _network()
    numeric, categorical = _inputs(16, seed=0)
    logits = network(numeric, categorical)
    assert logits.shape == (16,)
    assert logits.dtype == torch.float32
    logits.sum().backward()
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_inputs_of_one_kind_alone() -> None:
    numeric_only = _network(cardinalities=()).eval()
    numeric, _ = _inputs(3, seed=2)
    assert numeric_only(numeric, torch.zeros(3, 0, dtype=torch.int64)).shape == (3,)
    categorical_only = _network(n_numeric=0).eval()
    _, categorical = _inputs(3, seed=2)
    assert categorical_only(torch.zeros(3, 0), categorical).shape == (3,)


def test_weight_decay_spares_the_tokenizer_normalisation_and_biases(
    settings: ModelSettings,
) -> None:
    model = _model(settings)
    network = model.build_network(model.check_architecture(FT_TRANSFORMER))
    decayed, undecayed = model.parameter_groups(network, 1e-4)
    assert (decayed["weight_decay"], undecayed["weight_decay"]) == (1e-4, 0.0)
    names = {id(parameter): name for name, parameter in network.named_parameters()}
    decayed_names = {names[id(parameter)] for parameter in decayed["params"]}
    undecayed_names = {names[id(parameter)] for parameter in undecayed["params"]}
    assert decayed_names | undecayed_names == set(names.values())
    assert all(name.endswith(".weight") for name in decayed_names)
    assert not any(name.startswith("tokenizer.") for name in decayed_names)
    assert not any("norm" in name for name in decayed_names)
    assert "last_normalization.weight" in undecayed_names  # the paper's rule (REF-087, E.3)
    assert "head.weight" in decayed_names
    assert "blocks.0.attention.query.weight" in decayed_names
    with pytest.raises(TypeError, match="FTTransformer"):
        model.parameter_groups(torch.nn.Linear(2, 1), 1e-4)


# --- Hyperparameters ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"d_embedding": 8}, "Unknown"),
        ({"n_layers": None}, "n_layers"),
        ({"d_token": 20}, "multiple of the 8 heads"),
        ({"d_token": 24.0}, "d_token must"),
        ({"d_token": 8, "d_ffn_factor": 0.1}, "hidden size"),
        ({"d_ffn_factor": 0}, "d_ffn_factor"),
        ({"attention_dropout": 1.0}, "attention_dropout"),
        ({"ffn_dropout": -0.1}, "ffn_dropout"),
        ({"residual_dropout": 1.5}, "residual_dropout"),
    ],
)
def test_invalid_hyperparameters_are_rejected(
    settings: ModelSettings,
    train: NeuralData,
    valid: NeuralData,
    changes: dict[str, Any],
    message: str,
) -> None:
    hyperparameters = {**FT_HYPERPARAMETERS, **changes}
    for name, value in changes.items():
        if value is None:
            del hyperparameters[name]
    with pytest.raises(SpecificationError, match=message):
        _model(settings).fit(train, valid, hyperparameters)


def test_only_ft_transformer_configurations(settings: ModelSettings) -> None:
    for name in ("RES-u", "Cat-b"):
        with pytest.raises(ValueError, match="not a ft_transformer configuration"):
            _model(settings, name=name)


# --- Training, prediction, persistence ------------------------------------------------------------


def test_fit_applies_and_records_dr013_settings(fitted: FTTransformerModel) -> None:
    resolved = fitted.parameters["resolved"]
    precision = resolved["precision"]
    assert precision["matmul_fp32_precision"] == "ieee"
    assert precision["deterministic_algorithms"] is True
    assert precision["gpu_attention_backend"] == "efficient"
    assert precision["attention"] == "scaled_dot_product_attention, PyTorch CPU backend selection"
    observed = resolved["observed_dtypes"]
    assert observed["training_logits"] == ["torch.bfloat16"]
    assert set(observed["optimizer_state"]) == {"exp_avg", "exp_avg_sq", "step"}
    groups = resolved["optimizer"]["parameter_groups"]
    assert [group["weight_decay"] for group in groups] == [FT_HYPERPARAMETERS["weight_decay"], 0.0]
    architecture = resolved["architecture"]
    assert architecture["n_heads"] == 8
    assert (architecture["d_head"], architecture["d_head_padded"]) == (3, 8)
    assert architecture["n_tokens"] == 1 + 4 + 2


def test_the_network_learns_the_signal(fitted: FTTransformerModel, valid: NeuralData) -> None:
    baseline = average_precision_score(
        valid.labels, np.full(valid.inputs.rows, 0.5), sample_weight=valid.weights
    )
    assert fitted.parameters["resolved"]["schedule"]["best_v1_score"] > baseline + 0.1


def test_prediction_uses_float32_logits_and_repeats(
    fitted: FTTransformerModel, valid: NeuralData
) -> None:
    network = fitted._network_or_raise()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):  # a caller's autocast region
        logits = predict_logits(network, valid.inputs, device=torch.device("cpu"), batch_size=96)
        scores = fitted.predict(valid.inputs)
    assert logits.dtype == np.float32
    assert scores.dtype == np.float64  # probabilities of the float32 logits, in float64
    expected = torch.sigmoid(torch.from_numpy(logits).double()).numpy()
    assert np.array_equal(scores, expected)
    assert np.array_equal(scores, fitted.predict(valid.inputs))  # no dropout at prediction


def test_the_same_seed_repeats_exactly(
    settings: ModelSettings, train: NeuralData, valid: NeuralData, fitted: FTTransformerModel
) -> None:
    again = _fit(settings, train, valid)
    assert np.array_equal(again.predict(valid.inputs), fitted.predict(valid.inputs))


def test_save_and_load_reproduce_the_model(
    fitted: FTTransformerModel, valid: NeuralData, settings: ModelSettings, tmp_path: Path
) -> None:
    fitted.save(tmp_path)
    loaded = FTTransformerModel.load(
        tmp_path, configuration("FTT-b"), 42, settings=settings, layout=LAYOUT, device="cpu"
    )
    assert np.array_equal(loaded.predict(valid.inputs), fitted.predict(valid.inputs))
    assert jsonable(loaded.parameters) == jsonable(fitted.parameters)


def test_load_rejects_changed_parameter_groups(
    fitted: FTTransformerModel, settings: ModelSettings, tmp_path: Path
) -> None:
    fitted.save(tmp_path)
    path = tmp_path / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    groups = manifest["training"]["optimizer"]["parameter_groups"]
    groups[0]["parameters"].append(groups[1]["parameters"].pop())  # decay one more parameter
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="parameter groups"):
        FTTransformerModel.load(
            tmp_path, configuration("FTT-b"), 42, settings=settings, layout=LAYOUT, device="cpu"
        )


def test_load_rejects_a_changed_attention_backend(
    fitted: FTTransformerModel, settings: ModelSettings, tmp_path: Path
) -> None:
    fitted.save(tmp_path)
    path = tmp_path / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["training"]["precision"]["gpu_attention_backend"] = "math"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="precision settings"):
        FTTransformerModel.load(
            tmp_path, configuration("FTT-b"), 42, settings=settings, layout=LAYOUT, device="cpu"
        )


def test_run_record_carries_the_attention_backend(fitted: FTTransformerModel) -> None:
    _, snapshot = load_settings_with_snapshot(REPO_PATHS)
    record = model_run_record(
        model=fitted,
        settings=snapshot,
        hyperparameters=FT_HYPERPARAMETERS,
        lightgbm_build=None,
        environment={},
        code={},
    )
    assert (record["model"], record["configuration"]) == ("ft_transformer", "FTT-b")
    json.dumps(record, allow_nan=False)
    parameters = record["parameters"]
    assert isinstance(parameters, dict)
    resolved = parameters["resolved"]
    assert isinstance(resolved, dict)
    precision = resolved["precision"]
    assert isinstance(precision, dict)
    assert precision["gpu_attention_backend"] == "efficient"
