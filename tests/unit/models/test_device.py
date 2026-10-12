"""DR-013 device controls: the LightGBM build check, PyTorch settings, seeding and the
environment record (no data read; no GPU required). Global PyTorch and random-generator state
changed by a test is restored."""

from __future__ import annotations

import random
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from review_capacity.config import LightGBMBuild, load_model_settings
from review_capacity.errors import EnvironmentMismatchError
from review_capacity.hashing import file_sha256
from review_capacity.models import device
from review_capacity.models.device import (
    configure_torch,
    environment_record,
    installed_lightgbm,
    seed_everything,
    verify_lightgbm_build,
)
from support.synthetic import REPO_PATHS


@pytest.fixture
def library(tmp_path: Path) -> Path:
    path = tmp_path / "lib_lightgbm.so"
    path.write_bytes(b"recorded build")
    return path


def test_recorded_build_is_accepted(library: Path) -> None:
    expected = LightGBMBuild(version="4.7.0", library_sha256=file_sha256(library))
    identity = verify_lightgbm_build(expected, installed=("4.7.0", library))
    assert identity.sha256 == expected.library_sha256
    assert identity.library == library


def test_rebuilt_library_is_rejected(library: Path) -> None:
    expected = LightGBMBuild(version="4.7.0", library_sha256="0" * 64)
    with pytest.raises(EnvironmentMismatchError, match="not the recorded build"):
        verify_lightgbm_build(expected, installed=("4.7.0", library))


def test_other_version_is_rejected(library: Path) -> None:
    expected = LightGBMBuild(version="4.7.0", library_sha256=file_sha256(library))
    with pytest.raises(EnvironmentMismatchError, match=r"4\.6\.0 is installed"):
        verify_lightgbm_build(expected, installed=("4.6.0", library))


def test_missing_library_is_rejected(tmp_path: Path) -> None:
    expected = LightGBMBuild(version="4.7.0", library_sha256="0" * 64)
    with pytest.raises(EnvironmentMismatchError, match="not found"):
        verify_lightgbm_build(expected, installed=("4.7.0", tmp_path / "absent.so"))


def test_missing_lightgbm_library_is_an_environment_mismatch() -> None:
    def missing() -> tuple[str, object]:
        # LightGBM 4.7.0 raises a bare Exception with this message (lightgbm/libpath.py)
        raise Exception(f"{device.MISSING_LIGHTGBM_LIBRARY}\n/lib/lib_lightgbm.so")

    with pytest.raises(EnvironmentMismatchError, match="cannot be imported"):
        installed_lightgbm(missing)


def test_lightgbm_import_error_is_an_environment_mismatch() -> None:
    def broken() -> tuple[str, object]:
        raise ImportError("no module named lightgbm")

    with pytest.raises(EnvironmentMismatchError, match="cannot be imported"):
        installed_lightgbm(broken)


def test_other_lightgbm_failure_propagates_as_a_defect() -> None:
    def failing() -> tuple[str, object]:
        raise RuntimeError("unrelated")

    with pytest.raises(RuntimeError, match="unrelated"):
        installed_lightgbm(failing)


def test_the_loaded_lightgbm_library_is_identified() -> None:
    """Guards the private ``lightgbm.libpath._LIB`` boundary of the pinned version."""
    from lightgbm.libpath import _LIB

    version, library = installed_lightgbm()
    assert version == load_model_settings(REPO_PATHS).lightgbm_build.version
    assert library == Path(_LIB._name).resolve()
    assert library.name == "lib_lightgbm.so"
    assert library.is_file()


@pytest.fixture
def restore_torch() -> Iterator[None]:
    matmul = torch.backends.cuda.matmul
    saved = (
        matmul.fp32_precision,
        (
            matmul.allow_bf16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction_split_k,
        ),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
    )
    yield
    matmul.fp32_precision = saved[0]
    matmul.allow_bf16_reduced_precision_reduction = saved[1]
    torch.use_deterministic_algorithms(saved[2], warn_only=saved[3])
    torch.backends.cudnn.benchmark = saved[4]


@pytest.mark.usefixtures("restore_torch")
def test_torch_settings_follow_dr013() -> None:
    # Start from the opposite of every setting, so that an omitted assignment fails the test.
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.benchmark = True
    effective = configure_torch(load_model_settings(REPO_PATHS).neural)
    assert effective == {
        "matmul_fp32_precision": "ieee",
        "allow_bf16_reduced_precision_reduction": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
    }
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    assert torch.are_deterministic_algorithms_enabled()


@pytest.fixture
def restore_generators() -> Iterator[None]:
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.set_rng_state(saved[2])


@pytest.mark.usefixtures("restore_generators")
def test_seeding_repeats_every_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda_seeds: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)

    def draws() -> tuple[float, float, float]:
        seed_everything(42)
        return random.random(), float(np.random.random()), float(torch.rand(1))

    assert draws() == draws()
    assert cuda_seeds == [42, 42]  # forwarded to every GPU without touching a real one


@pytest.mark.usefixtures("restore_generators")
@pytest.mark.parametrize(
    ("seed", "error"), [(-1, ValueError), (2**32, ValueError), (0.5, TypeError), (True, TypeError)]
)
def test_invalid_seed_is_a_defect(seed: Any, error: type[Exception]) -> None:
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    with pytest.raises(error):
        seed_everything(seed)
    # Checked before any generator changes: Python's generator is seeded first, torch's last.
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_visible_memory_is_read_from_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       1000 kB\nMemFree:        1 kB\n", encoding="utf-8")
    assert device._visible_memory_bytes(meminfo) == 1000 * 1024
    assert device._visible_memory_bytes(tmp_path / "absent") is None
    meminfo.write_text("MemFree:        1 kB\n", encoding="utf-8")
    assert device._visible_memory_bytes(meminfo) is None


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=stdout)


def test_driver_version_is_queried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: _completed("592.01\n"))
    assert device._nvidia_driver_version() == "592.01"


def test_unavailable_driver_query_is_recorded_as_null(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert device._nvidia_driver_version() is None


def test_two_driver_versions_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: _completed("592.01\n580.10\n"))
    with pytest.raises(EnvironmentMismatchError, match="one NVIDIA driver version"):
        device._nvidia_driver_version()


def test_gpu_record_with_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    properties = SimpleNamespace(
        name="Test GPU", major=12, minor=0, total_memory=8 * 2**30, multi_processor_count=26
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: properties)
    # cudnn.version() checks the visible devices itself, so it is mocked too: the test must not
    # depend on a real GPU (it runs in CI on a CPU-only runner).
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91900)
    monkeypatch.setattr(device, "_nvidia_driver_version", lambda: "592.01")
    record = device._gpu_record()
    assert record["available"] is True
    assert record["cudnn"] == 91900
    assert record["compute_capability"] == "12.0"
    assert record["memory_bytes"] == 8 * 2**30
    assert record["driver"] == "592.01"


def test_gpu_record_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device, "_nvidia_driver_version", lambda: None)
    record = device._gpu_record()
    assert record["available"] is False
    assert "compute_capability" not in record


def test_environment_record_contents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(device, "_nvidia_driver_version", lambda: None)
    record = environment_record()
    variables = record["environment_variables"]
    assert variables["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert variables["OMP_NUM_THREADS"] is None  # recorded as found, including unset
    assert "TORCH_CUBLASLT_UNIFIED_WORKSPACE" in variables
    assert set(record["packages"]) >= {"lightgbm", "catboost", "torch"}
    assert set(record["cpu"]) == {"logical_processors", "affinity", "visible_memory_bytes"}
    assert record["torch_threads"]["intraop"] >= 1
    assert record["torch_threads"]["interop"] >= 1
    assert "driver" in record["gpu"]
