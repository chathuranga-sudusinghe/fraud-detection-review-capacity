"""Training-device controls of DR-013: the LightGBM build check, the neural-network precision and
determinism settings, seeding and the run-environment record.

* ``verify_lightgbm_build`` checks at start-up that the installed LightGBM library is the
  recorded CUDA build (version and SHA-256 of ``lib_lightgbm.so``, DR-013 §2);
* ``configure_torch`` applies the precision and determinism settings of DR-013 §4 and returns
  their effective values;
* ``seed_everything`` seeds the Python, NumPy and PyTorch generators (DR-013 §6);
* ``environment_record`` collects what DR-013 §6 records with every run: software versions, the
  GPU and its driver, PyTorch's thread counts, CPU affinity, the memory visible to WSL and the
  relevant environment variables. Each library's own thread settings are recorded with the
  model that uses them.

The model libraries are imported inside the functions that need them, so that the rest of the
package works without them.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from review_capacity.config import SEED_LIMIT, LightGBMBuild, NeuralSettings
from review_capacity.errors import EnvironmentMismatchError
from review_capacity.hashing import file_sha256

logger = logging.getLogger(__name__)

RECORDED_PACKAGES = ("lightgbm", "catboost", "torch", "numpy", "pandas", "scikit-learn")
RECORDED_VARIABLES = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_WORKSPACE_SIZE",
    "TORCH_CUBLASLT_UNIFIED_WORKSPACE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_DYNAMIC",
    "OMP_THREAD_LIMIT",
    "OMP_SCHEDULE",
    "MKL_NUM_THREADS",
)


@dataclass(frozen=True)
class LightGBMIdentity:
    """The installed LightGBM: version, library file and its SHA-256."""

    version: str
    library: Path
    sha256: str


# The message of the bare Exception that LightGBM 4.7.0 raises when it finds no library file
# (lightgbm/libpath.py); recognised only for the pinned version.
MISSING_LIGHTGBM_LIBRARY = "Cannot find lightgbm library file in following paths:"


def _load_lightgbm() -> tuple[str, object]:
    import lightgbm
    from lightgbm.libpath import _LIB

    return lightgbm.__version__, _LIB._name


def installed_lightgbm(
    load: Callable[[], tuple[str, object]] = _load_lightgbm,
) -> tuple[str, Path]:
    """Version of the installed LightGBM and the shared library it actually loaded.

    LightGBM 4.7.0 loads the first ``lib_lightgbm.so`` found in ``site-packages/``,
    ``lightgbm/bin/`` and ``lightgbm/lib/`` (``lightgbm/libpath.py``), so the loaded file, not
    the expected wheel path, is the one identified. ``_LIB`` is a private attribute of the
    pinned version; a unit test guards this boundary. An import failure or a missing library
    is an environment mismatch; any other exception is a defect and propagates.
    """
    try:
        version, name = load()
    except (ImportError, OSError) as error:
        raise EnvironmentMismatchError(f"LightGBM cannot be imported: {error}") from error
    except Exception as error:
        if type(error) is not Exception or not str(error).startswith(MISSING_LIGHTGBM_LIBRARY):
            raise
        raise EnvironmentMismatchError(f"LightGBM cannot be imported: {error}") from error
    if not isinstance(name, str) or not name:
        raise EnvironmentMismatchError("LightGBM reports no loaded library file")
    return version, Path(name).resolve()


def verify_lightgbm_build(
    expected: LightGBMBuild, *, installed: tuple[str, Path] | None = None
) -> LightGBMIdentity:
    """Check that the installed LightGBM is the recorded build; raise otherwise.

    A rebuild is recorded anew with its hash before any run uses it (DR-013 §2), so a
    different hash stops the run instead of training with an unrecorded library.
    """
    version, library = installed_lightgbm() if installed is None else installed
    if version != expected.version:
        raise EnvironmentMismatchError(
            f"LightGBM {version} is installed; the recorded build is {expected.version}"
        )
    if not library.is_file():
        raise EnvironmentMismatchError(f"LightGBM library not found: {library}")
    digest = file_sha256(library)
    if digest != expected.library_sha256:
        raise EnvironmentMismatchError(
            f"{library} has SHA-256 {digest}, not the recorded build "
            f"{expected.library_sha256}; record a rebuild in config/models.yaml before using it"
        )
    logger.info("LightGBM %s: the recorded CUDA build (SHA-256 %s...)", version, digest[:16])
    return LightGBMIdentity(version=version, library=library, sha256=digest)


def configure_torch(settings: NeuralSettings) -> dict[str, Any]:
    """Apply the precision and determinism settings of DR-013 §4; return the effective values.

    ``fp32_precision`` and the bf16 reduction switch are separate PyTorch controls; the older
    ``allow_tf32`` interface is not used, so the two interfaces are not mixed.
    """
    import torch

    torch.backends.cuda.matmul.fp32_precision = settings.matmul_fp32_precision
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        settings.allow_bf16_reduced_precision_reduction
    )
    torch.use_deterministic_algorithms(settings.deterministic_algorithms, warn_only=False)
    torch.backends.cudnn.benchmark = settings.cudnn_benchmark
    effective = {
        "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    logger.info("PyTorch precision and determinism settings: %s", effective)
    return effective


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy (global) and PyTorch (CPU and every GPU) generators.

    Code that needs its own random numbers takes an explicit generator derived from the same
    seed; the global generators are seeded for library code that uses them. The seed is checked
    before any generator changes; ``torch.manual_seed`` also seeds every CUDA device.
    """
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, not {seed!r}")
    if not 0 <= seed < SEED_LIMIT:
        raise ValueError(f"Seed {seed} outside 0 to {SEED_LIMIT - 1}")
    random.seed(seed)
    np.random.seed(seed)  # the legacy global generator, used by some library code
    torch.manual_seed(seed)
    logger.info("Random generators seeded with %d", seed)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            logger.warning("Package %s is not installed; recorded as null", name)
            versions[name] = None
    return versions


def _visible_memory_bytes(meminfo: Path = Path("/proc/meminfo")) -> int | None:
    """Usable RAM visible to the operating system: under WSL, the guest-visible RAM total
    (``MemTotal``), not the exact ``.wslconfig`` setting or a lower per-process limit."""
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError as error:
        logger.warning("%s unreadable (%s); visible memory recorded as null", meminfo, error)
        return None
    logger.warning("%s has no MemTotal line; visible memory recorded as null", meminfo)
    return None


def _nvidia_driver_version() -> str | None:
    """The NVIDIA driver version from ``nvidia-smi``; null, with a warning, if unavailable.

    ``torch.version.cuda`` is the CUDA version PyTorch was built with, not the driver.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("NVIDIA driver version unavailable (%s); recorded as null", error)
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    if len(versions) != 1:
        raise EnvironmentMismatchError(f"Expected one NVIDIA driver version, found {versions}")
    return versions[0]


def _gpu_record() -> dict[str, Any]:
    import torch

    record: dict[str, Any] = {
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "available": torch.cuda.is_available(),
        "driver": _nvidia_driver_version(),
    }
    if record["available"]:
        properties = torch.cuda.get_device_properties(0)
        record.update(
            name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            memory_bytes=properties.total_memory,
            multiprocessors=properties.multi_processor_count,
        )
    return record


def environment_record() -> dict[str, Any]:
    """The run environment recorded with every run (DR-013 §6)."""
    import torch

    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cpu": {
            "logical_processors": os.cpu_count(),
            "affinity": affinity,
            "visible_memory_bytes": _visible_memory_bytes(),
        },
        "torch_threads": {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
        "gpu": _gpu_record(),
        "environment_variables": {name: os.environ.get(name) for name in RECORDED_VARIABLES},
    }
