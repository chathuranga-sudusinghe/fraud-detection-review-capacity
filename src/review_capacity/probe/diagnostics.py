"""Measurements of the GPU acceptance probe: score files and hashes, rankings and top-K
comparisons, timing and memory (``docs/gpu-acceptance-probe.md``, checks C8–C13).

* **Score files:** scores are stored as little-endian float64 (``<f8``) and row identifiers as
  little-endian int64 (``<i8``); the SHA-256 of these canonical bytes identifies a set of scores.
* **Rankings:** descending score, ties broken by ascending synthetic row identifier (C9).
* **GPU memory:** the whole-device maximum sampled by ``nvidia-smi`` every 0.5 seconds; a sampled
  maximum, not the true peak.
"""

from __future__ import annotations

import hashlib
import logging
import resource
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt

from review_capacity.errors import EnvironmentMismatchError
from review_capacity.files import write_atomically

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_MS = 500
WARM_UP_CALLS = 3
TIMED_REPETITIONS = 5
LATENCY_ROWS = 128

type Scores = npt.NDArray[np.float64]
type Ids = npt.NDArray[np.int64]


def canonical_sha256(values: npt.NDArray[Any], dtype: str) -> str:
    """SHA-256 of ``values`` as contiguous bytes of ``dtype`` (``<f8`` or ``<i8``)."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def save_scores(directory: Path, key: str, scores: Scores, ids: Ids) -> dict[str, Any]:
    """Write ``scores`` and their row identifiers as a comparison file; return its record."""
    name = f"{key}.npz"
    canonical = np.ascontiguousarray(scores, dtype="<f8")
    order = np.ascontiguousarray(ids, dtype="<i8")

    def write(target: Path) -> None:
        with target.open("wb") as handle:
            np.savez(handle, scores=canonical, ids=order)

    write_atomically(directory / name, write)
    return {
        "path": name,
        "sha256": canonical_sha256(canonical, "<f8"),
        "dtype": "<f8",
        "shape": list(canonical.shape),
        "row_order_sha256": canonical_sha256(order, "<i8"),
    }


def load_scores(directory: Path, record: dict[str, Any]) -> tuple[Scores, Ids]:
    """The scores and row identifiers of a comparison file, checked against its record."""
    with np.load(directory / record["path"]) as stored:
        scores = np.asarray(stored["scores"], dtype="<f8")
        ids = np.asarray(stored["ids"], dtype="<i8")
    if (
        canonical_sha256(scores, "<f8") != record["sha256"]
        or canonical_sha256(ids, "<i8") != record["row_order_sha256"]
    ):
        name = record["path"]
        raise EnvironmentMismatchError(f"Comparison file {name} does not match its record")
    return scores, ids


def ranking(scores: Scores, ids: Ids) -> npt.NDArray[np.intp]:
    """Row positions by descending score, ties by ascending row identifier."""
    return np.lexsort((ids, -scores))


def top_k_summary(scores: Scores, ids: Ids, k: int) -> dict[str, Any]:
    """Ties among the top-K scores and the score margin at the cutoff (C9)."""
    order = ranking(scores, ids)
    top = scores[order[:k]]
    _, counts = np.unique(scores, return_counts=True)
    repeated = set(np.unique(scores)[counts > 1].tolist())
    margin = float(scores[order[k - 1]] - scores[order[k]]) if len(scores) > k else None
    return {
        "k": k,
        "top_k_rows_with_tied_scores": int(sum(value in repeated for value in top.tolist())),
        "cutoff_margin": margin,
    }


def bitwise_equal(first: npt.NDArray[Any], second: npt.NDArray[Any]) -> bool:
    """T2 and T3: the same shape, dtype and bytes (so -0.0 and 0.0, or two NaN payloads, differ)."""
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes()
    )


def files_identical(first: Path, second: Path, chunk: int = 1 << 20) -> bool:
    """Whether two files hold the same bytes, read in chunks (C4, T4)."""
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            block = left.read(chunk)
            if block != right.read(chunk):
                return False
            if not block:
                return True


def compare_scores(first: Scores, second: Scores, ids: Ids, k: int) -> dict[str, Any]:
    """Bitwise equality, score differences and top-K membership changes of two score sets."""
    if first.shape != second.shape:
        raise ValueError("Score sets of different shapes cannot be compared")
    difference = np.abs(first - second)
    top_first = set(ids[ranking(first, ids)[:k]].tolist())
    top_second = set(ids[ranking(second, ids)[:k]].tolist())
    return {
        "bitwise_identical": bitwise_equal(first, second),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "top_k_membership_changes": len(top_first - top_second),
        "k": k,
    }


def logit_comparison(
    required: npt.NDArray[np.float32], reference: npt.NDArray[np.float32], tolerance: float
) -> dict[str, Any]:
    """T1: the float32 path's logits against the float32 reference (C8)."""
    difference = np.abs(required.astype(np.float64) - reference.astype(np.float64))
    scale = np.maximum(np.abs(reference.astype(np.float64)), 1.0)
    sigmoid = 1.0 / (1.0 + np.exp(-required.astype(np.float64)))
    sigmoid_reference = 1.0 / (1.0 + np.exp(-reference.astype(np.float64)))
    maximum = float(difference.max())
    return {
        "tolerance": tolerance,
        "max_abs_logit_difference": maximum,
        "mean_abs_logit_difference": float(difference.mean()),
        "max_scaled_logit_difference": float((difference / scale).max()),
        "max_abs_score_difference": float(np.abs(sigmoid - sigmoid_reference).max()),
        "max_abs_logit": float(np.abs(reference).max()),
        "passed": maximum <= tolerance,
    }


def latency_summary(seconds: list[float]) -> dict[str, float]:
    """Median, 95th and 99th percentiles of single-row latencies, in milliseconds."""
    values = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "median_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def timed_calls(call: Callable[[], object], synchronise: Callable[[], None]) -> dict[str, Any]:
    """``WARM_UP_CALLS`` untimed calls, then ``TIMED_REPETITIONS`` timed calls, synchronised at
    every timing boundary."""
    for _ in range(WARM_UP_CALLS):
        call()
    synchronise()
    durations = []
    for _ in range(TIMED_REPETITIONS):
        start = time.perf_counter()
        call()
        synchronise()
        durations.append(time.perf_counter() - start)
    return {
        "warm_up_calls": WARM_UP_CALLS,
        "repetitions": TIMED_REPETITIONS,
        "seconds": durations,
        "median_seconds": float(np.median(durations)),
    }


def host_peak_bytes() -> int:
    """Peak resident memory of this process since it started (``ru_maxrss``, kibibytes on
    Linux)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


NVIDIA_SMI = (
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader,nounits",
    "--id=0",
    f"--loop-ms={SAMPLE_INTERVAL_MS}",
)


class GpuMemorySampler:
    """Whole-device GPU memory sampled by ``nvidia-smi`` every ``SAMPLE_INTERVAL_MS`` milliseconds
    in a background thread.

    ``start`` waits for a first valid sample and stops the run if none arrives (no monitoring
    means no acceptance evidence). Output lines that are not two whole numbers (for example
    ``[N/A]`` under WSL), an early end of ``nvidia-smi``, a failure of the reading thread and a
    stall (no valid sample for more than ``stall_timeout`` seconds between samples or before
    ``stop``) are recorded as problems; ``snapshot`` gives the sampled maximum with those problems.
    """

    def __init__(
        self,
        command: Sequence[str] = NVIDIA_SMI,
        first_sample_timeout: float = 15.0,
        stall_timeout: float = 15.0,
    ) -> None:
        self.command = tuple(command)
        self.first_sample_timeout = first_sample_timeout
        self.stall_timeout = stall_timeout
        self._process: subprocess.Popen[str] | None = None
        self._errors: IO[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.samples = 0
        self.invalid_lines = 0
        self.first_used_mib: int | None = None
        self.max_used_mib = 0
        self.total_mib: int | None = None
        self.problems: list[str] = []
        self._last_sample: float | None = None  # time.monotonic() of the latest valid sample
        self.max_gap_seconds = 0.0
        self.final_sample_age_seconds: float | None = None

    def _read(self, stream: Iterator[str]) -> None:
        try:
            for line in stream:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 2 or not all(part.isdigit() for part in parts):
                    with self._lock:
                        self.invalid_lines += 1
                    continue
                used, total = int(parts[0]), int(parts[1])
                now = time.monotonic()
                with self._lock:
                    if self._last_sample is not None:
                        self.max_gap_seconds = max(self.max_gap_seconds, now - self._last_sample)
                    self._last_sample = now
                    self.samples += 1
                    if self.first_used_mib is None:
                        self.first_used_mib = used
                    self.max_used_mib = max(self.max_used_mib, used)
                    self.total_mib = total
        except Exception as error:  # recorded: missing evidence must block acceptance
            logger.exception("GPU memory sampling failed")
            with self._lock:
                self.problems.append(f"reader failed: {error!r}")

    def _error_text(self) -> str:
        if self._errors is None:
            return ""
        self._errors.seek(0)
        return self._errors.read().strip()

    def start(self) -> None:
        """Start sampling; stop the run if no valid sample arrives in time."""
        # Held for the sampler's lifetime and closed in stop(), so not a with-block.
        self._errors = tempfile.TemporaryFile("w+", encoding="utf-8")  # noqa: SIM115
        try:
            self._process = subprocess.Popen(
                list(self.command), stdout=subprocess.PIPE, stderr=self._errors, text=True
            )
        except OSError as error:
            raise EnvironmentMismatchError(f"nvidia-smi cannot be started: {error}") from error
        if self._process.stdout is None:
            raise EnvironmentMismatchError("nvidia-smi gave no output stream")
        self._thread = threading.Thread(
            target=self._read, args=(self._process.stdout,), name="gpu-memory", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + self.first_sample_timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.samples:
                    return
            if self._process.poll() is not None and not self._thread.is_alive():
                break
            time.sleep(0.05)
        message = self._error_text() or "no message"
        self.stop()
        raise EnvironmentMismatchError(
            "GPU memory monitoring gave no valid sample "
            f"({self.invalid_lines} unusable lines; nvidia-smi: {message})"
        )

    def stop(self) -> None:
        """Stop sampling; an earlier end of ``nvidia-smi`` or a stall is recorded as a problem."""
        with self._lock:
            if self._last_sample is not None:
                self.final_sample_age_seconds = time.monotonic() - self._last_sample
        if self._process is not None:
            code = self._process.poll()
            if code is not None:
                with self._lock:
                    self.problems.append(f"nvidia-smi ended early with exit code {code}")
            else:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=10)
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                with self._lock:
                    self.problems.append("the reading thread did not stop")
        message = self._error_text()
        if self._errors is not None:
            self._errors.close()
            self._errors = None
        if message:
            with self._lock:
                self.problems.append(f"nvidia-smi error output: {message}")
        with self._lock:
            if self.invalid_lines:
                self.problems.append(f"{self.invalid_lines} unusable output lines")
            longest = max(self.max_gap_seconds, self.final_sample_age_seconds or 0.0)
            if longest > self.stall_timeout:
                self.problems.append(
                    f"sampling stalled: no valid sample for {longest:.1f} s "
                    f"(limit {self.stall_timeout:.1f} s)"
                )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sampled_max_used_mib": self.max_used_mib,
                "baseline_used_mib": self.first_used_mib,
                "total_mib": self.total_mib,
                "samples": self.samples,
                "interval_ms": SAMPLE_INTERVAL_MS,
                "max_gap_seconds": self.max_gap_seconds,
                "final_sample_age_seconds": self.final_sample_age_seconds,
                "units": "MiB",
                "kind": "sampled maximum, not the true peak",
                "problems": list(self.problems),
            }


@contextmanager
def sampled_gpu_memory(enabled: bool) -> Iterator[GpuMemorySampler | None]:
    """A running ``GpuMemorySampler`` when ``enabled``, stopped on exit (``snapshot`` stays
    available after the block)."""
    if not enabled:
        yield None
        return
    sampler = GpuMemorySampler()
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()
