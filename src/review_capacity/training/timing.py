"""The computational-cost measurements of the development runs (DR-015 §9; Issue #121).

This module measures; it reads no data and trains nothing. The search runner passes it the
callables to time and the facts of the run, and receives measurements and ``timing`` records
(``development.records``, version 2):

* **Timed calls.** Every timed call follows ``synchronise, start the clock, work, synchronise,
  stop the clock``; the synchronisation is that of the device (``torch.cuda.synchronize`` for the
  GPU, nothing for the CPU), passed in as ``Clock.synchronise`` so that the order of events is
  testable.
* **Scoring.** Untimed warm-up calls, then the timed full passes; the median is the pass time.
* **Latency.** Batch-of-one calls over a fixed, label-free, deterministic sample of rows: untimed
  warm-up calls, then rounds through the same rows; the percentiles are those of
  ``numpy.percentile`` with the ``linear`` method, per round and over all timed calls.
* **Memory.** GPU: the sampled whole-device maximum, its baseline and the neural allocator's peak
  counters; host: the process high-water mark (``VmHWM``) reset through
  ``/proc/self/clear_refs`` where the kernel supports it (``host_reset_supported`` checks that with
  a synthetic allocation), otherwise a sampler of the resident size. A method is always named.
* **Session.** The conditions of the session, recorded as context: the load average, the
  affinity, the kernel and memory limits, and GPU telemetry where the system exposes it.

The values (passes, warm-up counts, rounds, the sample size) come from the ``timing`` block of
``config/training.yaml`` (``training.declaration.TimingDeclaration``); nothing here fixes them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import numpy.typing as npt

from review_capacity.development.records import RECORD_CONTRACTS, TIMING_PROCEDURE_VERSION
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    SpecificationError,
)
from review_capacity.evaluation.queue import id_digest
from review_capacity.logging_setup import ProgressLogger, log_step
from review_capacity.models.base import Keys

logger = logging.getLogger(__name__)

PROCEDURE_VERSION = TIMING_PROCEDURE_VERSION
PERCENTILE_METHOD = "linear"
BOUNDARIES = ("model_only", "end_to_end")
ROW_RULE = "smallest SHA-256 digests of the decimal TransactionID (UTF-8), ties by identifier"
GPU_MEMORY_SCOPE = "sampled whole-device GPU memory maximum (not a true peak)"
HOST_MEMORY_SCOPE = "peak resident memory of the parent process during the phase"
PROC = Path("/proc/self")
SAMPLER_JOIN_SECONDS = 10.0
# Settings the builders own: a caller's ``Timed.extra`` may not set or override them.
RESERVED_SETTINGS = frozenset(
    {
        "procedure_version",
        "session",
        "memory_problems",
        "gpu_memory",
        "allocator_peaks",
        "host_memory_method",
        "host_sample_interval_ms",
    }
)


@dataclass(frozen=True)
class Clock:
    """The clock and the device synchronisation of a measurement (injectable for tests)."""

    now: Callable[[], float] = time.perf_counter
    synchronise: Callable[[], None] = lambda: None


def timed(call: Callable[[], object], clock: Clock) -> float:
    """Seconds of one ``call``: synchronise, start, work, synchronise, stop."""
    clock.synchronise()
    start = clock.now()
    call()
    clock.synchronise()
    return clock.now() - start


def gpu_clock(clock: Clock) -> Clock:
    """``clock`` with the explicit synchronisation of CUDA device 0 before the clock starts and
    after the work completes (DR-015 §9); a process without CUDA has nothing to synchronise."""
    import torch

    def synchronise() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device("cuda", 0))

    return Clock(clock.now, synchronise)


def settle(seconds: int, sleep: Callable[[float], None] = time.sleep) -> None:
    """The settling interval before a standalone scoring block (DR-015 §9)."""
    if seconds > 0:
        with log_step(logger, f"Settling for {seconds} s"):
            sleep(float(seconds))


# --- scoring -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringMeasurement:
    """Timed full passes of one partition and boundary."""

    warmup: int
    pass_seconds: tuple[float, ...]

    @property
    def seconds(self) -> float:
        """The pass time: the median of the passes."""
        return float(median(self.pass_seconds))


def measure_scoring(
    timed_pass: Callable[[], object],
    warmup_calls: Sequence[Callable[[], object]],
    passes: int,
    clock: Clock,
) -> ScoringMeasurement:
    """Run the untimed ``warmup_calls``, then ``passes`` timed passes of ``timed_pass``; every
    pass is kept, slow ones too."""
    if passes < 1:
        raise SpecificationError("A scoring measurement needs at least one pass")
    with log_step(logger, f"Scoring: {len(warmup_calls)} warm-up calls, {passes} timed passes"):
        for call in warmup_calls:
            call()
        clock.synchronise()
        durations = tuple(timed(timed_pass, clock) for _ in range(passes))
    return ScoringMeasurement(len(warmup_calls), durations)


# --- latency -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Latency:
    """Median, 95th and 99th percentile of single-transaction latencies, in milliseconds."""

    median: float
    p95: float
    p99: float

    def as_record(self) -> dict[str, float]:
        return {"median": self.median, "p95": self.p95, "p99": self.p99}


def latency_percentiles(seconds: Sequence[float], method: str = PERCENTILE_METHOD) -> Latency:
    """Percentiles of ``seconds`` (as milliseconds) with the stated ``numpy.percentile`` method."""
    if method != PERCENTILE_METHOD:
        raise SpecificationError(f"The percentile method must be {PERCENTILE_METHOD!r}")
    if len(seconds) == 0:
        raise DataValidationError("No latency was measured")
    values = np.asarray(seconds, dtype=np.float64) * 1000.0
    p50, p95, p99 = (float(x) for x in np.percentile(values, [50, 95, 99], method="linear"))
    return Latency(p50, p95, p99)


@dataclass(frozen=True)
class LatencyMeasurement:
    """Batch-of-one latencies: ``rows`` distinct rows, ``warmup`` untimed calls and one entry per
    round; ``pooled`` covers every timed call."""

    rows: int
    warmup: int
    rounds: tuple[Latency, ...]
    pooled: Latency
    timed_calls: int


def measure_latency(
    call_one: Callable[[int], object],
    rows: int,
    warmup_calls: int,
    rounds: int,
    clock: Clock,
) -> LatencyMeasurement:
    """``warmup_calls`` untimed calls on the first rows, then ``rounds`` rounds in which every
    row ``0 .. rows - 1`` is scored once, in that fixed order, as a timed call."""
    if rows < 1 or rounds < 1 or warmup_calls < 0:
        raise SpecificationError("Latency needs rows and rounds of at least 1")
    everything: list[float] = []
    per_round: list[Latency] = []
    with log_step(logger, f"Latency: {warmup_calls} warm-up calls, {rounds} rounds of {rows:,}"):
        for i in range(warmup_calls):
            call_one(i % rows)
        clock.synchronise()
        for number in range(1, rounds + 1):
            progress = ProgressLogger(logger, f"Latency round {number}/{rounds}", rows)
            seconds: list[float] = []
            for row in range(rows):
                seconds.append(timed(partial(call_one, row), clock))
                progress.advance()
            per_round.append(latency_percentiles(seconds))
            everything += seconds
    return LatencyMeasurement(
        rows, warmup_calls, tuple(per_round), latency_percentiles(everything), len(everything)
    )


def select_latency_rows(ids: Keys, count: int) -> npt.NDArray[np.intp]:
    """Positions of the ``count`` rows with the smallest SHA-256 digests of the decimal
    ``TransactionID`` (UTF-8; identical digests by ascending identifier), in that order.

    Neither labels nor scores enter the selection, and the result does not depend on the order
    of ``ids``.
    """
    if ids.ndim != 1 or ids.dtype != np.int64 or len(np.unique(ids)) != len(ids):
        raise DataValidationError("TransactionIDs must be a one-dimensional unique int64 array")
    if not 1 <= count <= len(ids):
        raise DataValidationError(f"The latency sample of {count} rows needs {count} rows")
    order = sorted(range(len(ids)), key=lambda row: (id_digest(int(ids[row])), int(ids[row])))
    return np.asarray(order[:count], dtype=np.intp)


def rows_identity(ids: Keys) -> str:
    """SHA-256 of the ordered identifiers as little-endian int64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i8").tobytes()).hexdigest()


# --- memory --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HostPeak:
    """Peak resident memory of a phase, the method that measured it and, for the sampled
    method, its interval."""

    peak_mib: float
    method: str
    sample_interval_ms: int | None = None


def _read_status() -> str:
    return (PROC / "status").read_text(encoding="utf-8")


def _reset_peak() -> None:
    (PROC / "clear_refs").write_text("5", encoding="ascii")


@dataclass(frozen=True)
class HostInterface:
    """Reads of the process's status and the reset of its high-water mark, injectable for tests
    (the default is the Linux ``/proc/self``)."""

    read_status: Callable[[], str] = _read_status
    reset_peak: Callable[[], None] = _reset_peak


def status_kib(status: str, name: str) -> int:
    """A ``kB`` field (``VmRSS``, ``VmHWM``) of ``/proc/<pid>/status``, in kibibytes."""
    for line in status.splitlines():
        if line.startswith(f"{name}:"):
            return int(line.split()[1])
    raise DataValidationError(f"{name} is not in the process status")


def _kib(interface: HostInterface, name: str) -> int:
    """``status_kib`` of the interface's current status; a failed read is a project error."""
    try:
        text = interface.read_status()
    except OSError as error:
        raise FileAccessError(f"The process status cannot be read ({error})") from error
    return status_kib(text, name)


def _reset(interface: HostInterface) -> None:
    try:
        interface.reset_peak()
    except OSError as error:
        raise FileAccessError(f"The peak resident memory cannot be reset ({error})") from error


def synthetic_allocation(megabytes: int) -> object:
    """``megabytes`` of memory that is resident (every page written), to be released by the
    caller."""
    return np.ones(megabytes * 1024 * 1024, dtype=np.uint8)


def host_reset_supported(
    interface: HostInterface,
    allocate: Callable[[int], object] = synthetic_allocation,
    megabytes: int = 64,
) -> bool:
    """Whether writing ``5`` to ``clear_refs`` resets the high-water mark on this machine.

    The mark is reset first, so that an earlier, higher mark does not hide the test. A synthetic
    allocation of ``megabytes`` then raises it; after the allocation is released and the mark is
    reset again, the mark must fall by at least half of the allocation. A reset that the kernel
    refuses means "not supported".
    """
    try:
        _reset(interface)
        baseline = _kib(interface, "VmHWM")
        held = allocate(megabytes)
        raised = _kib(interface, "VmHWM")
        del held
        _reset(interface)
    except FileAccessError:
        return False
    reset = _kib(interface, "VmHWM")
    logger.info("Host peak reset check: %d, %d, %d KiB", baseline, raised, reset)
    return raised - baseline >= megabytes * 512 and raised - reset >= megabytes * 512


class _HostSampler:
    """The resident size of the process, read every interval by a background thread. A failed
    read or a thread that does not stop means the phase has no valid measurement: ``stop``
    raises, never returning a peak that a failed sampler produced."""

    def __init__(self, interface: HostInterface, interval_ms: int) -> None:
        self.interface, self.interval = interface, interval_ms / 1000.0
        self.peak_kib = _kib(interface, "VmRSS")
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="host-memory", daemon=True)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                self.peak_kib = max(self.peak_kib, _kib(self.interface, "VmRSS"))
        except Exception as error:  # kept: stop() raises it with the phase's result
            logger.exception("Host memory sampling failed")
            self.error = error

    def start(self) -> None:
        self._thread.start()

    def _join(self) -> bool:
        self._stop.set()
        self._thread.join(timeout=SAMPLER_JOIN_SECONDS)
        return not self._thread.is_alive()

    def abort(self) -> None:
        """End sampling when the measured block itself failed; its error is the one to report."""
        if not self._join():
            logger.warning("The host memory sampler did not stop")

    def stop(self) -> int:
        if not self._join():
            raise EnvironmentMismatchError("The host memory sampler did not stop in time")
        if self.error is not None:
            raise EnvironmentMismatchError("Host memory sampling failed") from self.error
        return max(self.peak_kib, _kib(self.interface, "VmRSS"))


@dataclass
class HostPhase:
    """The result of ``host_memory_phase``, available after the block."""

    peak: HostPeak | None = None


@contextmanager
def host_memory_phase(
    interface: HostInterface, method: str, sample_interval_ms: int = 100
) -> Iterator[HostPhase]:
    """Measure the peak resident memory of the block by ``method``: ``clear_refs`` (reset the
    high-water mark, read it after) or ``sampled`` (the resident size every interval)."""
    phase = HostPhase()
    if method == "clear_refs":
        _reset(interface)
        yield phase
        phase.peak = HostPeak(_kib(interface, "VmHWM") / 1024, method)
    elif method == "sampled":
        sampler = _HostSampler(interface, sample_interval_ms)
        sampler.start()
        try:
            yield phase
        except BaseException:
            sampler.abort()
            raise
        phase.peak = HostPeak(sampler.stop() / 1024, method, sample_interval_ms)
    else:
        raise SpecificationError(f"Unknown host memory method {method!r}")


def gpu_memory_fields(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The settings entry for a ``GpuMemorySampler.snapshot``: the absolute sampled maximum, the
    baseline and the increase above it, kept apart (the increase is not memory attributable to
    the model alone)."""
    if snapshot is None:
        return None
    maximum, baseline = snapshot["sampled_max_used_mib"], snapshot["baseline_used_mib"]
    return {
        "sampled_max_mib": maximum,
        "baseline_mib": baseline,
        "increase_above_baseline_mib": None if baseline is None else maximum - baseline,
        "samples": snapshot["samples"],
        "interval_ms": snapshot["interval_ms"],
        "max_gap_seconds": snapshot["max_gap_seconds"],
        "problems": list(snapshot["problems"]),
        "kind": snapshot["kind"],
    }


@dataclass(frozen=True)
class MemoryReading:
    """What a phase measured: GPU sampler fields, the neural allocator's peaks, host peak, and
    ``scope``, the caller's statement of what the phase covered (resident inputs, models,
    whether warm-up is inside the interval), so that different boundaries stay distinguishable;
    ``problems`` name each component that could not be measured (its value is then absent)."""

    scope: str
    gpu: Mapping[str, Any] | None = None
    allocator: Mapping[str, int] | None = None
    host: HostPeak | None = None
    problems: tuple[str, ...] = ()


def allocator_peaks(torch: Any, device: Any) -> dict[str, int] | None:
    """The neural allocator's maximum allocated and reserved bytes since the last reset, or None
    without CUDA."""
    if not torch.cuda.is_available():
        return None
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def reset_allocator_peaks(torch: Any, device: Any) -> None:
    """Start a phase of the neural allocator's counters (the reset does not release memory)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


# --- session -------------------------------------------------------------------------------------


def _meminfo_mib(field_name: str, text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith(f"{field_name}:"):
            return int(line.split()[1]) // 1024
    return None


def gpu_telemetry(run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Temperature, clocks and power of GPU 0 where ``nvidia-smi`` reports them (under WSL some
    fields are ``[N/A]``); an unusable reading is recorded as unavailable, never as a pass."""
    query = "temperature.gpu,clocks.sm,clocks.max.sm,power.draw"
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "--id=0"]
    try:
        done = run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": repr(error)}
    parts = [part.strip() for part in str(done.stdout).strip().split(",")]
    if done.returncode != 0 or len(parts) != 4:
        return {"available": False, "reason": f"exit {done.returncode}: {done.stdout!r}"}

    def number(text: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            return None

    values = [number(part) for part in parts]
    return {
        "available": any(v is not None for v in values),
        "temperature_c": values[0],
        "clock_sm_mhz": values[1],
        "clock_max_sm_mhz": values[2],
        "power_w": values[3],
    }


def session_conditions(
    *,
    loadavg: Callable[[], tuple[float, float, float]] = os.getloadavg,
    affinity: Callable[[int], set[int]] = os.sched_getaffinity,
    meminfo: Callable[[], str] = lambda: Path("/proc/meminfo").read_text(encoding="utf-8"),
    telemetry: Callable[[], dict[str, Any]] = gpu_telemetry,
) -> dict[str, Any]:
    """The conditions of a timing session (DR-015 §9), recorded as context with what the system
    reports; no condition is required. The load average is context only; WSL memory and swap are
    the limits the guest sees."""
    memory = meminfo()
    return {
        "load_average_1m": loadavg()[0],
        "visible_cpus": os.cpu_count(),
        "allowed_cpus": sorted(affinity(0)),
        "kernel": platform.release(),
        "memory_total_mib": _meminfo_mib("MemTotal", memory),
        "swap_total_mib": _meminfo_mib("SwapTotal", memory),
        "gpu": telemetry(),
    }


# --- records -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Timed:
    """What a ``timing`` record describes: the model, its role and where it ran."""

    configuration: str
    model_role: str | None
    device: str
    threads: int | None = None
    cpu_affinity: tuple[int, ...] | None = None
    attention_backend: str | None = None
    subject: Mapping[str, str] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = sorted(set(self.extra) & RESERVED_SETTINGS)
        if reserved:
            raise SpecificationError(f"Timed.extra may not set the builder-owned keys {reserved}")


def _record(
    identity: Mapping[str, Any],
    who: Timed,
    measurement: str,
    session: Mapping[str, Any],
    memory: MemoryReading | None,
    details: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    cpu = who.device == "cpu"
    gpu_fields = None if memory is None else gpu_memory_fields(memory.gpu)
    host = None if memory is None else memory.host
    if memory is not None and not memory.scope.strip():
        raise SpecificationError("A memory reading states its scope (inputs, models, warm-up)")
    scopes = [] if memory is None else [memory.scope]
    scopes += [s for s, used in ((GPU_MEMORY_SCOPE, gpu_fields), (HOST_MEMORY_SCOPE, host)) if used]
    record: dict[str, Any] = {
        "record": "timing",
        "schema_version": RECORD_CONTRACTS["timing"].version,
        "identity": dict(identity),
        "configuration": who.configuration,
        "measurement": measurement,
        "model_role": who.model_role,
        "subject": None if who.subject is None else dict(who.subject),
        "partition": None,
        "device": who.device,
        "boundary": None,
        "threads": who.threads if cpu else None,
        "cpu_affinity": None if not cpu or who.cpu_affinity is None else list(who.cpu_affinity),
        "synchronised": who.device == "gpu",
        "attention_backend": who.attention_backend,
        "row_selection": None,
        "rows_sha256": None,
        "batch_rows": None,
        "warmup": None,
        "repetitions": None,
        "rows": None,
        "seconds": None,
        "rows_per_second": None,
        "latency_ms": None,
        "peak_gpu_mib": None if gpu_fields is None else gpu_fields["sampled_max_mib"],
        "peak_host_mib": None if host is None else host.peak_mib,
        "memory_scope": "; ".join(scopes) if scopes else None,
    }
    record |= details
    clash = sorted(set(who.extra) & set(settings))
    if clash:
        raise SpecificationError(f"Timed.extra repeats the measurement's settings {clash}")
    full: dict[str, Any] = {
        **who.extra,
        **settings,
        # the authoritative entries come last: nothing above can change them
        "procedure_version": PROCEDURE_VERSION,
        "session": dict(session),
        "memory_problems": [] if memory is None else list(memory.problems),
    }
    if gpu_fields is not None:
        full["gpu_memory"] = gpu_fields
    if memory is not None and memory.allocator is not None:
        full["allocator_peaks"] = dict(memory.allocator)
    if host is not None:
        full["host_memory_method"] = host.method
        if host.sample_interval_ms is not None:
            full["host_sample_interval_ms"] = host.sample_interval_ms
    record["settings"] = full
    return record


def fit_record(
    identity: Mapping[str, Any],
    who: Timed,
    session: Mapping[str, Any],
    *,
    seconds: float,
    completed_length: int,
    retained_length: int,
    startup_seconds: float | None,
    preprocessing_seconds: float | None,
    persistence_seconds: float | None = None,
    startup_scope: str | None = None,
    startup_status: str | None = None,
    memory: MemoryReading | None = None,
) -> dict[str, Any]:
    """The selected trial's own elapsed fit time with its lengths (DR-015 §9, Fit).

    ``startup_seconds`` is the shared start-up of the library (once per process, with its
    ``startup_scope``); ``preprocessing_seconds`` is the shared preprocessing artifact's fit
    (none for the tree-based models); ``persistence_seconds`` is the saving of the model, its
    manifest and its model-run record. None of them is inside ``seconds``. ``startup_status`` is
    ``measured``, or ``not_measured`` when the start-up could not be measured, in which case the
    fit's time may include the library and device initialisation (null: no shared start-up)."""
    settings = {
        "completed_length": completed_length,
        "retained_length": retained_length,
        "startup_seconds": startup_seconds,
        "startup_scope": startup_scope,
        "startup_status": startup_status,
        "preprocessing_seconds": preprocessing_seconds,
        "persistence_seconds": persistence_seconds,
    }
    return _record(identity, who, "fit", session, memory, {"seconds": seconds}, settings)


def search_record(
    identity: Mapping[str, Any],
    who: Timed,
    session: Mapping[str, Any],
    *,
    seconds: float,
    trials_completed: int,
    trials_declared: int,
    interrupted: bool,
    memory: MemoryReading | None = None,
) -> dict[str, Any]:
    """The sequential trial loop of a configuration, with the declared trial count
    (``TrainingDeclaration.trials``); an interrupted loop says so and counts as partial."""
    settings = {
        "trials_completed": trials_completed,
        "trials_declared": trials_declared,
        "interrupted": interrupted,
    }
    return _record(identity, who, "search", session, memory, {"seconds": seconds}, settings)


def scoring_record(
    identity: Mapping[str, Any],
    who: Timed,
    session: Mapping[str, Any],
    *,
    partition: str,
    boundary: str,
    rows: int,
    batch_rows: int,
    measurement: ScoringMeasurement,
    memory: MemoryReading | None = None,
) -> dict[str, Any]:
    """Full-partition scoring: the median pass, every pass, and throughput = rows / median."""
    if boundary not in BOUNDARIES:
        raise SpecificationError(f"The boundary must be one of {BOUNDARIES}")
    seconds = measurement.seconds
    details = {
        "partition": partition,
        "boundary": boundary,
        "rows": rows,
        "batch_rows": batch_rows,
        "warmup": measurement.warmup,
        "repetitions": len(measurement.pass_seconds),
        "seconds": seconds,
        "rows_per_second": rows / seconds if seconds > 0 else 0.0,
    }
    settings = {
        "pass_seconds": list(measurement.pass_seconds),
        "pass_min_seconds": min(measurement.pass_seconds),
        "pass_max_seconds": max(measurement.pass_seconds),
    }
    return _record(identity, who, "scoring", session, memory, details, settings)


def latency_record(
    identity: Mapping[str, Any],
    who: Timed,
    session: Mapping[str, Any],
    *,
    boundary: str,
    sample_ids: Keys,
    measurement: LatencyMeasurement,
    memory: MemoryReading | None = None,
) -> dict[str, Any]:
    """Batch-of-one latency over the ordered sample ``sample_ids`` (DR-015 §9, Latency)."""
    if boundary not in BOUNDARIES:
        raise SpecificationError(f"The boundary must be one of {BOUNDARIES}")
    if len(sample_ids) != measurement.rows:
        raise SpecificationError("The sample identifiers and the measured rows differ")
    details = {
        "boundary": boundary,
        "row_selection": ROW_RULE,
        "rows_sha256": rows_identity(sample_ids),
        "batch_rows": 1,
        "warmup": measurement.warmup,
        "repetitions": len(measurement.rounds),
        "rows": measurement.rows,
        "latency_ms": measurement.pooled.as_record(),
    }
    settings = {
        "percentile_method": PERCENTILE_METHOD,
        "timed_calls": measurement.timed_calls,
        "round_latency_ms": [r.as_record() for r in measurement.rounds],
    }
    return _record(identity, who, "latency", session, memory, details, settings)
