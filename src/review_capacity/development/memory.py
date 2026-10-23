"""Memory measurement of the phases of a development run (DR-015 §9, Issue #143).

``run_measured`` runs one phase (a trial's fit, a scoring boundary) and returns its result with
what was measured: the process's peak resident memory (the high-water mark reset at the start of
the phase where the kernel supports that, otherwise a sampler of the resident size) and, for a
phase on the GPU, the sampled whole-device maximum. A reading is absolute: memory that stays
resident from before the phase (inputs, retained models, cached allocations) is part of it.

A component that cannot be measured (a reset the kernel refuses, a sampler that does not start,
stalls or does not stop) is absent from the reading and named in its ``problems``; the phase's
work is never stopped for it, and a value from a failed sampler is never returned as a peak. If
the work itself fails, the monitors are stopped and the work's error is raised.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from review_capacity.errors import DataValidationError, EnvironmentMismatchError, FileAccessError
from review_capacity.probe.diagnostics import GpuMemorySampler
from review_capacity.training.timing import (
    HostInterface,
    HostPeak,
    MemoryReading,
    allocator_peaks,
    host_memory_phase,
    host_reset_supported,
    reset_allocator_peaks,
)

logger = logging.getLogger(__name__)

# The expected failures of a monitor: the process status cannot be read or reset, it lacks a field,
# or a sampler does not start, fails or does not stop.
MONITOR_ERRORS = (FileAccessError, DataValidationError, EnvironmentMismatchError)
# What stopping the GPU sampler's ``nvidia-smi`` process can raise besides the above.
STOP_ERRORS = (OSError, subprocess.TimeoutExpired, EnvironmentMismatchError)


@dataclass(frozen=True)
class MemorySetup:
    """How a run measures memory: the host method, the sampling interval, whether the GPU is
    monitored (only phases on the GPU use it) and the injectable interfaces."""

    host_method: str
    host_sample_interval_ms: int
    gpu_monitoring: bool
    interface: HostInterface = field(default_factory=HostInterface)
    gpu_sampler: Callable[[], GpuMemorySampler] = GpuMemorySampler

    @classmethod
    def detect(cls, host_sample_interval_ms: int, *, gpu_monitoring: bool) -> MemorySetup:
        """Choose the host method for this machine: the high-water-mark reset where a synthetic
        allocation shows that it works, otherwise the resident-size sampler."""
        method = "clear_refs" if host_reset_supported(HostInterface()) else "sampled"
        logger.info("Host memory method: %s", method)
        return cls(method, host_sample_interval_ms, gpu_monitoring)


@dataclass
class PhaseMemory:
    """What one phase measured, with the components that could not be measured."""

    scope: str
    host: HostPeak | None = None
    gpu: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)
    allocator: dict[str, int] | None = None

    def reading(self) -> MemoryReading:
        return MemoryReading(
            self.scope,
            gpu=self.gpu,
            allocator=self.allocator,
            host=self.host,
            problems=tuple(self.problems),
        )


def run_measured[T](
    setup: MemorySetup | None,
    scope: str,
    work: Callable[[], T],
    *,
    gpu: bool = False,
    allocator: bool = False,
) -> tuple[T, PhaseMemory | None]:
    """Run ``work`` and measure its memory; ``scope`` states what the phase covers. Without a
    ``setup`` nothing is measured. ``gpu`` marks a phase on the GPU; ``allocator`` also records
    PyTorch's allocator peaks of the phase (its counters are reset at the start; a neural phase).

    Monitors are acquired inside the protected block and every one that was acquired is released
    whatever happens: a failing release is a recorded problem (a warning when the work itself
    failed) and never replaces the work's own exception."""
    if setup is None:
        return work(), None
    phase = PhaseMemory(scope)
    sampler: GpuMemorySampler | None = None
    host: tuple[AbstractContextManager[Any], Any] | None = None
    try:
        sampler = _start_gpu(setup, phase) if gpu and setup.gpu_monitoring else None
        host = _start_host(setup, phase)
        if allocator:
            _reset_allocator()
        result = work()
        if allocator:
            phase.allocator = _read_allocator()
    except BaseException:
        _release(host, sampler, phase, keep=False)
        raise
    _release(host, sampler, phase, keep=True)
    return result, phase


def _reset_allocator() -> None:
    import torch

    reset_allocator_peaks(torch, torch.device("cuda", 0))


def _read_allocator() -> dict[str, int] | None:
    """The allocator's peaks since the reset, read after the phase's GPU work has finished."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return allocator_peaks(torch, torch.device("cuda", 0))


def _release(
    host: tuple[AbstractContextManager[Any], Any] | None,
    sampler: GpuMemorySampler | None,
    phase: PhaseMemory,
    *,
    keep: bool,
) -> None:
    """Release the host and the GPU monitor, each whatever happens to the other."""
    try:
        _finish_host(host, phase, keep=keep)
    finally:
        _stop_gpu(sampler, phase, keep=keep)


def _start_gpu(setup: MemorySetup, phase: PhaseMemory) -> GpuMemorySampler | None:
    sampler = setup.gpu_sampler()
    try:
        sampler.start()
    except EnvironmentMismatchError as error:
        phase.problems.append(f"GPU memory not measured: {error}")
        return None
    return sampler


def _stop_gpu(sampler: GpuMemorySampler | None, phase: PhaseMemory, *, keep: bool) -> None:
    if sampler is None:
        return
    try:
        sampler.stop()
        snapshot = sampler.snapshot()
    except STOP_ERRORS as error:
        if keep:
            phase.problems.append(f"GPU memory not measured: the sampler did not stop ({error})")
        else:
            logger.warning("The GPU memory sampler did not stop cleanly: %s", error)
        return
    if snapshot["problems"]:
        if keep:
            phase.problems.append("GPU memory not measured: " + "; ".join(snapshot["problems"]))
        return
    if keep:
        phase.gpu = snapshot


def _start_host(
    setup: MemorySetup, phase: PhaseMemory
) -> tuple[AbstractContextManager[Any], Any] | None:
    manager = host_memory_phase(setup.interface, setup.host_method, setup.host_sample_interval_ms)
    try:
        return manager, manager.__enter__()
    except MONITOR_ERRORS as error:
        phase.problems.append(f"host memory not measured: {error}")
        return None


def _finish_host(
    started: tuple[AbstractContextManager[Any], Any] | None, phase: PhaseMemory, *, keep: bool
) -> None:
    if started is None:
        return
    manager, state = started
    try:
        manager.__exit__(None, None, None)
    except MONITOR_ERRORS as error:
        if keep:
            phase.problems.append(f"host memory not measured: {error}")
        else:
            logger.warning("Host memory monitoring did not stop cleanly: %s", error)
        return
    if keep:
        phase.host = state.peak


def trials_maximum(trials: Sequence[tuple[int, PhaseMemory | None]], scope: str) -> PhaseMemory:
    """The maximum over the fit phases of the trials, as the search record's memory: the trial
    with the highest host peak and the one with the highest sampled GPU maximum, each with its own
    readings, and every problem named with its trial. It is not a whole-search peak, and a
    component is reported only when every trial has a valid reading of it: otherwise it is absent
    and the trials without one are named, because their unknown peak could be the highest."""
    result = PhaseMemory(scope)
    for number, phase in trials:
        if phase is not None:
            result.problems += [f"trial {number}: {problem}" for problem in phase.problems]
    hosts = {n: p.host for n, p in trials if p is not None and p.host is not None}
    gpus = {n: p.gpu for n, p in trials if p is not None and p.gpu is not None}
    allocators = {n: p.allocator for n, p in trials if p is not None and p.allocator is not None}
    if trials and len(hosts) == len(trials):
        result.host = max(hosts.values(), key=lambda peak: peak.peak_mib)
    elif hosts:
        missing = [n for n, _ in trials if n not in hosts]
        result.problems.append(f"host maximum not established: no reading for trials {missing}")
    if trials and allocators and len(allocators) == len(trials):
        result.allocator = {
            name: max(peaks[name] for peaks in allocators.values())
            for name in ("max_allocated_bytes", "max_reserved_bytes")
        }
    elif allocators:
        missing = [n for n, _ in trials if n not in allocators]
        result.problems.append(
            f"allocator maximum not established: no reading for trials {missing}"
        )
    if trials and len(gpus) == len(trials):
        result.gpu = max(gpus.values(), key=lambda snap: snap["sampled_max_used_mib"])
    elif gpus:
        missing = [n for n, _ in trials if n not in gpus]
        result.problems.append(f"GPU maximum not established: no reading for trials {missing}")
    return result
