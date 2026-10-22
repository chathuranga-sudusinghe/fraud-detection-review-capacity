"""Fake memory interfaces for the tests of the memory measurement of a run (Issue #143)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import count
from typing import Any, ClassVar

from review_capacity.development.memory import MemorySetup
from review_capacity.errors import EnvironmentMismatchError
from review_capacity.training.timing import HostInterface

FAKE_KIB = 512 * 1024


def _nothing() -> None:
    return None


def host_interface(
    peaks_kib: Iterable[int] | None = None,
    *,
    on_reset: Callable[[], None] = _nothing,
    on_read: Callable[[], None] = _nothing,
) -> HostInterface:
    """A host whose high-water mark is the next value of ``peaks_kib`` at every status read (one
    read per phase with the ``clear_refs`` method); a constant 512 MiB by default. ``on_reset``
    and ``on_read`` run at every reset and read, for tests that move a clock there."""
    values = iter(peaks_kib) if peaks_kib is not None else (FAKE_KIB for _ in count())

    def status() -> str:
        on_read()
        value = next(values)
        return f"VmHWM:\t{value} kB\nVmRSS:\t{value // 2} kB\n"

    return HostInterface(read_status=status, reset_peak=on_reset)


class FakeGpuSampler:
    """A GPU sampler with a scripted outcome: a maximum, or problems, or a start failure."""

    created: ClassVar[list[FakeGpuSampler]] = []

    def __init__(
        self,
        *,
        maximum: int = 4000,
        problems: tuple[str, ...] = (),
        fail_start: bool = False,
        fail_stop: bool = False,
    ) -> None:
        self.maximum, self.problems, self.fail_start = maximum, list(problems), fail_start
        self.fail_stop = fail_stop
        self.started = self.stopped = False
        FakeGpuSampler.created.append(self)

    def start(self) -> None:
        if self.fail_start:
            raise EnvironmentMismatchError("nvidia-smi cannot be started")
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self.fail_stop:
            raise OSError("the process could not be terminated")

    def snapshot(self) -> dict[str, Any]:
        return {
            "sampled_max_used_mib": self.maximum,
            "baseline_used_mib": 1000,
            "total_mib": 8000,
            "samples": 5,
            "interval_ms": 100,
            "max_gap_seconds": 0.2,
            "final_sample_age_seconds": 0.1,
            "units": "MiB",
            "kind": "sampled maximum, not the true peak",
            "problems": list(self.problems),
        }


def memory_setup(
    peaks_kib: Iterable[int] | None = None,
    *,
    gpu_monitoring: bool = False,
    gpu_sampler: Callable[[], Any] | None = None,
    on_reset: Callable[[], None] = _nothing,
    on_read: Callable[[], None] = _nothing,
) -> MemorySetup:
    """A memory setup on the ``clear_refs`` method with fake interfaces."""
    return MemorySetup(
        "clear_refs",
        100,
        gpu_monitoring,
        host_interface(peaks_kib, on_reset=on_reset, on_read=on_read),
        gpu_sampler or FakeGpuSampler,  # type: ignore[arg-type]
    )
