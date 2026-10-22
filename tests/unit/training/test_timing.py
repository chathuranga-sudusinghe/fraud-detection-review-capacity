"""The computational-cost measurements: the order of events around every timed call, warm-up
excluded, percentiles, the latency sample, host and GPU memory methods, the session and the
``timing`` records (fake clocks, fake kernel and fake nvidia-smi; no data, no GPU)."""

from __future__ import annotations

import hashlib
import subprocess
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from review_capacity.development.records import validate_record
from review_capacity.errors import (
    DataValidationError,
    EnvironmentMismatchError,
    FileAccessError,
    RecordContractError,
    SpecificationError,
)
from review_capacity.training import timing
from review_capacity.training.timing import (
    Clock,
    HostInterface,
    HostPeak,
    MemoryReading,
    Timed,
)
from support.development import IDENTITY, PARTITIONS, file


class FakeClock:
    """A clock that logs its events; the work advances ``time`` itself."""

    def __init__(self) -> None:
        self.time = 0.0
        self.events: list[str] = []

    def now(self) -> float:
        self.events.append("now")
        return self.time

    def synchronise(self) -> None:
        self.events.append("sync")

    def clock(self) -> Clock:
        return Clock(self.now, self.synchronise)


def test_a_timed_call_synchronises_before_the_start_and_after_the_work() -> None:
    fake = FakeClock()

    def work() -> None:
        fake.events.append("work")
        fake.time += 2.5

    assert timing.timed(work, fake.clock()) == 2.5
    assert fake.events == ["sync", "now", "work", "sync", "now"]


def test_scoring_excludes_the_warm_up_and_keeps_every_pass() -> None:
    fake = FakeClock()
    durations = iter([3.0, 1.0, 2.0])
    calls: list[str] = []

    def warm(name: str) -> Callable[[], None]:
        def call() -> None:
            calls.append(name)
            fake.time += 100.0  # would show up in a pass if it were timed

        return call

    def one_pass() -> None:
        calls.append("pass")
        fake.time += next(durations)

    result = timing.measure_scoring(one_pass, [warm("full"), warm("short")], 3, fake.clock())
    assert calls == ["full", "short", "pass", "pass", "pass"]
    assert result.warmup == 2
    assert result.pass_seconds == (3.0, 1.0, 2.0)
    assert result.seconds == 2.0  # the median, not the mean or the best


def test_scoring_needs_a_pass() -> None:
    with pytest.raises(SpecificationError, match="at least one pass"):
        timing.measure_scoring(lambda: None, [], 0, Clock())


def test_percentiles_use_the_linear_method() -> None:
    latency = timing.latency_percentiles([i / 1000 for i in range(1, 101)])  # 1 .. 100 ms
    assert (latency.median, latency.p95, latency.p99) == pytest.approx((50.5, 95.05, 99.01))
    with pytest.raises(SpecificationError, match="linear"):
        timing.latency_percentiles([0.001], "lower")
    with pytest.raises(DataValidationError, match="No latency"):
        timing.latency_percentiles([])


def test_latency_times_every_row_once_per_round_after_untimed_warm_up() -> None:
    fake = FakeClock()
    scored: list[int] = []

    def call_one(row: int) -> None:
        scored.append(row)
        fake.time += (row + 1) / 1000  # row r takes r + 1 ms

    result = timing.measure_latency(call_one, 100, 5, 3, fake.clock())
    assert scored[:5] == [0, 1, 2, 3, 4]
    assert scored[5:] == list(range(100)) * 3  # fixed order, the same rows in every round
    assert (result.rows, result.warmup, result.timed_calls) == (100, 5, 300)
    assert len(result.rounds) == 3
    for latency in (*result.rounds, result.pooled):
        assert (latency.median, latency.p95, latency.p99) == pytest.approx((50.5, 95.05, 99.01))
    assert fake.events.count("sync") == 1 + 2 * 300  # after the warm-up, around every timed call


def test_each_latency_round_has_its_own_percentiles_and_the_pool_covers_all() -> None:
    fake = FakeClock()
    factors = (1.0, 2.0, 3.0)
    calls = {"n": 0}

    def call_one(row: int) -> None:
        factor = factors[calls["n"] // 100]  # round 1, 2, 3 are 1x, 2x, 3x as slow
        calls["n"] += 1
        fake.time += factor * (row + 1) / 1000

    result = timing.measure_latency(call_one, 100, 0, 3, fake.clock())
    for latency, factor in zip(result.rounds, factors, strict=True):
        expected = (50.5 * factor, 95.05 * factor, 99.01 * factor)
        assert (latency.median, latency.p95, latency.p99) == pytest.approx(expected)
    every = np.concatenate([np.arange(1, 101) * factor for factor in factors])
    pooled = np.percentile(every, [50, 95, 99], method="linear")
    assert (result.pooled.median, result.pooled.p95, result.pooled.p99) == pytest.approx(
        tuple(pooled)
    )
    assert result.pooled.median != result.rounds[0].median


def test_latency_needs_rows_and_rounds() -> None:
    with pytest.raises(SpecificationError, match="at least 1"):
        timing.measure_latency(lambda _row: None, 0, 0, 1, Clock())
    with pytest.raises(SpecificationError, match="at least 1"):
        timing.measure_latency(lambda _row: None, 1, 0, 0, Clock())


def test_the_latency_sample_is_the_smallest_digests_whatever_the_order() -> None:
    ids = np.array([5, 17, 101, 3, 999, 42, 7, 1_000_001, 64, 12], dtype=np.int64)
    expected = sorted(ids.tolist(), key=lambda i: (hashlib.sha256(str(i).encode()).digest(), i))
    expected = expected[:4]
    positions = timing.select_latency_rows(ids, 4)
    assert ids[positions].tolist() == expected
    order = np.random.default_rng(1).permutation(len(ids))
    again = timing.select_latency_rows(ids[order], 4)
    assert ids[order][again].tolist() == expected
    chosen = ids[positions]
    digest = hashlib.sha256(chosen.astype("<i8").tobytes()).hexdigest()
    assert timing.rows_identity(chosen) == digest
    assert timing.rows_identity(chosen[::-1]) != timing.rows_identity(chosen)


def test_equal_digests_are_ordered_by_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing, "id_digest", lambda _identifier: b"the same digest")
    ids = np.array([30, 10, 20], dtype=np.int64)
    assert ids[timing.select_latency_rows(ids, 2)].tolist() == [10, 20]


def test_the_latency_sample_is_checked() -> None:
    ids = np.arange(1, 6, dtype=np.int64)
    with pytest.raises(DataValidationError, match="needs 6 rows"):
        timing.select_latency_rows(ids, 6)
    with pytest.raises(DataValidationError, match="needs 0 rows"):
        timing.select_latency_rows(ids, 0)
    with pytest.raises(DataValidationError, match="unique int64"):
        timing.select_latency_rows(np.array([1, 1, 2], dtype=np.int64), 1)
    with pytest.raises(DataValidationError, match="unique int64"):
        timing.select_latency_rows(np.array([1, 2], dtype=np.int32), 1)


def test_settling_sleeps_only_when_it_has_an_interval() -> None:
    slept: list[float] = []
    timing.settle(60, slept.append)
    timing.settle(0, slept.append)
    assert slept == [60.0]


# --- host memory ---------------------------------------------------------------------------------


class FakeKernel:
    """A process whose resident size and high-water mark behave like Linux's."""

    def __init__(self, resets: bool = True, raises: bool = False) -> None:
        self.rss = self.hwm = 10_000  # KiB
        self.resets, self.raises = resets, raises
        self.reads = 0  # status reads, also by a sampler thread
        self.target: int | None = None
        self.seen = threading.Event()  # set once ``reads`` reaches ``target``

    def status(self) -> str:
        self.reads += 1
        if self.target is not None and self.reads >= self.target:
            self.seen.set()
        return f"Name:\tpython\nVmHWM:\t{self.hwm} kB\nVmRSS:\t{self.rss} kB\n"

    def reset(self) -> None:
        if self.raises:
            raise PermissionError("clear_refs")
        if self.resets:
            self.hwm = self.rss  # 5 resets to the current resident size, not to zero

    def grow(self, megabytes: int) -> None:
        self.rss += megabytes * 1024
        self.hwm = max(self.hwm, self.rss)

    def shrink(self, megabytes: int) -> None:
        self.rss -= megabytes * 1024

    def interface(self) -> HostInterface:
        return HostInterface(self.status, self.reset)

    def allocate(self, megabytes: int) -> object:
        """Memory held until the returned object is released (as a real allocation is)."""
        return Held(self, megabytes)


class Held:
    def __init__(self, kernel: FakeKernel, megabytes: int) -> None:
        self.kernel, self.megabytes = kernel, megabytes
        kernel.grow(megabytes)

    def __del__(self) -> None:
        self.kernel.shrink(self.megabytes)


def test_a_kernel_that_resets_the_high_water_mark_is_detected() -> None:
    kernel = FakeKernel()
    assert timing.host_reset_supported(kernel.interface(), kernel.allocate, 64)
    assert kernel.rss == 10_000  # the allocation was released


@pytest.mark.parametrize("kernel", [FakeKernel(resets=False), FakeKernel(raises=True)])
def test_a_kernel_without_the_reset_is_detected(kernel: FakeKernel) -> None:
    assert not timing.host_reset_supported(kernel.interface(), kernel.allocate, 64)


def test_the_reset_phase_reports_the_high_water_mark_since_the_reset() -> None:
    kernel = FakeKernel()
    kernel.grow(500)  # an earlier phase's peak must not count
    kernel.shrink(500)
    with timing.host_memory_phase(kernel.interface(), "clear_refs") as phase:
        kernel.grow(200)
        kernel.shrink(150)
    assert phase.peak == HostPeak((10_000 + 200 * 1024) / 1024, "clear_refs")


def test_the_sampled_phase_reports_the_largest_resident_size() -> None:
    kernel = FakeKernel()
    with timing.host_memory_phase(kernel.interface(), "sampled", sample_interval_ms=1) as phase:
        kernel.grow(300)
        kernel.target = kernel.reads + 2  # two reads after the growth: the sampler has seen it
        assert kernel.seen.wait(5)
        kernel.shrink(300)
    assert phase.peak is not None
    assert (phase.peak.method, phase.peak.sample_interval_ms) == ("sampled", 1)
    assert phase.peak.peak_mib == pytest.approx((10_000 + 300 * 1024) / 1024)


def failing_after(reads: int, kernel: FakeKernel, failed: threading.Event) -> HostInterface:
    """A status that fails (and says so) from its ``reads``-th read on."""
    count = {"n": 0}

    def status() -> str:
        count["n"] += 1
        if count["n"] >= reads:
            failed.set()
            raise PermissionError("denied")
        return kernel.status()

    return HostInterface(status, kernel.reset)


def test_a_failing_sampler_invalidates_the_phase() -> None:
    failed = threading.Event()
    interface = failing_after(3, FakeKernel(), failed)
    with (
        pytest.raises(EnvironmentMismatchError, match="sampling failed"),
        timing.host_memory_phase(interface, "sampled", 1),
    ):
        assert failed.wait(5)


def test_a_failed_block_keeps_its_own_error_over_a_sampler_failure() -> None:
    failed = threading.Event()
    interface = failing_after(3, FakeKernel(), failed)

    def block() -> None:
        with timing.host_memory_phase(interface, "sampled", 1):
            assert failed.wait(5)
            raise ValueError("the block")

    with pytest.raises(ValueError, match="the block"):
        block()


def test_a_sampler_that_does_not_stop_invalidates_the_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = FakeKernel()
    entered, release = threading.Event(), threading.Event()
    count = {"n": 0}

    def status() -> str:
        count["n"] += 1
        if count["n"] >= 2:
            entered.set()
            release.wait(10)  # a read that blocks
        return kernel.status()

    monkeypatch.setattr(timing, "SAMPLER_JOIN_SECONDS", 0.05)
    with (
        pytest.raises(EnvironmentMismatchError, match="did not stop"),
        timing.host_memory_phase(HostInterface(status, kernel.reset), "sampled", 1),
    ):
        assert entered.wait(5)
    release.set()


def test_failed_status_reads_and_resets_are_project_errors() -> None:
    kernel = FakeKernel()

    def unreadable() -> str:
        raise PermissionError("no /proc")

    with (
        pytest.raises(FileAccessError, match="cannot be read"),
        timing.host_memory_phase(HostInterface(unreadable, kernel.reset), "clear_refs"),
    ):
        pass
    with (
        pytest.raises(FileAccessError, match="cannot be reset"),
        timing.host_memory_phase(FakeKernel(raises=True).interface(), "clear_refs"),
    ):
        pass


def test_the_reset_check_is_not_hidden_by_an_earlier_higher_mark() -> None:
    kernel = FakeKernel()
    kernel.grow(500)  # an earlier phase left the mark far above the resident size
    kernel.shrink(500)
    assert timing.host_reset_supported(kernel.interface(), kernel.allocate, 64)


def test_an_unknown_host_method_and_a_missing_field_are_errors() -> None:
    with (
        pytest.raises(SpecificationError, match="Unknown host memory method"),
        timing.host_memory_phase(FakeKernel().interface(), "guess"),
    ):
        pass
    with pytest.raises(DataValidationError, match="VmHWM is not in"):
        timing.status_kib("VmRSS:\t1 kB\n", "VmHWM")


# --- GPU memory ----------------------------------------------------------------------------------

SNAPSHOT = {
    "sampled_max_used_mib": 5200,
    "baseline_used_mib": 1200,
    "total_mib": 8151,
    "samples": 40,
    "interval_ms": 500,
    "max_gap_seconds": 0.6,
    "final_sample_age_seconds": 0.1,
    "units": "MiB",
    "kind": "sampled maximum, not the true peak",
    "problems": [],
}


def test_gpu_memory_keeps_the_maximum_the_baseline_and_the_increase_apart() -> None:
    fields = timing.gpu_memory_fields(SNAPSHOT)
    assert fields is not None
    assert (fields["sampled_max_mib"], fields["baseline_mib"]) == (5200, 1200)
    assert fields["increase_above_baseline_mib"] == 4000
    assert timing.gpu_memory_fields(None) is None
    assert timing.gpu_memory_fields(SNAPSHOT | {"baseline_used_mib": None}) is not None


def test_the_allocator_peaks_need_cuda() -> None:
    resets: list[Any] = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        max_memory_allocated=lambda _device: 111,
        max_memory_reserved=lambda _device: 222,
        reset_peak_memory_stats=lambda device: resets.append(device),
    )
    torch = SimpleNamespace(cuda=cuda)
    assert timing.allocator_peaks(torch, "cuda:0") == {
        "max_allocated_bytes": 111,
        "max_reserved_bytes": 222,
    }
    timing.reset_allocator_peaks(torch, "cuda:0")
    assert resets == ["cuda:0"]
    torch.cuda.is_available = lambda: False
    assert timing.allocator_peaks(torch, "cuda:0") is None
    timing.reset_allocator_peaks(torch, "cuda:0")
    assert resets == ["cuda:0"]


# --- session -------------------------------------------------------------------------------------


def done(code: int, out: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, "")


def test_gpu_telemetry_is_recorded_as_unavailable_not_as_a_pass() -> None:
    ok = timing.gpu_telemetry(lambda *_a, **_k: done(0, "55, 1500, 2500, 40.5\n"))
    assert ok == {
        "available": True,
        "temperature_c": 55.0,
        "clock_sm_mhz": 1500.0,
        "clock_max_sm_mhz": 2500.0,
        "power_w": 40.5,
    }
    unsupported = timing.gpu_telemetry(lambda *_a, **_k: done(0, "[N/A], [N/A], [N/A], [N/A]\n"))
    assert unsupported["available"] is False
    assert timing.gpu_telemetry(lambda *_a, **_k: done(9, "boom"))["available"] is False

    def missing(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError("nvidia-smi")

    assert timing.gpu_telemetry(missing)["available"] is False


def conditions() -> dict[str, Any]:
    return timing.session_conditions(
        loadavg=lambda: (0.25, 0.5, 0.75),
        affinity=lambda _pid: {3, 1, 2},
        meminfo=lambda: "MemTotal:  33554432 kB\nSwapTotal:  8388608 kB\n",
        telemetry=lambda: {"available": False, "reason": "none"},
    )


def test_the_session_records_what_the_system_reports_as_context() -> None:
    session = conditions()
    assert session["load_average_1m"] == 0.25
    assert session["allowed_cpus"] == [1, 2, 3]
    assert (session["memory_total_mib"], session["swap_total_mib"]) == (32768, 8192)
    assert session["gpu"] == {"available": False, "reason": "none"}
    assert "confirmed" not in conditions()


# --- records -------------------------------------------------------------------------------------

CPU = Timed("LGB-u", "selected", "cpu", threads=24, cpu_affinity=(0, 1, 2), subject=file("f.json"))
GPU = Timed("RES-u", "selected", "gpu", subject=file("fits/RES-u/trial-05.json"))
MEMORY = MemoryReading(
    scope="parent process: resident inputs and the fitted model; warm-up outside the interval",
    gpu=SNAPSHOT,
    allocator={"max_allocated_bytes": 1, "max_reserved_bytes": 2},
    host=HostPeak(512.0, "clear_refs"),
)


def check(record: dict[str, Any]) -> dict[str, Any]:
    validate_record(record, PARTITIONS, "timing")
    return record


def test_a_fit_record_carries_the_lengths_and_the_session() -> None:
    record = check(
        timing.fit_record(
            IDENTITY,
            CPU,
            conditions(),
            seconds=12.5,
            completed_length=170,
            retained_length=120,
            startup_seconds=1.0,
            preprocessing_seconds=None,
        )
    )
    assert (record["measurement"], record["seconds"], record["device"]) == ("fit", 12.5, "cpu")
    assert record["threads"] == 24
    assert record["cpu_affinity"] == [0, 1, 2]
    assert record["synchronised"] is False
    assert record["settings"]["completed_length"] == 170
    assert "session_confirmed" not in record["settings"]
    assert record["settings"]["session"]["load_average_1m"] == 0.25
    assert record["settings"]["memory_problems"] == []
    assert record["settings"]["persistence_seconds"] is None  # a caller that does not measure it


def test_a_fit_record_states_the_shared_startup_and_the_persistence() -> None:
    record = check(
        timing.fit_record(
            IDENTITY,
            CPU,
            conditions(),
            seconds=12.5,
            completed_length=170,
            retained_length=120,
            startup_seconds=1.0,
            preprocessing_seconds=None,
            persistence_seconds=0.75,
            startup_scope="library import, once per process",
        )
    )
    assert record["seconds"] == 12.5
    assert record["settings"]["startup_seconds"] == 1.0
    assert record["settings"]["startup_scope"] == "library import, once per process"
    assert record["settings"]["persistence_seconds"] == 0.75


def test_memory_problems_are_recorded_beside_the_readings_that_stand() -> None:
    record = check(
        timing.fit_record(
            IDENTITY,
            CPU,
            conditions(),
            seconds=1.0,
            completed_length=2,
            retained_length=1,
            startup_seconds=None,
            preprocessing_seconds=None,
            memory=timing.MemoryReading(
                "a phase", host=timing.HostPeak(10.0, "clear_refs"), problems=("GPU: stalled",)
            ),
        )
    )
    assert record["peak_host_mib"] == 10.0
    assert record["peak_gpu_mib"] is None
    assert record["settings"]["memory_problems"] == ["GPU: stalled"]


def test_memory_problems_is_builder_owned() -> None:
    with pytest.raises(SpecificationError, match="builder-owned"):
        Timed("LGB-u", "selected", "cpu", threads=24, extra={"memory_problems": []})


def test_a_search_record_says_whether_it_was_interrupted() -> None:
    record = check(
        timing.search_record(
            IDENTITY,
            Timed("LGB-u", None, "cpu", threads=24, cpu_affinity=(0,)),
            conditions(),
            seconds=3600.0,
            trials_completed=4,
            trials_declared=10,
            interrupted=True,
        )
    )
    assert record["model_role"] is None
    settings = record["settings"]
    assert (settings["trials_completed"], settings["trials_declared"]) == (4, 10)
    assert settings["interrupted"] is True


def test_a_search_that_stopped_early_cannot_claim_completion() -> None:
    record = timing.search_record(
        IDENTITY,
        Timed("LGB-u", None, "cpu", threads=24, cpu_affinity=(0,)),
        conditions(),
        seconds=10.0,
        trials_completed=4,
        trials_declared=10,
        interrupted=False,
    )
    with pytest.raises(RecordContractError, match="completed every declared trial"):
        check(record)


def test_a_scoring_record_has_the_median_pass_and_throughput_from_it() -> None:
    result = timing.ScoringMeasurement(3, (2.5, 2.0, 1.9))
    record = check(
        timing.scoring_record(
            IDENTITY,
            GPU,
            conditions(),
            partition="V2",
            boundary="end_to_end",
            rows=77_822,
            batch_rows=4096,
            measurement=result,
            memory=MEMORY,
        )
    )
    assert record["seconds"] == 2.0
    assert record["rows_per_second"] == 77_822 / 2.0
    assert record["synchronised"] is True
    assert record["threads"] is None
    assert (record["warmup"], record["repetitions"]) == (3, 3)
    assert record["settings"]["pass_seconds"] == [2.5, 2.0, 1.9]
    # unequal passes: the slow pass is kept and the range is explicit
    assert (record["settings"]["pass_min_seconds"], record["settings"]["pass_max_seconds"]) == (
        1.9,
        2.5,
    )


def test_memory_is_recorded_with_its_methods_and_scope() -> None:
    record = check(
        timing.scoring_record(
            IDENTITY,
            GPU,
            conditions(),
            partition="V1",
            boundary="model_only",
            rows=PARTITIONS.v1.rows,
            batch_rows=4096,
            measurement=timing.ScoringMeasurement(3, (1.0, 1.1, 1.2)),
            memory=MEMORY,
        )
    )
    assert record["peak_gpu_mib"] == 5200
    assert record["peak_host_mib"] == 512.0
    assert "sampled whole-device GPU memory maximum" in record["memory_scope"]
    assert "resident memory" in record["memory_scope"]
    settings = record["settings"]
    assert settings["gpu_memory"]["increase_above_baseline_mib"] == 4000
    assert settings["allocator_peaks"] == {"max_allocated_bytes": 1, "max_reserved_bytes": 2}
    assert settings["host_memory_method"] == "clear_refs"


def test_a_latency_record_names_the_sample_the_rounds_and_the_calls() -> None:
    ids = np.arange(1, 11, dtype=np.int64)
    sample = ids[timing.select_latency_rows(ids, 10)]
    fake = FakeClock()

    def call_one(row: int) -> None:
        fake.time += (row + 1) / 1000

    result = timing.measure_latency(call_one, 10, 4, 3, fake.clock())
    record = check(
        timing.latency_record(
            IDENTITY,
            GPU,
            conditions(),
            boundary="model_only",
            sample_ids=sample,
            measurement=result,
        )
    )
    assert (record["batch_rows"], record["rows"], record["repetitions"]) == (1, 10, 3)
    assert record["rows_sha256"] == timing.rows_identity(sample)
    assert record["settings"]["timed_calls"] == 30
    assert len(record["settings"]["round_latency_ms"]) == 3
    assert record["row_selection"] == timing.ROW_RULE
    with pytest.raises(SpecificationError, match="differ"):
        timing.latency_record(
            IDENTITY,
            GPU,
            conditions(),
            boundary="model_only",
            sample_ids=sample[:5],
            measurement=result,
        )


def test_the_boundary_is_model_only_or_end_to_end() -> None:
    sample = np.arange(1, 4, dtype=np.int64)
    with pytest.raises(SpecificationError, match="boundary"):
        timing.scoring_record(
            IDENTITY,
            GPU,
            conditions(),
            partition="V2",
            boundary="partial",
            rows=1,
            batch_rows=1,
            measurement=timing.ScoringMeasurement(1, (1.0,)),
        )
    latency = timing.LatencyMeasurement(3, 0, (), timing.Latency(1, 1, 1), 0)
    with pytest.raises(SpecificationError, match="boundary"):
        timing.latency_record(
            IDENTITY,
            GPU,
            conditions(),
            boundary="partial",
            sample_ids=sample,
            measurement=latency,
        )


def test_extra_settings_cannot_override_the_session_or_the_measurement() -> None:
    with pytest.raises(SpecificationError, match="builder-owned"):
        Timed("LGB-u", "selected", "cpu", threads=24, extra={"session": {}})
    clash = Timed("LGB-u", "selected", "cpu", threads=24, extra={"completed_length": 1})
    with pytest.raises(SpecificationError, match="repeats"):
        timing.fit_record(
            IDENTITY,
            clash,
            conditions(),
            seconds=1.0,
            completed_length=2,
            retained_length=1,
            startup_seconds=None,
            preprocessing_seconds=None,
        )
    fine = Timed("LGB-u", "selected", "cpu", threads=24, cpu_affinity=(0,), extra={"note": "x"})
    record = timing.fit_record(
        IDENTITY,
        fine,
        conditions(),
        seconds=1.0,
        completed_length=2,
        retained_length=1,
        startup_seconds=None,
        preprocessing_seconds=None,
    )
    assert record["settings"]["note"] == "x"
    assert "session_confirmed" not in record["settings"]
    assert "confirmed" not in record["settings"]["session"]


def test_a_memory_reading_states_its_scope() -> None:
    with pytest.raises(SpecificationError, match="states its scope"):
        timing.fit_record(
            IDENTITY,
            CPU,
            conditions(),
            seconds=1.0,
            completed_length=2,
            retained_length=1,
            startup_seconds=None,
            preprocessing_seconds=None,
            memory=MemoryReading(scope=" ", host=HostPeak(1.0, "clear_refs")),
        )


def test_sampled_host_memory_records_its_interval() -> None:
    def record(peak: HostPeak) -> dict[str, Any]:
        return timing.fit_record(
            IDENTITY,
            CPU,
            conditions(),
            seconds=1.0,
            completed_length=2,
            retained_length=1,
            startup_seconds=None,
            preprocessing_seconds=None,
            memory=MemoryReading(scope="parent process; warm-up excluded", host=peak),
        )

    sampled = check(record(HostPeak(100.0, "sampled", 100)))
    assert sampled["settings"]["host_sample_interval_ms"] == 100
    assert "parent process; warm-up excluded" in sampled["memory_scope"]
    with pytest.raises(RecordContractError, match="host_sample_interval_ms"):
        check(record(HostPeak(100.0, "sampled")))
