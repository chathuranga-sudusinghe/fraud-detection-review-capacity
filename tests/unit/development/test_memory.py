"""Memory measurement of the phases of a run (DR-015 §9, Issue #143): readings, components that
cannot be measured, cleanup and the maximum over trials. Fake host and GPU interfaces only."""

from __future__ import annotations

import time

import pytest

from review_capacity.development.memory import (
    MemorySetup,
    PhaseMemory,
    run_measured,
    trials_maximum,
)
from review_capacity.errors import ModelError
from review_capacity.training.timing import HostInterface, HostPeak
from support.memory import FakeGpuSampler, host_interface, memory_setup

SCOPE = "a phase"


@pytest.fixture(autouse=True)
def fresh_samplers() -> None:
    FakeGpuSampler.created.clear()


def test_without_a_setup_nothing_is_measured() -> None:
    result, phase = run_measured(None, SCOPE, lambda: 7)
    assert (result, phase) == (7, None)


def test_a_phase_returns_its_result_and_its_host_peak() -> None:
    result, phase = run_measured(memory_setup([2048 * 1024]), SCOPE, lambda: "done")
    assert result == "done"
    assert phase is not None
    assert phase.host == HostPeak(2048.0, "clear_refs")
    assert phase.gpu is None
    reading = phase.reading()
    assert (reading.scope, reading.problems) == (SCOPE, ())


def test_a_gpu_phase_adds_the_sampled_maximum() -> None:
    setup = memory_setup(gpu_monitoring=True)
    _, phase = run_measured(setup, SCOPE, lambda: None, gpu=True)
    assert phase is not None
    assert phase.gpu is not None
    assert phase.gpu["sampled_max_used_mib"] == 4000
    (sampler,) = FakeGpuSampler.created
    assert sampler.started
    assert sampler.stopped


def test_a_cpu_phase_never_starts_the_gpu_sampler() -> None:
    run_measured(memory_setup(gpu_monitoring=True), SCOPE, lambda: None, gpu=False)
    run_measured(memory_setup(gpu_monitoring=False), SCOPE, lambda: None, gpu=True)
    assert FakeGpuSampler.created == []


def test_a_gpu_sampler_that_does_not_start_is_a_recorded_problem_not_a_stop() -> None:
    setup = memory_setup(gpu_monitoring=True, gpu_sampler=lambda: FakeGpuSampler(fail_start=True))
    result, phase = run_measured(setup, SCOPE, lambda: "ran", gpu=True)
    assert result == "ran"
    assert phase is not None
    assert phase.gpu is None
    assert phase.host is not None  # the other component stands
    assert phase.problems == ["GPU memory not measured: nvidia-smi cannot be started"]


def test_a_sampler_snapshot_with_problems_is_never_a_peak() -> None:
    def stalled() -> FakeGpuSampler:
        return FakeGpuSampler(maximum=9999, problems=("sampling stalled",))

    setup = memory_setup(gpu_monitoring=True, gpu_sampler=stalled)
    _, phase = run_measured(setup, SCOPE, lambda: None, gpu=True)
    assert phase is not None
    assert phase.gpu is None
    assert phase.problems == ["GPU memory not measured: sampling stalled"]
    assert phase.reading().gpu is None


def test_a_host_reset_that_fails_is_a_recorded_problem() -> None:
    def refuse() -> None:
        raise OSError("not permitted")

    interface = HostInterface(read_status=host_interface().read_status, reset_peak=refuse)
    setup = MemorySetup("clear_refs", 100, False, interface)
    result, phase = run_measured(setup, SCOPE, lambda: 3)
    assert result == 3
    assert phase is not None
    assert phase.host is None
    assert len(phase.problems) == 1
    assert phase.problems[0].startswith("host memory not measured")


def test_a_failing_phase_stops_the_monitors_and_raises_its_own_error() -> None:
    setup = memory_setup(gpu_monitoring=True)

    def fail() -> None:
        raise ModelError("the fit failed")

    with pytest.raises(ModelError, match="the fit failed"):
        run_measured(setup, SCOPE, fail, gpu=True)
    (sampler,) = FakeGpuSampler.created
    assert sampler.stopped


def test_an_interrupted_phase_stops_the_monitors_and_is_raised_again() -> None:
    setup = memory_setup(gpu_monitoring=True)

    def interrupt() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_measured(setup, SCOPE, interrupt, gpu=True)
    assert FakeGpuSampler.created[0].stopped


def test_a_failure_to_stop_the_host_monitor_does_not_hide_the_work_error() -> None:
    calls = {"reads": 0}

    def status() -> str:
        calls["reads"] += 1
        raise OSError("gone")

    setup = MemorySetup("clear_refs", 100, False, HostInterface(status, lambda: None))

    def fail() -> None:
        raise ModelError("the fit failed")

    with pytest.raises(ModelError, match="the fit failed"):
        run_measured(setup, SCOPE, fail)
    assert calls["reads"] >= 1


def test_a_gpu_sampler_that_cannot_be_stopped_is_a_recorded_problem() -> None:
    setup = memory_setup(gpu_monitoring=True, gpu_sampler=lambda: FakeGpuSampler(fail_stop=True))
    result, phase = run_measured(setup, SCOPE, lambda: "ran", gpu=True)
    assert result == "ran"
    assert phase is not None
    assert phase.gpu is None
    assert phase.host is not None  # the other component stands
    assert phase.problems == [
        "GPU memory not measured: the sampler did not stop (the process could not be terminated)"
    ]


def test_a_sampler_that_cannot_be_stopped_does_not_replace_the_work_error() -> None:
    setup = memory_setup(gpu_monitoring=True, gpu_sampler=lambda: FakeGpuSampler(fail_stop=True))

    def fail() -> None:
        raise ModelError("the fit failed")

    with pytest.raises(ModelError, match="the fit failed"):
        run_measured(setup, SCOPE, fail, gpu=True)
    assert FakeGpuSampler.created[0].stopped


def test_a_failure_while_starting_the_host_monitor_still_stops_the_gpu_sampler() -> None:
    def broken() -> None:
        raise RuntimeError("an unexpected defect")

    setup = memory_setup(gpu_monitoring=True, on_reset=broken)
    with pytest.raises(RuntimeError, match="unexpected defect"):
        run_measured(setup, SCOPE, lambda: None, gpu=True)
    assert FakeGpuSampler.created[0].stopped


def test_a_failed_resident_size_sampler_is_a_recorded_problem() -> None:
    reads = {"n": 0}

    def status() -> str:
        reads["n"] += 1
        if reads["n"] > 1:
            raise OSError("gone")
        return "VmHWM:\t100 kB\nVmRSS:\t100 kB\n"

    setup = MemorySetup("sampled", 10, False, HostInterface(status, lambda: None))

    def slow() -> str:
        time.sleep(0.3)  # long enough for the sampler to read, and fail, once
        return "ran"

    result, phase = run_measured(setup, SCOPE, slow)
    assert result == "ran"
    assert phase is not None
    assert phase.host is None
    assert phase.problems == ["host memory not measured: Host memory sampling failed"]


def _phase(host_mib: float | None, gpu_mib: int | None, *problems: str) -> PhaseMemory:
    host = None if host_mib is None else HostPeak(host_mib, "clear_refs")
    gpu = None if gpu_mib is None else FakeGpuSampler(maximum=gpu_mib).snapshot()
    return PhaseMemory("fit", host, gpu, list(problems))


def test_the_maximum_over_trials_keeps_each_trials_own_readings() -> None:
    trials = [(1, _phase(900.0, 4000)), (2, _phase(700.0, 6000)), (3, _phase(800.0, 5000))]
    result = trials_maximum(trials, "maximum over fits")
    assert result.host == HostPeak(900.0, "clear_refs")
    assert result.gpu is not None
    assert result.gpu["sampled_max_used_mib"] == 6000
    assert result.problems == []
    assert result.scope == "maximum over fits"


def test_a_trials_problems_are_named_with_its_trial() -> None:
    trials = [(1, _phase(900.0, None)), (2, _phase(700.0, None, "GPU memory not measured: x"))]
    result = trials_maximum(trials, "maximum over fits")
    assert result.gpu is None
    assert result.problems == ["trial 2: GPU memory not measured: x"]


def test_a_component_without_a_reading_in_every_trial_has_no_maximum() -> None:
    trials = [(1, _phase(900.0, 4000)), (2, _phase(None, None)), (3, None)]
    result = trials_maximum(trials, "maximum over fits")
    assert result.host is None
    assert result.gpu is None
    assert result.problems == [
        "host maximum not established: no reading for trials [2, 3]",
        "GPU maximum not established: no reading for trials [2, 3]",
    ]


def test_no_completed_trial_has_no_maximum() -> None:
    result = trials_maximum([], "maximum over fits")
    assert (result.host, result.gpu, result.problems) == (None, None, [])
