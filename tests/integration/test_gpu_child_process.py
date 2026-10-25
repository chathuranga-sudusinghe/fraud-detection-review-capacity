"""The GPU memory of a child process next to its parent (Issue #175).

Opt-in, as ``test_data_contract``: run only with ``REVIEW_CAPACITY_INTEGRATION=1`` and a CUDA
device. It reads no data and writes nothing. It sizes its requests from the device memory that
no process uses, read device-wide with ``nvidia-smi``, and skips when too little is left, so that
it does not ask for anything near the physical limit (an allocation beyond the physical memory
would spill into system memory under WSL2, which is what the allocator cap exists to prevent). A
fresh process does not inherit its parent's allocator cap, and the parent's cached memory stays
in use on the device until the parent releases it. Two synthetic checks cover the two actions the
refit takes:

* the cap: a child applies ``cap_allocator`` with a small allowed size (1 GiB of the device) and
  is refused by its own allocator beyond that size, long before the physical limit;
* the handoff: the device-wide memory in use (``nvidia-smi``, whichever process holds it) still
  counts the parent's cached memory after its tensors are freed, and no longer after the parent
  called ``release_gpu_memory``. A child's own ``cudaMemGetInfo`` is not used: under WSL2 the
  device memory is virtualised and a process does not see another's cache in its free memory (the
  reason the cap exists), so only the device-wide reading shows what a child competes with.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from review_capacity.training.preflight import release_gpu_memory  # noqa: E402

ENABLED = os.environ.get("REVIEW_CAPACITY_INTEGRATION") == "1"
ALLOWED_MIB = 1024  # what the capped child may use
BLOCK_MIB = 256
CACHED_MIB = 2048  # what the parent caches in the handoff check
TOLERANCE_MIB = 256  # the device reading moves a little with other processes
CONTEXT_MIB = 1536  # allowance for the CUDA contexts of this process and of a child
MAX_SECONDS = 60  # a refusal by the allocator is immediate, not a stall

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ENABLED, reason="opt-in: set REVIEW_CAPACITY_INTEGRATION=1"),
]

CAPPED = textwrap.dedent(
    """
    import json
    import time
    import torch
    from review_capacity.training.preflight import cap_allocator

    total = torch.cuda.get_device_properties(0).total_memory // (1 << 20)
    cap_allocator(total - {allowed})  # the margin that leaves {allowed} MiB to the allocator
    held = []
    started = time.perf_counter()
    refused = False
    try:
        for _ in range({asked} // {block}):
            held.append(torch.empty({block} * (1 << 18), dtype=torch.float32, device="cuda"))
    except torch.OutOfMemoryError:
        refused = True
    print(json.dumps({{"blocks": len(held), "refused": refused,
                       "seconds": time.perf_counter() - started}}))
    """
)


def _run(code: str) -> dict[str, float]:
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300
    )
    result: dict[str, float] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def _need_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def _used_mib() -> int:
    """The device-wide memory in use as ``nvidia-smi`` reports it, whichever process holds it."""
    done = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return int(done.stdout.strip())


def _settled(condition: Callable[[int], bool]) -> int:
    """The device-wide memory in use once ``condition`` holds (the reading lags a little)."""
    used = _used_mib()
    for _ in range(20):
        if condition(used):
            break
        time.sleep(0.5)
        used = _used_mib()
    return used


def _headroom_mib() -> int:
    """The device memory that no process uses, less an allowance for the CUDA contexts of this
    process and of a child. It is read device-wide (``nvidia-smi``): under WSL2 a process's own
    ``cudaMemGetInfo`` does not see another process's cache, and would overstate the headroom."""
    total = int(torch.cuda.get_device_properties(0).total_memory) // (1 << 20)
    return total - _used_mib() - CONTEXT_MIB


def test_a_child_is_refused_by_its_own_allocator_beyond_its_cap() -> None:
    _need_gpu()
    asked = 2 * ALLOWED_MIB
    if _headroom_mib() < ALLOWED_MIB:  # what the capped child may use must fit physically
        pytest.skip("not enough unused device memory for a safe check")
    code = CAPPED.format(allowed=ALLOWED_MIB, asked=asked, block=BLOCK_MIB)
    result = _run(code)
    assert result["refused"], "the allocator should refuse a request beyond its cap"
    assert result["blocks"] * BLOCK_MIB <= ALLOWED_MIB  # it stopped at the cap, not at the device
    assert result["seconds"] < MAX_SECONDS  # refused at once, not after a stall


def test_the_memory_the_parent_released_leaves_the_memory_in_use_on_the_device() -> None:
    _need_gpu()
    torch.zeros(1, device="cuda")
    release_gpu_memory()
    baseline = _settled(lambda _used: True)
    cached = min(CACHED_MIB, _headroom_mib() // 2) // BLOCK_MIB * BLOCK_MIB
    if cached < 2 * TOLERANCE_MIB:  # the cache must be well above the reading's tolerance
        pytest.skip("not enough unused device memory for a safe check")
    blocks: list[Any] = []
    try:
        blocks.extend(
            torch.empty(BLOCK_MIB * (1 << 18), dtype=torch.float32, device="cuda")
            for _ in range(cached // BLOCK_MIB)
        )
        blocks.clear()  # freed to PyTorch's cache, not to the device
        assert torch.cuda.memory_reserved() >= cached * (1 << 20) - TOLERANCE_MIB * (1 << 20)
        # the device still counts the cache as in use: this is what a child process would meet
        held = _settled(lambda used: used >= baseline + cached - TOLERANCE_MIB)
        release_gpu_memory()
        assert torch.cuda.memory_reserved() == 0
        released = _settled(lambda used: used <= baseline + TOLERANCE_MIB)
    finally:
        blocks.clear()
        release_gpu_memory()
    assert held >= baseline + cached - TOLERANCE_MIB
    assert released <= baseline + TOLERANCE_MIB
