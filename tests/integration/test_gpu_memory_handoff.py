"""The GPU memory handoff between configurations on the project GPU (Issue #171).

Opt-in, as ``test_data_contract``: run only with ``REVIEW_CAPACITY_INTEGRATION=1`` and a CUDA
device. It reads no data and writes nothing. After PyTorch allocates and frees memory, its cache
still holds it, so a library with its own allocator (CatBoost) cannot use it; ``release_gpu_memory``
hands it back, and the device reports it free again. The test establishes the release of the cache.
It does not establish that CatBoost then trains in the same process: that is the CatBoost
acceptance and the first development run.

The allocation is bounded by the memory that is free at the start, with a margin, and PyTorch's
allocator is capped at the declared margin (``cap_allocator``): under WSL2 the device otherwise
spills into system memory instead of failing.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from review_capacity.training.preflight import (  # noqa: E402
    cap_allocator,
    release_gpu_memory,
)

ENABLED = os.environ.get("REVIEW_CAPACITY_INTEGRATION") == "1"
MARGIN_MIB = 1024  # kept free of the allocation (the declared preflight margin)
BLOCK_MIB = 512
TOLERANCE_MIB = 256  # the device reading moves a little with other processes

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ENABLED, reason="opt-in: set REVIEW_CAPACITY_INTEGRATION=1"),
]


def _free_mib() -> int:
    free, _ = torch.cuda.mem_get_info()
    return int(free) // (1 << 20)


def test_the_release_returns_the_cached_memory_of_a_finished_model_to_the_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    torch.zeros(1, device="cuda")  # initialise the CUDA context before the baseline
    cap_allocator(MARGIN_MIB)
    release_gpu_memory()
    baseline = _free_mib()
    blocks = max((baseline - 2 * MARGIN_MIB) // BLOCK_MIB, 1)
    tensors: list[Any] = []
    try:
        tensors.extend(
            torch.empty(BLOCK_MIB * (1 << 18), dtype=torch.float32, device="cuda")
            for _ in range(blocks)
        )
        tensors.clear()  # freed to PyTorch's cache, not to the device
        assert torch.cuda.memory_reserved() >= blocks * BLOCK_MIB * (1 << 20)
        assert _free_mib() <= baseline - (blocks * BLOCK_MIB) + TOLERANCE_MIB
        release_gpu_memory()
        assert torch.cuda.memory_reserved() == 0
        assert _free_mib() >= baseline - TOLERANCE_MIB
    finally:
        tensors.clear()
        release_gpu_memory()
