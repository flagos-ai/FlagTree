# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""IPC exports allocation roots, so a symmetric window cannot be a suballocation."""

import ctypes

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape,dtype", [
    ((1792,), torch.uint8), ((3, 64), torch.float32), ((7_292_672,), torch.uint8),
])
def test_symmetric_window_starts_at_cuda_allocation_base(monkeypatch, shape, dtype):
    from triton.experimental.tle.language import communication

    if not communication.enabled:
        pytest.skip("FlagCX is unavailable")
    # Allocation itself needs no communicator. Leave live allocations in the
    # ordinary cache so aligning a view inside its next block cannot pass.
    monkeypatch.setattr(communication, "get_mem_pool", lambda: None)
    _cache_allocations = [torch.empty(257 + index * 13, dtype=torch.uint8, device="cuda") for index in range(3)]
    window = communication.allocate_dist_tensor(shape, dtype=dtype, device="cuda")
    base = ctypes.c_uint64()
    size = ctypes.c_size_t()
    query = ctypes.CDLL("libcuda.so.1").cuMemGetAddressRange_v2
    query.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint64]
    query.restype = ctypes.c_int
    assert query(ctypes.byref(base), ctypes.byref(size), window.data_ptr()) == 0
    assert window.data_ptr() == base.value, "IPC maps the allocation base, not an aligned caching-pool subview"
    assert window.data_ptr() % communication._FLAGCX_WINDOW_ALIGNMENT == 0
    assert tuple(window.shape) == shape
    assert window.dtype == dtype
