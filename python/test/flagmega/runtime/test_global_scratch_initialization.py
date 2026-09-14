# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

import triton.flagmega.runtime.prepared as prepared_runtime


class _Device:
    type = "cuda"


class _Argument:
    device = _Device()


class _Metadata:
    global_scratch_size = 8
    global_scratch_align = 16
    num_ctas = 1


class _Compiled:
    metadata = _Metadata()


class _Storage:
    def __init__(self, size, *, zeroed):
        self.size = int(size)
        self.zeroed = bool(zeroed)

    def data_ptr(self):
        return 4096

    def narrow(self, dimension, start, length):
        assert dimension == 0
        assert start == 0
        assert length <= self.size
        return _Buffer(self, length)


class _Buffer:
    def __init__(self, storage, nbytes):
        self.storage = storage
        self.nbytes = int(nbytes)

    def data_ptr(self):
        return self.storage.data_ptr()


def test_grid_barrier_scratch_starts_from_a_zero_counter(monkeypatch):
    allocations = []
    synchronized = []

    def allocate(size, *, zeroed):
        allocations.append((int(size), zeroed))
        return _Storage(size, zeroed=zeroed)

    monkeypatch.setattr(
        torch,
        "empty",
        lambda size, **_kwargs: allocate(size, zeroed=False),
    )
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda size, **_kwargs: allocate(size, zeroed=True),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronized.append(device),
    )

    scratch = prepared_runtime._prepare_global_scratch(
        _Compiled(), (_Argument(),), (4, 1, 1)
    )

    assert scratch is not None
    assert allocations == [(47, True)]
    assert scratch.storage.zeroed is True
    assert scratch.nbytes == 32
    assert synchronized == [_Argument.device]


def test_prepared_scratch_allocation_skips_launcher_clear(monkeypatch):
    """The launcher clears returned tensors before cooperative launches.

    The prepared scratch is zero-initialized once and its barrier counters
    stay phase-safe across launches, so the allocated view must not expose
    ``zero_``/``fill_``; otherwise every launch pays a redundant clear.
    """

    monkeypatch.setattr(
        torch, "zeros", lambda size, **_kwargs: _Storage(size, zeroed=True)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    scratch = prepared_runtime._prepare_global_scratch(
        _Compiled(), (_Argument(),), (4, 1, 1)
    )
    view = scratch(scratch.nbytes, 16, None)
    assert not hasattr(view, "zero_")
    assert not hasattr(view, "fill_")
    assert view.data_ptr() == scratch.buffer.data_ptr()

    import pytest

    from triton.flagmega.errors import RuntimeContractError

    with pytest.raises(RuntimeContractError):
        scratch(scratch.nbytes + 1, 16, None)
    with pytest.raises(RuntimeContractError):
        scratch(scratch.nbytes, 32, None)
