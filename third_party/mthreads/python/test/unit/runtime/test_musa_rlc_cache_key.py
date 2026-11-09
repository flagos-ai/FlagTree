"""RLC switches and active policies must identify distinct MUSA compilations."""

import pytest

from triton.backends.compiler import GPUTarget
from triton.backends.mthreads.compiler import MUSABackend, _rlc_policy_signature


def _keys(backend):
    options = backend.parse_options({})
    return backend.hash(), _rlc_policy_signature(), options.rlc_policy, options.hash()


def test_rlc_keys_track_master_and_phase_mask(monkeypatch):
    backend = MUSABackend(GPUTarget("musa", 31, 32))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", "0")
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", "15")
    off = _keys(backend)
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", "1")
    on = _keys(backend)
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", "3")
    phase3 = _keys(backend)
    assert all(a != b and b != c for a, b, c in zip(off, on, phase3))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", "15")
    assert _keys(backend) == on
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", "0")
    assert _keys(backend) == off


@pytest.mark.parametrize("key,first,second", [
    ("FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO", "1", "2"),
    ("FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY", "0", "1"),
])
@pytest.mark.parametrize("enhance", [False, True])
@pytest.mark.parametrize("mask", range(16))
def test_phase2_policy_keys(monkeypatch, key, first, second, enhance, mask):
    backend = MUSABackend(GPUTarget("musa", 31, 32))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", str(int(enhance)))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", str(mask))
    monkeypatch.setenv(key, first)
    before = _keys(backend)
    monkeypatch.setenv(key, second)
    after = _keys(backend)
    active = enhance and mask in (5, 7, 13, 15)
    assert all((a != b) == active for a, b in zip(before, after))
    monkeypatch.setenv(key, first)
    assert _keys(backend) == before


@pytest.mark.parametrize("enhance,mask,expected", [
    (False, 15, "fixed-cost-v1-late-proof-v2-0-15-0-0"),
    (True, 3, "fixed-cost-v1-late-proof-v2-1-3-0-0"),
    (True, 5, "fixed-cost-v1-late-proof-v2-1-5-2-1"),
    (True, 15, "fixed-cost-v1-late-proof-v2-1-15-2-1"),
    (True, 4, "fixed-cost-v1-late-proof-v2-1-4-0-0"),
    (True, 12, "fixed-cost-v1-late-proof-v2-1-12-0-0"),
])
def test_rlc_policy_cache_format(monkeypatch, enhance, mask, expected):
    # Keep the versioned format for effective policies; only inactive Phase 2
    # controls are omitted when cost-based resolution is disabled.
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", str(int(enhance)))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", str(mask))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO", "2")
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY", "1")
    backend = MUSABackend(GPUTarget("musa", 31, 32))
    assert _rlc_policy_signature() == expected
    assert backend.parse_options({}).rlc_policy == expected
