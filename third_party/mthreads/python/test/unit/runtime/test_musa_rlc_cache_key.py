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


@pytest.mark.parametrize("key,first,second", [
    ("FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO", "1", "2"),
    ("FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY", "0", "1"),
])
@pytest.mark.parametrize("enhance,mask,active", [
    (True, 5, True),
    (True, 3, False),
    (False, 15, False),
])
def test_phase2_policy_keys(monkeypatch, key, first, second, enhance, mask, active):
    backend = MUSABackend(GPUTarget("musa", 31, 32))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", str(int(enhance)))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", str(mask))
    monkeypatch.setenv(key, first)
    before = _keys(backend)
    monkeypatch.setenv(key, second)
    after = _keys(backend)
    assert all((a != b) == active for a, b in zip(before, after))
