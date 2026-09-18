"""RLC knobs must change MUSA compile keys in the same process.

The 2026-08-18 FlagGems 4→4 result was a harness false negative: site
`hash()` omitted RLC, `@functools.lru_cache()` froze the first env, and
`JITFunction.device_caches` keyed only on launch kwargs. MThreads loads
`triton/spec/mthreads/triton/runtime/jit.py`, not `triton/runtime/jit.py`.
"""

from __future__ import annotations

import os

from triton.backends.compiler import GPUTarget
from triton.backends.mthreads.compiler import MUSABackend, _rlc_policy_signature


def _backend():
    return MUSABackend(GPUTarget("musa", 31, 32))


def _set_rlc(enhance: bool, mask: int = 15):
    os.environ["FLAGTREE_MUSA_RLC_ENHANCE"] = "1" if enhance else "0"
    os.environ["FLAGTREE_MUSA_RLC_PHASE_MASK"] = str(mask)


def test_backend_hash_tracks_in_process_rlc_flip():
    previous = {k: os.environ.get(k) for k in ("FLAGTREE_MUSA_RLC_ENHANCE", "FLAGTREE_MUSA_RLC_PHASE_MASK")}
    backend = _backend()
    try:
        _set_rlc(False, 15)
        off = backend.hash()
        sig_off = _rlc_policy_signature()
        _set_rlc(True, 15)
        on = backend.hash()
        sig_on = _rlc_policy_signature()
        _set_rlc(True, 3)
        mask3 = backend.hash()
        assert sig_off != sig_on, (sig_off, sig_on)
        assert off != on, (off, on)
        assert on != mask3, (on, mask3)
        assert "-rlc" in off and "-rlc" in on
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_options_include_live_rlc_policy():
    previous = {k: os.environ.get(k) for k in ("FLAGTREE_MUSA_RLC_ENHANCE", "FLAGTREE_MUSA_RLC_PHASE_MASK")}
    backend = _backend()
    try:
        _set_rlc(False)
        off = backend.parse_options({})
        _set_rlc(True)
        on = backend.parse_options({})
        assert off.rlc_policy != on.rlc_policy, (off.rlc_policy, on.rlc_policy)
        assert str(off) != str(on)
        assert off.hash() != on.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
