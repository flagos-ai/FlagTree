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


def test_backend_hash_tracks_atomic_writeback_policy():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 5)
        key = "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO"
        os.environ[key] = "1"
        ratio1 = backend.hash()
        options1 = backend.parse_options({})
        os.environ[key] = "2"
        ratio2 = backend.hash()
        options2 = backend.parse_options({})
        assert ratio1 != ratio2, (ratio1, ratio2)
        assert options1.rlc_policy != options2.rlc_policy
        assert options1.hash() != options2.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_ignores_atomic_policy_without_phase2():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 3)
        key = "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO"
        os.environ[key] = "1"
        ratio1 = backend.hash()
        options1 = backend.parse_options({})
        os.environ[key] = "2"
        ratio2 = backend.hash()
        options2 = backend.parse_options({})
        assert ratio1 == ratio2, (ratio1, ratio2)
        assert options1.rlc_policy == options2.rlc_policy
        assert options1.hash() == options2.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_tracks_int_to_fp_contiguity_policy():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 5)
        key = "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY"
        os.environ[key] = "1"
        guarded = backend.hash()
        guarded_options = backend.parse_options({})
        os.environ[key] = "0"
        unguarded = backend.hash()
        unguarded_options = backend.parse_options({})
        assert guarded != unguarded, (guarded, unguarded)
        assert guarded_options.rlc_policy != unguarded_options.rlc_policy
        assert guarded_options.hash() != unguarded_options.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_ignores_int_to_fp_policy_without_phase2():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 3)
        key = "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY"
        os.environ[key] = "1"
        guarded = backend.hash()
        guarded_options = backend.parse_options({})
        os.environ[key] = "0"
        unguarded = backend.hash()
        unguarded_options = backend.parse_options({})
        assert guarded == unguarded, (guarded, unguarded)
        assert guarded_options.rlc_policy == unguarded_options.rlc_policy
        assert guarded_options.hash() == unguarded_options.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
