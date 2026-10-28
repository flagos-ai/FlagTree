# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega compiles gfx1201 megakernels without ConvertToBufferOps, scoped to
that compile: the env-mirrored knob must reach the backend (and its cache key)
only inside the scope, and ordinary kernels keep the backend default."""

import os

import pytest

from triton import knobs
from triton.flagmega.runtime.nvidia_launch import compilation_knobs

_ENV = "AMDGCN_USE_BUFFER_OPS"


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    knobs.amd.reset()
    yield
    knobs.amd.reset()


def test_gfx1201_disables_buffer_ops_only_inside_scope(clean_env):
    assert knobs.amd.use_buffer_ops
    with compilation_knobs("amd-gfx1201"):
        assert not knobs.amd.use_buffer_ops
        # The cache key reads the environment, so the knob must be mirrored there.
        assert os.environ[_ENV] == "0"
    assert knobs.amd.use_buffer_ops
    assert _ENV not in os.environ


def test_scope_restores_on_error(clean_env):
    with pytest.raises(RuntimeError):
        with compilation_knobs("amd-gfx1201"):
            raise RuntimeError
    assert knobs.amd.use_buffer_ops
    assert _ENV not in os.environ


@pytest.mark.parametrize("target", ["amd-gfx1100", "nvidia-sm90", "iluvatar-bi-v150", ""])
def test_other_targets_are_untouched(clean_env, target):
    with compilation_knobs(target):
        assert knobs.amd.use_buffer_ops
        assert _ENV not in os.environ


@pytest.mark.parametrize("value", ["0", "1"])
def test_explicit_user_setting_wins(clean_env, monkeypatch, value):
    monkeypatch.setenv(_ENV, value)
    with compilation_knobs("amd-gfx1201"):
        assert knobs.amd.use_buffer_ops == (value == "1")
    assert os.environ[_ENV] == value


def _hip_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        pytest.skip("AMD GPU is required")
    return torch


def test_scope_reaches_amd_compile_and_cache_key(clean_env):
    torch = _hip_gpu()
    import triton
    import triton.language as tl

    @triton.jit
    def copy(src, dst, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)

    src = torch.arange(1000, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    default = copy.warmup(src, dst, 1000, BLOCK=256, grid=(4, ))
    with compilation_knobs("amd-gfx1201"):
        scoped = copy.warmup(src, dst, 1000, BLOCK=256, grid=(4, ))
    again = copy.warmup(src, dst, 1000, BLOCK=256, grid=(4, ))

    assert "buffer_load" in default.asm["amdgcn"]
    assert "buffer_load" not in scoped.asm["amdgcn"]
    assert "buffer_load" in again.asm["amdgcn"]
    assert scoped.hash != default.hash

    with compilation_knobs("amd-gfx1201"):
        copy[(4, )](src, dst, 1000, BLOCK=256)
    torch.cuda.synchronize()
    assert torch.equal(dst, src)
