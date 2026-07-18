# flagtree tle
"""
TLE negative / contract tests (sunrise)
=======================================

SKILL requires every semantic surface to have a negative (contract) case, not
just positive ones. These assert that misuse raises at compile time rather than
miscompiling or crashing, and that Hopper-only TLE features fail with a clear
error on sunrise instead of reaching an NVVM lowering that cannot legalize.

All checks fire during JIT tracing, so the frontend ValueError is wrapped in a
triton CompilationError (original is in __cause__). We accept either.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# JIT tracing wraps frontend ValueError in CompilationError; accept both.
try:
    from triton.compiler.errors import CompilationError
    _RAISES = (CompilationError, ValueError)
except Exception:
    _RAISES = (Exception, )


# --- extract_tile: tile_shape rank must match source rank ---
@triton.jit
def _extract_tile_rank_mismatch(x_ptr):
    offs = tl.arange(0, 64)
    x = tl.load(x_ptr + offs[:, None] * 64 + tl.arange(0, 64)[None, :])
    # 2D source but 1D tile_shape -> rank mismatch -> ValueError
    t = tle.extract_tile(x, index=0, tile_shape=[32])
    tl.store(x_ptr + offs, tl.sum(t))


def test_extract_tile_rank_mismatch_raises():
    x = torch.zeros(64 * 64, device=DEVICE, dtype=torch.float32)
    with pytest.raises(_RAISES):
        _extract_tile_rank_mismatch[(1, )](x)


# --- insert_tile: source rank must match tile rank ---
@triton.jit
def _insert_tile_rank_mismatch(x_ptr, y_ptr):
    offs = tl.arange(0, 64)
    x = tl.load(x_ptr + offs[:, None] * 64 + tl.arange(0, 64)[None, :])
    y = tl.load(y_ptr + tl.arange(0, 32))  # 1D tile vs 2D source
    z = tle.insert_tile(x, y, index=0)
    tl.store(x_ptr + offs[:, None] * 64 + tl.arange(0, 64)[None, :], z)


def test_insert_tile_rank_mismatch_raises():
    x = torch.zeros(64 * 64, device=DEVICE, dtype=torch.float32)
    y = torch.zeros(32, device=DEVICE, dtype=torch.float32)
    with pytest.raises(_RAISES):
        _insert_tile_rank_mismatch[(1, )](x, y)


# --- tle.gpu.alloc: source dim not divisible by tile dim (extract) ---
@triton.jit
def _extract_tile_not_divisible(x_ptr):
    offs = tl.arange(0, 64)
    x = tl.load(x_ptr + offs[:, None] * 64 + tl.arange(0, 64)[None, :])
    # 64 not divisible by tile dim 48 -> ValueError
    t = tle.extract_tile(x, index=[0, 0], tile_shape=[48, 48])
    tl.store(x_ptr + offs, tl.sum(t))


def test_extract_tile_not_divisible_raises():
    x = torch.zeros(64 * 64, device=DEVICE, dtype=torch.float32)
    with pytest.raises(_RAISES):
        _extract_tile_not_divisible[(1, )](x)


# --- tle.pipe: capacity must be positive (Hopper-only feature; also a contract
#     check that fires before any NVWS lowering) ---
@triton.jit
def _pipe_bad_capacity(out_ptr):
    buf = tle.gpu.alloc([1, 16, 16], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    # capacity must be > 0 -> ValueError
    _ = tle.pipe(capacity=0, data=buf)
    tl.store(out_ptr + tl.arange(0, 16), tl.zeros([16], tl.float32))


def test_pipe_bad_capacity_raises():
    out = torch.zeros(16, device=DEVICE, dtype=torch.float32)
    with pytest.raises(_RAISES):
        _pipe_bad_capacity[(1, )](out)


# --- tle.pipe: scope must be 'cta' in the MVP ---
@triton.jit
def _pipe_bad_scope(out_ptr):
    buf = tle.gpu.alloc([2, 16, 16], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    _ = tle.pipe(capacity=2, scope="gpu", data=buf)  # only 'cta' supported
    tl.store(out_ptr + tl.arange(0, 16), tl.zeros([16], tl.float32))


def test_pipe_bad_scope_raises():
    out = torch.zeros(16, device=DEVICE, dtype=torch.float32)
    with pytest.raises(_RAISES):
        _pipe_bad_scope[(1, )](out)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
