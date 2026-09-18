"""SQMMA vs WMMA selection for MThreads AccelerateMUSAMatmul.

This is the Gate B hole from the MUSA RLC port: phase IR already covers
WMMA/SQMMA encodings on RLC, but not the matmul pass choosing between them.
"""

import os
import tempfile

import pytest
import triton
import triton.language as tl

from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _blocked_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m * BLOCK_K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n)
    acc = tl.dot(a, b)
    tl.store(c_ptr + offs_m * BLOCK_N + offs_n, acc)


def _compile_dot(block_m, block_n, block_k, extra_env=None):
    extra_env = extra_env or {}
    previous = {}
    cache_dir = tempfile.mkdtemp(prefix="musa-mma-select-")
    extra_env = dict(extra_env)
    extra_env.setdefault("TRITON_CACHE_DIR", cache_dir)
    try:
        for key, value in extra_env.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return compile_musa(
            _blocked_dot_kernel,
            {
                "a_ptr": "*fp16",
                "b_ptr": "*fp16",
                "c_ptr": "*fp32",
            },
            constexprs={
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
            },
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_sqmma_selected_for_legal_128_tile():
    compiled = _compile_dot(128, 128, 64)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "#ttg.musa_sqmma" in ttgir, ttgir
    assert "#ttg.musa_wmma" not in ttgir, ttgir
    assert "llvm.musa.sqmma" in llir, llir


def test_wmma_fallback_when_sqmma_disabled():
    compiled = _compile_dot(128, 128, 64, extra_env={"DISABLE_SQMMA": "1"})
    ttgir = compiled.asm["ttgir"]
    assert "#ttg.musa_wmma" in ttgir, ttgir
    assert "#ttg.musa_sqmma" not in ttgir, ttgir


def test_wmma_when_shape_is_sqmma_ineligible():
    compiled = _compile_dot(16, 16, 16)
    ttgir = compiled.asm["ttgir"]
    assert "#ttg.musa_wmma" in ttgir, ttgir
    assert "#ttg.musa_sqmma" not in ttgir, ttgir


def test_wmma_fallback_runtime_precision():
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    compiled = _compile_dot(16, 16, 16)
    ttgir = compiled.asm["ttgir"]
    assert "#ttg.musa_wmma" in ttgir, ttgir

    torch.manual_seed(1234)
    a_cpu = torch.randn((16, 16), dtype=torch.float16)
    b_cpu = torch.randn((16, 16), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((16, 16), device="musa", dtype=torch.float32)
    _blocked_dot_kernel[(1, )](
        a,
        b,
        out,
        16,
        16,
        16,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()
    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=5e-2, rtol=5e-2)
