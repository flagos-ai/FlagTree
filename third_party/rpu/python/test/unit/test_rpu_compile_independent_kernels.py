"""Independent relu / maximum / reduce_sum_axis / broadcast_add kernels.

These exercise the executable-IR direct lowering path (no plan/DSL mirror).
"""
import os
from pathlib import Path

import pytest

from _rpu_env import require_rpu_toolchain


def _common_setup(tmp_path, monkeypatch):
    llvm_root = require_rpu_toolchain()
    monkeypatch.setenv("RPU_LLVM_ROOT", str(llvm_root))
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.delenv("RPU_BOARD_LOAD_CONTIG_NVEC", raising=False)
    import triton.language as tl
    monkeypatch.setitem(globals(), "tl", tl)


def _compile_kernel(fn, n_args, constants, options=None):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    arg_names = fn.arg_names
    signature = {arg_names[i]: "*fp16" for i in range(n_args)}
    constexprs = {(idx, ): val for idx, val in constants.items()}
    src = ASTSource(fn=fn, constexprs=constexprs, signature=signature)
    return triton.compile(
        src=src,
        target=GPUTarget("rpu", "rpu-v1", 1),
        options=options or {"num_warps": 1},
    )


def test_rpu_compile_relu_n16(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    import triton
    import triton.language as tl

    @triton.jit
    def k_relu(out, x, n: tl.constexpr):
        o = tl.arange(0, n)
        v = tl.load(x + o)
        tl.store(out + o, tl.maximum(v, 0.0).to(tl.float16))

    kernel = _compile_kernel(k_relu, n_args=2, constants={2: 16})
    assert "rpubin" in kernel.asm
    assert isinstance(kernel.asm["rpubin"], bytes)
    assert len(kernel.asm["rpubin"]) > 0
    assert kernel.metadata.rpu_pattern == "relu"
    assert "rpu.relu" in kernel.asm["rpuexec"]
    assert "rpu::maximum" in kernel.asm["rpurc"]
    assert "ctx.full<half" in kernel.asm["rpurc"]


def test_rpu_compile_relu_n128(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    import triton
    import triton.language as tl

    @triton.jit
    def k_relu(out, x, n: tl.constexpr):
        o = tl.arange(0, n)
        v = tl.load(x + o)
        tl.store(out + o, tl.maximum(v, 0.0).to(tl.float16))

    kernel = _compile_kernel(k_relu, n_args=2, constants={2: 128})
    assert kernel.metadata.rpu_pattern == "relu"
    # Relu follows sqrt's nvec=n/16 convention (warp count, not element count).
    assert "ctx.load_contig<8>" in kernel.asm["rpurc"]
    assert "rpu::maximum" in kernel.asm["rpurc"]


def test_rpu_compile_maximum_n16(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    import triton
    import triton.language as tl

    @triton.jit
    def k_max(out, a, b, n: tl.constexpr):
        o = tl.arange(0, n)
        av = tl.load(a + o)
        bv = tl.load(b + o)
        tl.store(out + o, tl.maximum(av, bv))

    kernel = _compile_kernel(k_max, n_args=3, constants={3: 16})
    assert "rpubin" in kernel.asm
    assert kernel.metadata.rpu_pattern == "maximum"
    assert "rpu.max" in kernel.asm["rpuexec"]
    assert "rpu::max" in kernel.asm["rpurc"] or "rpu::maximum" in kernel.asm["rpurc"]


def test_rpu_compile_reduce_sum_axis0_16x16(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    import triton
    import triton.language as tl

    @triton.jit
    def k_reduce_sum_axis0(out, x, M: tl.constexpr, N: tl.constexpr):
        om = tl.arange(0, M)[:, None]
        on = tl.arange(0, N)[None, :]
        v = tl.load(x + om * N + on)
        s = tl.sum(v, axis=0)
        tl.store(out + tl.arange(0, N), s)

    kernel = _compile_kernel(k_reduce_sum_axis0, n_args=2, constants={2: 16, 3: 16})
    assert kernel.metadata.rpu_pattern == "reduce_sum_axis0"
    assert "rpu.reduce_sum_axis" in kernel.asm["rpuexec"]
    assert "ctx.reduce_sum<0>" in kernel.asm["rpurc"]


def test_rpu_compile_reduce_sum_axis1_16x16(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    import triton
    import triton.language as tl

    @triton.jit
    def k_reduce_sum_axis1(out, x, M: tl.constexpr, N: tl.constexpr):
        om = tl.arange(0, M)[:, None]
        on = tl.arange(0, N)[None, :]
        v = tl.load(x + om * N + on)
        s = tl.sum(v, axis=1)
        tl.store(out + tl.arange(0, M), s)

    kernel = _compile_kernel(k_reduce_sum_axis1, n_args=2, constants={2: 16, 3: 16})
    assert kernel.metadata.rpu_pattern == "reduce_sum_axis1"
    assert "rpu.reduce_sum_axis" in kernel.asm["rpuexec"]
    assert "ctx.reduce_sum<1>" in kernel.asm["rpurc"]


def test_rpu_compile_broadcast_add_16x16(tmp_path, monkeypatch):
    _common_setup(tmp_path, monkeypatch)
    import triton
    import triton.language as tl

    @triton.jit
    def k_bcast_nc_plus_n(out, a, b, N: tl.constexpr, C: tl.constexpr):
        on = tl.arange(0, N)[:, None]
        oc = tl.arange(0, C)[None, :]
        av = tl.load(a + on * C + oc)
        bv = tl.load(b + tl.arange(0, N))
        tl.store(out + on * C + oc, av + bv[:, None])

    kernel = _compile_kernel(k_bcast_nc_plus_n, n_args=3, constants={3: 16, 4: 16})
    assert kernel.metadata.rpu_pattern == "broadcast_add"
    assert "rpu.broadcast_add" in kernel.asm["rpuexec"]
    # The board emit drops the from_raw + BCASTOP_R2_M chain and instead
    # composes load_contig + rank-1 add + store_contig. The harness preloads
    # `b` in a V256 broadcast layout, so a rank-1 add already realises the
    # broadcast.
    assert "ctx.load_contig<1>(arg2 + 0)" in kernel.asm["rpurc"]
    assert "ctx.store_contig<1>(arg0 + 0," in kernel.asm["rpurc"]
