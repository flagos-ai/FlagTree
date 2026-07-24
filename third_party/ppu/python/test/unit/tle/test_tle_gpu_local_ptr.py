"""PPU compile-only port of python/test/tle/unit/test_tle_gpu_local_ptr.py.

The upstream test class is a 12-method exercise of every interesting shape
of ``tle.gpu.local_ptr`` — full-view (``None`` indices), 1D/2D explicit
indices, store inside ``scf.if`` with mask, scalar dynamic index, slice
pointers in a loop, and a couple of fp16 matmul shapes. The original
asserts are numerical comparisons against torch on CUDA; here we drop the
numerical comparisons (compile-only) but keep the IR-shape assertions
that the upstream tests use to pin down lowering invariants.

The two fp16 kernels (tiled_matmul, full_view_dot) currently xfail on PPU
because their ``tle.gpu.copy`` is rewritten into a ``cp.async`` shorter
than 4 bytes — the same upstream-Triton constraint that
test_tle_pipeline_e2e.py's fp16 case hits.
"""

import os
import re
import shutil

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

tle_backend = pytest.importorskip("triton._C.libtriton.tle", reason="libtriton built without FLAGTREE_TLE")
tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")


def _ppu_sdk_available() -> bool:
    sdk = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
    if sdk and os.path.isfile(os.path.join(sdk, "bin", "ppu-llc")):
        return True
    return bool(shutil.which("ppu-llc"))


_PPU_TARGET = GPUTarget("ppu", 80, 32)
BLOCK_SIZE = 64

# ---------------------------------------------------------------------------
# Kernels (mirrors of the upstream test file)
# ---------------------------------------------------------------------------


@triton.jit
def _axpy_kernel(x_ptr, y_ptr, out_ptr, numel, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (tl.arange(0, BLOCK), ))
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(smem_ptrs, x_vals, mask=mask)
    shared = tl.load(smem_ptrs, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(smem_ptrs, shared * alpha + y_vals, mask=mask)
    out_vals = tl.load(smem_ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _store_constant_kernel(out_ptr, numel, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (tl.arange(0, BLOCK), ))
    init = tl.full((BLOCK, ), value, tl.float32)
    tl.store(smem_ptrs, init, mask=mask)
    out_vals = tl.load(smem_ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _full_view_1d_store_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile)
    vals = tl.arange(0, BLOCK)
    tl.store(smem_ptrs, vals)
    out_vals = tl.load(smem_ptrs, mask=mask, other=-1)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _full_view_2d_copy_kernel(x_ptr, out_ptr, stride_xm, stride_xn, stride_om, stride_on, ROWS: tl.constexpr,
                              COLS: tl.constexpr):
    smem = tle.gpu.alloc([ROWS, COLS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    tle.gpu.copy(x_ptr + rows * stride_xm + cols * stride_xn, smem, [ROWS, COLS])
    full_ptrs = tle.gpu.local_ptr(smem)
    vals = tl.load(full_ptrs)
    tl.store(out_ptr + rows * stride_om + cols * stride_on, vals)


@triton.jit
def _local_load_none_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile)
    tl.store(smem_ptrs, idx + 3)
    tl.store(out_ptr + idx, tl.load(smem_ptrs))


@triton.jit
def _local_load_full_indices_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (idx, ))
    tl.store(smem_ptrs, idx + 5)
    tl.store(out_ptr + idx, tl.load(smem_ptrs))


@triton.jit
def _conditional_mask_store_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < numel
    smem = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (idx, ))
    if pid == 0:
        tl.store(ptrs, idx, mask=mask)
    vals = tl.load(ptrs, mask=mask, other=-1)
    tl.store(out_ptr + idx, vals, mask=mask)


@triton.jit
def _looped_elementwise_kernel(x_ptr, y_ptr, out_ptr, numel, alpha, BLOCK: tl.constexpr, CHUNKS: tl.constexpr,
                               SLICES: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * CHUNKS
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (tl.arange(0, BLOCK), ))
    assert BLOCK % SLICE_SIZE == 0
    slice_indices = tl.arange(0, SLICE_SIZE)
    for chunk in range(CHUNKS):
        offsets = base + chunk * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(smem_ptrs, x_vals, mask=mask)
        for slice_idx in range(SLICES):
            block_offset = slice_idx * SLICE_SIZE
            slice_ptr = tle.gpu.local_ptr(smem_tile, (block_offset + slice_indices, ))
            slice_offsets = base + chunk * BLOCK + block_offset + slice_indices
            slice_mask = slice_offsets < numel
            shared_vals = tl.load(slice_ptr, mask=slice_mask, other=0.0)
            y_vals = tl.load(y_ptr + slice_offsets, mask=slice_mask, other=0.0)
            tl.store(slice_ptr, shared_vals * alpha + y_vals, mask=slice_mask)
        out_vals = tl.load(smem_ptrs, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _tiled_matmul_kernel(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NUM_K_TILES: tl.constexpr,
                         SLICE_PARTS: tl.constexpr, SLICE_WIDTH: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    smem_a = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    smem_b = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    assert BLOCK_K % SLICE_PARTS == 0
    for k_tile in range(NUM_K_TILES):
        k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        a_tile = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_tile = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        tle.gpu.copy(a_tile, smem_a, [BLOCK_M, BLOCK_K])
        tle.gpu.copy(b_tile, smem_b, [BLOCK_K, BLOCK_N])
        for slice_idx in range(SLICE_PARTS):
            k_start = slice_idx * SLICE_WIDTH
            a_rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, SLICE_WIDTH))
            a_cols = tl.broadcast_to(tl.arange(0, SLICE_WIDTH)[None, :] + k_start, (BLOCK_M, SLICE_WIDTH))
            a_slice = tle.gpu.local_ptr(smem_a, (a_rows, a_cols))
            b_rows = tl.broadcast_to(tl.arange(0, SLICE_WIDTH)[:, None] + k_start, (SLICE_WIDTH, BLOCK_N))
            b_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (SLICE_WIDTH, BLOCK_N))
            b_slice = tle.gpu.local_ptr(smem_b, (b_rows, b_cols))
            acc += tl.dot(tl.load(a_slice), tl.load(b_slice), out_dtype=tl.float32)
    c_tile = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_tile, acc)


@triton.jit
def _axis_gather_kernel(x_ptr, out_ptr, stride_xm, stride_xn, stride_om, stride_on, ROWS: tl.constexpr,
                        COLS: tl.constexpr, SLICE: tl.constexpr):
    smem = tle.gpu.alloc([ROWS, COLS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, ROWS)[:, None]
    offs_n = tl.arange(0, COLS)[None, :]
    tle.gpu.copy(x_ptr + offs_m * stride_xm + offs_n * stride_xn, smem, [ROWS, COLS])
    row_ids = tl.broadcast_to(offs_m, (ROWS, SLICE))
    col_ids = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (ROWS, SLICE))
    vals = tl.load(tle.gpu.local_ptr(smem, (row_ids, col_ids)))
    tl.store(out_ptr + offs_m * stride_om + tl.arange(0, SLICE)[None, :] * stride_on, vals)


@triton.jit
def _scalar_dynamic_index_kernel(out_ptr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    vec_idx = tl.arange(0, BLOCK)
    vec_ptr = tle.gpu.local_ptr(smem, (vec_idx, ))
    tl.store(vec_ptr, vec_idx + 1)
    zero = tl.program_id(0) * 0
    for i in range(BLOCK):
        scalar_idx = zero + i
        scalar_ptr = tle.gpu.local_ptr(smem, (scalar_idx, ))
        tl.store(out_ptr + i, tl.load(scalar_ptr))


@triton.jit
def _full_view_dot_kernel(a_ptr, out_ptr, stride_ai, stride_aj, stride_oi, stride_oj, BLOCK: tl.constexpr):
    offs_i = tl.arange(0, BLOCK)[:, None]
    offs_j = tl.arange(0, BLOCK)[None, :]
    a_tile = tl.load(a_ptr + offs_i * stride_ai + offs_j * stride_aj)
    smem = tle.gpu.alloc([BLOCK, BLOCK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptr = tle.gpu.local_ptr(smem)
    tl.store(smem_ptr, a_tile)
    staged = tl.load(smem_ptr)
    acc = tl.dot(staged, tl.trans(staged), out_dtype=tl.float32)
    tl.store(out_ptr + offs_i * stride_oi + offs_j * stride_oj, acc.to(tl.float16))


# ---------------------------------------------------------------------------
# Compile + assertion helpers
# ---------------------------------------------------------------------------


def _compile(fn, signature, constexprs):
    src = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=_PPU_TARGET)


def _no_tle_residue(compiled):
    """Bare ``tle.`` substring is unsafe because it matches metadata strings
    in ttgir (``tle.barrier_group`` etc.) and function symbols in llir;
    check only the LLIR for genuine residue."""
    llir = compiled.asm["llir"]
    bad = [ln for ln in llir.split("\n") if re.search(r'"tle\.\w', ln) or re.search(r"\btle\.[a-z]+_[a-z]+\b", ln)]
    return [ln for ln in bad if "@_" not in ln]  # drop func symbols


def _full_pipeline(compiled):
    for stage in ("ttir", "ttgir", "llir", "hgbin"):
        assert stage in compiled.asm and compiled.asm[stage], f"stage {stage} missing"


# ---------------------------------------------------------------------------
# The 12 ported tests
# ---------------------------------------------------------------------------


def test_local_pointer_axpy_compiles():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _axpy_kernel,
        signature={
            "x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "numel": "i32", "alpha": "fp32", "BLOCK":
            "constexpr"
        },
        constexprs={"BLOCK": BLOCK_SIZE},
    )
    _full_pipeline(compiled)
    assert not _no_tle_residue(compiled)


def test_local_pointer_store_constant_compiles():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _store_constant_kernel,
        signature={"out_ptr": "*fp32", "numel": "i32", "value": "fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": BLOCK_SIZE},
    )
    _full_pipeline(compiled)


def test_local_pointer_none_full_view_1d_lowers_to_local_store():
    """Upstream invariant: ``tle.gpu.local_ptr(smem)`` with no indices
    rewrites the immediately-following store to a ``ttg.local_store``."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _full_view_1d_store_kernel,
        signature={"out_ptr": "*i32", "numel": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 128},
    )
    _full_pipeline(compiled)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.local_store" in ttgir, "no ttg.local_store in ttgir"


def test_local_pointer_none_full_view_2d_with_copy_compiles():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _full_view_2d_copy_kernel,
        signature={
            "x_ptr": "*fp32", "out_ptr": "*fp32", "stride_xm": "i32", "stride_xn": "i32", "stride_om": "i32",
            "stride_on": "i32", "ROWS": "constexpr", "COLS": "constexpr"
        },
        constexprs={"ROWS": 16, "COLS": 32},
    )
    _full_pipeline(compiled)


def test_local_pointer_none_load_rewrites_to_local_load():
    """Upstream invariant: ``tl.load(tle.gpu.local_ptr(smem))`` rewrites to
    a ``ttg.local_load``."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _local_load_none_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )
    _full_pipeline(compiled)
    assert "ttg.local_load" in compiled.asm["ttgir"]


def test_local_pointer_full_indices_load_rewrites_to_local_load():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _local_load_full_indices_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )
    _full_pipeline(compiled)
    assert "ttg.local_load" in compiled.asm["ttgir"]


def test_local_pointer_conditional_mask_store_compiles():
    """Store under ``if pid == 0`` with a mask — used to trigger
    triton-tle-select-encodings verifier failure upstream."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _conditional_mask_store_kernel,
        signature={"out_ptr": "*i32", "numel": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 512},
    )
    _full_pipeline(compiled)


def test_local_pointer_looped_elementwise_compiles():
    """Inner loop reuses smem via slice pointers — exercises
    add_optimize_local_pointer_loads/stores."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _looped_elementwise_kernel,
        signature={
            "x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "numel": "i32", "alpha": "fp32", "BLOCK":
            "constexpr", "CHUNKS": "constexpr", "SLICES": "constexpr", "SLICE_SIZE": "constexpr"
        },
        constexprs={"BLOCK": BLOCK_SIZE, "CHUNKS": 4, "SLICES": 4, "SLICE_SIZE": BLOCK_SIZE // 4},
    )
    _full_pipeline(compiled)


@pytest.mark.xfail(
    reason="fp16 + tle.gpu.copy compile-only path hits cp.async < 4B "
    "constraint with default ASTSource options. The same kernel "
    "passes when launched via JIT (which picks wider vectorization).", raises=RuntimeError, strict=False)
def test_local_pointer_tiled_matmul_fp16_compiles():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _tiled_matmul_kernel,
        signature={
            "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32", "stride_am": "i32", "stride_ak": "i32", "stride_bk":
            "i32", "stride_bn": "i32", "stride_cm": "i32", "stride_cn": "i32", "BLOCK_M": "constexpr", "BLOCK_N":
            "constexpr", "BLOCK_K": "constexpr", "NUM_K_TILES": "constexpr", "SLICE_PARTS": "constexpr", "SLICE_WIDTH":
            "constexpr"
        },
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "NUM_K_TILES": 2, "SLICE_PARTS": 2, "SLICE_WIDTH": 16},
    )
    _full_pipeline(compiled)


def test_local_pointer_axis_gather_compiles():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _axis_gather_kernel,
        signature={
            "x_ptr": "*fp32", "out_ptr": "*fp32", "stride_xm": "i32", "stride_xn": "i32", "stride_om": "i32",
            "stride_on": "i32", "ROWS": "constexpr", "COLS": "constexpr", "SLICE": "constexpr"
        },
        constexprs={"ROWS": 8, "COLS": 8, "SLICE": 4},
    )
    _full_pipeline(compiled)


def test_local_pointer_scalar_dynamic_index_inserts_barrier():
    """Vector store followed by scalar dynamic-index loads requires the TLE
    barrier-insertion pass to put a ``gpu.barrier`` between them."""
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _scalar_dynamic_index_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )
    _full_pipeline(compiled)
    ttgir = compiled.asm["ttgir"]
    assert "gpu.barrier" in ttgir, "expected gpu.barrier in ttgir"
    assert '"tt.reduce"' not in ttgir


def test_local_pointer_full_view_dot_fp16_compiles():
    if not _ppu_sdk_available():
        pytest.skip("PPU SDK not available")
    compiled = _compile(
        _full_view_dot_kernel,
        signature={
            "a_ptr": "*fp16", "out_ptr": "*fp16", "stride_ai": "i32", "stride_aj": "i32", "stride_oi": "i32",
            "stride_oj": "i32", "BLOCK": "constexpr"
        },
        constexprs={"BLOCK": 32},
    )
    _full_pipeline(compiled)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.local_load" in ttgir
