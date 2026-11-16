"""TLE local_pointers test suite for the GCU backend.

Covers all tle.local_pointers code paths in:
  - TleToTritonGCU.cpp         (canonicalize Cases A-D + Path A/B)
  - TritonGCULocalMemOptimize.cpp  (FuseMemDescTransLoadLocalAllocPattern / B2)
  - TTSmemLowerToGCU.cpp       (MemDesc view op lowerings)

Memdesc view ops produced by canonicalizeLocalPointers:
  Case A -> ttg.memdesc_index    (scalar dim-0 peel on multi-dim buffer)
  Case B -> ttg.memdesc_subslice (aligned offset sub-region on 2D smem)
  Case C -> ttg.memdesc_trans    (transposed 2D access; B2 folds into DTE copy)
  Case D -> ttg.memdesc_reshape  (1D buffer accessed with multi-dim index)

Fallback paths:
  Path A  (full-block)  -> ttg.local_load / ttg.local_store
  Path B  (lowerSubViewAccess) -> triton_gcu.memdesc_to_ptr + pointer arithmetic
  Masked  (non-trivial mask)   -> Path B (DTE handles mask on pointer arithmetic)
  Scalar  (dynamic scalar index in loop) -> Path B + gpu.barrier
"""

import importlib.util

import pytest
import torch
import triton
import triton.language as tl

if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton  # noqa: F401  (register GCU backend)
from torch_gcu import transfer_to_gcu  # noqa: F401  (enables .gcu() on tensors)
import triton.experimental.tle.language as tle


def _is_enflame_backend():
    return triton.runtime.driver.active.get_current_target().backend == "gcu"


def _require_device():
    try:
        if not _is_enflame_backend():
            torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"device init failed: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _device_guard():
    _require_device()


def _get_ir(compiled):
    if _is_enflame_backend():
        return compiled.asm["gcuir"]
    return compiled.asm["ttgir"]


# ---------------------------------------------------------------------------
# Problem-1 regression: compute-filled smem + non-contiguous sub-tile read.
# The smem is filled by tl.store of an arithmetic result (NOT a
# tle.gpu.copy), so no FuseMemDescSubslice pattern can fire and the
# local_load(subslice) reaches convert-triton-to-gcu unfused. The 2D
# sub-tile (row stride N > width S) must be staged via
# stageViewToContiguous (strided memref.load/store + gpu.barrier).
# ---------------------------------------------------------------------------

@triton.jit
def _compute_fill_subtile_kernel(x_ptr, out_ptr, sm, sn, om, on,
                                 M: tl.constexpr, N: tl.constexpr,
                                 OFF: tl.constexpr, S: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = tl.load(x_ptr + offs_m * sm + offs_n * sn)
    # Compute fill: store an arithmetic result through full-view pointers.
    row2 = tl.broadcast_to(tl.arange(0, M)[:, None], (M, N))
    col2 = tl.broadcast_to(tl.arange(0, N)[None, :], (M, N))
    full = tle.gpu.local_ptr(smem, (row2, col2))
    tl.store(full, x_tile * 2.0)
    # Read a 2D sub-tile at column offset OFF (non-contiguous view).
    row = tl.broadcast_to(tl.arange(0, M)[:, None], (M, S))
    col = tl.broadcast_to((OFF + tl.arange(0, S))[None, :], (M, S))
    ptrs = tle.gpu.local_ptr(smem, (row, col))
    vals = tl.load(ptrs)
    out_offs = tl.arange(0, M)[:, None] * om + tl.arange(0, S)[None, :] * on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Problem-1 regression: aligned 2D subslice + non-trivial mask.
# Case B canonicalization fires (aligned offset) but the mask forces Path B
# (lowerViaMemDescToPtr). The base pointer must address the view's first
# element (MemDescToPtrOpLowering folds the view offset) and the row stride
# must be the ROOT row stride (resolveViewStrides), not row-major of the
# sub-tile shape.
# ---------------------------------------------------------------------------

@triton.jit
def _subslice_masked_kernel(x_ptr, out_ptr, sm, sn, om, on,
                            M: tl.constexpr, N: tl.constexpr,
                            OFF: tl.constexpr, S: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = x_ptr + offs_m * sm + offs_n * sn
    tle.gpu.copy(x_tile, smem, [M, N])
    row = tl.broadcast_to(tl.arange(0, M)[:, None], (M, S))
    col = tl.broadcast_to((OFF + tl.arange(0, S))[None, :], (M, S))
    ptrs = tle.gpu.local_ptr(smem, (row, col))
    # Non-trivial mask over the sub-tile columns.
    m = col < (OFF + S // 2)
    vals = tl.load(ptrs, mask=m, other=0.0)
    out_offs = tl.arange(0, M)[:, None] * om + tl.arange(0, S)[None, :] * on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Case C extension: transposed access of an OFFSET sub-region.
# result[r, c] = smem[OFF0 + c, OFF1 + r] for r in [0,S1), c in [0,S0).
# canonicalizeTranspose emits subslice[OFF0, OFF1] + trans; ON mode folds
# both into the DTE copy (FuseMemDescTrans through the subslice), OFF mode
# stages the transposed view.
# ---------------------------------------------------------------------------

@triton.jit
def _memdesc_trans_offset_kernel(x_ptr, out_ptr, stride_xm, stride_xn,
                                 stride_om, stride_on,
                                 M: tl.constexpr, N: tl.constexpr,
                                 OFF0: tl.constexpr, OFF1: tl.constexpr,
                                 S0: tl.constexpr, S1: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, [M, N])

    # Transposed sub-region: result [S1, S0].
    # idx0 (dim0) varies along axis 1 of the result (covers S0 at OFF0),
    # idx1 (dim1) varies along axis 0 of the result (covers S1 at OFF1).
    row = tl.broadcast_to((OFF0 + tl.arange(0, S0))[None, :], (S1, S0))
    col = tl.broadcast_to((OFF1 + tl.arange(0, S1))[:, None], (S1, S0))
    ptrs = tle.gpu.local_ptr(smem, (row, col))
    vals = tl.load(ptrs)

    out_offs = tl.arange(0, S1)[:, None] * stride_om         + tl.arange(0, S0)[None, :] * stride_on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Path A: full-block access -> local_load / local_store
# ---------------------------------------------------------------------------

@triton.jit
def _full_view_1d_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem = tle.gpu.alloc((BLOCK,), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(ptrs, x_vals, mask=mask)
    out_vals = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _full_view_2d_kernel(x_ptr, out_ptr, stride_xm, stride_xn,
                         stride_om, stride_on,
                         M: tl.constexpr, N: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, [M, N])

    full_ptrs = tle.gpu.local_ptr(smem)
    vals = tl.load(full_ptrs)

    out_tile = out_ptr + offs_m * stride_om + offs_n * stride_on
    tl.store(out_tile, vals)


# ---------------------------------------------------------------------------
# Case A: memdesc_index — scalar dim-0 peel on 2D buffer
# ---------------------------------------------------------------------------

@triton.jit
def _memdesc_index_kernel(x_ptr, out_ptr, stride_xm, stride_xn,
                          stride_om, stride_on,
                          M: tl.constexpr, N: tl.constexpr,
                          ROW: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, [M, N])

    # 1D indices on 2D buffer: row_idx is a splat (tl.full),
    # col_idx is a range. canonicalizeDim0Index peels dim-0
    # via memdesc_index, reducing to 1D local_load.
    row_idx = tl.full((N,), ROW, tl.int32)
    col_idx = tl.arange(0, N)
    ptrs = tle.gpu.local_ptr(smem, (row_idx, col_idx))
    vals = tl.load(ptrs)

    out_offs = tl.arange(0, N) * stride_on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Case B: memdesc_subslice — aligned offset sub-region on 2D smem
# ---------------------------------------------------------------------------

@triton.jit
def _memdesc_subslice_kernel(x_ptr, out_ptr, stride_xm, stride_xn,
                             stride_om, stride_on,
                             M: tl.constexpr, N: tl.constexpr,
                             OFF: tl.constexpr, S: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, [M, N])

    # Aligned offset range: OFF must be aligned to S
    # (MemDescSubsliceOp verifier: offset & (resultSize - 1) == 0)
    row = tl.broadcast_to(tl.arange(0, M)[:, None], (M, S))
    col = tl.broadcast_to((OFF + tl.arange(0, S))[None, :], (M, S))
    ptrs = tle.gpu.local_ptr(smem, (row, col))
    vals = tl.load(ptrs)

    out_offs = tl.arange(0, M)[:, None] * stride_om \
        + tl.arange(0, S)[None, :] * stride_on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Case C: memdesc_trans — transposed 2D access + DTE fold (B2)
# ---------------------------------------------------------------------------

@triton.jit
def _memdesc_trans_kernel(x_ptr, out_ptr, stride_xm, stride_xn,
                          stride_om, stride_on,
                          M: tl.constexpr, N: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, [M, N])

    # Transposed access: result [N, M] from smem [M, N].
    # idx0 varies along axis 1 of result (covers M),
    # idx1 varies along axis 0 of result (covers N).
    row = tl.broadcast_to(tl.arange(0, M)[None, :], (N, M))
    col = tl.broadcast_to(tl.arange(0, N)[:, None], (N, M))
    ptrs = tle.gpu.local_ptr(smem, (row, col))
    vals = tl.load(ptrs)

    out_offs = tl.arange(0, N)[:, None] * stride_om \
        + tl.arange(0, M)[None, :] * stride_on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Case D: memdesc_reshape — 1D buffer with multi-dim index
# ---------------------------------------------------------------------------

@triton.jit
def _memdesc_reshape_kernel(x_ptr, out_ptr, stride_om, stride_on,
                            TOTAL: tl.constexpr,
                            M: tl.constexpr, N: tl.constexpr):
    smem = tle.gpu.alloc((TOTAL,), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs = tl.arange(0, TOTAL)
    x_tile = x_ptr + offs
    tle.gpu.copy(x_tile, smem, [TOTAL])

    # 1D buffer accessed with 2D index -> canonicalizeReshape
    # creates memdesc_reshape (1D -> 2D)
    flat_idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    ptrs = tle.gpu.local_ptr(smem, (flat_idx,))
    vals = tl.load(ptrs)

    out_offs = tl.arange(0, M)[:, None] * stride_om \
        + tl.arange(0, N)[None, :] * stride_on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Path B: lowerSubViewAccess — unaligned offset fallback
# ---------------------------------------------------------------------------

@triton.jit
def _unaligned_offset_kernel(x_ptr, out_ptr, stride_xm, stride_xn,
                             stride_om, stride_on,
                             M: tl.constexpr, N: tl.constexpr,
                             OFF: tl.constexpr, S: tl.constexpr):
    smem = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, [M, N])

    # Unaligned offset: OFF not aligned to S -> canonicalizeSubslice
    # rejects (MemDescSubsliceOp verifier) -> falls through to Path B
    # (memdesc_to_ptr + pointer arithmetic)
    row = tl.broadcast_to(tl.arange(0, M)[:, None], (M, S))
    col = tl.broadcast_to((OFF + tl.arange(0, S))[None, :], (M, S))
    ptrs = tle.gpu.local_ptr(smem, (row, col))
    vals = tl.load(ptrs)

    out_offs = tl.arange(0, M)[:, None] * stride_om \
        + tl.arange(0, S)[None, :] * stride_on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Path B: masked access — non-trivial mask -> Path B
# ---------------------------------------------------------------------------

@triton.jit
def _masked_access_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem = tle.gpu.alloc((BLOCK,), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Non-trivial mask -> optimizeLocalPointerOps skips -> Path B
    tl.store(ptrs, x_vals, mask=mask)
    out_vals = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


# ---------------------------------------------------------------------------
# Path B: scalar dynamic index -> barrier insertion
# ---------------------------------------------------------------------------

@triton.jit
def _scalar_dynamic_index_kernel(out_ptr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc((BLOCK,), dtype=tl.int32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    vec_idx = tl.arange(0, BLOCK)
    vec_ptr = tle.gpu.local_ptr(smem, (vec_idx,))
    tl.store(vec_ptr, vec_idx + 1)
    zero = tl.program_id(0) * 0

    # Scalar dynamic index in loop -> Path B + gpu.barrier
    for i in range(BLOCK):
        scalar_idx = zero + i
        scalar_ptr = tle.gpu.local_ptr(smem, (scalar_idx,))
        scalar_val = tl.load(scalar_ptr)
        tl.store(out_ptr + i, scalar_val)


# ---------------------------------------------------------------------------
# Path A: trivially-true mask -> local_load / local_store (strip mask)
# ---------------------------------------------------------------------------

@triton.jit
def _trivially_true_mask_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc((BLOCK,), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))

    x_vals = tl.load(x_ptr + tl.arange(0, BLOCK))
    # Trivially-true mask: all-true constant -> Path A strips it.
    true_mask = tl.full((BLOCK,), True, tl.int1)
    tl.store(ptrs, x_vals, mask=true_mask)
    out_vals = tl.load(ptrs, mask=true_mask, other=0.0)
    tl.store(out_ptr + tl.arange(0, BLOCK), out_vals)


# ---------------------------------------------------------------------------
# Path B: gather reshape (non-iota flat index) -> fallback
# ---------------------------------------------------------------------------

@triton.jit
def _gather_reshape_kernel(x_ptr, out_ptr, stride_om, stride_on,
                           TOTAL: tl.constexpr,
                           M: tl.constexpr, N: tl.constexpr):
    smem = tle.gpu.alloc((TOTAL,), dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs = tl.arange(0, TOTAL)
    x_tile = x_ptr + offs
    tle.gpu.copy(x_tile, smem, [TOTAL])

    # Gather: reversed index — NOT row-major iota.
    # canonicalizeReshape rejects (isRowMajorIota fails) -> Path B.
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    forward = row * N + col
    flat_idx = (TOTAL - 1) - forward
    ptrs = tle.gpu.local_ptr(smem, (flat_idx,))
    vals = tl.load(ptrs)

    out_offs = row * stride_om + col * stride_on
    tl.store(out_ptr + out_offs, vals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTLELocalPointers:
    """Verify all tle.local_pointers code paths on the GCU backend."""

    # -- Path A: full-block access ------------------------------------------

    def test_full_view_1d_path_a(self):
        BLOCK = 64
        numel = BLOCK * 3
        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        grid = (triton.cdiv(numel, BLOCK),)
        _full_view_1d_kernel[grid](x, out, numel, BLOCK=BLOCK)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_full_view_1d_ir_no_memdesc_to_ptr(self):
        BLOCK = 64
        out = torch.empty(BLOCK, device="cuda", dtype=torch.float32)
        compiled = _full_view_1d_kernel.warmup(
            torch.empty(BLOCK, device="cuda", dtype=torch.float32), out,
            BLOCK, BLOCK=BLOCK, grid=(1,))
        ir = _get_ir(compiled)
        assert "memdesc_to_ptr" not in ir
        assert "gcu.memref2ptr" in ir or "slice_start" in ir or "deslice" in ir

    def test_full_view_2d_path_a(self):
        M, N = 16, 32
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        _full_view_2d_kernel[(1,)](
            x, out, x.stride(0), x.stride(1),
            out.stride(0), out.stride(1), M=M, N=N)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_full_view_2d_ir_no_memdesc_to_ptr(self):
        M, N = 16, 32
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        compiled = _full_view_2d_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, N, 1, M=M, N=N, grid=(1,))
        ir = _get_ir(compiled)
        assert "local_load" in ir
        assert "memdesc_to_ptr" not in ir

    # -- Case A: memdesc_index ---------------------------------------------

    def test_memdesc_index_correctness(self):
        M, N = 4, 16
        ROW = 2
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.empty(N, device="cuda", dtype=torch.float32)
        _memdesc_index_kernel[(1,)](
            x, out, x.stride(0), x.stride(1), 1, 1, M=M, N=N, ROW=ROW)
        torch.testing.assert_close(out, x[ROW, :], atol=1e-6, rtol=1e-6)

    def test_memdesc_index_ir(self):
        M, N = 4, 16
        out = torch.empty(N, device="cuda", dtype=torch.float32)
        compiled = _memdesc_index_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, 1, 1, M=M, N=N, ROW=2, grid=(1,))
        ir = _get_ir(compiled)
        assert "memdesc_to_ptr" not in ir, (
            "Case A should use memdesc_index, not memdesc_to_ptr (Path B)")
        # DTE fold: memdesc_index should be folded into copy_global_to_local
        # via runtime pointer adjustment, producing slice_start (DTE copy).
        assert "slice_start" in ir, (
            "Case A DTE fold should produce slice_start")

    # -- Case B: memdesc_subslice (aligned offset) -------------------------

    def test_memdesc_subslice_correctness(self):
        # OFF=8 aligned to S=8 (8 & 7 == 0) -> creates memdesc_subslice
        M, N, OFF, S = 8, 16, 8, 8
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.empty(M, S, device="cuda", dtype=torch.float32)
        _memdesc_subslice_kernel[(1,)](
            x, out, x.stride(0), x.stride(1),
            out.stride(0), out.stride(1), M=M, N=N, OFF=OFF, S=S)
        torch.testing.assert_close(
            out, x[:, OFF:OFF + S], atol=1e-6, rtol=1e-6)

    def test_memdesc_subslice_ir(self):
        # OFF=8 aligned to S=8 -> canonicalizeSubslice creates memdesc_subslice
        # -> TTMemDescSubsliceOpLowering lowers via memref.subview.
        M, N, OFF, S = 8, 16, 8, 8
        out = torch.empty(M, S, device="cuda", dtype=torch.float32)
        compiled = _memdesc_subslice_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, S, 1, M=M, N=N, OFF=OFF, S=S, grid=(1,))
        ir = _get_ir(compiled)
        assert "memdesc_to_ptr" not in ir, (
            "Case B (aligned) should use memdesc_subslice, not Path B")
        assert "memdesc_subslice" in ir, (
            "Case B should produce memdesc_subslice border annotation")

    def test_memdesc_subslice_unaligned_path_b(self):
        # OFF=4 not aligned to S=8 (4 & 7 = 4 != 0) -> canonicalizeSubslice
        # skips alignment check -> Path B fallback (pointer arithmetic).
        M, N, OFF, S = 8, 16, 4, 8
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.empty(M, S, device="cuda", dtype=torch.float32)
        _memdesc_subslice_kernel[(1,)](
            x, out, x.stride(0), x.stride(1),
            out.stride(0), out.stride(1), M=M, N=N, OFF=OFF, S=S)
        torch.testing.assert_close(
            out, x[:, OFF:OFF + S], atol=1e-6, rtol=1e-6)

    def test_memdesc_subslice_unaligned_ir(self):
        # OFF=4 not aligned -> Path B fallback.
        M, N, OFF, S = 8, 16, 4, 8
        out = torch.empty(M, S, device="cuda", dtype=torch.float32)
        compiled = _memdesc_subslice_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, S, 1, M=M, N=N, OFF=OFF, S=S, grid=(1,))
        ir = _get_ir(compiled)
        assert "gcu.memref2ptr" in ir, (
            "Unaligned offset should use Path B (gcu.memref2ptr)")

    # -- Case C: memdesc_trans + B2 (DTE fold) ------------------------------

    def test_memdesc_trans_correctness(self):
        M, N = 8, 16
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.empty(N, M, device="cuda", dtype=torch.float32)
        _memdesc_trans_kernel[(1,)](
            x, out, x.stride(0), x.stride(1),
            out.stride(0), out.stride(1), M=M, N=N)
        torch.testing.assert_close(out, x.t(), atol=1e-6, rtol=1e-6)

    def test_memdesc_trans_ir(self):
        M, N = 8, 16
        out = torch.empty(N, M, device="cuda", dtype=torch.float32)
        compiled = _memdesc_trans_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, M, 1, M=M, N=N, grid=(1,))
        ir = _get_ir(compiled)
        assert "memdesc_to_ptr" not in ir, (
            "Case C should use memdesc_trans, not Path B")
        # B2 DTE fold: memdesc_trans folded into copy_global_to_local,
        # DTE writes transposed layout directly via SliceTransposeStartOp.
        assert "slice_transpose" in ir.lower() or "memdesc_trans" in ir, (
            "Case C B2 fold should produce transpose DTE op")

    # -- Case D: memdesc_reshape --------------------------------------------

    def test_memdesc_reshape_correctness(self):
        M, N = 8, 16
        TOTAL = M * N
        x = torch.randn(TOTAL, device="cuda", dtype=torch.float32)
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        _memdesc_reshape_kernel[(1,)](
            x, out, out.stride(0), out.stride(1),
            TOTAL=TOTAL, M=M, N=N)
        torch.testing.assert_close(out, x.reshape(M, N), atol=1e-6, rtol=1e-6)

    def test_memdesc_reshape_ir(self):
        M, N = 8, 16
        TOTAL = M * N
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        compiled = _memdesc_reshape_kernel.warmup(
            torch.empty(TOTAL, device="cuda", dtype=torch.float32), out,
            N, 1, TOTAL=TOTAL, M=M, N=N, grid=(1,))
        ir = _get_ir(compiled)
        assert "memdesc_to_ptr" not in ir, (
            "Case D should use memdesc_reshape, not Path B")
        # DTE fold: memdesc_reshape folded into copy_global_to_local.
        assert "slice_start" in ir, (
            "Case D DTE fold should produce slice_start")

    # -- Path B: unaligned offset ------------------------------------------

    def test_unaligned_offset_path_b(self):
        M, N, OFF, S = 8, 16, 1, 8  # OFF=1 not aligned to S=8 -> Path B
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.empty(M, S, device="cuda", dtype=torch.float32)
        _unaligned_offset_kernel[(1,)](
            x, out, x.stride(0), x.stride(1),
            out.stride(0), out.stride(1), M=M, N=N, OFF=OFF, S=S)
        torch.testing.assert_close(
            out, x[:, OFF:OFF + S], atol=1e-6, rtol=1e-6)

    def test_unaligned_offset_ir_uses_path_b(self):
        M, N, OFF, S = 8, 16, 1, 8
        out = torch.empty(M, S, device="cuda", dtype=torch.float32)
        compiled = _unaligned_offset_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, S, 1, M=M, N=N, OFF=OFF, S=S, grid=(1,))
        ir = _get_ir(compiled)
        assert "gcu.memref2ptr" in ir, (
            "Unaligned offset should use Path B (gcu.memref2ptr)")

    # -- Path B: masked access ---------------------------------------------

    def test_masked_access_path_b(self):
        BLOCK = 64
        numel = BLOCK * 3 - 10
        x = torch.randn(numel + 10, device="cuda", dtype=torch.float32)
        out = torch.zeros_like(x)
        grid = (triton.cdiv(BLOCK * 3, BLOCK),)
        _masked_access_kernel[grid](x, out, numel, BLOCK=BLOCK)
        expected = x.clone()
        expected[numel:] = 0.0
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_masked_access_ir_uses_path_b(self):
        BLOCK = 64
        numel = BLOCK * 3 - 10
        out = torch.zeros(numel + 10, device="cuda", dtype=torch.float32)
        compiled = _masked_access_kernel.warmup(
            torch.empty(numel + 10, device="cuda", dtype=torch.float32), out,
            numel, BLOCK=BLOCK, grid=(1,))
        ir = _get_ir(compiled)
        assert "gcu.memref2ptr" in ir, (
            "Non-trivial mask should use Path B (gcu.memref2ptr)")

    # -- Path B: scalar dynamic index -> barrier ----------------------------

    def test_scalar_dynamic_index_inserts_barrier(self):
        BLOCK = 64
        out = torch.empty(BLOCK, device="cuda", dtype=torch.int32)
        compiled = _scalar_dynamic_index_kernel.warmup(
            out, BLOCK=BLOCK, grid=(1,))
        ir = _get_ir(compiled)
        assert "gpu.barrier" in ir, (
            "Scalar dynamic index after vector store must insert barrier")
        assert '"tt.reduce"' not in ir

        _scalar_dynamic_index_kernel[(1,)](out, BLOCK=BLOCK)
        torch.testing.assert_close(
            out, torch.arange(1, BLOCK + 1, device="cuda", dtype=torch.int32),
            atol=0, rtol=0)

    # -- Path A: trivially-true mask -> Path A (strip mask) ----------------

    def test_trivially_true_mask_path_a(self):
        BLOCK = 64
        x = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        _trivially_true_mask_kernel[(1,)](x, out, BLOCK=BLOCK)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_trivially_true_mask_ir(self):
        BLOCK = 64
        out = torch.empty(BLOCK, device="cuda", dtype=torch.float32)
        compiled = _trivially_true_mask_kernel.warmup(
            torch.empty(BLOCK, device="cuda", dtype=torch.float32), out,
            BLOCK=BLOCK, grid=(1,))
        ir = _get_ir(compiled)
        # Trivially-true mask should use Path A (no memdesc_to_ptr).
        assert "memdesc_to_ptr" not in ir, (
            "Trivially-true mask should use Path A, not Path B")
        assert "slice_start" in ir or "deslice" in ir, (
            "Trivially-true mask should produce DTE ops (Path A)")

    # -- Case C extension: transposed offset sub-region ---------------------

    def test_memdesc_trans_offset_correctness(self):
        M, N, OFF0, OFF1, S0, S1 = 8, 16, 0, 8, 8, 8
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.zeros(S1, S0, device="cuda", dtype=torch.float32)
        grid = (1,)
        _memdesc_trans_offset_kernel[grid](
            x, out, x.stride(0), x.stride(1), out.stride(0), out.stride(1),
            M=M, N=N, OFF0=OFF0, OFF1=OFF1, S0=S0, S1=S1)
        # out[r, c] = x[OFF0 + c, OFF1 + r]
        torch.testing.assert_close(out, x[OFF0:OFF0 + S0, OFF1:OFF1 + S1].t(),
                                   atol=1e-6, rtol=1e-6)

    def test_memdesc_trans_offset_ir(self):
        M, N, OFF0, OFF1, S0, S1 = 8, 16, 0, 8, 8, 8
        out = torch.zeros(S1, S0, device="cuda", dtype=torch.float32)
        compiled = _memdesc_trans_offset_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, N, 1, M=M, N=N, OFF0=OFF0, OFF1=OFF1, S0=S0, S1=S1,
            grid=(1,))
        ir = _get_ir(compiled)
        # ON mode: the subslice + trans pair is folded into the DTE copy
        # (single copy_global_to_local with the sub-region offsets baked in)
        # and no local_load on a view op remains.
        assert "ttg.memdesc_trans" not in ir, (
            "transposed offset sub-region should fold into the DTE copy")

    # -- Problem-1 regression: compute-filled smem + sub-tile ---------------

    def test_compute_fill_subtile_correctness(self):
        M, N, OFF, S = 8, 16, 4, 4
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.zeros(M, S, device="cuda", dtype=torch.float32)
        grid = (1,)
        _compute_fill_subtile_kernel[grid](
            x, out, x.stride(0), x.stride(1), out.stride(0), out.stride(1),
            M=M, N=N, OFF=OFF, S=S)
        torch.testing.assert_close(out, 2.0 * x[:, OFF:OFF + S],
                                   atol=1e-6, rtol=1e-6)

    def test_compute_fill_subtile_ir_stages_view(self):
        M, N, OFF, S = 8, 16, 4, 4
        out = torch.zeros(M, S, device="cuda", dtype=torch.float32)
        compiled = _compute_fill_subtile_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, N, 1, M=M, N=N, OFF=OFF, S=S, grid=(1,))
        ir = _get_ir(compiled)
        # The unfused non-contiguous sub-tile local_load must be staged with
        # strided memref.load (stageViewToContiguous), never silently
        # reinterpreted as identity (which drops the view offset/strides).
        assert "memref.load" in ir and "strided" in ir, (
            "compute-filled sub-tile read must stage the strided view")

    def test_subslice_masked_correctness(self):
        M, N, OFF, S = 8, 16, 4, 4
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = torch.zeros(M, S, device="cuda", dtype=torch.float32)
        grid = (1,)
        _subslice_masked_kernel[grid](
            x, out, x.stride(0), x.stride(1), out.stride(0), out.stride(1),
            M=M, N=N, OFF=OFF, S=S)
        expected = x.clone()
        expected[:, OFF + S // 2:OFF + S] = 0.0
        torch.testing.assert_close(out, expected[:, OFF:OFF + S],
                                   atol=1e-6, rtol=1e-6)

    def test_subslice_masked_ir_folds_view_offset(self):
        M, N, OFF, S = 8, 16, 4, 4
        out = torch.zeros(M, S, device="cuda", dtype=torch.float32)
        compiled = _subslice_masked_kernel.warmup(
            torch.empty(M, N, device="cuda", dtype=torch.float32), out,
            N, 1, N, 1, M=M, N=N, OFF=OFF, S=S, grid=(1,))
        ir = _get_ir(compiled)
        # A non-trivially masked load gates OFF Cases B/C/D
        # (hasNonTriviallyMaskedLoadUser): the masked Path B load is later
        # converted to a DTE transfer whose size derivation assumes a ROOT
        # buffer base, so a mid-buffer view base would read out of bounds.
        # The IR must therefore keep the root base + offset-carrying index
        # arithmetic: no memdesc_subslice view, and the root row stride
        # (dense<N>) multiplies the row index in the pointer arithmetic.
        assert "ttg.memdesc_subslice" not in ir, (
            "masked sub-tile access must NOT canonicalize to a memdesc view "
            "(DTE transfer would read out of bounds from the view base)")
        assert f"value = {N} : index" in ir, (
            "masked sub-tile Path B must use the ROOT row stride in the "
            "pointer arithmetic")


    # -- Path B: gather reshape (non-iota) -> fallback ----------------------

    def test_gather_reshape_path_b(self):
        M, N = 8, 16
        TOTAL = M * N
        x = torch.randn(TOTAL, device="cuda", dtype=torch.float32)
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        _gather_reshape_kernel[(1,)](
            x, out, out.stride(0), out.stride(1),
            TOTAL=TOTAL, M=M, N=N)
        # Gather: out[i,j] = x[TOTAL - 1 - (i*N + j)]
        row = torch.arange(M, device="cuda")[:, None]
        col = torch.arange(N, device="cuda")[None, :]
        forward = row * N + col
        flat_idx = (TOTAL - 1) - forward
        torch.testing.assert_close(
            out, x[flat_idx.cpu()], atol=1e-6, rtol=1e-6)

    def test_gather_reshape_ir_uses_path_b(self):
        M, N = 8, 16
        TOTAL = M * N
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        compiled = _gather_reshape_kernel.warmup(
            torch.empty(TOTAL, device="cuda", dtype=torch.float32), out,
            N, 1, TOTAL=TOTAL, M=M, N=N, grid=(1,))
        ir = _get_ir(compiled)
        # Non-iota gather should fall to Path B (gcu.memref2ptr).
        assert "gcu.memref2ptr" in ir, (
            "Gather reshape should use Path B (gcu.memref2ptr)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
