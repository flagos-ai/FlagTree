"""TLE-Struct buffered_tensor view API on the GCU backend.

Covers the view-producing methods of ``tle.gpu.buffered_tensor``
(``slot`` / ``subslice`` / ``load`` / ``store`` / ``reshape``) used as a DTE
destination, i.e. ``tle.gpu.copy(src, view, shape)``.

Implementation side (kurama):
  - TritonLoadStoreToDma.cpp        : the <= 8KB rank-1 load must NOT take the
                                     elementwise-fusion bypass when its only
                                     consumers are stores into *contiguous*
                                     shared-memory views
                                     (``hasOnlySharedStoreUsers``)
  - TritonGCULocalMemOptimize.cpp   : ``ttg.local_store`` + ``triton_gcu.load``
                                     fusion into ``copy_global_to_local``, with
                                     a contiguity guard on the dst view
                                     (``isContiguousMemDescView``)
  - TTSmemLowerToGCU.cpp            : view offset -> DTE dst address; staging
                                     through a contiguous scratch buffer when a
                                     fusion was not possible
  - TritonToGCU.cpp                 : ``GCUStoreOpLowering`` rejects a DTE
                                     store whose destination shared-memory
                                     view is not contiguous

Two behaviours are pinned here:

  * a copy into a contiguous view must stay a DTE (``memref_ext.slice_*`` and no
    ``memref_ext.deslice_start``, which is the
    "maskedload + TAR + private chunk + deslice" degradation);
  * a copy into a NON-contiguous view is not fused into a DTE -- no DTE can
    describe the view strides -- but still produces the correct result: the
    lowering stages the value through a contiguous scratch buffer and writes
    the view back element-wise.

Note: ``memref_ext.slice_deslice_start`` is a normal part of the DTE sequence.
``memref_ext.deslice_start`` is *also* emitted by a 2-D full-block SMEM
read-back, so the "no deslice_start" assertion is only applied to kernels whose
SMEM traffic is 1-D.
"""

import importlib.util
import os

import pytest
import torch
import triton
import triton.language as tl

if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton  # noqa: F401  (register GCU backend)
from torch_gcu import transfer_to_gcu  # noqa: F401  (enables .gcu() on tensors)
import triton.experimental.tle.language as tle

# Recompile on every launch: the gcuir assertions must look at freshly lowered
# IR, never at a cached binary.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

# Kernel-visible constant: must be a tl.constexpr to be captured by @jit.
SENTINEL = tl.constexpr(-7.0)


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


def _assert_dte_into_view(compiled, strict=True):
    """The copy must lower to a shared DTE whose dst is the view."""
    ir = _get_ir(compiled)
    # ConfigGcuLoad picks one of the slice_* variants depending on rank /
    # broadcast / mask, so accept any of them and only reject the
    # elementwise fallback below.
    dte_markers = ("memref_ext.slice_start", "memref_ext.slice_pad_start",
                   "memref_ext.slice_broadcast_start")
    assert any(marker in ir for marker in dte_markers), (
        "copy into a buffered_tensor view must go through the DTE "
        "(no memref_ext.slice_* op found)")
    if strict:
        assert "memref_ext.deslice_start" not in ir, (
            "copy into a buffered_tensor view degraded into the elementwise "
            "fallback (maskedload + TAR + private chunk + deslice)")
    # The DTE lowering tags the fused GMEM -> SMEM copy with a border carrying
    # the name of the op it came from, so the marker proves the fusion happened.
    assert 'enter = "triton_gcu.copy_global_to_local"' in ir, (
        "the copy was not fused into a DTE (no copy_global_to_local border)")


def _sentinel(*shape):
    return torch.full(shape, SENTINEL.value, device="cuda", dtype=torch.float32)


# ---------------------------------------------------------------------------
# slot: alloc((2, N)) + buf.slot(stage) as DTE destination
# ---------------------------------------------------------------------------

@triton.jit
def _slot_copy_kernel(x_ptr, fill_ptr, out_ptr, stage, N: tl.constexpr):
    buf = tle.gpu.alloc((2, N), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs = tl.arange(0, N)
    # Fill every stage with a sentinel through the DTE (a compute-filled store
    # would add its own TAR/deslice traffic and hide what we assert below).
    r2 = tl.arange(0, 2)[:, None]
    rn = tl.arange(0, N)[None, :]
    tle.gpu.copy(fill_ptr + r2 * N + rn, buf, [2, N])

    # `gpu.copy` takes a POINTER tensor on the GMEM side; the load it
    # generates has the view store as its only user, which is exactly the
    # pattern the DTE exemption in TritonLoadStoreToDma.cpp targets.
    slot = buf.slot(stage)
    tle.gpu.copy(x_ptr + offs, slot, [N])

    tl.store(out_ptr + offs, tl.load(tle.gpu.local_ptr(buf.slot(0), (offs,))))
    tl.store(out_ptr + N + offs, tl.load(tle.gpu.local_ptr(slot, (offs,))))


def _run_slot_copy(n):
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    out = torch.empty(2 * n, device="cuda", dtype=torch.float32)
    compiled = _slot_copy_kernel[(1,)](x, _sentinel(2, n), out, 1, N=n)
    expected = torch.empty_like(out)
    expected[:n] = SENTINEL.value
    expected[n:] = x
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    return compiled


class TestBufferedTensorView:
    """TLE-Struct buffered_tensor subview API on GCU400."""

    # -- slot (memdesc_index, runtime stage) --------------------------------

    def test_slot_copy_small(self):
        """N * 4 bytes <= 8192 used to hit the elementwise-fusion bypass."""
        _run_slot_copy(256)

    def test_slot_copy_large(self):
        """Large copy: already DTE before this change, kept as a regression."""
        _run_slot_copy(8192)

    def test_slot_copy_ir_is_dte(self):
        compiled = _run_slot_copy(256)
        _assert_dte_into_view(compiled)

    # -- subslice (memdesc_subslice, compile-time offsets) ------------------

    @staticmethod
    def _subslice_1d(x, total, start, length):
        out = torch.empty(total, device="cuda", dtype=torch.float32)
        compiled = _subslice_1d_kernel[(1,)](x, _sentinel(total), out,
                                            TOTAL=total, START=start,
                                            LEN=length)
        expected = _sentinel(total)
        expected[start:start + length] = x
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        return compiled

    def test_subslice_1d_copy_second_half(self):
        total, length = 2048, 1024
        x = torch.arange(length, device="cuda", dtype=torch.float32)
        self._subslice_1d(x, total, total - length, length)

    def test_subslice_1d_copy_offset_preserved(self):
        # The offset of a memdesc subslice must be a multiple of its length,
        # so this is not the trivial "second half" case: the slice starts at
        # 512 of 2048 and is followed by 1024 untouched elements.
        total, length, start = 2048, 512, 512
        x = torch.arange(length, device="cuda", dtype=torch.float32)
        self._subslice_1d(x, total, start, length)

    def test_subslice_1d_copy_large(self):
        """32KB copy: already DTE before this change, kept as a regression."""
        total, length = 16384, 8192
        x = torch.arange(length, device="cuda", dtype=torch.float32)
        self._subslice_1d(x, total, length, length)

    def test_subslice_1d_copy_ir_is_dte(self):
        total, length = 2048, 1024
        x = torch.arange(length, device="cuda", dtype=torch.float32)
        compiled = self._subslice_1d(x, total, total - length, length)
        _assert_dte_into_view(compiled)

    # -- subslice on dim 0: still contiguous, must stay a DTE ---------------

    def test_subslice_2d_dim0_copy(self):
        m, n, start, size = 4, 64, 2, 2
        x = torch.arange(size * n, device="cuda", dtype=torch.float32).reshape(
            size, n)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        compiled = _subslice_dim0_kernel[(1,)](x, _sentinel(m, n), out, M=m,
                                              N=n, START=start, S=size)
        expected = _sentinel(m, n)
        expected[start:start + size, :] = x
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        # The 2-D full-block read-back emits deslice_start for the register
        # deslice, so only the DTE marker is asserted here; the exact lowering
        # of the copy is pinned by
        # triton_to_gcu_copy_global_to_local_gcu400.mlir.
        _assert_dte_into_view(compiled, strict=False)

    # -- subslice on the inner dim: non-contiguous views -------------------

    def test_subslice_2d_dim1_copy_non_contiguous(self):
        """Inner-dim sub-tile: not DTE-fusable, but correct.

        `subslice(0, S, dim=1)` of an (M, N) buffer with S < N has row stride N
        but width S, so no single DTE can describe the destination. The copy
        must therefore not be fused into `copy_global_to_local` (the local-mem
        fusion guards the dst view with `isContiguousMemDescView`), yet it must
        still produce the right tile: the lowering stages the value through a
        contiguous scratch buffer and writes the view back element-wise.
        """
        m, n, size = 4, 64, 32
        row = torch.arange(m, device="cuda")[:, None]
        col = torch.arange(n, device="cuda")[None, :]
        # The kernel addresses the source with row stride N, so the tensor must
        # be allocated as (M, N); the copied tile is its first S columns.
        x = (row * n + col).to(torch.float32)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        compiled = _subslice_dim1_kernel[(1,)](x, out, M=m, N=n, S=size)
        expected = _sentinel(m, n)
        expected[:, :size] = x[:, :size]
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        ir = _get_ir(compiled)
        assert 'enter = "triton_gcu.copy_global_to_local"' not in ir, (
            "a copy into a non-contiguous view must not be fused into a DTE; "
            "the staging path must handle it")

    # -- buffered_tensor.store / .load -------------------------------------

    def test_buffer_load_store_roundtrip(self):
        n = 1024
        x = torch.arange(n, device="cuda", dtype=torch.float32)
        out = torch.empty(n, device="cuda", dtype=torch.float32)
        _buffer_load_store_kernel[(1,)](x, out, N=n)
        torch.testing.assert_close(out, x, atol=0, rtol=0)

    # -- buffered_tensor.reshape -------------------------------------------

    def test_buffer_reshape_roundtrip(self):
        m, n = 32, 64
        total = m * n
        x = torch.arange(total, device="cuda",
                         dtype=torch.float32).reshape(m, n)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        compiled = _reshape_kernel[(1,)](x, out, TOTAL=total, M=m, N=n)
        torch.testing.assert_close(out, x, atol=0, rtol=0)
        # 2-D store/load on the reshaped view: see the note in
        # test_subslice_2d_dim0_copy for the non-strict assertion.
        _assert_dte_into_view(compiled, strict=False)


@triton.jit
def _subslice_1d_kernel(x_ptr, fill_ptr, out_ptr, TOTAL: tl.constexpr,
                        START: tl.constexpr, LEN: tl.constexpr):
    buf = tle.gpu.alloc((TOTAL,), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    all_idx = tl.arange(0, TOTAL)
    tle.gpu.copy(fill_ptr + all_idx, buf, [TOTAL])

    sub = buf.subslice(START, LEN)
    tle.gpu.copy(x_ptr + tl.arange(0, LEN), sub, [LEN])

    tl.store(out_ptr + all_idx, tl.load(tle.gpu.local_ptr(buf, (all_idx,))))


@triton.jit
def _subslice_dim0_kernel(x_ptr, fill_ptr, out_ptr, M: tl.constexpr,
                          N: tl.constexpr, START: tl.constexpr,
                          S: tl.constexpr):
    buf = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    tle.gpu.copy(fill_ptr + row * N + col, buf, [M, N])

    sub = buf.subslice(START, S, dim=0)
    tle.gpu.copy(x_ptr + tl.arange(0, S)[:, None] * N + col, sub, [S, N])

    tl.store(out_ptr + row * N + col, tl.load(tle.gpu.local_ptr(buf)))


@triton.jit
def _subslice_dim1_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                          S: tl.constexpr):
    buf = tle.gpu.alloc((M, N), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    # A constant fill keeps this kernel's ONLY `tle.gpu.copy` the non-contiguous
    # one, so the "no copy_global_to_local" assertion below is unambiguous.
    buf.store(tl.full((M, N), SENTINEL.value, tl.float32))

    # Inner-dim sub-tile: row stride N but width S -> non-contiguous view.
    sub = buf.subslice(0, S, dim=1)
    tle.gpu.copy(x_ptr + row * N + tl.arange(0, S)[None, :], sub, [M, S])

    tl.store(out_ptr + row * N + col, tl.load(tle.gpu.local_ptr(buf)))


@triton.jit
def _buffer_load_store_kernel(x_ptr, out_ptr, N: tl.constexpr):
    buf = tle.gpu.alloc((N,), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    buf.store(tl.load(x_ptr + tl.arange(0, N)))
    tl.store(out_ptr + tl.arange(0, N), buf.load())


@triton.jit
def _reshape_kernel(x_ptr, out_ptr, TOTAL: tl.constexpr, M: tl.constexpr,
                    N: tl.constexpr):
    buf = tle.gpu.alloc((TOTAL,), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = buf.reshape((M, N))
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    view.store(tl.load(x_ptr + offs))
    tl.store(out_ptr + offs, view.load())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
