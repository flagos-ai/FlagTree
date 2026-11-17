"""``tle.gpu.memory_space(...)`` on the GCU backend.

``tle.gpu.memory_space(value, space)`` attaches the ``tt.memory_space`` hint to
the defining op of ``value``. Nothing in the GCU pipeline used to consume it,
so the hint was silently dropped. ``gcu-tle-assign-memory-space``
(``Transforms/GCUTleAssignMemorySpace.cpp``) now materializes it, in a single
run placed after ``gcu-triton-fusion`` and before
``triton-gpu-local-mem-optimize`` (see that pass for why this one position
satisfies the load and the compute constraints alike).

``shared_memory``
    The hinted value is staged through a ``ttg.local_alloc(src)`` +
    ``ttg.local_load`` pair - the same value-level form of the blocked -> shared
    staging that ``rewriteLocalLoad`` produces for an explicit staging, so the
    existing ``TTSmemLocalAllocOpLowering`` / ``TTSmemLocalLoadOpLowering`` code
    paths apply unchanged. On a load result the pair is created right before
    ``triton-gpu-local-mem-optimize``, so that pass folds it into a
    ``copy_global_to_local``. On a computed value the producer lowering writes
    straight into the shared tile through a per-warp view, so each subthread
    writes its own subview and the private buffer + deslice copy is skipped.

``register``
    The producer's output buffer becomes a ``memref.alloca``
    (``vector.load``/``vector.store``) instead of a ``memref.alloc`` (the
    ``gcu.tar_load``/``gcu.tar_store`` path). This is only meaningful for a
    producer whose lowering writes its result with plain memref stores, i.e.
    the elementwise (fused or not) path; other producers ignore the marker and
    keep their default buffer.

``gcu-triton-fusion`` rewrites the value's definition: the hinted op is cloned
into a ``triton_gcu.elementwise_fusion_region`` and, when the value is consumed
by an op that is not fused (here: ``tt.reduce``), it leaves the region as a
region result. The hint is consequently applied to the region result. When the
consumer *is* fused in - a plain store, for instance - the value never escapes
the region, there is nothing to stage, and the hint is dropped with a remark
(the elementwise lowering already keeps the value in registers). The
elementwise tests below therefore feed the hinted value into a ``tl.sum`` so
that the hint is actually exercised, and use a shape whose per-thread run is
longer than the fusion pass' full-unroll limit: that is exactly the case where
``register`` changes the default decision from the tar path to ``memref.alloca``
and ``shared_memory`` from the private buffer to the tile view.

The fast path is only taken under the same condition the lowering already needs
for an alloca output (at least one full vector-length run per thread), so a
small block would silently fall back to the private-buffer path and the IR
assertions below would be vacuous.
"""

import importlib.util
import os
import re

import pytest
import torch
import triton
import triton.language as tl

if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton  # noqa: F401  (register GCU backend)
from torch_gcu import transfer_to_gcu  # noqa: F401  (enables .gcu() on tensors)
import triton.experimental.tle.language as tle

# Recompile on every launch: the IR assertions must look at freshly lowered IR,
# never at a cached binary.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

# 32768 f32 elements with 4 warps puts 256..128 elements in each subthread,
# i.e. at least one 512B vector per iteration for both 32- and 64-wide warps.
BLOCK = 32768
NUM_WARPS = 4
# Elementwise -> reduce: 16384 f32 elements put 4096 in each subthread (32 rows
# x 128 columns), i.e. 32 vectors per thread - more than the 16 the fusion pass
# is willing to fully unroll - while the last axis (128 elements * 4B = 512B)
# is still a legal alloca consumer for the reduce.
ELT_ROWS = 128
ELT_COLS = 128
# Hinted reduce result: small enough that the whole tile stays in one
# allocation.
REDUCE_ROWS = 64
REDUCE_COLS = 64


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


@triton.jit
def _store_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr, HINT: tl.constexpr,
                  SPACE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs) * 2.0 + 1.0
    if HINT:
        v = tle.gpu.memory_space(v, SPACE)
    tl.store(o_ptr + offs, v)


@triton.jit
def _eltwise_reduce_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr,
                           HINT: tl.constexpr, SPACE: tl.constexpr):
    rows = tl.arange(0, R)
    cols = tl.arange(0, C)
    v = tl.load(x_ptr + rows[:, None] * C + cols[None, :]) * 2.0 + 1.0
    if HINT:
        v = tle.gpu.memory_space(v, SPACE)
    r = tl.sum(v, axis=1)
    tl.store(o_ptr + rows, r)


@triton.jit
def _sum_combine(a, b):
    return a + b


@triton.jit
def _reduce_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr,
                   HINT: tl.constexpr):
    rows = tl.arange(0, R)
    cols = tl.arange(0, C)
    x = tl.load(x_ptr + rows[:, None] * C + cols[None, :])
    # ``tl.reduce`` with an explicit combine instead of ``tl.sum``: the hint is
    # attached to the defining op of the value, and a value returned from a
    # @triton.jit helper (``tl.sum`` is one) loses the attribute when the
    # frontend inlines the call (the attribute would sit on the call result,
    # which the inliner replaces). The reduce op built here lives in the
    # kernel body, so the hint reaches the backend.
    r = tl.reduce(x, 1, _sum_combine)
    if HINT:
        # The hint is attached to the tt.reduce, whose output buffer is the
        # one that ends up in shared memory.
        r = tle.gpu.memory_space(r, "shared_memory")
    tl.store(o_ptr + rows, r)


def _run_store(space):
    torch.manual_seed(0)
    x = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    compiled = _store_kernel[(1,)](x,
                                   out,
                                   BLOCK=BLOCK,
                                   HINT=space is not None,
                                   SPACE=space or "shared_memory",
                                   num_warps=NUM_WARPS)
    return compiled, out, x * 2.0 + 1.0


def _run_eltwise_reduce(space):
    torch.manual_seed(0)
    x = torch.randn(ELT_ROWS, ELT_COLS, device="cuda", dtype=torch.float32)
    out = torch.empty(ELT_ROWS, device="cuda", dtype=torch.float32)
    compiled = _eltwise_reduce_kernel[(1,)](x,
                                            out,
                                            R=ELT_ROWS,
                                            C=ELT_COLS,
                                            HINT=space is not None,
                                            SPACE=space or "shared_memory",
                                            num_warps=NUM_WARPS)
    return compiled, out, (x * 2.0 + 1.0).sum(dim=1)


def _run_reduce(hint):
    torch.manual_seed(0)
    x = torch.randn(REDUCE_ROWS, REDUCE_COLS, device="cuda",
                    dtype=torch.float32)
    out = torch.empty(REDUCE_ROWS, device="cuda", dtype=torch.float32)
    compiled = _reduce_kernel[(1,)](x,
                                    out,
                                    R=REDUCE_ROWS,
                                    C=REDUCE_COLS,
                                    HINT=hint,
                                    num_warps=NUM_WARPS)
    return compiled, out, x.sum(dim=1)


def _smem_alloc_count(ir):
    """Number of shared-memory (address space 2) memref allocations."""
    # The GCU IR prints generic ops, i.e. `"memref.alloc"() <{...}> : () ->
    # memref<..., 2>`; accept the custom-format spelling as well.
    return len(re.findall(r'memref\.alloc"?\(\)[^\n]*memref<[^>\n]*, 2>', ir))


def _alloca_count(ir):
    """Number of register (stack) memref allocations."""
    return len(re.findall(r"memref\.alloca", ir))


def _smem_tile_view_count(ir):
    """Number of per-warp views into a shared allocation.

    The fast path re-derives each subthread's chunk as a ``memref.view`` over
    the shared tile (``memref2ptr`` on the space-2 alloc -> ``ptr2memref`` ->
    ``memref.view``) and stores through it, so a view over a shared allocation
    is the signature of "the producer wrote the tile directly". A private buffer
    that is later desliced into the tile has no such view.
    """
    return len(re.findall(r'"memref\.view"', ir))


class TestElementwise:
    """Hint on an elementwise value consumed by a reduce."""

    def test_shared_memory_numerics(self):
        _, out, expected = _run_eltwise_reduce("shared_memory")
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_shared_memory_tile_is_allocated(self):
        baseline = _smem_alloc_count(_get_ir(_run_eltwise_reduce(None)[0]))
        hinted_ir = _get_ir(_run_eltwise_reduce("shared_memory")[0])
        hinted = _smem_alloc_count(hinted_ir)
        assert baseline == 0, "the unhinted kernel should not use shared memory"
        assert hinted >= 1, (
            "the shared_memory hint must stage the value in shared memory:\n"
            f"{hinted_ir}")

    def test_shared_memory_writes_the_tile_directly(self):
        """The producer must land in the tile, not in a private copy.

        The point of the hint is to skip the private ``memref.alloc`` +
        ``memref_ext.slice_deslice_start`` copy into the tile: the elementwise
        lowering should re-derive each subthread's chunk as a ``memref.view`` of
        the shared allocation and store through it.
        """
        baseline_ir = _get_ir(_run_eltwise_reduce(None)[0])
        hinted_ir = _get_ir(_run_eltwise_reduce("shared_memory")[0])
        assert _smem_tile_view_count(baseline_ir) == 0
        assert _smem_tile_view_count(hinted_ir) >= 1, (
            "the elementwise producer must write the shared tile through a "
            f"per-warp view:\n{hinted_ir}")

    def test_register_numerics(self):
        _, out, expected = _run_eltwise_reduce("register")
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_register_allocates_a_stack_buffer(self):
        """The register hint switches the producer from the tar path to alloca.

        The per-thread run is longer than the fusion pass' full-unroll limit,
        so the default decision (computed only on the short-loop path) is the
        tar ``memref.alloc`` buffer; the hint forces the ``memref.alloca``
        form that the vector.load/vector.store path uses.
        """
        baseline = _alloca_count(_get_ir(_run_eltwise_reduce(None)[0]))
        hinted_ir = _get_ir(_run_eltwise_reduce("register")[0])
        hinted = _alloca_count(hinted_ir)
        assert baseline == 0, "the unhinted kernel should use the tar path"
        assert hinted >= 1, (
            "the register hint must use a memref.alloca buffer:\n"
            f"{hinted_ir}")

    def test_store_hint_is_dropped(self):
        """A value that never leaves its fusion region has nothing to stage.

        The store is fused into the elementwise region, so the hinted value is
        consumed inside it and the pass drops the hint (with a remark) instead
        of inserting a staging pair inside the region. The kernel must still
        compute the right result and must not gain a shared allocation.
        """
        _, out, expected = _run_store("shared_memory")
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
        baseline = _smem_alloc_count(_get_ir(_run_store(None)[0]))
        hinted = _smem_alloc_count(_get_ir(_run_store("shared_memory")[0]))
        assert hinted == baseline, (
            "a hint on a value that does not escape its fusion region must not "
            f"add a shared allocation (baseline {baseline}, hinted {hinted})")


class TestReduce:
    """Hint on the output of a reduce."""

    def test_shared_memory_numerics(self):
        _, out, expected = _run_reduce(True)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_shared_memory_tile_is_allocated(self):
        hinted_ir = _get_ir(_run_reduce(True)[0])
        assert _smem_alloc_count(hinted_ir) >= 1, (
            "the shared_memory hint must allocate an address-space-2 memref:\n"
            f"{hinted_ir}")


@triton.jit
def _ptr_to_int_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr, HINT: tl.constexpr,
                       SPACE: tl.constexpr):
    """Hint a ``tt.ptr_to_int`` result.

    The op is not an elementwise-mappable arith op, so the fusion pass leaves
    the value as a region *operand* rather than cloning it into the region, and
    the hint reaches ``TTElementwiseOpLowering`` (the lowering dedicated to
    ptr_to_int/int_to_ptr/mulhiui) instead of the fusion lowering covered by
    the tests above. Casting the value back to i32 and storing it keeps the
    kernel's output bounded and comparable against the host-side pointer
    arithmetic.
    """
    offs = tl.arange(0, BLOCK)
    v = (x_ptr + offs).to(tl.int64)
    if HINT:
        v = tle.gpu.memory_space(v, SPACE)
    w = (v & 65535).to(tl.int32)
    tl.store(o_ptr + offs, w)


def _run_ptr_to_int(space):
    torch.manual_seed(0)
    x = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    out = torch.empty(BLOCK, device="cuda", dtype=torch.int32)
    compiled = _ptr_to_int_kernel[(1,)](x,
                                        out,
                                        BLOCK=BLOCK,
                                        HINT=space is not None,
                                        SPACE=space or "shared_memory",
                                        num_warps=NUM_WARPS)
    # GCU pointers are not torch pointers: compare against the device address of
    # the input buffer, whose low 16 bits identify the byte offset of each
    # element (4 bytes apart) unambiguously for this block size.
    offs = torch.arange(BLOCK, dtype=torch.int64)
    expected = ((x.data_ptr() + 4 * offs) & 65535).to(torch.int32)
    return compiled, out, expected


class TestPtrToInt:
    """Hint on a producer lowered by TTElementwiseOpLowering."""

    def test_shared_memory_numerics(self):
        _, out, expected = _run_ptr_to_int("shared_memory")
        assert torch.equal(out.cpu(), expected.cpu())

    def test_shared_memory_tile_is_allocated(self):
        baseline = _smem_alloc_count(_get_ir(_run_ptr_to_int(None)[0]))
        hinted_ir = _get_ir(_run_ptr_to_int("shared_memory")[0])
        assert baseline == 0, "the unhinted kernel should not use shared memory"
        assert _smem_alloc_count(hinted_ir) >= 1, (
            "the shared_memory hint must stage the ptr_to_int result in shared "
            f"memory:\n{hinted_ir}")

    def test_shared_memory_writes_the_tile_directly(self):
        baseline_ir = _get_ir(_run_ptr_to_int(None)[0])
        hinted_ir = _get_ir(_run_ptr_to_int("shared_memory")[0])
        assert _smem_tile_view_count(baseline_ir) == 0
        assert _smem_tile_view_count(hinted_ir) >= 1, (
            "TTElementwiseOpLowering must write the shared tile through a "
            f"per-warp view:\n{hinted_ir}")

    def test_register_numerics(self):
        _, out, expected = _run_ptr_to_int("register")
        assert torch.equal(out.cpu(), expected.cpu())

    def test_register_allocates_a_stack_buffer(self):
        baseline = _alloca_count(_get_ir(_run_ptr_to_int(None)[0]))
        hinted_ir = _get_ir(_run_ptr_to_int("register")[0])
        assert baseline == 0, "the unhinted kernel should use the tar path"
        assert _alloca_count(hinted_ir) >= 1, (
            "the register hint must use a memref.alloca buffer:\n"
            f"{hinted_ir}")
