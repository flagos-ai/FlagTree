"""``tle.gpu.copy(..., other=...)`` on the GCU backend.

Covers the ``other`` option of the TLE-Struct global-memory -> SMEM copy: the
value written for the elements the copy does not read, i.e. the lanes excluded
by ``mask``. It becomes the ``other`` operand of the generated load, so it must
reach the DTE pad operand of ``memref_ext.slice_pad_start`` /
``memref_ext.memset_start`` unchanged. It requires a mask: Triton's load builder
has no fill value without one, so ``other`` without ``mask`` is rejected
instead of silently ignored.

Implementation side:
  - flagtree ``tle/language/gpu/core.py``: ``other`` becomes the ``other``
    operand of the generated masked load.
  - triton_gcu400: the value already travels
    tt.load ``other`` -> triton_gcu.load default_value ->
    copy_global_to_local default_value -> the DTE pad operand in
    TTSmemLowerToGCU.cpp. The head-mask path reuses the same operand as the
    fill value of the whole-tile memset.

A mask whose valid region *starts* inside the tile (``on >= k``, ``k != 0``)
still takes the DTE. The mask start is folded into the source base, so it
travels as a *destination* tile offset: PtrAnalysis marks the load
(``offsets_are_dst``) and the fused ``copy_global_to_local`` keeps the marker,
so the backend fills the whole tile with ``other``
(``memref_ext.memset_start``) and DTEs the valid block into its destination
columns (``memref_ext.slice_deslice_start``), see the ``isDstTileOffsetOp``
branch of ``ConfigGcuLoad`` (Utility.cpp). Without the marker the offsets are
plain source offsets (a sub-region of the source tile): reading the tile start
as one - what the code did before - applies it twice and collapses the slice to
zero elements, so the test pins all three offsets.

``other`` has to fold to a single scalar to become the DTE pad register. A
per-lane fill value cannot, so such a copy is deliberately kept off the DTE
(TritonLoadStoreToDma.cpp) and runs on the per-subthread elementwise path:
correct, but slower. A copy whose shape the destination-offset path cannot
express (transposed, dynamic stride, broadcast, reshaped) is kept off the DTE
the same way.

The small-load elementwise-fusion bypass is decided per thread
(``bpe * elemsPerThread[0] <= 8 KB``, TritonLoadStoreToDma.cpp); inside that
bypass a load stays on the DTE only when every user stores into a contiguous
SMEM view (``hasOnlySharedStoreUsers``), which
``test_tle_buffered_tensor_view.py`` pins. Without it the copy would degrade to
the per-subthread path and the DTE assertions below would fail.

A copy into a ``tle.gpu.alloc`` buffer has no frontend option for
asynchrony, so the backend makes it asynchronous by default
(``FuseLoadLocalStorePattern``) and places the wait at the destination's first
use (``TTSmemLowerToGCU``). The shared tag pool only has a spare tag when the
kernel has fewer warps than ``maxWarpPerSlot`` (``DTETagPool``), so a 4-warp
kernel falls back to a synchronous issue-and-wait; ``TestCopyAsync`` pins both.
"""

import importlib.util
import os
import re

import pytest
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton  # noqa: F401  (register GCU backend)
from torch_gcu import transfer_to_gcu  # noqa: F401  (enables .gcu() on tensors)
import triton.experimental.tle.language as tle

# Recompile on every launch: the gcuir assertions must look at freshly lowered
# IR, never at a cached binary.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

# Kernel-visible constants: must be tl.constexpr to be captured by @jit.
NEG_INF = tl.constexpr(float("-inf"))
NAN = tl.constexpr(float("nan"))


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


def _copy_border_body(ir):
    """The ops emitted for the `tle.gpu.copy` under test.

    `enterTritionOp` / `leaveTritionOp` bracket the lowering of one
    `triton_gcu.copy_global_to_local`, so the slice ops, the thread-0 guard and
    the wait are all inside this span.
    """
    enter = ir.index('enter = "triton_gcu.copy_global_to_local"')
    leave = ir.index('leave = "triton_gcu.copy_global_to_local"', enter)
    return enter, leave


# gcuir is printed in the generic op form, so the operands of a DTE op are the
# comma-separated SSA values inside its parenthesis.
_MREF_RE = re.compile(r'"memref_ext\.(\w+)"\(([^)]*)\)')


def _mref_ops(line):
    """(mnemonic, operands) of one generic-form `memref_ext.*` op line."""
    match = _MREF_RE.search(line)
    assert match, f"not a memref_ext op: {line}"
    return match.group(1), [op.strip() for op in match.group(2).split(",")]


def _pad_operand_of(line):
    """The fill-value SSA operand of a DTE pad op, by operand position.

    `slice_pad_start` ends with (..., fill_value, tag_memref, tag_index) whatever
    the rank is; `memset_start` is (dst, fill_value, tag_memref, tag_index).
    Pinning the position (rather than "the constant occurs among the operands")
    is what makes this an assertion about the fill value itself.
    """
    mnemonic, operands = _mref_ops(line)
    if mnemonic == "slice_pad_start":
        return operands[-3]
    if mnemonic == "memset_start":
        return operands[1]
    raise AssertionError(f"{mnemonic} has no pad operand")


def _constant_int_of(defs, value):
    """Integer constant assigned to `value`, following one `arith.index_cast`."""
    definition = defs.get(value, "")
    match = re.search(r"value = (-?\d+) : (?:i32|index)", definition)
    if match:
        return int(match.group(1))
    cast = re.search(r'"arith\.index_cast"\((%[\w.]+)\)', definition)
    if cast:
        return _constant_int_of(defs, cast.group(1))
    raise AssertionError(f"{value} is not an integer constant: {definition}")


def _deslice_dims(ir, line):
    """(source offset, slice shape, destination offset) of a rank-1 deslice.

    The head-mask path must read the mask start as a *destination* offset: the
    source offset stays 0 (the start is already folded into the base, so a
    non-zero source offset applies it twice) and the slice keeps the width of
    the valid block (a zero shape copies nothing). Pinning the whole tuple
    catches both failure modes.
    """
    mnemonic, operands = _mref_ops(line)
    assert mnemonic == "slice_deslice_start"
    defs = _ssa_defs(ir)
    return tuple(_constant_int_of(defs, operand) for operand in operands[2:5])


def _last_copy_border(ir):
    """(enter, leave) offsets of the *last* `copy_global_to_local` region.

    Kernels that prefill the buffer contain more than one copy; the copy under
    test is the one emitted last.
    """
    enter = ir.rindex('enter = "triton_gcu.copy_global_to_local"')
    leave = ir.index('leave = "triton_gcu.copy_global_to_local"', enter)
    return enter, leave


def _deslice_operand_values(ir, line):
    """Integer operands of a rank-N deslice: src offsets, slice, dst offsets.

    Operands are (dst, src, src offsets x N, slice shape x N, dst offsets x N,
    tag memref, tag index). Elementwise fallbacks aside, this tuple pins the
    whole data movement of the DTE.
    """
    _, operands = _mref_ops(line)
    defs = _ssa_defs(ir)
    return tuple(_constant_int_of(defs, operand) for operand in operands[2:-2])


def _def_of(ir, value):
    """Definition line of an SSA value ('' when it is an argument)."""
    return _ssa_defs(ir).get(value, "")


def _lines_with(body, op, dialect="memref_ext"):
    needle = f'"{dialect}.{op}"'
    return [line for line in body.splitlines() if needle in line]


def _ssa_defs(body):
    defs = {}
    for line in body.splitlines():
        match = re.match(r"\s*(%[\w.]+) = (.*)", line)
        if match:
            defs[match.group(1)] = match.group(2)
    return defs


def _enclosing_if_conds(lines, index):
    """SSA conditions of the `scf.if` blocks that enclose `lines[index]`."""
    conds, stack, depth = [], [], 0
    for line in lines[:index]:
        match = re.search(r'"scf\.if"\((%[\w.]+)\)', line)
        opens = line.count("{")
        closes = line.count("}")
        if match and opens:
            stack.append((match.group(1), depth))
        depth += opens - closes
        while stack and depth <= stack[-1][1]:
            stack.pop()
    return [cond for cond, _ in stack]


def _depends_on_thread0_eq(cond, defs):
    """True when `cond` is (or is an `and` of) `cmpi eq <thread_id>, 0`."""
    pending, seen = [cond], set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        definition = defs.get(value, "")
        match = re.search(r'"arith\.cmpi"\((%[\w.]+), (%[\w.]+)\)'
                          r" <\{predicate = 0 : ", definition)
        if match:
            lhs, rhs = match.groups()
            lhs_is_tid = '"gpu.thread_id"' in defs.get(lhs, "")
            rhs_is_zero = "value = 0 : index" in defs.get(rhs, "")
            if lhs_is_tid and rhs_is_zero:
                return True
        match = re.search(r'"arith\.andi"\((%[\w.]+), (%[\w.]+)\)',
                          definition)
        if match:
            pending.extend(match.groups())
    return False


def _assert_issued_by_thread0(body, dte_line, defs):
    """The DTE op is inside an `scf.if` guarded by `thread_id == 0`.

    A single `gpu.thread_id` in the body is not enough: every subthread reads
    the id somewhere. What matters is that the copy op itself is dominated by
    the compare-to-zero. `defs` is built from the whole function IR, because the
    zero constant lives before the copy's border marker.
    """
    lines = body.splitlines()
    index = next(i for i, line in enumerate(lines) if line == dte_line)
    conds = _enclosing_if_conds(lines, index)
    assert conds, f"the DTE op is not inside an scf.if: {dte_line}"
    assert any(_depends_on_thread0_eq(cond, defs) for cond in conds), (
        f"the DTE op is not guarded by thread_id == 0: {dte_line}")


def _check_tail(out, capacity, row_len, expected):
    torch.testing.assert_close(
        out[:row_len],
        torch.arange(row_len, device="cuda", dtype=out.dtype).to(torch.float32),
        atol=0,
        rtol=0)
    tail = out[row_len:capacity]
    if isinstance(expected, float) and expected != expected:  # NaN
        assert torch.isnan(tail).all(), "the padded tail must be NaN"
    else:
        torch.testing.assert_close(
            tail, torch.full_like(tail, expected), atol=0, rtol=0)


def _check_head_mask(out, capacity, head, expected):
    """`mask = on >= capacity - head`: the head lanes are the fill value."""
    start = capacity - head
    torch.testing.assert_close(
        out[start:capacity],
        torch.arange(start,
                     capacity,
                     device="cuda",
                     dtype=out.dtype).to(torch.float32),
        atol=0,
        rtol=0)
    head_part = out[:start]
    if isinstance(expected, float) and expected != expected:  # NaN
        assert torch.isnan(head_part).all(), "the head must be NaN"
    else:
        torch.testing.assert_close(
            head_part, torch.full_like(head_part, expected), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _other_kernel(x_ptr, out_ptr, row_len, L: tl.constexpr, OTHER: tl.constexpr,
                  USE_OTHER: tl.constexpr):
    """`row_len` is a runtime value smaller than the buffer capacity `L`."""
    buf = tle.gpu.alloc((2, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    on = tl.arange(0, L)
    if USE_OTHER:
        tle.gpu.copy(x_ptr + on, buf.slot(1), [L], mask=on < row_len,
                     other=OTHER)
    else:
        # Backward compatibility: no fill value means zero fill.
        tle.gpu.copy(x_ptr + on, buf.slot(1), [L], mask=on < row_len)
    tl.store(out_ptr + on, tl.load(tle.gpu.local_ptr(buf.slot(1), (on, ))))


@triton.jit
def _other_kernel_f16(x_ptr, out_ptr, row_len, L: tl.constexpr,
                      OTHER: tl.constexpr):
    """`other` is cast to the destination element type (f16 here)."""
    buf = tle.gpu.alloc((2, L), dtype=tl.float16, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    on = tl.arange(0, L)
    tle.gpu.copy(x_ptr + on, buf.slot(1), [L], mask=on < row_len,
                 other=OTHER)
    vals = tl.load(tle.gpu.local_ptr(buf.slot(1), (on, )))
    tl.store(out_ptr + on, vals.to(tl.float32))


@triton.jit
def _bad_other_direction_kernel(x_ptr, out_ptr, L: tl.constexpr):
    """SMEM -> GMEM copy with a fill value: no meaning, must be rejected."""
    buf = tle.gpu.alloc((L, ), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    on = tl.arange(0, L)
    tl.store(tle.gpu.local_ptr(buf, (on, )), on.to(tl.float32))
    tle.gpu.copy(buf, out_ptr + on, [L], other=-1.0)


@triton.jit
def _bad_other_no_mask_kernel(x_ptr, out_ptr, L: tl.constexpr):
    """A fill value without a mask: the fill has no lane to apply to."""
    buf = tle.gpu.alloc((2, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    on = tl.arange(0, L)
    tle.gpu.copy(x_ptr + on, buf.slot(1), [L], other=NEG_INF)
    tl.store(out_ptr + on, tl.load(tle.gpu.local_ptr(buf.slot(1), (on, ))))


@triton.jit
def _other_head_mask_kernel(x_ptr, out_ptr, L: tl.constexpr,
                            HEAD: tl.constexpr, OTHER: tl.constexpr):
    """Only lanes [L-HEAD, L) are read: a mask whose start is non-zero."""
    buf = tle.gpu.alloc((2, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    on = tl.arange(0, L)
    tle.gpu.copy(x_ptr + on, buf.slot(1), [L], mask=on >= L - HEAD, other=OTHER)
    tl.store(out_ptr + on, tl.load(tle.gpu.local_ptr(buf.slot(1), (on, ))))


@triton.jit
def _other_head_mask_dyn_stride_kernel(x_ptr, out_ptr, row_stride,
                                       L: tl.constexpr, S: tl.constexpr,
                                       KEEP: tl.constexpr,
                                       OTHER: tl.constexpr):
    """Head mask on a source tile whose row stride is a runtime value.

    The destination-offset path needs a static source description to place the
    block at the mask start, so this copy must stay off the DTE: otherwise the
    valid block would land at tile offset 0, i.e. in the masked-out lanes.
    """
    buf = tle.gpu.alloc((S, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    sm = tl.arange(0, S)[:, None]
    on = tl.arange(0, L)[None, :]
    tle.gpu.copy(x_ptr + sm * row_stride + on, buf, [S, L],
                 mask=on >= L - KEEP, other=OTHER)
    tl.store(out_ptr + sm * L + on, tl.load(tle.gpu.local_ptr(buf)))


@triton.jit
def _other_lane_kernel(x_ptr, other_ptr, out_ptr, L: tl.constexpr,
                       KEEP: tl.constexpr, MASK_HEAD: tl.constexpr):
    """`other` is a per-lane tensor, so it cannot become a DTE pad register.

    `KEEP` is the number of lanes the mask keeps, at the head of the tile when
    `MASK_HEAD` is set and at the tail otherwise.
    """
    buf = tle.gpu.alloc((2, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    on = tl.arange(0, L)
    other = tl.load(other_ptr + on)
    if MASK_HEAD:
        mask = on >= L - KEEP
    else:
        mask = on < KEEP
    tle.gpu.copy(x_ptr + on, buf.slot(1), [L], mask=mask, other=other)
    tl.store(out_ptr + on, tl.load(tle.gpu.local_ptr(buf.slot(1), (on, ))))


@triton.jit
def _other_lane_rank3_kernel(x_ptr, other_ptr, out_ptr, D0: tl.constexpr,
                             D1: tl.constexpr, D2: tl.constexpr):
    """A rank-3 copy cannot be fused, so a per-lane `other` cannot be lowered.

    The DTE carries one pad register per copy and the elementwise fallback only
    covers rank-1/2 pointer tensors, so there is no path that applies a per-lane
    fill value here. The backend must report a diagnostic for it.
    """
    buf = tle.gpu.alloc((D0, D1, D2), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    a = tl.arange(0, D0)[:, None, None]
    b = tl.arange(0, D1)[None, :, None]
    c = tl.arange(0, D2)[None, None, :]
    off = a * D1 * D2 + b * D2 + c
    other = tl.load(other_ptr + off)
    mask = c >= D2 - 8
    tle.gpu.copy(x_ptr + off, buf, [D0, D1, D2], mask=mask, other=other)
    tl.store(out_ptr + off, tl.load(tle.gpu.local_ptr(buf)))


@triton.jit
def _other_view_subslice_kernel(x_ptr, fill_ptr, out_ptr, L: tl.constexpr,
                                M: tl.constexpr, START: tl.constexpr,
                                S: tl.constexpr, KEEP: tl.constexpr,
                                MASK_HEAD: tl.constexpr,
                                OTHER: tl.constexpr):
    """Copy into a 2-D `.subslice(START, S, dim=0)` view of a larger buffer.

    The view offset is a compile-time `memref.subview` offset, the mask start a
    destination tile offset of the deslice; the copy must apply both and leave
    the rows outside the view at their prefill.
    """
    buf = tle.gpu.alloc((M, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    om = tl.arange(0, M)[:, None]
    on = tl.arange(0, L)[None, :]
    tle.gpu.copy(fill_ptr + om * L + on, buf, [M, L])
    view = buf.subslice(START, S, dim=0)
    sm = tl.arange(0, S)[:, None]
    if MASK_HEAD:
        mask = on >= L - KEEP
    else:
        mask = on < KEEP
    tle.gpu.copy(x_ptr + sm * L + on, view, [S, L], mask=mask, other=OTHER)
    tl.store(out_ptr + om * L + on, tl.load(tle.gpu.local_ptr(buf)))


@triton.jit
def _other_view_reshape_kernel(x_ptr, out_ptr, L: tl.constexpr,
                               KEEP: tl.constexpr, OTHER: tl.constexpr):
    """Copy into `.reshape((2 * L, ))`: the DTE destination is a memref.view."""
    buf = tle.gpu.alloc((2, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = buf.reshape((2 * L, ))
    on = tl.arange(0, 2 * L)
    tle.gpu.copy(x_ptr + on, view, [2 * L], mask=on >= 2 * L - KEEP,
                 other=OTHER)
    tl.store(out_ptr + on, tl.load(tle.gpu.local_ptr(buf.reshape((2 * L, )))))


@triton.jit
def _other_view_slot_kernel(x_ptr, out_ptr, L: tl.constexpr, KEEP: tl.constexpr,
                            stage, OTHER: tl.constexpr):
    """Copy into `buf.slot(stage)`: the view base is a runtime byte offset."""
    buf = tle.gpu.alloc((2, L), dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    on = tl.arange(0, L)
    tle.gpu.copy(x_ptr + on, buf.slot(stage), [L], mask=on >= L - KEEP,
                 other=OTHER)
    tl.store(out_ptr + on, tl.load(tle.gpu.local_ptr(buf.slot(stage), (on, ))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_other(capacity, row_len, value, use_other=True, num_warps=4):
    x = torch.arange(capacity, device="cuda", dtype=torch.float32)
    out = torch.empty(capacity, device="cuda", dtype=torch.float32)
    compiled = _other_kernel[(1,)](x, out, row_len, L=capacity, OTHER=value,
                                   USE_OTHER=use_other, num_warps=num_warps)
    return out, compiled


def _copy_wait_placement(ir):
    """Waits of the copy region vs. of the destination's first use.

    Returns ``(waits inside the copy_global_to_local region, waits between that
    region and the first ttg.local_load, the between-region text, SSA defs of
    the function prefix)``. The defs cover the prefix because the thread guard
    is defined before the copy's border marker.
    """
    enter, leave = _copy_border_body(ir)
    first_use = ir.index('enter = "ttg.local_load"', leave)
    body = ir[enter:leave]
    between = ir[leave:first_use]
    defs = _ssa_defs(ir[ir.rindex("func.func", 0, enter):first_use])
    return (_lines_with(body, "dma_wait", dialect="memref"),
            _lines_with(between, "dma_wait", dialect="memref"), between, defs)


class TestCopyOther:
    """`tle.gpu.copy(..., other=...)`: deterministic tail."""

    # The elementwise-fusion bypass only looks at the per-thread bytes of the
    # load (bpe * elemsPerThread[0] <= 8 KB, TritonLoadStoreToDma.cpp). On
    # GCU400 threads-per-warp is 1, so the default 4-warp kernel has 4 threads:
    # 4096 f32 elements are 1024 elements = 4 KB per thread and enter the
    # bypass, where a load stays on the DTE only through hasOnlySharedStoreUsers;
    # 16384 elements are 4096 elements = 16 KB per thread, never enter it, and
    # are kept here as a plain size regression.
    @pytest.mark.parametrize("capacity,row_len", [(4096, 3000), (16384, 15000)])
    def test_other_inf_tail(self, capacity, row_len):
        out, _ = _run_other(capacity, row_len, NEG_INF)
        _check_tail(out, capacity, row_len, float("-inf"))

    def test_other_default_zero(self):
        """`other=None` keeps the historical zero fill."""
        capacity, row_len = 4096, 3000
        out, _ = _run_other(capacity, row_len, NEG_INF, use_other=False)
        _check_tail(out, capacity, row_len, 0.0)

    def test_other_nan(self):
        capacity, row_len = 4096, 3000
        out, _ = _run_other(capacity, row_len, NAN)
        _check_tail(out, capacity, row_len, float("nan"))

    def test_other_nan_f16(self):
        """The fill value is converted to the destination element type."""
        capacity, row_len = 4096, 3000
        x = torch.arange(capacity, device="cuda", dtype=torch.float16)
        out = torch.empty(capacity, device="cuda", dtype=torch.float32)
        _other_kernel_f16[(1,)](x, out, row_len, L=capacity, OTHER=NAN)
        tail = out[row_len:capacity]
        assert torch.isnan(tail).all(), "the padded f16 tail must be NaN"
        # f16 cannot represent every integer above 2048, so compare against the
        # values that were actually copied, not against an f32 arange.
        torch.testing.assert_close(out[:row_len], x[:row_len].to(torch.float32),
                                   atol=0, rtol=0)

    def test_other_rejected_for_local_to_global(self):
        """`other` describes a GMEM -> SMEM transfer only."""
        capacity = 1024
        x = torch.arange(capacity, device="cuda", dtype=torch.float32)
        out = torch.empty(capacity, device="cuda", dtype=torch.float32)
        # The frontend ValueError is wrapped by the JIT compiler; the message
        # stays intact, which is what users see.
        with pytest.raises(CompilationError,
                           match="only supported for standard pointer-tensor"):
            _bad_other_direction_kernel[(1,)](x, out, L=capacity)

    def test_other_requires_mask(self):
        """`other` without a mask is a usage error, not a silent no-op."""
        capacity = 1024
        x = torch.arange(capacity, device="cuda", dtype=torch.float32)
        out = torch.empty(capacity, device="cuda", dtype=torch.float32)
        with pytest.raises(CompilationError, match="copy other requires a mask"):
            _bad_other_no_mask_kernel[(1,)](x, out, L=capacity)

    def test_other_large_tail_uses_memset(self):
        """A tail beyond the DTE pad register width (2047) is memset."""
        capacity, row_len = 4096, 512
        out, compiled = _run_other(capacity, row_len, NEG_INF)
        _check_tail(out, capacity, row_len, float("-inf"))
        ir = _get_ir(compiled)
        enter, leave = _copy_border_body(ir)
        body = ir[enter:leave]
        neg_inf = re.search(
            r'(%[\w.]+) = "arith\.constant"\(\) <\{value = 0xFF800000 : f32\}>',
            ir)
        assert neg_inf, "the -inf fill value was folded away or never created"
        # The 3584-element tail takes memset + deslice; slice_pad_start is the
        # other branch of the same scf.if and is not executed for this shape.
        memset = _lines_with(body, "memset_start")
        assert len(memset) == 1, "a 3584-element tail must take the memset branch"
        assert _pad_operand_of(memset[0]) == neg_inf.group(1), (
            "memset_start must fill the tail with the -inf constant")
        assert len(_lines_with(body, "slice_deslice_start")) == 1
        # Both branches of the pad decision are present in the IR, but the
        # memset branch is the one this shape selects.
        assert len(_lines_with(body, "slice_pad_start")) == 1


class TestCopyDteOnly:
    """The copy is one shared DTE issued by thread 0 only."""

    def test_sub0_only_dte(self):
        capacity, row_len = 4096, 3000
        _, compiled = _run_other(capacity, row_len, NEG_INF)
        ir = _get_ir(compiled)

        # No per-subthread private DTE: that is the
        # "maskedload + TAR + private chunk + deslice" degradation.
        assert "memref_ext.deslice_start" not in ir, (
            "the copy degraded into the elementwise fallback "
            "(maskedload + TAR + private chunk + deslice)")

        enter, leave = _copy_border_body(ir)
        body = ir[enter:leave]
        # The zero constant the guard compares against is emitted before the
        # copy's border marker, so resolve SSA defs from the whole function.
        defs = _ssa_defs(ir[ir.rindex("func.func", 0, enter):leave])
        # Issued (and waited for) by the master thread only: the DTE is a
        # shared engine, the other subthreads just hit the block-wide barrier.
        pad_ops = _lines_with(body, "slice_pad_start")
        assert len(pad_ops) == 1, "expected exactly one pad DTE op"
        _assert_issued_by_thread0(body, pad_ops[0], defs)
        wait_ops = _lines_with(body, "dma_wait", dialect="memref")
        assert wait_ops, "the synchronous copy must wait here"
        for wait_op in wait_ops:
            _assert_issued_by_thread0(body, wait_op, defs)
        # No per-subthread TAR traffic for the copy itself.
        assert "gcu.tar_store" not in body, (
            "the SMEM tile is written per subthread instead of by the DTE")

        # The fill value reaches the DTE as the -inf constant, not as a zero.
        neg_inf = re.search(
            r'(%[\w.]+) = "arith\.constant"\(\) <\{value = 0xFF800000 : f32\}>',
            ir)
        assert neg_inf, "the -inf fill value was folded away or never created"
        assert _pad_operand_of(pad_ops[0]) == neg_inf.group(1), (
            "the DTE pad operand must be the -inf constant")


class TestCopyHeadMask:
    """A non-zero mask start travels as a destination tile offset.

    The start is folded into the source base, so the fused
    ``copy_global_to_local`` must not read it as a source offset. The backend
    fills the whole tile with ``other`` and DTEs the valid block into its
    destination columns instead. Pinning the three deslice operands is what
    distinguishes this from the earlier bug, where the start was applied as a
    source offset and the slice collapsed to zero elements.
    """

    @pytest.mark.parametrize("capacity,head", [(4096, 8), (1024, 1023)])
    def test_head_mask_uses_dte_destination_offset(self, capacity, head):
        x = torch.arange(capacity, device="cuda", dtype=torch.float32)
        out = torch.empty(capacity, device="cuda", dtype=torch.float32)
        compiled = _other_head_mask_kernel[(1,)](x, out, L=capacity, HEAD=head,
                                                 OTHER=NEG_INF)
        _check_head_mask(out, capacity, head, float("-inf"))

        ir = _get_ir(compiled)
        assert '"triton_gcu.copy_global_to_local"' in ir, (
            "a non-zero mask start must stay on the DTE path")
        # No per-subthread private DTE: that is the elementwise fallback.
        assert "memref_ext.deslice_start" not in ir, (
            "the copy degraded into the elementwise fallback "
            "(maskedload + TAR + private chunk + deslice)")

        enter, leave = _copy_border_body(ir)
        body = ir[enter:leave]
        # The zero constant the guard compares against is emitted before the
        # copy's border marker, so resolve SSA defs from the whole function.
        defs = _ssa_defs(ir[ir.rindex("func.func", 0, enter):leave])

        memset = _lines_with(body, "memset_start")
        deslice = _lines_with(body, "slice_deslice_start")
        assert len(memset) == 1, "the whole tile must be filled by one memset"
        assert len(deslice) == 1, "the valid block must take one deslice"
        assert not _lines_with(body, "slice_pad_start"), (
            "slice_pad_start writes at tile offset 0 and would put the data "
            "in the masked-out lanes")

        # The fill value is `other`, not the zero a broken scalar extraction
        # produces.
        neg_inf = re.search(
            r'(%[\w.]+) = "arith\.constant"\(\) <\{value = 0xFF800000 : f32\}>',
            ir)
        assert neg_inf, "the -inf fill value was folded away or never created"
        assert _pad_operand_of(memset[0]) == neg_inf.group(1), (
            "the tile memset must use the `other` fill value")

        # source offset 0, the width of the valid block, the destination start.
        assert _deslice_dims(ir, deslice[0]) == (0, head, capacity - head), (
            "the mask start must be a destination offset, not a source offset")

        for op_line in memset + deslice:
            _assert_issued_by_thread0(body, op_line, defs)


    def test_dynamic_stride_head_mask_stays_elementwise(self):
        """A runtime source stride keeps the head-mask copy off the DTE.

        ConfigGcuLoad can only place the valid block at the mask start for a
        static, identity-order, non-broadcast source description. With a
        runtime row stride the backend must fall back to the elementwise path
        (which applies ``other`` per lane) instead of letting the block land
        at tile offset 0, i.e. in the masked-out lanes.
        """
        L, S, keep = 2048, 8, 16
        x = torch.arange(S * L, device="cuda", dtype=torch.float32)
        out = torch.empty(S * L, device="cuda", dtype=torch.float32)
        compiled = _other_head_mask_dyn_stride_kernel[(1,)](
            x, out, L, L=L, S=S, KEEP=keep, OTHER=NEG_INF)

        expect = x.view(S, L).clone()
        expect[:, :L - keep] = float("-inf")
        torch.testing.assert_close(out, expect.view(-1), atol=0, rtol=0)

        ir = _get_ir(compiled)
        assert '"triton_gcu.copy_global_to_local"' not in ir, (
            "a runtime source stride cannot place the block at the mask start")
        assert '"triton_gcu.elementwise_fusion_region"' in ir, (
            "a copy whose block cannot be placed at the mask start must run "
            "on the elementwise path")


class TestCopyViewDestination:
    """A TLE memdesc view as the DTE destination, combined with a mask.

    `tle.buffered_tensor.slot/.subslice/.reshape` address a region of one
    `tle.gpu.alloc` buffer. The view offset is part of the destination memref
    (`memref.subview` static offset with `.subslice`, `memref.view` byte offset
    with `.slot`, `memref.reinterpret_cast` with `.reshape`), the mask start an
    independent destination tile offset of the deslice. A copy must compose
    both - the valid block lands at ``view_base + dst_offset`` - instead of
    writing at the parent buffer's origin, and the bytes outside the view stay
    untouched.
    """

    def test_subslice_head_mask(self):
        L, M, start, s, keep = 512, 8, 4, 4, 8
        x = torch.arange(s * L, device="cuda", dtype=torch.float32)
        fill = torch.full((M, L), -7.0, device="cuda", dtype=torch.float32)
        out = torch.empty(M, L, device="cuda", dtype=torch.float32)
        compiled = _other_view_subslice_kernel[(1,)](
            x, fill, out, L=L, M=M, START=start, S=s, KEEP=keep, MASK_HEAD=True,
            OTHER=NEG_INF)

        expect = fill.clone()
        block = x.view(s, L).clone()
        block[:, :L - keep] = float("-inf")
        expect[start:start + s] = block
        torch.testing.assert_close(out, expect, atol=0, rtol=0)

        ir = _get_ir(compiled)
        enter, leave = _last_copy_border(ir)
        body = ir[enter:leave]
        defs = _ssa_defs(ir[ir.rindex("func.func", 0, enter):leave])

        memset = _lines_with(body, "memset_start")
        deslice = _lines_with(body, "slice_deslice_start")
        assert len(memset) == 1, "the whole view must be filled by one memset"
        assert len(deslice) == 1, "the valid block must take one deslice"
        assert not _lines_with(body, "slice_pad_start"), (
            "slice_pad_start writes at the view origin and would put the data "
            "in the masked-out lanes")
        assert not _lines_with(body, "deslice_start"), (
            "the copy degraded into the elementwise fallback "
            "(maskedload + TAR + private chunk + deslice)")

        # The DTE destination is the view, not the parent buffer.
        dst = _mref_ops(deslice[0])[1][0]
        dst_def = _def_of(ir, dst)
        assert "memref.subview" in dst_def, (
            f"the DTE destination must be the subview: {dst_def}")
        assert f"offset: {start * L}" in dst_def, (
            f"the subview must carry the view offset {start * L}: {dst_def}")
        assert _mref_ops(memset[0])[1][0] == dst, (
            "the memset and the deslice must target the same view")

        # Per dimension: source offsets, then the valid block, then the mask
        # start as a destination offset inside the view.
        assert _deslice_operand_values(ir, deslice[0]) == (0, 0, s, keep, 0,
                                                           L - keep), (
            "the mask start must be a destination offset inside the view")

        for op_line in memset + deslice:
            _assert_issued_by_thread0(body, op_line, defs)

    def test_subslice_tail_mask(self):
        """A tail mask on a view destination stays a single padded DTE."""
        L, M, start, s, keep = 512, 8, 4, 4, 8
        x = torch.arange(s * L, device="cuda", dtype=torch.float32)
        fill = torch.full((M, L), -7.0, device="cuda", dtype=torch.float32)
        out = torch.empty(M, L, device="cuda", dtype=torch.float32)
        compiled = _other_view_subslice_kernel[(1,)](
            x, fill, out, L=L, M=M, START=start, S=s, KEEP=keep,
            MASK_HEAD=False, OTHER=NEG_INF)

        expect = fill.clone()
        block = x.view(s, L).clone()
        block[:, keep:] = float("-inf")
        expect[start:start + s] = block
        torch.testing.assert_close(out, expect, atol=0, rtol=0)

        ir = _get_ir(compiled)
        enter, leave = _last_copy_border(ir)
        body = ir[enter:leave]
        pad = _lines_with(body, "slice_pad_start")
        assert len(pad) == 1
        assert not _lines_with(body, "memset_start")
        assert not _lines_with(body, "deslice_start")
        dst = _mref_ops(pad[0])[1][0]
        assert "memref.subview" in _def_of(ir, dst), (
            "the padded DTE must write into the view")
        neg_inf = re.search(
            r'(%[\w.]+) = "arith\.constant"\(\) <\{value = 0xFF800000 : f32\}>',
            ir)
        assert neg_inf, "the -inf fill value was folded away or never created"
        assert _pad_operand_of(pad[0]) == neg_inf.group(1), (
            "the DTE pad operand must be the -inf constant")

    @pytest.mark.parametrize("stage", [0, 1])
    def test_runtime_slot_head_mask(self, stage):
        """A runtime `.slot(stage)` view base composes with the mask start."""
        L, keep = 1024, 8
        x = torch.arange(L, device="cuda", dtype=torch.float32)
        out = torch.empty(L, device="cuda", dtype=torch.float32)
        compiled = _other_view_slot_kernel[(1,)](x, out, L=L, KEEP=keep,
                                                 stage=stage, OTHER=NEG_INF)
        _check_head_mask(out, L, keep, float("-inf"))

        ir = _get_ir(compiled)
        enter, leave = _last_copy_border(ir)
        body = ir[enter:leave]
        memset = _lines_with(body, "memset_start")
        deslice = _lines_with(body, "slice_deslice_start")
        assert len(memset) == 1
        assert len(deslice) == 1
        assert not _lines_with(body, "deslice_start")
        dst = _mref_ops(deslice[0])[1][0]
        assert "memref.view" in _def_of(ir, dst), (
            "the DTE destination must be the runtime slot view")
        assert _deslice_operand_values(ir, deslice[0]) == (0, keep, L - keep), (
            "the mask start must be a destination offset inside the view")

    def test_reshape_head_mask(self):
        """`.reshape` is a reinterpret_cast destination; the mask start applies."""
        L, keep = 512, 8
        x = torch.arange(2 * L, device="cuda", dtype=torch.float32)
        out = torch.empty(2 * L, device="cuda", dtype=torch.float32)
        compiled = _other_view_reshape_kernel[(1,)](x, out, L=L, KEEP=keep,
                                                    OTHER=NEG_INF)

        expect = x.clone()
        expect[:2 * L - keep] = float("-inf")
        torch.testing.assert_close(out, expect, atol=0, rtol=0)

        ir = _get_ir(compiled)
        enter, leave = _last_copy_border(ir)
        body = ir[enter:leave]
        memset = _lines_with(body, "memset_start")
        deslice = _lines_with(body, "slice_deslice_start")
        assert len(memset) == 1
        assert len(deslice) == 1
        assert not _lines_with(body, "deslice_start")
        dst = _mref_ops(deslice[0])[1][0]
        assert "memref.reinterpret_cast" in _def_of(ir, dst), (
            "the DTE destination must be the reshape view")
        assert _deslice_operand_values(ir, deslice[0]) == (0, keep,
                                                           2 * L - keep)


class TestCopyNonUniformOther:
    """A per-lane `other` must not take the DTE.

    The DTE carries one pad register per copy, so a per-lane fill value has no
    representation there. The backend declines such a copy - it used to build
    a null scalar and fill with 0 - and the elementwise path applies the
    per-lane value, which the numeric check pins.
    """

    @pytest.mark.parametrize("mask_head", [False, True])
    def test_per_lane_other_stays_elementwise(self, mask_head):
        capacity, keep = 4096, 1000
        x = torch.arange(capacity, device="cuda", dtype=torch.float32)
        other = -1 - torch.arange(capacity, device="cuda", dtype=torch.float32)
        out = torch.full((capacity,), 7.0, device="cuda", dtype=torch.float32)
        compiled = _other_lane_kernel[(1,)](x, other, out, L=capacity, KEEP=keep,
                                            MASK_HEAD=mask_head)
        if mask_head:
            expected = torch.cat([other[:capacity - keep], x[capacity - keep:]])
        else:
            expected = torch.cat([x[:keep], other[keep:]])
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

        ir = _get_ir(compiled)
        assert '"triton_gcu.copy_global_to_local"' not in ir, (
            "a per-lane `other` cannot become a DTE pad register")
        for op in ("slice_pad_start", "memset_start", "slice_deslice_start"):
            assert not _lines_with(ir, op), (
                f"the elementwise fallback must not emit the DTE op {op}")

    def test_per_lane_other_rank3_reports_error(self, capfd):
        """A per-lane `other` that no path can lower is a compile error.

        A rank-3 copy is not fused into an elementwise region and the DTE has
        no per-lane pad register, so TritonLoadOpLowering used to hit an
        `llvm_unreachable` - a compiler core dump. It must fail with a
        diagnostic instead.
        """
        d0, d1, d2 = 2, 4, 128
        n = d0 * d1 * d2
        x = torch.arange(n, device="cuda", dtype=torch.float32)
        other = torch.zeros(n, device="cuda", dtype=torch.float32)
        out = torch.empty(n, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Pipeline run failed"):
            _other_lane_rank3_kernel[(1,)](x, other, out, D0=d0, D1=d1, D2=d2)
        err = capfd.readouterr().err
        assert "per-lane `other` fill value that cannot be lowered" in err, (
            "the unsupported `other` must be reported by the backend instead "
            "of hitting an unreachable")


class TestCopyAsync:
    """Copies into `tle.gpu.alloc` buffers are asynchronous by default.

    The frontend API has no option for it, so the backend marks the fused
    ``copy_global_to_local`` ``tt.load.async`` and ``TTSmemLowerToGCU`` waits at
    the destination's first use. A kernel whose shared tag pool has no spare tag
    (4 warps without warp specialization, ``DTETagPool``) keeps the synchronous
    issue-and-wait inside the copy region.
    """

    @pytest.mark.parametrize("num_warps", [1, 4])
    def test_wait_placement(self, num_warps):
        capacity, row_len = 4096, 3000
        out, compiled = _run_other(capacity, row_len, NEG_INF,
                                   num_warps=num_warps)
        _check_tail(out, capacity, row_len, float("-inf"))

        body_waits, use_waits, between, defs = _copy_wait_placement(
            _get_ir(compiled))
        if num_warps == 1:
            # The memset waits for its own fill DTE; the copy is only awaited
            # at the first use of the destination.
            assert len(body_waits) == 1, (
                "the asynchronous copy must not wait inside its own region")
            assert len(use_waits) == 1, (
                "the copy must be awaited at the destination's first use")
            _assert_issued_by_thread0(between, use_waits[0], defs)
        else:
            assert len(body_waits) == 2, (
                "without a spare shared tag the copy waits synchronously "
                "inside its own region")
            assert not use_waits, (
                "the synchronous fallback must not leave the wait to the "
                "first use")

    def test_head_mask_async(self):
        """The memset + deslice head-mask path awaits at the first use too."""
        capacity, head = 4096, 8
        x = torch.arange(capacity, device="cuda", dtype=torch.float32)
        out = torch.empty(capacity, device="cuda", dtype=torch.float32)
        compiled = _other_head_mask_kernel[(1,)](x, out, L=capacity, HEAD=head,
                                                 OTHER=NEG_INF, num_warps=1)
        _check_head_mask(out, capacity, head, float("-inf"))

        body_waits, use_waits, _, _ = _copy_wait_placement(_get_ir(compiled))
        # Only the memset waits for its own fill; the deslice is awaited at the
        # first use of the destination.
        assert len(body_waits) == 1
        assert len(use_waits) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
