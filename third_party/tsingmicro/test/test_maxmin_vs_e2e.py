"""End-to-end checks for vector-scalar max/min lowering on the TsingMicro backend.

The IR tests require only the compiler (no device).  The numeric tests are
opt-in via ``TX81_RUN_DEVICE_TESTS=1`` because some hosts enumerate a TX device
that blocks in synchronize().
"""

import math
import os

import pytest
import triton
import triton.language as tl
from triton.compiler import ASTSource, compile

F32_SIG = {"x_ptr": "*fp32", "y_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}
VEC_SIG = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "z_ptr": "*fp32",
    "n": "i32",
    "BLOCK": "constexpr",
}


@triton.jit
def max_scalar_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def min_scalar_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.minimum(x, 0.0), mask=mask)


@triton.jit
def max_scalar_all_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.maximum(x, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def max_scalar_lhs_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(0.0, x), mask=mask)


@triton.jit
def min_scalar_lhs_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.minimum(0.0, x), mask=mask)


@triton.jit
def max_vector_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, tl.maximum(x, y), mask=mask)


@triton.jit
def min_vector_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, tl.minimum(x, y), mask=mask)


# propagate_nan=ALL keeps the raw vector-vector instruction (tx.maxvv/minvv)
# without the maxnum mask-move sequence, so these kernels observe the
# hardware's own NaN behaviour.
@triton.jit
def max_vector_all_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    tl.store(z_ptr + offs, z, mask=mask)


@triton.jit
def min_vector_all_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL)
    tl.store(z_ptr + offs, z, mask=mask)


def _txir(fn, signature, block=1024):
    target = triton.runtime.driver.active.get_current_target()
    source = ASTSource(fn=fn, signature=signature, constexprs={"BLOCK": block})
    return compile(source, target=target).asm["txir"]


def test_scalar_max_uses_maxvs():
    ir = _txir(max_scalar_kernel, F32_SIG)
    # A non-NaN splat constant needs no NaN handling at all: the tensor stays
    # operand 0, so the hardware returns the constant on NaN lanes.
    assert "tx.maxvs" in ir
    assert "tx.maxvv" not in ir
    assert "tx.unequalvv" not in ir
    assert "tx.equalvs" not in ir
    assert "tx.mask_move" not in ir


def test_scalar_min_uses_minvs():
    ir = _txir(min_scalar_kernel, F32_SIG)
    assert "tx.minvs" in ir
    assert "tx.minvv" not in ir
    assert "tx.unequalvv" not in ir
    assert "tx.equalvs" not in ir
    assert "tx.mask_move" not in ir


def test_scalar_max_lhs_uses_maxvs():
    # Constant on the left: operands are swapped so the tensor is operand 0,
    # after which the hardware already implements maxNum and no mask is
    # needed (max/min are not commutative under NaN, so the order matters).
    ir = _txir(max_scalar_lhs_kernel, F32_SIG)
    assert "tx.maxvs" in ir
    assert "tx.maxvv" not in ir
    assert "tx.unequalvv" not in ir
    assert "tx.equalvs" not in ir
    assert "tx.mask_move" not in ir


def test_scalar_min_lhs_uses_minvs():
    ir = _txir(min_scalar_lhs_kernel, F32_SIG)
    assert "tx.minvs" in ir
    assert "tx.minvv" not in ir
    assert "tx.unequalvv" not in ir
    assert "tx.equalvs" not in ir
    assert "tx.mask_move" not in ir


def test_propagate_nan_all_is_not_fused():
    # arith.maximumf is still kept off the new scalar fusion by the
    # mk.minmax_nan_masked guard.
    ir = _txir(max_scalar_all_kernel, F32_SIG)
    assert "tx.maxvv" in ir
    assert "tx.maxvs" not in ir


def test_vector_max_keeps_maxvv():
    ir = _txir(max_vector_kernel, VEC_SIG)
    assert "tx.maxvv" in ir
    assert "tx.maxvs" not in ir
    # Only operand 1's NaN check is needed: the hardware drops operand 0's NaN
    # and the mask-move restores the maxNum result for operand 1.
    assert ir.count("tx.unequalvv") == 1
    assert ir.count("tx.equalvs") == 1
    assert ir.count("tx.mask_move") == 1


def _torch_device():
    try:
        import torch

        device = triton.runtime.driver.active.get_active_torch_device()
        torch.zeros(1, device=device)
        return device
    except Exception:
        return None


# Device execution is opt-in: on some hosts the TX runtime enumerates a device
# that is not actually usable and a launch blocks in synchronize().  The IR
# tests above need no device and always run.
RUN_DEVICE = os.getenv("TX81_RUN_DEVICE_TESTS", "1").lower() in ("1", "true", "yes")
DEVICE = _torch_device() if RUN_DEVICE else None


@pytest.mark.skipif(DEVICE is None, reason="no Tx81 device available")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_numeric_maxnum(dtype):
    import torch

    torch_dtype = getattr(torch, dtype)
    values = [-float("inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")]
    x = torch.tensor(values, dtype=torch_dtype, device=DEVICE)
    out = torch.empty_like(x)
    n = x.numel()
    max_scalar_kernel[(1, )](x, out, n, BLOCK=16)

    # torch.fmax implements the IEEE maxNum semantics that tl.maximum(NONE)
    # lowers to: if one operand is NaN, return the other.
    expected = torch.fmax(x, torch.zeros_like(x))
    torch.testing.assert_close(out, expected, equal_nan=True)

    min_scalar_kernel[(1, )](x, out, n, BLOCK=16)
    expected = torch.fmin(x, torch.zeros_like(x))
    torch.testing.assert_close(out, expected, equal_nan=True)


@pytest.mark.skipif(DEVICE is None, reason="no Tx81 device available")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_numeric_maxnum_vector(dtype):
    """maxnum on two tensors: NaN in either operand, both NaN, infinities, ±0.

    This is the path the mask-move sequence must keep correct when the lhs
    check is removed.  Compare exactly against torch.fmax/fmin, including the
    sign of zero (torch returns operand 0 on a tie, as the hardware does).
    """
    import torch

    torch_dtype = getattr(torch, dtype)
    xs = [float("nan"), 1.0, float("nan"), float("nan"), -float("inf"), -0.0, 0.0, 2.5]
    ys = [1.0, float("nan"), float("nan"), 5.0, float("nan"), 0.0, -0.0, -3.0]
    x_cpu = torch.tensor(xs, dtype=torch_dtype)
    y_cpu = torch.tensor(ys, dtype=torch_dtype)
    n = x_cpu.numel()
    x = x_cpu.to(DEVICE)
    y = y_cpu.to(DEVICE)
    out = torch.empty(n, dtype=torch_dtype, device=DEVICE)

    max_vector_kernel[(1, )](x, y, out, n, BLOCK=16)
    torch.txda.synchronize()
    _assert_hw_maxmin(out.cpu(), torch.fmax(x_cpu, y_cpu))

    min_vector_kernel[(1, )](x, y, out, n, BLOCK=16)
    torch.txda.synchronize()
    _assert_hw_maxmin(out.cpu(), torch.fmin(x_cpu, y_cpu))


@pytest.mark.skipif(DEVICE is None, reason="no Tx81 device available")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_numeric_maxnum_scalar_lhs(dtype):
    """maxnum with the constant on the left (``tl.maximum(0.0, x)``).

    NaN and finite values must match torch.fmax/fmin.  Signed zeros are a
    known pre-existing divergence, characterized below: the frontend
    canonicalizes ``tl.maximum(0.0, x)`` to ``maximumf(x, 0.0)``, and the
    hardware returns operand 0 on a tie, so ``x = -0.0`` yields ``-0.0`` while
    torch returns the constant ``+0.0``.
    """
    import torch

    torch_dtype = getattr(torch, dtype)
    values = [float("nan"), -float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")]
    neg_zero_lane = values.index(-0.0)
    x_cpu = torch.tensor(values, dtype=torch_dtype)
    n = x_cpu.numel()
    x = x_cpu.to(DEVICE)
    out = torch.empty(n, dtype=torch_dtype, device=DEVICE)
    zero = torch.full_like(x_cpu, 0.0)

    for kernel, ref in ((max_scalar_lhs_kernel, torch.fmax), (min_scalar_lhs_kernel, torch.fmin)):
        kernel[(1, )](x, out, n, BLOCK=16)
        torch.txda.synchronize()
        got = out.cpu()
        expected = ref(zero, x_cpu)

        # NaN lanes and values match torch; assert_close treats -0.0 == +0.0.
        torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)

        # Known divergence, pinned so a future ±0 fix is an intentional change.
        assert not torch.signbit(expected[neg_zero_lane]).item(), \
            "torch.fmax/fmin should pick the constant (+0.0) on a tie"
        assert torch.signbit(got[neg_zero_lane]).item(), \
            "current lowering picks the tensor operand on a tie (-0.0)"


# External CoreIR can put a NaN constant on the left.  LinalgFusion turns the
# fill into operand 1 (with the tensor in operand 0), so the maxnum lowering
# must keep the tensor first and mask the constant; otherwise the hardware
# returns the constant's NaN on every lane.
_NAN_LHS_MAXNUM_MODULE = """#map = affine_map<(d0) -> (d0)>
module {{
  func.func @{name}(%arg0: memref<*xf32>, %arg1: memref<*xf32>) {{
    %cst = arith.constant dense<0x7FC00000> : tensor<{block}xf32>
    %in = memref.reinterpret_cast %arg0 to offset: [0], sizes: [{block}], strides: [1] : memref<*xf32> to memref<{block}xf32, strided<[1], offset: ?>>
    %out = memref.reinterpret_cast %arg1 to offset: [0], sizes: [{block}], strides: [1] : memref<*xf32> to memref<{block}xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<{block}xf32>
    memref.copy %in, %alloc : memref<{block}xf32, strided<[1], offset: ?>> to memref<{block}xf32>
    %tin = bufferization.to_tensor %alloc restrict writable : memref<{block}xf32> to tensor<{block}xf32>
    %er = tensor.empty() : tensor<{block}xf32>
    %res = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}} ins(%cst, %tin : tensor<{block}xf32>, tensor<{block}xf32>) outs(%er : tensor<{block}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.maxnumf %a, %b : f32
      linalg.yield %v : f32
    }} -> tensor<{block}xf32>
    bufferization.materialize_in_destination %res in writable %out : (tensor<{block}xf32>, memref<{block}xf32, strided<[1], offset: ?>>) -> ()
    return
  }}
}}
"""


@pytest.mark.skipif(DEVICE is None, reason="no Tx81 device available")
def test_numeric_maxnum_nan_constant_lhs():
    """maxnumf(NaN constant, x) must return x, not NaN."""
    import torch
    from triton.backends.tsingmicro.core_ir import (
        ArgumentSpec,
        CompileOptions,
        Grid,
        KernelSignature,
        compile_core_ir,
    )
    from triton.backends.tsingmicro.runtime import load_core_ir_artifact

    block = 16
    values = [float("nan"), 1.0, -float("inf"), -0.0, 0.0, 2.5]
    values += [0.0] * (block - len(values))
    x_cpu = torch.tensor(values, dtype=torch.float32)
    x = x_cpu.to(DEVICE)

    artifact = compile_core_ir(
        module=_NAN_LHS_MAXNUM_MODULE.format(name="nan_lhs_maxnum", block=block),
        kernel_name="nan_lhs_maxnum",
        signature=KernelSignature((
            ArgumentSpec("x", "float32", shape=(block, ), readonly=True),
            ArgumentSpec("out", "float32", shape=(block, )),
        )),
        grid=Grid(1),
        options=CompileOptions(),
    )

    out = torch.full((block, ), 12345.0, dtype=torch.float32, device=DEVICE)
    load_core_ir_artifact(artifact).launch(x, out, device=DEVICE.index)
    torch.txda.synchronize()

    # maxNum(NaN, x) returns x on every lane, including x = NaN.
    _assert_hw_maxmin(out.cpu(), x_cpu)


# ---------------------------------------------------------------------------
# Raw hardware max/min NaN semantics (opt-in, device only)
# ---------------------------------------------------------------------------
#
# Measured on TX8110-256 (card 27, rcs1fw v1.1.0, 2026-10-22) for f32 and f16:
# the TX81 max/min ALU is a compare-and-select
#
#     max(a, b) = (a >= b) ? a : b        min(a, b) = (a <= b) ? a : b
#
# with unordered comparisons false.  A NaN in operand 0 is therefore dropped
# (operand 1 is returned), while a NaN in operand 1 is returned unchanged.
# This is neither IEEE maxNum (return the non-NaN operand on either side) nor
# IEEE maximum (return NaN when either operand is NaN).
#
# The tests below pin that behaviour so a firmware update cannot silently
# invalidate the assumptions behind the maxnum mask-move lowering and the
# mk.minmax_nan_masked fusion guard.

_HW_NAN = float("nan")

# (a, b) lanes: NaN in each position, both NaN, infinities, and signed zeros
# (equal values must return operand 0).
HW_MAXMIN_LANES = (
    (_HW_NAN, 1.0),
    (1.0, _HW_NAN),
    (_HW_NAN, _HW_NAN),
    (_HW_NAN, float("inf")),
    (-float("inf"), _HW_NAN),
    (-0.0, 0.0),
    (0.0, -0.0),
    (-3.0, 2.0),
)


def _assert_hw_maxmin(got, expected):
    """Exact comparison: NaN lanes must match and signed zeros are preserved."""
    import torch

    got = got.float()
    expected = expected.float()
    assert torch.equal(torch.isnan(got), torch.isnan(expected)), \
        f"NaN lanes differ: got={got.tolist()}, expected={expected.tolist()}"
    finite = ~torch.isnan(expected)
    assert torch.equal(got[finite], expected[finite]), \
        f"values differ: got={got.tolist()}, expected={expected.tolist()}"
    zeros = finite & (expected == 0)
    assert torch.equal(torch.signbit(got[zeros]), torch.signbit(expected[zeros])), \
        f"signed zeros differ: got={got.tolist()}, expected={expected.tolist()}"


@pytest.mark.skipif(DEVICE is None, reason="no Tx81 device available")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_hw_vector_maxmin_nan_semantics(dtype):
    """tx.maxvv/tx.minvv: NaN in operand 0 is dropped, in operand 1 propagates."""
    import torch

    torch_dtype = getattr(torch, dtype)
    a_cpu = torch.tensor([lane[0] for lane in HW_MAXMIN_LANES], dtype=torch_dtype)
    b_cpu = torch.tensor([lane[1] for lane in HW_MAXMIN_LANES], dtype=torch_dtype)
    n = a_cpu.numel()
    a = a_cpu.to(DEVICE)
    b = b_cpu.to(DEVICE)
    out = torch.full((16, ), 12345.0, dtype=torch_dtype, device=DEVICE)

    max_vector_all_kernel[(1, )](a, b, out, n, BLOCK=16)
    torch.txda.synchronize()
    _assert_hw_maxmin(out[:n].cpu(), torch.where(a_cpu >= b_cpu, a_cpu, b_cpu))

    min_vector_all_kernel[(1, )](a, b, out, n, BLOCK=16)
    torch.txda.synchronize()
    _assert_hw_maxmin(out[:n].cpu(), torch.where(a_cpu <= b_cpu, a_cpu, b_cpu))


# tl.maximum/minimum promote bf16 to f32 in the frontend, so a Triton kernel
# cannot observe the raw bf16 vector-vector instruction.  Feed the bf16
# generic through the external CoreIR entry instead.
_RAW_VV_BF16_MODULE = """#map = affine_map<(d0) -> (d0)>
module {{
  func.func @{name}(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {{
    %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [{block}], strides: [1] : memref<*xbf16> to memref<{block}xbf16, strided<[1], offset: ?>>
    %b = memref.reinterpret_cast %arg1 to offset: [0], sizes: [{block}], strides: [1] : memref<*xbf16> to memref<{block}xbf16, strided<[1], offset: ?>>
    %out = memref.reinterpret_cast %arg2 to offset: [0], sizes: [{block}], strides: [1] : memref<*xbf16> to memref<{block}xbf16, strided<[1], offset: ?>>
    %aa = memref.alloc() : memref<{block}xbf16>
    %bb = memref.alloc() : memref<{block}xbf16>
    memref.copy %a, %aa : memref<{block}xbf16, strided<[1], offset: ?>> to memref<{block}xbf16>
    memref.copy %b, %bb : memref<{block}xbf16, strided<[1], offset: ?>> to memref<{block}xbf16>
    %ta = bufferization.to_tensor %aa restrict writable : memref<{block}xbf16> to tensor<{block}xbf16>
    %tb = bufferization.to_tensor %bb restrict writable : memref<{block}xbf16> to tensor<{block}xbf16>
    %er = tensor.empty() : tensor<{block}xbf16>
    %res = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}} ins(%ta, %tb : tensor<{block}xbf16>, tensor<{block}xbf16>) outs(%er : tensor<{block}xbf16>) {{
    ^bb0(%x: bf16, %y: bf16, %o: bf16):
      %v = {op} %x, %y : bf16
      linalg.yield %v : bf16
    }} -> tensor<{block}xbf16>
    bufferization.materialize_in_destination %res in writable %out : (tensor<{block}xbf16>, memref<{block}xbf16, strided<[1], offset: ?>>) -> ()
    return
  }}
}}
"""


@pytest.mark.skipif(DEVICE is None, reason="no Tx81 device available")
def test_hw_vector_maxmin_nan_semantics_bf16():
    """Raw bf16 tx.maxvv/tx.minvv follow the same asymmetric select rule."""
    import torch
    from triton.backends.tsingmicro.core_ir import (
        ArgumentSpec,
        CompileOptions,
        Grid,
        KernelSignature,
        compile_core_ir,
    )
    from triton.backends.tsingmicro.runtime import load_core_ir_artifact

    block = 16
    a_cpu = torch.tensor([lane[0] for lane in HW_MAXMIN_LANES], dtype=torch.bfloat16)
    b_cpu = torch.tensor([lane[1] for lane in HW_MAXMIN_LANES], dtype=torch.bfloat16)
    while a_cpu.numel() < block:
        a_cpu = torch.cat((a_cpu, torch.zeros(1, dtype=torch.bfloat16)))
        b_cpu = torch.cat((b_cpu, torch.zeros(1, dtype=torch.bfloat16)))
    n = len(HW_MAXMIN_LANES)
    a = a_cpu.to(DEVICE)
    b = b_cpu.to(DEVICE)

    cases = (
        ("arith.maximumf", "tx.maxvv", True),
        ("arith.minimumf", "tx.minvv", False),
    )
    for op, tx_op, is_max in cases:
        name = f"raw_{tx_op.replace('.', '_')}_bf16"
        module = _RAW_VV_BF16_MODULE.format(name=name, block=block, op=op)
        artifact = compile_core_ir(
            module=module,
            kernel_name=name,
            signature=KernelSignature((
                ArgumentSpec("a", "bfloat16", shape=(block, ), readonly=True),
                ArgumentSpec("b", "bfloat16", shape=(block, ), readonly=True),
                ArgumentSpec("out", "bfloat16", shape=(block, )),
            )),
            grid=Grid(1),
            options=CompileOptions(),
        )

        txir = artifact.dumps["txir"]
        assert txir.count(tx_op) == 1, txir
        for masked in ("tx.mask_move", "tx.bf16_fp32", "tx.fp32_bf16"):
            assert masked not in txir, txir

        out = torch.full((block, ), 12345.0, dtype=torch.bfloat16, device=DEVICE)
        load_core_ir_artifact(artifact).launch(a, b, out, device=DEVICE.index)
        torch.txda.synchronize()

        expected = (torch.where(a_cpu >= b_cpu, a_cpu, b_cpu) if is_max else torch.where(a_cpu <= b_cpu, a_cpu, b_cpu))
        _assert_hw_maxmin(out[:n].cpu(), expected[:n])


# Hand-written CoreIR for the raw vector-scalar instruction.  Triton's own
# maxnum path always wraps the instruction in mask-moves, and the
# mk.minmax_nan_masked guard keeps maximumf/minimumf off the fusion, so a
# Triton kernel cannot observe tx.maxvs/tx.minvs directly.  Feeding the marked
# generic through the public external-CoreIR entry bypasses the maxnum rewrite
# (it only matches maxnumf) and lets LinalgFusion scalarize the fill.
_RAW_MAXMIN_MODULE = """#map = affine_map<(d0) -> (d0)>
module {{
  func.func @{name}(%arg0: memref<*x{ty}>, %arg1: memref<*x{ty}>) {{
    %cst = arith.constant {const} : {ty}
    %in = memref.reinterpret_cast %arg0 to offset: [0], sizes: [{block}], strides: [1] : memref<*x{ty}> to memref<{block}x{ty}, strided<[1], offset: ?>>
    %out = memref.reinterpret_cast %arg1 to offset: [0], sizes: [{block}], strides: [1] : memref<*x{ty}> to memref<{block}x{ty}, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<{block}x{ty}>
    memref.copy %in, %alloc : memref<{block}x{ty}, strided<[1], offset: ?>> to memref<{block}x{ty}>
    %tin = bufferization.to_tensor %alloc restrict writable : memref<{block}x{ty}> to tensor<{block}x{ty}>
    %ef = tensor.empty() : tensor<{block}x{ty}>
    %er = tensor.empty() : tensor<{block}x{ty}>
    %fill = linalg.fill ins(%cst : {ty}) outs(%ef : tensor<{block}x{ty}>) -> tensor<{block}x{ty}>
    %res = linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], mk.minmax_nan_masked}} ins(%tin, %fill : tensor<{block}x{ty}>, tensor<{block}x{ty}>) outs(%er : tensor<{block}x{ty}>) {{
    ^bb0(%x: {ty}, %c: {ty}, %o: {ty}):
      %v = {op} %x, %c : {ty}
      linalg.yield %v : {ty}
    }} -> tensor<{block}x{ty}>
    bufferization.materialize_in_destination %res in writable %out : (tensor<{block}x{ty}>, memref<{block}x{ty}, strided<[1], offset: ?>>) -> ()
    return
  }}
}}
"""

# MLIR hex float literals are the raw IEEE bit patterns of the target type.
_RAW_MAXMIN_NAN_LITERAL = {
    "float32": "0x7FC00000",
    "float16": "0x7E00",
    "bfloat16": "0x7FC0",
}
_RAW_MAXMIN_MLIR_TYPE = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
}

HW_MAXMIN_SCALAR_LANES = (
    _HW_NAN,
    -float("inf"),
    -1.0,
    -0.0,
    0.0,
    1.0,
    float("inf"),
)


@pytest.mark.skipif(DEVICE is None, reason="no Tx81 device available")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "op, const, tx_op",
    [
        ("arith.maximumf", 0.0, "tx.maxvs"),
        ("arith.maximumf", _HW_NAN, "tx.maxvs"),
        ("arith.minimumf", 0.0, "tx.minvs"),
        ("arith.minimumf", _HW_NAN, "tx.minvs"),
    ],
    ids=["max-zero", "max-nan", "min-zero", "min-nan"],
)
def test_hw_scalar_maxmin_nan_semantics(dtype, op, const, tx_op):
    """tx.maxvs/tx.minvs: NaN in operand 0 is dropped, in operand 1 propagates."""
    import torch
    from triton.backends.tsingmicro.core_ir import (
        ArgumentSpec,
        CompileOptions,
        Grid,
        KernelSignature,
        compile_core_ir,
    )
    from triton.backends.tsingmicro.runtime import load_core_ir_artifact

    block = 16
    ty = _RAW_MAXMIN_MLIR_TYPE[dtype]
    torch_dtype = getattr(torch, dtype)
    is_max = op == "arith.maximumf"
    is_nan_const = math.isnan(const)
    name = (f"raw_{tx_op.replace('.', '_')}_"
            f"{'nan' if is_nan_const else 'zero'}_{ty}")
    const_literal = (_RAW_MAXMIN_NAN_LITERAL[dtype] if is_nan_const else repr(float(const)))
    module = _RAW_MAXMIN_MODULE.format(name=name, block=block, const=const_literal, op=op, ty=ty)

    lanes = list(HW_MAXMIN_SCALAR_LANES)
    lanes += [0.0] * (block - len(lanes))
    x_cpu = torch.tensor(lanes, dtype=torch_dtype)
    x = x_cpu.to(DEVICE)

    artifact = compile_core_ir(
        module=module,
        kernel_name=name,
        signature=KernelSignature((
            ArgumentSpec("x", dtype, shape=(block, ), readonly=True),
            ArgumentSpec("out", dtype, shape=(block, )),
        )),
        grid=Grid(1),
        options=CompileOptions(),
    )

    # Make sure the fusion actually produced the bare instruction, otherwise
    # the numeric check below would be testing something else.  The f16 case
    # must also stay on the native instruction, not a f16->f32->f16 round trip.
    txir = artifact.dumps["txir"]
    assert txir.count(tx_op) == 1, txir
    for masked in ("tx.mask_move", "tx.unequalvv", "tx.equalvs", "tx.fp16_fp32", "tx.fp32_fp16", "tx.bf16_fp32",
                   "tx.fp32_bf16"):
        assert masked not in txir, txir

    out = torch.full((block, ), 12345.0, dtype=torch_dtype, device=DEVICE)
    load_core_ir_artifact(artifact).launch(x, out, device=DEVICE.index)
    torch.txda.synchronize()

    c_cpu = torch.full_like(x_cpu, const)
    expected = (torch.where(x_cpu >= c_cpu, x_cpu, c_cpu) if is_max else torch.where(x_cpu <= c_cpu, x_cpu, c_cpu))
    n = len(HW_MAXMIN_SCALAR_LANES)
    _assert_hw_maxmin(out[:n].cpu(), expected[:n])
