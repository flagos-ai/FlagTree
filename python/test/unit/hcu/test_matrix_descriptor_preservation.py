"""Block-pointer descriptor preservation and exact-index fallback contracts."""
import re
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource


def rewrite(tmp_path, body, enabled=True):
    attr = ' attributes {ttg.hcu.prefer_wide_async_copy = 1 : i32}' if enabled else ''
    path = tmp_path / "descriptor.mlir"
    path.write_text("module" + attr + " {\n" + body + "\n}")
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.passes.ttir.add_rewrite_tensor_pointer(manager)
    manager.run(module, "matrix_descriptor_preservation")
    return str(module)


@pytest.mark.parametrize("enabled,async_,volatile,padding,wide_shape,expected", [
    (True, True, False, 1, False, True),
    (True, True, False, None, False, True),
    (False, True, False, 1, False, False),
    (True, False, False, 1, False, True),
    (True, True, True, 1, False, False),
    (True, True, False, 2, False, False),
    (True, True, False, 1, True, False),
])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_descriptor_preserves_semantics_or_falls_back(
        tmp_path, enabled, async_, volatile, padding, wide_shape, expected, dtype):
    shape = "%wide" if wide_shape else "%n64"
    pad = f", padding = {padding} : i32" if padding is not None else ""
    text = f'''tt.func public @copy(%p: !tt.ptr<{dtype}>, %n: i32,
      %wide: i64, %stride: i64, %offset: i32) -> tensor<16x16x{dtype}> {{
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i64
      %n64 = arith.extsi %n : i32 to i64
      %bp = tt.make_tensor_ptr %p, [%n64, {shape}], [%stride, %one],
        [%offset, %zero] {{order = array<i32: 1, 0>}}
        : !tt.ptr<tensor<16x16x{dtype}>>
      %x = tt.load %bp {{boundaryCheck = array<i32: 0, 1>{pad},
        tt.load.async = {str(async_).lower()}, isVolatile = {str(volatile).lower()}}}
        : !tt.ptr<tensor<16x16x{dtype}>>
      tt.return %x : tensor<16x16x{dtype}>
    }}'''
    result = rewrite(tmp_path, text, enabled)
    assert ("tt.matrix_load" in result) == expected
    assert "tt.make_tensor_ptr" not in result
    if expected:
        assert re.search(r"boundaryCheck = (?:array<i32: 0, 1>|\[0, 1\])", result)
        assert "tt.load.async = " + str(async_).lower() in result
        assert "tt.addptr" not in result
    else:
        assert "tt.load " in result and "tt.addptr" in result


@pytest.mark.parametrize("advance,initial,expected", [
    (16, 7, True), (8, 7, False), (16, 0, False),
])
def test_loop_descriptor_requires_exact_induction_recurrence(
        tmp_path, advance, initial, expected):
    text = f'''tt.func public @loop(%p: !tt.ptr<f16>, %n: i32,
      %stride: i64, %row: i32) -> tensor<16x16xf16> {{
      %zero = arith.constant 0 : i32
      %lower = arith.constant 7 : i32
      %initial = arith.constant {initial} : i32
      %step = arith.constant 16 : i32
      %delta = arith.constant {advance} : i32
      %one = arith.constant 1 : i64
      %n64 = arith.extsi %n : i32 to i64
      %empty = arith.constant dense<0.0> : tensor<16x16xf16>
      %bp = tt.make_tensor_ptr %p, [%n64, %n64], [%stride, %one],
        [%row, %initial] {{order = array<i32: 1, 0>}}
        : !tt.ptr<tensor<16x16xf16>>
      %result:2 = scf.for %i = %lower to %n step %step
        iter_args(%q = %bp, %acc = %empty)
        -> (!tt.ptr<tensor<16x16xf16>>, tensor<16x16xf16>) : i32 {{
        %x = tt.load %q {{boundaryCheck = array<i32: 0, 1>,
          padding = 1 : i32, tt.load.async = true}}
          : !tt.ptr<tensor<16x16xf16>>
        %next = tt.advance %q, [%zero, %delta]
          : !tt.ptr<tensor<16x16xf16>>
        %sum = arith.addf %acc, %x : tensor<16x16xf16>
        scf.yield %next, %sum
          : !tt.ptr<tensor<16x16xf16>>, tensor<16x16xf16>
      }}
      tt.return %result#1 : tensor<16x16xf16>
    }}'''
    result = rewrite(tmp_path, text)
    assert ("tt.matrix_load" in result) == expected
    assert "tt.make_tensor_ptr" not in result and "tt.advance" not in result


@triton.jit
def _unpipelined_descriptor_dot(A, B, C, BK: tl.constexpr, ASYNC: tl.constexpr, COL):
    a = tl.make_block_ptr(A, (1, BK), (BK, 1), (0, 0), (128, BK), (1, 0))
    b = tl.make_block_ptr(B, (BK, 16), (16, 1), (0, COL), (BK, 128), (1, 0))
    av = tle.load(a, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC)
    bv = tle.load(b, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC)
    value = tl.dot(av, bv)
    m = tl.arange(0, 128)
    n = tl.arange(0, 128)
    tl.store(C + m[:, None] * 16 + n[None, :], value,
             (m[:, None] < 1) & (n[None, :] < 16))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("block_k", [16, 32])
@pytest.mark.parametrize("kpack", [1, 2])
@pytest.mark.parametrize("async_", [False, True])
def test_unpipelined_descriptor_is_lowered_before_pointer_canonicalization(dtype, block_k, kpack, async_):
    source = ASTSource(
        _unpipelined_descriptor_dot,
        {"A": "*" + dtype, "B": "*" + dtype, "C": "*" + dtype,
         "BK": "constexpr", "ASYNC": "constexpr", "COL": "constexpr"},
        {"BK": block_k, "ASYNC": async_, "COL": 0},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=4, kpack=kpack,
                     schedule_hint="local-prefetch", mmac_warps_m=2,
                     mmac_transposed=True, optimize_epilogue=True),
    )
    if async_:
        assert "tt.matrix_load " not in kernel.asm["ttgir"]
        assert "hcug.matrix_load_to_local" in kernel.asm["ttgir"]
        assert re.search(r"ttg.local_load[^\n]* token ", kernel.asm["ttgir"])
        assert "!noalias" in kernel.asm["llir"]
        assert '\"dereferenceable\"(ptr addrspace(3)' in kernel.asm["llir"]
    else:
        assert "hcu.structured_register" in kernel.asm["ttgir"]
        assert "llvm.amdgcn.struct.ptr.buffer.load." in kernel.asm["llir"]

    # The descriptor is not a fixed-width native MLS instruction. Honor the
    # ordinary dot option, while K16 still clamps the illegal packed K group.
    expected = 8 if kpack == 2 and block_k >= 32 else 4
    assert set(re.findall(r"kWidth = (\d+)", kernel.asm["ttgir"])) == {str(expected)}


def test_async_descriptor_with_unproved_column_alignment_falls_back():
    source = ASTSource(
        _unpipelined_descriptor_dot,
        {"A": "*bf16", "B": "*bf16", "C": "*bf16", "COL": "i32",
         "BK": "constexpr", "ASYNC": "constexpr"},
        {"BK": 32, "ASYNC": True},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(3)},
    )
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=4, kpack=2,
                     schedule_hint="local-prefetch", mmac_warps_m=2, mmac_transposed=True),
    )
    assert kernel.asm["ttgir"].count("hcug.matrix_load_to_local") == 1
    assert "tt.matrix_load " not in kernel.asm["ttgir"]


@triton.jit
def _aligned_loop_descriptor_dot(A, B, C):
    value = tl.full((128, 128), 0, tl.float32)
    for k in tl.range(0, 128, 32, num_stages=3):
        a = tl.make_block_ptr(A, (128, 128), (128, 1), (0, k), (128, 32), (1, 0))
        b = tl.make_block_ptr(B, (128, 128), (1, 128), (k, 0), (32, 128), (0, 1))
        av = tle.load(a, boundary_check=(0, 1), padding_option="zero", is_async=True)
        bv = tle.load(b, boundary_check=(0, 1), padding_option="zero", is_async=True)
        value = tl.dot(av, bv, value)
    m = tl.arange(0, 128)
    n = tl.arange(0, 128)
    tl.store(C + m[:, None] * 128 + n[None, :], value)


def test_async_descriptor_accepts_aligned_loop_induction_in_both_orders():
    source = ASTSource(
        _aligned_loop_descriptor_dot, {"A": "*bf16", "B": "*bf16", "C": "*bf16"},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(3)},
    )
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=3, kpack=2,
                     schedule_hint="local-prefetch", mmac_warps_m=2, mmac_transposed=True),
    )
    assert kernel.asm["ttgir"].count("hcug.matrix_load_to_local") >= 2
