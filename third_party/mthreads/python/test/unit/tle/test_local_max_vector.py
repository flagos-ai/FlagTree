"""Opt-in local maxnum tree; ordinary tl.max and sum semantics are isolated."""
import os
import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _reduce(x, out, SUM: tl.constexpr, L: tl.constexpr):
    i = tl.arange(0, 32)[:, None] * 128 + tl.arange(0, 128)[None, :]
    values = tle.gpu.set_layout(tl.load(x + i), L)
    if SUM:
        result = tl.sum(values, 1)
    else:
        result = tl.max(values, 1)
    tl.store(out + tl.arange(0, 32), result)


@pytest.mark.parametrize("width", [1, 4])
def test_local_max_lowering(width):
    k = compile_musa(_reduce, {"x": "*fp32", "out": "*fp32", "SUM": "constexpr", "L": "constexpr"},
                     {"SUM": False, "L": tle.gpu.BlockEncoding([1, 16], [4, 8], [4, 1], [1, 0])},
                     options={"num_warps": 4, "local_max_vector_width": width})
    assert ("llvm.maxnum.v4f32" in k.asm["llir"]) == (width == 4)


def test_local_sum_unchanged():
    args = (_reduce, {"x": "*fp32", "out": "*fp32", "SUM": "constexpr", "L":
                      "constexpr"}, {"SUM": True, "L": tle.gpu.BlockEncoding([1, 16], [4, 8], [4, 1], [1, 0])})
    a = compile_musa(*args, options={"num_warps": 4, "local_max_vector_width": 1})
    b = compile_musa(*args, options={"num_warps": 4, "local_max_vector_width": 4})
    assert a.asm["llir"] == b.asm["llir"]


@triton.jit
def _sum_linear(x, out, L: tl.constexpr):
    i = tl.arange(0, 64)[:, None] * 128 + tl.arange(0, 128)[None, :]
    values = tle.gpu.set_layout(tl.load(x + i), L)
    four = tl.sum(values.reshape(64, 4, 4, 8), 1)
    result = tl.sum(tl.sum(four, 1), 1)
    tl.store(out + tl.arange(0, 64), result)


@pytest.mark.skipif(
    not (hasattr(ir.builder, "get_tle_linear_encoding")
         or os.environ.get("TRITON_MUSA_TEST_LAYOUT_INTEGRATION") == "1"),
    reason="requires companion explicit-layout propagation PR #1606")
@pytest.mark.parametrize("width", [1, 4])
def test_independent_linear_sum_chains_with_layout_companion(width):
    k = compile_musa(_sum_linear, {"x": "*fp32", "out": "*fp32", "L": "constexpr"},
                     {"L": tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])},
                     options={"num_warps": 8, "fp32_vector_width": width, "enable_fp_fusion": False})
    assert ("fadd <4 x float>" in k.asm["llir"]) == (width == 4)
    assert " reassoc " not in k.asm["llir"]


@pytest.mark.parametrize("width", [0, 2, True, 4.0, "4"])
def test_invalid_max_width(width):
    _, backend = mthreads_backend()
    with pytest.raises(ValueError, match="local_max_vector_width must be 1 or 4"):
        backend.parse_options({"local_max_vector_width": width})


@pytest.mark.parametrize("width", [1, 4])
def test_standalone_linear_sum_chains(tmp_path, width):
    """Four independent rows per thread, preserving each four-element sum chain.

    Explicit TTGIR separates numerical lowering from frontend reshape/layout
    propagation. Pointers and reduced values have the same declared slice.
    """
    from test_tle_utils import musa_target
    path = tmp_path / "linear_sum.ttgir"
    path.write_text("""
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[4, 0], [8, 0]], block = []}>
#rows = #ttg.slice<{dim = 1, parent = #linear}>
#cols = #ttg.slice<{dim = 0, parent = #linear}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @linear_sum(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #rows>
    %c = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #cols>
    %stride = arith.constant dense<128> : tensor<16xi32, #rows>
    %offset = arith.muli %r, %stride : tensor<16xi32, #rows>
    %r2 = tt.expand_dims %offset {axis = 1 : i32} : tensor<16xi32, #rows> -> tensor<16x1xi32, #linear>
    %c2 = tt.expand_dims %c {axis = 0 : i32} : tensor<128xi32, #cols> -> tensor<1x128xi32, #linear>
    %rb = tt.broadcast %r2 : tensor<16x1xi32, #linear> -> tensor<16x128xi32, #linear>
    %cb = tt.broadcast %c2 : tensor<1x128xi32, #linear> -> tensor<16x128xi32, #linear>
    %index = arith.addi %rb, %cb : tensor<16x128xi32, #linear>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<16x128x!tt.ptr<f32>, #linear>
    %ptr = tt.addptr %base, %index : tensor<16x128x!tt.ptr<f32>, #linear>, tensor<16x128xi32, #linear>
    %values = tt.load %ptr : tensor<16x128x!tt.ptr<f32>, #linear>
    %sum = "tt.reduce"(%values) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %next = arith.addf %a, %b : f32
      tt.reduce.return %next : f32
    }) : (tensor<16x128xf32, #linear>) -> tensor<16xf32, #rows>
    %out = tt.splat %output : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #rows>
    %outptr = tt.addptr %out, %r : tensor<16x!tt.ptr<f32>, #rows>, tensor<16xi32, #rows>
    tt.store %outptr, %sum : tensor<16x!tt.ptr<f32>, #rows>
    tt.return
  }
}
""")
    kernel = triton.compile(str(path), target=musa_target(),
                            options={"num_warps": 4, "fp32_vector_width": width, "enable_fp_fusion": False})
    assert kernel.module is None and kernel.function is None
    assert ("fadd <4 x float>" in kernel.asm["llir"]) == (width == 4)
    assert " reassoc " not in kernel.asm["llir"]
