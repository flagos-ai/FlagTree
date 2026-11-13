"""Widen loads using address groups, without changing register ownership."""
import re

import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def source(dtype, layout="permuted", alignment=16, column=0):
    registers = "[[0, 32], [0, 1], [0, 2], [0, 4], [64, 0]]"
    lanes = "[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]]"
    if layout == "ordered":
        registers = "[[0, 1], [0, 2], [0, 4], [0, 32], [64, 0]]"
    elif layout == "lane_split":
        registers = "[[0, 32], [0, 8], [0, 2], [0, 4], [64, 0]]"
        lanes = "[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 16]]"
    return f'''#layout = #ttg.linear<{{register = {registers}, lane = {lanes}, warp = [[16, 0], [32, 0]], block = []}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @vectors(%p: !tt.ptr<{dtype}> {{tt.divisibility = {alignment} : i32}},
                           %q: !tt.ptr<{dtype}>, %M: i32, %row: i32) {{
    %zero = arith.constant 0 : i32
    %column = arith.constant {column} : i32
    %one = arith.constant 1 : i32
    %K = arith.constant 64 : i32
    %stride = arith.constant 64 : i64
    %unit = arith.constant 1 : i64
    %tile = tt.matrix_load %p, shape = [%M, %K], strides = [%stride, %unit],
      tensorShape = [128, 64], indices = [%row, %column, %one]
      {{hcu.structured_register, hcu.structured_register.internal_width = 8 : i32,
        hcu.structured_register.internal_order = array<i32: 999>}}
      : <{dtype}> -> tensor<128x64x{dtype}, #layout>
    %m = tt.make_range {{start = 0 : i32, end = 128 : i32}}
      : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #layout}}>>
    %n = tt.make_range {{start = 0 : i32, end = 64 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 0, parent = #layout}}>>
    %ms = tt.expand_dims %m {{axis = 1 : i32}}
      : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #layout}}>> -> tensor<128x1xi32, #layout>
    %ns = tt.expand_dims %n {{axis = 0 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 0, parent = #layout}}>> -> tensor<1x64xi32, #layout>
    %mb = tt.broadcast %ms : tensor<128x1xi32, #layout> -> tensor<128x64xi32, #layout>
    %nb = tt.broadcast %ns : tensor<1x64xi32, #layout> -> tensor<128x64xi32, #layout>
    %stride_tile = arith.constant dense<64> : tensor<128x64xi32, #layout>
    %row_offset = arith.muli %mb, %stride_tile : tensor<128x64xi32, #layout>
    %offset = arith.addi %row_offset, %nb : tensor<128x64xi32, #layout>
    %base = tt.splat %q : !tt.ptr<{dtype}> -> tensor<128x64x!tt.ptr<{dtype}>, #layout>
    %ptrs = tt.addptr %base, %offset : tensor<128x64x!tt.ptr<{dtype}>, #layout>, tensor<128x64xi32, #layout>
    tt.store %ptrs, %tile : tensor<128x64x!tt.ptr<{dtype}>, #layout>
    tt.return
  }}
}}'''


def lower(tmp_path, dtype, layout, alignment, column):
    path = tmp_path / "vectors.ttgir"
    path.write_text(source(dtype, layout, alignment, column))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "structured_register_vectors")
    result = str(module)
    (tmp_path / "vectors.llvm.mlir").write_text(result)
    return result


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("layout,alignment,column,width,calls", [
    ("ordered", 16, 0, "v4i32", 4),
    ("permuted", 16, 0, "v4i32", 4),
    ("permuted", 8, 0, "v2i32", 8),
    ("permuted", 2, 0, "i16", 32),
    ("permuted", 16, 1, "i16", 32),
    ("lane_split", 16, 0, "i16", 32),
])
def test_address_grouping_respects_thread_alignment_and_column(
    tmp_path, dtype, layout, alignment, column, width, calls
):
    result = lower(tmp_path, dtype, layout, alignment, column)
    name = "llvm.amdgcn.struct.ptr.buffer.load." + width
    assert len(re.findall(r"llvm.call @" + re.escape(name) + r"\(", result)) == calls
    assert "bpermute" not in result and "llvm.alloca" not in result
