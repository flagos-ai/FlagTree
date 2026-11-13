"""A closed view tree must keep a single register load at its memory event."""
import re

import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def lower(tmp_path, dtype, negative=None, pre_split=False):
    root = f"tensor<128x64x{dtype}, #blocked>"
    reshaped = f"tensor<128x2x32x{dtype}, #reshaped>"
    transposed = f"tensor<128x32x2x{dtype}, #transposed>"
    half = f"tensor<128x32x{dtype}, #blocked>"
    dot = f"tensor<128x32x{dtype}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>"
    other = dot.replace("kWidth = 8", "kWidth = 4") if negative == "incompatible" else dot
    volatile = ", isVolatile = true" if negative == "volatile" else ""
    extra = f"tt.store %pointers, %value : tensor<128x64x!tt.ptr<{dtype}>, #blocked>" if negative == "fanout" else ""
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}}>
#reshaped = #ttg.blocked<{{sizePerThread = [1, 2, 2], threadsPerWarp = [4, 1, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}}>
#transposed = #ttg.blocked<{{sizePerThread = [1, 2, 2], threadsPerWarp = [4, 16, 1], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}}>
#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @views(%p: !tt.ptr<{dtype}>, %q: !tt.ptr<{dtype}>) {{
    ttg.warp_specialize() attributes {{requestedRegisters = array<i32: 32>}}
    default {{
      %pointers = tt.splat %p : !tt.ptr<{dtype}> -> tensor<128x64x!tt.ptr<{dtype}>, #blocked>
      %extent = arith.constant 128 : i32
      %columns = arith.constant 64 : i32
      %stride = arith.constant 64 : i64
      %unit_stride = arith.constant 1 : i64
      %origin = arith.constant 0 : i32
      %enabled = arith.constant 1 : i32
      %value = tt.matrix_load %p, shape = [%extent, %columns], strides = [%stride, %unit_stride], tensorShape = [128, 64], indices = [%origin, %origin, %enabled] {{hcu.structured_register, tt.load.async = false{volatile}}} : <{dtype}> -> {root}
      %zero = arith.constant 0 : i32
      "hcug.cta_barrier"() : () -> ()
      %reshape = tt.reshape %value : {root} -> {reshaped}
      %transpose = tt.trans %reshape {{order = array<i32: 0, 2, 1>}} : {reshaped} -> {transposed}
      %lo, %hi = tt.split %transpose : {transposed} -> {half}
      %a = ttg.convert_layout %lo : {half} -> {dot}
      %b = ttg.convert_layout %hi : {half} -> {other}
      %qa = tt.splat %q : !tt.ptr<{dtype}> -> tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>
      %qb = tt.splat %q : !tt.ptr<{dtype}> -> tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {4 if negative == 'incompatible' else 8}}}>>
      tt.store %qa, %a : tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>
      tt.store %qb, %b : tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {4 if negative == 'incompatible' else 8}}}>>
      {extra}
      ttg.warp_yield
    }}
    partition0() num_warps(4) {{ ttg.warp_return }} : () -> ()
    tt.return
  }}
}}'''
    # The real typed CTA barrier is retained as an ordering witness.
    text = text.replace('"hcug.cta_barrier"() : () -> ()',
                        'ttg.local_barrier {hcu.cta_scope}')
    if pre_split:
        chart = "#target = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 4, 0], [64, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 8, 0], [0, 16, 0]], warp = [[16, 0, 0], [32, 0, 0]], block = []}>\n"
        text = chart + text
        target = f"tensor<128x32x2x{dtype}, #target>"
        text = text.replace(
            f"%lo, %hi = tt.split %transpose : {transposed} -> {half}",
            f"%view = ttg.convert_layout %transpose : {transposed} -> {target}\n"
            f"      %lo, %hi = tt.split %view : {target} -> {dot}")
        text = text.replace(f"%lo : {half} ->", f"%lo : {dot} ->")
        text = text.replace(f"%hi : {half} ->", f"%hi : {dot} ->")
    path = tmp_path / "views.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_optimize_dot_operands(manager, "gfx936")
    manager.run(module, "register_load_views")
    return before, str(module)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_split_consumers_infer_one_full_load(tmp_path, dtype):
    before, after = lower(tmp_path, dtype)
    assert before.count("tt.matrix_load ") == after.count("tt.matrix_load ") == 1
    assert re.search(r"tt.matrix_load .*tensor<128x64x" + dtype + r", #linear", after)
    assert after.index("tt.matrix_load ") < after.index("ttg.local_barrier")
    assert "ttg.local_alloc" not in after


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("negative", ["incompatible", "fanout", "volatile"])
def test_view_load_falls_back_without_changing_memory(tmp_path, dtype, negative):
    before, after = lower(tmp_path, dtype, negative)
    assert before.count("tt.matrix_load ") == after.count("tt.matrix_load ") == 1
    load = next(line for line in after.splitlines() if "tt.matrix_load " in line)
    assert "#blocked" in load
    assert after.index("tt.matrix_load ") < after.index("ttg.local_barrier")


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_conversion_before_split_retargets_original_descriptor(tmp_path, dtype):
    before, after = lower(tmp_path, dtype, pre_split=True)
    assert before.count("tt.matrix_load ") == after.count("tt.matrix_load ") == 1
    assert re.search(r"tt.matrix_load .*tensor<128x64x" + dtype + r", #linear", after)
    assert after.index("tt.matrix_load ") < after.index("ttg.local_barrier")
    assert "ttg.local_alloc" not in after
