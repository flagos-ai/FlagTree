"""Pure-view folding must retain shared-read completion and shared users."""
import re
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)
from triton.backends.hcu.compiler import HIPBackend


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("fanout", [False, True])
@pytest.mark.parametrize("through_loop", [False, True])
def test_shared_read_view_conversion_preserves_token_and_fanout(tmp_path, dtype, fanout, through_loop):
    shared = "!ttg.memdesc<16x256x" + dtype + ", #shared, #smem, mutable>"
    original = "tensor<16x256x" + dtype + ", #blocked>"
    split = "tensor<16x8x16x2x" + dtype + ", #split>"
    transposed = "tensor<16x8x2x16x" + dtype + ", #transposed>"
    output = "tensor<16x8x2x16x" + dtype + ", #linear>"
    returns = "(" + output + ", " + original + ")" if fanout else output
    operands = "%out, %value" if fanout else "%out"
    return_types = output + ", " + original if fanout else output
    read = f"%value = ttg.local_load %mem token %ready : {shared} -> {original}"
    if through_loop:
        read = f"""
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %initial = ttg.local_load %mem token %ready : {shared} -> {original}
    %value = scf.for %i = %zero to %iterations step %one
        iter_args(%held = %initial) -> ({original}) : i32 {{
      %next = ttg.local_load %mem token %ready : {shared} -> {original}
      scf.yield %next : {original}
    }}
"""
    source = f"""
#blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}}>
#split = #ttg.blocked<{{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 2, 16, 2], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}}>
#transposed = #ttg.blocked<{{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 2, 2, 16], warpsPerCTA = [1, 4, 1, 1], order = [2, 3, 1, 0]}}>
#linear = #ttg.linear<{{register = [[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 4, 0, 0]], lane = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [4, 0, 0, 0], [8, 0, 0, 0]], warp = [[0, 2, 0, 0], [0, 0, 0, 0]], block = []}}>
#shared = #ttg.shared_linear<{{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [1, 0], [2, 0], [4, 0], [8, 0]]}}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32}} {{
  tt.func public @view(%mem: {shared}, %ready: !ttg.async.token, %iterations: i32) -> {returns} {{
    {read}
    %split = tt.reshape %value : {original} -> {split}
    %transposed = tt.trans %split {{order = array<i32: 0, 1, 3, 2>}} : {split} -> {transposed}
    %out = ttg.convert_layout %transposed : {transposed} -> {output}
    tt.return {operands} : {return_types}
  }}
}}
"""
    path = tmp_path / "view.mlir"
    path.write_text(source)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_mls_lowering_pass(manager)
    manager.run(module, "shared_read_view_layout")
    after = str(module)
    assert after.count("ttg.local_load ") == (2 if through_loop else 1)
    if through_loop:
        assert "scf.for" in after
    token = r"ttg.local_load (\S+) token (\S+)"
    assert re.findall(token, before) == re.findall(token, after)
    assert ("ttg.convert_layout " in after) == fanout
