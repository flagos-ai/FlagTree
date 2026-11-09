"""Layout rematerialization preserves the shared-memory value being read."""
import re

import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu") or not hasattr(libtriton, "tle"):
    pytest.skip("HCU/TLE compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


@pytest.mark.parametrize("warp_size", [32, 64])
@pytest.mark.parametrize("paired_reads", [False, True])
def test_layout_rematerialization_keeps_read_before_overwrite(tmp_path, warp_size, paired_reads):
    target = "cuda:90" if warp_size == 32 else "hip:gfx936"
    second_read = """
    %later = tt.load %ptr : tensor<128x!tt.ptr<i32, 3>, #blocked>
    %sum = arith.addi %first, %later : tensor<128xi32, #blocked>
""" if paired_reads else ""
    value = "%sum" if paired_reads else "%first"
    source = f"""
#blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [{warp_size}], warpsPerCTA = [2], order = [0]}}>
#other = #ttg.blocked<{{sizePerThread = [2], threadsPerWarp = [{warp_size}], warpsPerCTA = [2], order = [0]}}>
#shared = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "{target}", "ttg.threads-per-warp" = {warp_size} : i32}} {{
  tt.func public @read_before_overwrite(%mem: !ttg.memdesc<2x128xi32, #shared, #smem, mutable>, %slot: i32) -> tensor<128xi32, #other> {{
    %row = tt.splat %slot : i32 -> tensor<128xi32, #blocked>
    %col = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #blocked>
    %ptr = "tle.local_pointers"(%mem, %row, %col) : (!ttg.memdesc<2x128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128x!tt.ptr<i32, 3>, #blocked>
    %first = tt.load %ptr : tensor<128x!tt.ptr<i32, 3>, #blocked>
    ttg.local_barrier
    %zero = arith.constant dense<0> : tensor<128xi32, #blocked>
    tt.store %ptr, %zero : tensor<128x!tt.ptr<i32, 3>, #blocked>
    {second_read}
    %out = ttg.convert_layout {value} : tensor<128xi32, #blocked> -> tensor<128xi32, #other>
    tt.return %out : tensor<128xi32, #other>
  }}
}}
"""
    path = tmp_path / "read_order.mlir"
    path.write_text(source)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    libtriton.tle.load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.tle.passes.add_optimize_local_pointer_loads(manager)
    manager.run(module, "preserve_shared_read_position")
    after = str(module)
    loads = [m.start() for m in re.finditer(r"\btt.load ", after)]
    barrier = after.index("ttg.local_barrier")
    store = after.index("tt.store")
    assert len(loads) == (2 if paired_reads else 1)
    assert loads[0] < barrier < store
    if paired_reads:
        assert store < loads[1]
    assert "ttg.convert_layout" not in after
