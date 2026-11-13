"""Swizzled buffer copies can recompute masks when all wave lanes execute."""
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def lower_copy(tmp_path, divergent):
    limit = "%lane" if divergent else "%trips"
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}}>
#shared = #ttg.swizzled_shared<{{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @copy(%ptr: !tt.ptr<bf16>) {{
    ttg.warp_specialize(%ptr) attributes {{requestedRegisters = array<i32: 32>}}
    default {{ ttg.warp_yield }}
    partition0(%p: !tt.ptr<bf16>) num_warps(4) {{
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %trips = arith.constant 3 : i32
      %tile_k = arith.constant 64 : i32
      %lane = rocdl.workitem.id.x : i32
      %rows = tt.make_range {{end = 256 : i32, start = 0 : i32}}
        : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
      %expanded_rows = tt.expand_dims %rows {{axis = 0 : i32}}
        : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x256xi32, #blocked>
      %stride = arith.constant dense<256> : tensor<1x256xi32, #blocked>
      %row_offsets = arith.muli %expanded_rows, %stride : tensor<1x256xi32, #blocked>
      %rows_broadcast = tt.broadcast %row_offsets : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %columns = tt.make_range {{end = 64 : i32, start = 0 : i32}}
        : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
      %expanded_columns = tt.expand_dims %columns {{axis = 1 : i32}}
        : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<64x1xi32, #blocked>
      %columns_broadcast = tt.broadcast %expanded_columns : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %local = arith.addi %rows_broadcast, %columns_broadcast : tensor<64x256xi32, #blocked>
      %bound = arith.constant dense<255> : tensor<1x256xi32, #blocked>
      %row_mask = arith.cmpi ult, %expanded_rows, %bound : tensor<1x256xi32, #blocked>
      %mask = tt.broadcast %row_mask : tensor<1x256xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %dest = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
      scf.for %iv = %zero to {limit} step %one : i32 {{
        %origin = arith.muli %iv, %tile_k : i32
        %base = tt.splat %origin : i32 -> tensor<64x256xi32, #blocked>
        %offset = arith.addi %base, %local : tensor<64x256xi32, #blocked>
        %copy = hcug.buffer_load_to_local %p[%offset] mask = %mask into %dest
          {{contiguity = 8 : i32}} : <bf16>[tensor<64x256xi32, #blocked>]
          -> <64x256xbf16, #shared, #smem, mutable>
        %commit = ttg.async_commit_group tokens %copy
        %wait = hcug.async_wait %commit {{num_inst = 0 : i32}}
      }}
      ttg.warp_return
    }} : (!tt.ptr<bf16>) -> ()
    tt.return
  }}
}}'''
    path = tmp_path / "copy.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    target_passes = libtriton.hcu.passes.ttgpuir
    target_passes.add_warp_specialize_to_llvm(manager, "gfx936", 4, 4, False, 0, 0, 0)
    libtriton.passes.convert.add_scf_to_cf(manager)
    target_passes.add_allocate_shared_memory(manager)
    target_passes.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "full_wave_buffer_copy")
    return str(module)


@pytest.mark.parametrize("divergent", [False, True])
def test_buffer_copy_retains_ballot_for_divergent_loop(tmp_path, divergent):
    lowered = lower_copy(tmp_path, divergent)
    assert "buffer.load.lds" in lowered
    assert ("rocdl.ballot" in lowered) == divergent
