"""Selected-lane address rematerialization reuses proven uniform quotients."""
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def lower(tmp_path, divisor, varying=False, dynamic=False):
    leaf = (
        "%source = rocdl.workitem.id.x : i32"
        if varying
        else "%source = tt.get_program_id x : i32"
    )
    divisor_value = "%dynamic" if dynamic else "%divisor"
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}}>
#shared = #ttg.swizzled_shared<{{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @copy(%p: !tt.ptr<bf16> {{tt.divisibility = 16 : i32,
      tt.pointer_range = 32 : i32}}, %dynamic: i32) {{
    %divisor = arith.constant {divisor} : i32
    %width = arith.constant 256 : i32
    {leaf}
    %group = arith.remui %source, {divisor_value} : i32
    %origin = arith.muli %group, %width : i32
    %columns = tt.make_range {{end = 256 : i32, start = 0 : i32}}
      : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %base = tt.splat %origin
      : i32 -> tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %column = arith.addi %base, %columns
      : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %expanded = tt.expand_dims %column {{axis = 0 : i32}}
      : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
      -> tensor<1x256xi32, #blocked>
    %stride = arith.constant dense<2048> : tensor<1x256xi32, #blocked>
    %rows = arith.muli %expanded, %stride : tensor<1x256xi32, #blocked>
    %row_offsets = tt.broadcast %rows
      : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %krange = tt.make_range {{end = 64 : i32, start = 0 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
    %kexpanded = tt.expand_dims %krange {{axis = 1 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
      -> tensor<64x1xi32, #blocked>
    %k = tt.broadcast %kexpanded
      : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %offset = arith.addi %row_offsets, %k : tensor<64x256xi32, #blocked>
    %dest = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %copy = hcug.buffer_load_to_local %p[%offset] into %dest
      {{contiguity = 8 : i32}} : <bf16>[tensor<64x256xi32, #blocked>]
      -> <64x256xbf16, #shared, #smem, mutable>
    %commit = ttg.async_commit_group tokens %copy
    %wait = hcug.async_wait %commit {{num_inst = 0 : i32}}
    tt.return
  }}
}}'''
    path = tmp_path / "uniform_division.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_allocate_shared_memory(manager)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "uniform_division")
    return str(module)


@pytest.mark.parametrize("divisor", [3, 4, 5, 12, 192])
def test_uniform_nonzero_constant_division_needs_no_lane_shuffle(tmp_path, divisor):
    assert "bpermute" not in lower(tmp_path, divisor)


@pytest.mark.parametrize("divisor,varying,dynamic", [
    (12, True, False), (192, True, False), (12, False, True),
])
def test_lane_varying_and_unproved_divisors_keep_original_shuffle(
    tmp_path, divisor, varying, dynamic
):
    assert "bpermute" in lower(tmp_path, divisor, varying, dynamic)
