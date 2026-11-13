"""SharedLinear byte ranges must remove only proved-disjoint membar edges."""
import pytest

from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)
compiler = pytest.importorskip("triton.backends.hcu.compiler")


def module_source(second_offset=64, index="bounded", select_load=False):
    index_op = {"bounded": "%index = arith.andi %idx, %one : i32",
                "unknown": "%index = arith.addi %idx, %zero : i32",
                "constant": "%index = arith.constant 0 : i32"}[index]
    types = "tensor<8x8xbf16, #blocked>"
    md = "!ttg.memdesc<8x8xbf16, #xor, #smem, mutable>"
    # A legal smaller tile partially overlaps A at [32,64); a 64-element
    # tile at offset32 is rejected by the shared-view verifier itself.
    second_size = 32 if second_offset == 32 else 64
    types_b = "tensor<4x8xbf16, #blocked>" if second_size == 32 else types
    md_b = "!ttg.memdesc<4x8xbf16, #xor_b, #smem, mutable>" if second_size == 32 else md
    if select_load:
        operations = f'''ttg.local_store %ones, %a : {types} -> {md}
    ttg.local_barrier
    %choice = arith.select %cond, %a, %b : {md}
    %value = ttg.local_load %choice : {md} -> {types}
    ttg.local_store %ones_b, %b : {types_b} -> {md_b}'''
    else:
        operations = f'''ttg.local_store %ones, %a : {types} -> {md}
    ttg.local_store %ones_b, %b : {types_b} -> {md_b}
    %value = ttg.local_load %a : {md} -> {types}'''
    return f'''#flat = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}}>
#xor = #ttg.shared_linear<{{offset = [[0, 1], [0, 2], [0, 4], [1, 1], [2, 2], [4, 4]]}}, alignment = 16>
#xor_b = #ttg.shared_linear<{{offset = [[0, 1], [0, 2], [0, 4], [1, 1], [2, 2]]}}, alignment = 16>
#blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @regions(%idx: i32, %cond: i1) {{
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    {index_op}
    %ones = arith.constant dense<1.0> : {types}
    %ones_b = arith.constant dense<2.0> : {types_b}
    %root = ttg.local_alloc {{alignment = 16 : i32}} : () -> !ttg.memdesc<2x128xbf16, #flat, #smem, mutable>
    %frame = ttg.memdesc_index %root[%index] : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<128xbf16, #flat, #smem, mutable>
    %sa = ttg.memdesc_subslice %frame[0] : !ttg.memdesc<128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<64xbf16, #flat, #smem, mutable, 128>
    %sb = ttg.memdesc_subslice %frame[{second_offset}] : !ttg.memdesc<128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<{second_size}xbf16, #flat, #smem, mutable, 128>
    %a = ttg.memdesc_reinterpret %sa : !ttg.memdesc<64xbf16, #flat, #smem, mutable, 128> -> {md}
    %b = ttg.memdesc_reinterpret %sb : !ttg.memdesc<{second_size}xbf16, #flat, #smem, mutable, 128> -> {md_b}
    {operations}
    ttg.local_dealloc %root : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable>
    tt.return
  }}
}}
'''


@pytest.mark.parametrize("second_offset,index,select_load,barriers", [
    (64, "bounded", False, 1),
    (64, "constant", False, 1),
    (32, "bounded", False, 2),
    (0, "bounded", False, 2),
    (64, "unknown", False, 2),
    # A read that can select B must retain its WAR barrier before writing B.
    (64, "bounded", True, 2),
])
def test_shared_linear_disjointness(tmp_path, second_offset, index, select_load, barriers):
    result = lower_source(tmp_path, module_source(second_offset, index, select_load))
    assert result.count("rocdl.s.barrier") == barriers, result


def lower_source(tmp_path, source, update_async_wait=False):
    path = tmp_path / "regions.mlir"
    path.write_text(source)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    pm = libtriton.ir.pass_manager(context)
    if update_async_wait:
        libtriton.hcu.passes.ttgpuir.add_update_async_wait_count(pm, "gfx936")
    libtriton.hcu.passes.ttgpuir.add_allocate_shared_memory(pm)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx936", False)
    pm.run(module, "physical_shared_regions_contract")
    return str(module)


def test_same_block_conditional_forwarding(tmp_path):
    text = module_source(select_load=True)
    md = "!ttg.memdesc<8x8xbf16, #xor, #smem, mutable>"
    text = text.replace(f"%choice = arith.select %cond, %a, %b : {md}",
                        f"cf.cond_br %cond, ^merge(%a : {md}), ^merge(%b : {md})\n  ^merge(%choice: {md}):")
    # Both edges from the same predecessor must be included; B aliases the
    # second incoming view and still needs the read-before-write barrier.
    assert lower_source(tmp_path, text).count("rocdl.s.barrier") == 2


def test_unmodeled_multibuffer_reinterpret_falls_back(tmp_path):
    text = module_source(index="constant")
    old = "%frame = ttg.memdesc_index %root[%index] : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<128xbf16, #flat, #smem, mutable>"
    new = "%frame = ttg.memdesc_reinterpret %root : !ttg.memdesc<2x128xbf16, #flat, #smem, mutable> -> !ttg.memdesc<256xbf16, #flat, #smem, mutable>"
    assert old in text
    text = text.replace(old, new).replace("!ttg.memdesc<128xbf16, #flat", "!ttg.memdesc<256xbf16, #flat").replace("mutable, 128>", "mutable, 256>")
    # SharedLinear conversion helpers assert if handed a whole buffer stack
    # with a lower-rank encoding. This legal reinterpret must compile safely
    # and keep generic barriers until its physical chart is modeled.
    assert lower_source(tmp_path, text).count("rocdl.s.barrier") == 2
