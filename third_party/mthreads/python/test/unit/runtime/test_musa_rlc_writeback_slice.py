"""Bound Phase 3 address and mask cloning without disabling small slices."""

import pytest
from triton._C import libtriton
from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend

pytestmark = pytest.mark.skipif(
    not hasattr(libtriton, "mthreads"), reason="MThreads backend is required"
)


@pytest.mark.parametrize("mma", ["wmma", "sqmma"])
@pytest.mark.parametrize("phase_mask", [8, 15])
@pytest.mark.parametrize("operand", ["address", "mask"])
@pytest.mark.parametrize("length", [8, 300])
def test_writeback_slice_limit(mma, phase_mask, operand, length, tmp_path):
    ty = "tensor<16x64xi32, #b>"
    chain = "\n".join(
        f"%v{i} = arith.addi %v{i - 1}, %seed : {ty}" for i in range(1, length + 1)
    )
    offset = f"%v{length}" if operand == "address" else "%v0"
    mask_value = f"%v{length}" if operand == "mask" else "%v0"
    source = f"""
#b = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}}>
#mma = #ttg.musa_{mma}<{{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}}>
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"}} {{
  tt.func @writeback(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>, %a: i32, %b: i32) {{
    %v0 = tt.splat %a : i32 -> {ty}
    %seed = tt.splat %b : i32 -> {ty}
    {chain}
    %bases = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #b>
    %ptrs = tt.addptr %bases, {offset} : tensor<16x64x!tt.ptr<f32>, #b>, {ty}
    %mask = arith.cmpi slt, {mask_value}, %seed : {ty}
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #b>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value, %mask : (tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xf32, #b>, tensor<16x64xi1, #b>) -> tensor<16x64xf32, #b>
    tt.return
  }}
}}
"""
    path = tmp_path / "writeback.ttgir"
    path.write_text(source)
    context = ir.context()
    ir.load_dialects(context)
    make_backend(GPUTarget("musa", 31, 32)).load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    passes.ttgpuir.add_remove_layout_conversions(pm, True, phase_mask)
    pm.run(module, "rlc_writeback_slice")
    result = module.str_nodebug()
    assert result.count("tt.atomic_rmw") == 1
    assert result.count("ttg.convert_layout") == (1 if length > 256 else 0)
