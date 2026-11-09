"""Descriptor captures retain the original scalar ABI and alignment proofs."""
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def lower(tmp_path, dtype, alignment, divergent=False):
    extra = (
        '%offset = "hcug.warp_group_id"() {groupSize = 1 : i32} : () -> i32\n'
        "    %derived = tt.addptr %p, %offset : !tt.ptr<DT>, i32"
    ).replace("DT", dtype) if divergent else ""
    base = "%derived" if divergent else "%p"
    text = f'''#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @captured_descriptor(%p: !tt.ptr<{dtype}> {{tt.divisibility = {alignment} : i32}}) {{
    {extra}
    ttg.warp_specialize({base}) attributes {{requestedRegisters = array<i32: 32>}}
    default {{ ttg.warp_yield }}
    partition0(%base: !tt.ptr<{dtype}>) num_warps(4) {{
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %M = arith.constant 128 : i32
      %K = arith.constant 64 : i32
      %stride = arith.constant 64 : i64
      %unit_stride = arith.constant 1 : i64
      %tile = tt.matrix_load %base, shape = [%M, %K], strides = [%stride, %unit_stride],
        tensorShape = [128, 32], indices = [%zero, %zero, %one]
        {{hcu.structured_register}} : <{dtype}> -> tensor<128x32x{dtype}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>
      ttg.warp_return
    }} : (!tt.ptr<{dtype}>) -> ()
    tt.return
  }}
}}'''
    path = tmp_path / "capture.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.passes.ttgpuir.add_allocate_warp_groups(manager)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "structured_capture")
    return str(module)


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("alignment,width", [(16, "v4i32"), (2, "i16")])
def test_descriptor_capture_uses_abi_base_and_alignment(
    tmp_path, dtype, alignment, width
):
    result = lower(tmp_path, dtype, alignment)
    assert "llvm.amdgcn.struct.ptr.buffer.load." + width in result


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("alignment", [2, 16])
def test_capture_does_not_make_wave_varying_pointer_uniform(
    tmp_path, capfd, dtype, alignment
):
    with pytest.raises(RuntimeError):
        lower(tmp_path, dtype, alignment, True)
    diagnostic = capfd.readouterr().err
    (tmp_path / "diagnostic.log").write_text(diagnostic)
    assert "uniform AS1 base" in diagnostic
