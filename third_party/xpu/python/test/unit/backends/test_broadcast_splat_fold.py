from pathlib import Path

from triton._C.libtriton import ir, xpu


def test_allocate_after_vector_broadcast_of_scalar_splat():
    # The result's element type is vector<16xf32>, while the splat element is f32.
    ctx = ir.context()
    ir.load_dialects(ctx)
    xpu.load_dialects(ctx)
    path = Path(__file__).with_name("broadcast_splat_fold.mlir")
    module = ir.parse_mlir_module(str(path), ctx)
    module.context = ctx
    assert module.verify()

    pm = ir.pass_manager(ctx)
    xpu.passes.ttxpuir.add_allocate_xpu_shared_memory(pm)
    pm.run(module, "allocate-xpu-shared-memory")
    assert module.verify()
