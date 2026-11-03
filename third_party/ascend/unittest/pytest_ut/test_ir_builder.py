"""Builder construction must work without an NPU or a compiled kernel."""

import gc

import pytest

from triton._C.libtriton import ascend, buffer_ir, gluon_ir, ir


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """Override the parent device fixture: these tests only inspect host IR/cache."""
    yield


@pytest.mark.parametrize("mode", ["simd", "simt"])
def test_shared_builder_layout(mode):
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    ascend.load_dialects(context)
    for _ in range(8):
        builder = ir.builder(context, compile_mode=mode)
        npu_builder = ascend.ir.ascendnpu_ir_builder(context, "Ascend910B4")
        buf_builder = buffer_ir.buffer_builder(context)
        gluon_builder = gluon_ir.GluonOpBuilder(context)
        builder.set_loc(__file__, 1, 0)
        npu_builder.set_loc(__file__, 1, 0)
        buf_builder.set_loc(__file__, 1, 0)
        assert builder.is_simt_mode() == (mode == "simt")
        assert not npu_builder.is_simt_mode()
        del gluon_builder, buf_builder, npu_builder, builder
        gc.collect()
