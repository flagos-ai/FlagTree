"""Builder construction must work without an NPU or a compiled kernel."""

import gc

import pytest

from triton._C.libtriton import ascend, buffer_ir, gluon_ir, ir


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """The builder test does not need a device."""
    yield


@pytest.mark.parametrize("mode", [None, "simd", "simt"])
def test_shared_builder_layout(mode):
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    ascend.load_dialects(context)
    for _ in range(8):
        builder = ir.builder(context) if mode is None else ir.builder(context, compile_mode=mode)
        npu_builder = ascend.ir.ascendnpu_ir_builder(context, "Ascend910B4")
        buf_builder = buffer_ir.buffer_builder(context)
        gluon_builder = gluon_ir.GluonOpBuilder(context)
        assert builder.is_simt_mode() == (mode == "simt")
        for specialized_builder in (npu_builder, buf_builder, gluon_builder):
            assert isinstance(specialized_builder, ir.builder)
            assert not specialized_builder.is_simt_mode()
            specialized_builder.set_loc(__file__, 1, 0)
        builder.set_loc(__file__, 1, 0)
        del specialized_builder
        del gluon_builder, buf_builder, npu_builder, builder
        gc.collect()
