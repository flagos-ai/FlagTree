"""Builder construction must work without an NPU or a compiled kernel."""

import gc

import pytest

from triton._C.libtriton import ascend, buffer_ir, gluon_ir, ir


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """The builder test does not need a device."""
    yield


@pytest.mark.parametrize("args, kwargs, is_simt", [
    pytest.param((), {}, False, id="default"),
    pytest.param((), {"compile_mode": "simd"}, False, id="simd-keyword"),
    pytest.param((), {"compile_mode": "simt"}, True, id="simt-keyword"),
    pytest.param(("simt", ), {}, True, id="simt-positional"),
])
def test_shared_builder_layout(args, kwargs, is_simt):
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    ascend.load_dialects(context)
    for _ in range(8):
        builder = ir.builder(context, *args, **kwargs)
        npu_builder = ascend.ir.ascendnpu_ir_builder(context, "Ascend910B4")
        buf_builder = buffer_ir.buffer_builder(context)
        gluon_builder = gluon_ir.GluonOpBuilder(context)
        assert builder.is_simt_mode() == is_simt
        for specialized_builder in (npu_builder, buf_builder, gluon_builder):
            assert isinstance(specialized_builder, ir.builder)
            assert not specialized_builder.is_simt_mode()
            specialized_builder.set_loc(__file__, 1, 0)
            assert isinstance(specialized_builder.create_module(), ir.module)
        builder.set_loc(__file__, 1, 0)
        del specialized_builder
        del gluon_builder, buf_builder, npu_builder, builder
        gc.collect()


def test_builder_mode_type_checks():
    with pytest.raises(TypeError, match="ir.builder"):
        ir.builder.is_simt_mode(123)

    context = ir.context()
    builder = ir.builder(context)
    builder._ascend_is_simt = "simd"
    with pytest.raises(TypeError, match="must be a bool"):
        builder.is_simt_mode()
