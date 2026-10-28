"""Allocation must remain compatible with backend-local builder bindings."""

from types import SimpleNamespace

import pytest
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu import core as gpu_core


class _ShapeAllocBuilder:
    """MetaX provides the shape overload without get_memdesc_type."""

    def get_int32_ty(self):
        return "i32"

    def get_float_ty(self):
        return "f32"

    def make_swizzled_shared_encoding_attr(self, *args):
        return "shared_layout"

    def create_local_alloc(self, shape, element_type, layout):
        self.alloc_args = (list(shape), element_type, layout)
        return "alloc_handle"


class _ValueAllocBuilder(_ShapeAllocBuilder):
    """Iluvatar and Mthreads require a Value in the typed overload."""

    def get_memdesc_type(self, shape, element_type, layout, space):
        return (tuple(shape), element_type, layout, space)

    def create_local_alloc(self, *args):
        if len(args) == 2:
            if args[1] is None:
                raise TypeError("the typed create_local_alloc overload requires a Value")
            self.alloc_args = args
            return "alloc_handle"
        return super().create_local_alloc(*args)


@pytest.mark.parametrize("builder_type", [_ShapeAllocBuilder, _ValueAllocBuilder])
@pytest.mark.parametrize("shape,dtype,element_type", [
    ([256], tl.int32, "i32"),
    ([64, 64], tl.float32, "f32"),
    ([2, 16, 32], tl.float32, "f32"),
])
def test_uninitialized_alloc_with_backend_local_bindings(monkeypatch, builder_type, shape, dtype, element_type):
    # Exercise the shared Python frontend with each backend's binding contract,
    # independently of the backend installed on the test machine.
    monkeypatch.setattr(gpu_core.tle_semantic, "COMMON_IR_ENABLED", False)
    monkeypatch.setattr(gpu_core.mthreads_common, "enabled", lambda: False)
    builder = builder_type()
    semantic = SimpleNamespace(builder=builder)
    layout = tle.gpu.swizzled_shared_layout.make_default(len(shape))

    buffer = tle.gpu.alloc(shape, dtype, layout=layout, _semantic=semantic)

    assert buffer.handle == "alloc_handle"
    assert buffer.shape == shape
    assert buffer.dtype == dtype
    assert builder.alloc_args == (shape, element_type, "shared_layout")
