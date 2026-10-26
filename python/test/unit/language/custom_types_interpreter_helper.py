"""Subprocess runner for the custom-types interpreter tests.

``triton.runtime.interpreter`` patches ``triton.language`` in place the first
time an interpreted kernel runs (see ``_patch_lang``).  That mutation is never
undone, so running these tests in the same process as the compiled unit tests
would corrupt every later compilation.  They are therefore executed here, in a
child process, and only the exit status is checked by the parent.

Usage: ``python custom_types_interpreter_helper.py <case>``
"""

import sys

import triton.language as tl

from test_custom_types import (
    Outer,
    TensorView,
    WithArray,
    _FakeDeviceBuffer,
    _interp,
    _k_array_of_struct,
    _k_copy_base,
    _k_nested_write,
    _k_write_ndim,
    _k_write_shape,
)


def scalar_and_array_field():
    st = TensorView._triton_struct_type
    buf = _FakeDeviceBuffer(st.size)
    tv = TensorView.from_buffer(buf)
    tv.ndim = 41
    tv.shape[2] = 5
    view = tl.StructView(st, buf)
    _interp(_k_write_ndim.fn).run(view, grid=(1, ))
    _interp(_k_write_shape.fn).run(view, 2, grid=(1, ))
    assert tv.ndim == 42
    assert tv.shape[2] == 15


def nested_struct():
    st = Outer._triton_struct_type
    buf = _FakeDeviceBuffer(st.size)
    o = Outer.from_buffer(buf)
    o.inner.a = 10
    view = tl.StructView(st, buf)
    _interp(_k_nested_write.fn).run(view, grid=(1, ))
    assert o.inner.a == 13


def array_of_struct():
    st = WithArray._triton_struct_type
    buf = _FakeDeviceBuffer(st.size)
    wa = WithArray.from_buffer(buf)
    wa.arr[1].b = 3.5
    view = tl.StructView(st, buf)
    _interp(_k_array_of_struct.fn).run(view, 1, grid=(1, ))
    assert abs(wa.arr[1].b - 5.0) < 1e-9


def pointer_field():
    st = TensorView._triton_struct_type
    buf = _FakeDeviceBuffer(st.size)
    tv = TensorView.from_buffer(buf)
    tv.base = 0x1234567890
    view = tl.StructView(st, buf)
    _interp(_k_copy_base.fn).run(view, grid=(1, ))
    # `base` (a pointer field) is copied verbatim into a uint64 field.
    assert tv.shape[3] == 0x1234567890


CASES = {
    "scalar_and_array_field": scalar_and_array_field,
    "nested_struct": nested_struct,
    "array_of_struct": array_of_struct,
    "pointer_field": pointer_field,
}

if __name__ == "__main__":
    CASES[sys.argv[1]]()
