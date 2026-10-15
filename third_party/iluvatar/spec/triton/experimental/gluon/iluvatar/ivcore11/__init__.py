from __future__ import annotations

from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.experimental.gluon.language._layouts import DotOperandLayout

from .._layouts import IluvatarMMALayout
from . import async_copy

__all__ = ["async_copy", "mma"]


@builtin
def mma(a, b, acc, input_precision=None, _semantic=None):
    input_precision = _unwrap_if_constexpr(input_precision)
    assert acc is not None, "acc is required"

    layout = acc.type.layout
    assert isinstance(layout, IluvatarMMALayout) and layout.version[0] == 1, \
        "Expected IluvatarMMALayout with major version 1"

    for name, operand in (("a", a), ("b", b)):
        op_layout = operand.type.layout
        assert isinstance(op_layout, DotOperandLayout) and isinstance(op_layout.parent, IluvatarMMALayout) \
            and op_layout.parent.version[0] == 1, \
            f"Expected {name}'s layout to be DotOperandLayout with IluvatarMMALayout parent"

    handle = _semantic.dot(a, b, acc, input_precision=input_precision, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
