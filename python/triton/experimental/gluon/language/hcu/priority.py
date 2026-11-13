"""Typed HCU wave scheduling boundaries; no memory completion is implied."""
from .. import _core as core
from .._core import builtin, _unwrap_if_constexpr


@builtin
def accumulator_fence(acc, priority=0, _semantic=None):
    """Return unchanged accumulator values across a wave priority boundary.

    The backend derives the complete register cohort from the tensor layout.
    This operation carries SSA, wave-state and memory scheduling effects. It
    is not a memory wait or a thread barrier: asynchronous completion and
    cross-wave publication/retirement still require their usual dependencies.
    """
    from triton._C.libtriton import hcu
    priority = _unwrap_if_constexpr(priority)
    if type(priority) is not int or priority not in (0, 1):
        raise ValueError("accumulator fence priority must be 0 or 1")
    acc = _semantic.to_tensor(acc)
    if not acc.type.is_block() or acc.dtype not in (core.float32, core.int32):
        raise ValueError("accumulator fence requires an f32 or i32 tensor")
    builder = _semantic.builder
    handle = hcu.create_accumulator_fence(
        builder.get_op_builder(), builder.get_loc(), acc.handle, priority)
    return core.tensor(handle, acc.type)
