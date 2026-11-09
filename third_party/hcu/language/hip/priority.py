"""Typed HCU scheduling boundaries for ordinary Triton and TLE kernels."""

from triton.language import core


@core.builtin
def cta_barrier(_semantic=None):
    """Rendezvous the complete CTA, including every warp-specialized role.

    Every role must execute the same sequence of these barriers. This publishes
    completed shared accesses; it does not complete outstanding async copies.
    """
    from triton._C.libtriton import hcu

    builder = _semantic.builder
    hcu.create_cta_barrier(builder.get_op_builder(), builder.get_loc())


@core.builtin
def accumulator_fence(acc, priority=0, _semantic=None):
    """Preserve the accumulator and scheduling effects across a priority change.

    This neither waits for asynchronous memory nor synchronizes threads.
    Completion and cross-wave publication still require their dependencies.
    """
    from triton._C.libtriton import hcu

    priority = core._unwrap_if_constexpr(priority)
    if type(priority) is not int or priority not in (0, 1):
        raise ValueError("accumulator fence priority must be 0 or 1")
    acc = _semantic.to_tensor(acc)
    if not acc.type.is_block() or acc.dtype not in (core.float32, core.int32):
        raise ValueError("accumulator fence requires an f32 or i32 tensor")
    builder = _semantic.builder
    handle = hcu.create_accumulator_fence(
        builder.get_op_builder(), builder.get_loc(), acc.handle, priority
    )
    return core.tensor(handle, acc.type)
