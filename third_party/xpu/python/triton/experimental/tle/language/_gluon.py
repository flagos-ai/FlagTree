"""Shared helper: reach the GluonOpBuilder that carries the TLE op builders."""

__all__ = ["gluon_builder"]


def gluon_builder(_semantic, _generator, probe):
    """Return a builder exposing `probe`, sharing the tracing insertion point.

    The TLE builder methods live on GluonOpBuilder rather than on the plain
    Triton builder, so a kernel traced through `@triton.jit` needs one wrapped
    around the same MLIRContext and insertion point.
    """
    builder = _semantic.builder
    if hasattr(builder, probe):
        return builder
    from triton._C.libtriton import gluon_ir
    context = _generator.context if _generator is not None else None
    if context is None:
        raise RuntimeError("Cannot access MLIRContext from builder. "
                           "Ensure the kernel uses @triton.jit.")
    gluon_builder_ = gluon_ir.GluonOpBuilder(context)
    gluon_builder_.restore_insertion_point(builder.get_insertion_point())
    return gluon_builder_
