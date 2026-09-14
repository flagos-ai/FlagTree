# Triton 3.6 XPU TLE - raw payload injection (`tle.raw`)
#
# FlagTree XPU overlay: cluster (SIMT) path only. `tle.raw.call` emits a
# `triton_xpu.raw` op carrying the user's LLVM IR payload as an attribute. The op
# is late-bound: it survives to TritonXPU->LLVM conversion, where it becomes an
# `always_inline llvm.call` spliced into the module (`RawOpToLLVM.cpp`).
#
# The SDNN counterpart (`sdnn.raw`, launched with `is_sdnn=True`) is not available
# here: its builder needs TritonSDNN's IR headers, which FlagTree does not ship --
# see ../dsa/__init__.py. A kernel that reaches `tle.raw.call` on the SDNN path is
# rejected with that explanation rather than silently emitting a cluster op into
# an SDNN pipeline.

import triton.language.core as tl

from .._gluon import gluon_builder

__all__ = ["call"]

_SDNN_MESSAGE = ("tle.raw.call: the SDNN path (is_sdnn=True) is not available in the FlagTree XPU build. "
                 "`sdnn.raw` is a TritonSDNN op and FlagTree consumes that dialect as prebuilt objects, "
                 "without the IR headers needed to build its Python bindings. Launch the kernel without "
                 "is_sdnn to use the cluster path (`triton_xpu.raw`), which takes global pointers and "
                 "scalars rather than tensors.")


def _handles(values, what):
    values = tl._unwrap_if_constexpr(values)
    if values is None:
        return [], []
    if isinstance(values, tl.tuple):
        values = tuple(values.values)
    elif isinstance(values, tl.tensor) or not isinstance(values, (tuple, list)):
        # Note: never compare a tl.tensor against a plain value here -- `==`/`!=`
        # dispatch to tensor builtins that need `_semantic`.
        values = (values, )
    handles, types = [], []
    for value in values:
        value = tl._unwrap_if_constexpr(value)
        if not isinstance(value, tl.tensor):
            raise ValueError(f"tle.raw.call: {what} must be tensors, got {type(value).__name__}")
        handles.append(value.handle)
        types.append(value.type)
    return handles, types


def _callee(func):
    name = getattr(func, "name", None)
    if name is None:
        raise ValueError("tle.raw.call: payload handle has no `name`")
    return name


def _is_sdnn(_semantic):
    """True when the kernel is being traced for the SDNN pipeline.

    `options.is_sdnn` is what the backend selects its pipeline from, and it is
    read at trace time.
    """
    options = getattr(_semantic.builder, "options", None)
    return bool(getattr(options, "is_sdnn", False))


# Which cluster op a payload handle maps to. The op is per-stack, not per-arch:
# `triton_xpu.raw` on the XPU cluster path (dialect_name "xpu").
_CLUSTER_OPS = {
    "xpu": ("create_xpu_raw", "create_xpu_raw_deferred"),
}


def _cluster_ops(func):
    dialect = getattr(func, "dialect_name", None)
    ops = _CLUSTER_OPS.get(dialect)
    if ops is None:
        raise ValueError(f"tle.raw.call: no cluster raw op for dialect '{dialect}'; "
                         f"expected one of {sorted(_CLUSTER_OPS)} "
                         '(e.g. @tle.raw.dialect("xpu3"))')
    return ops


@tl.builtin
def call(func, outputs=(), inputs=(), _semantic=None, _generator=None):
    """Call a user-provided payload on `outputs` (written) and `inputs` (read).

    Args:
        func: payload handle satisfying `make_llvm(ctx) -> str` and `.name`,
              e.g. produced by `@tle.raw.dialect("xpu3", file="kernel.xpu")`
        outputs: pointer/scalar operands the payload writes
        inputs: pointer/scalar operands the payload reads

    Returns nothing -- the payload writes through the pointers it is handed,
    which is also why the op is never dead-code eliminated. The payload sees one
    C++ parameter per operand, outputs first then inputs: a `tt.ptr<T>` arrives
    as `T*` and a scalar keeps its type. Tensor operands are rejected -- on the
    cluster path a tensor lives in registers, not memory, so there is no address
    to hand over. Payloads whose parameters are not split into outputs and inputs
    are simply passed as one sequence, e.g.
    `tle.raw.call(vec_add, (X, Y, Out, n))`.
    """
    if _is_sdnn(_semantic):
        raise NotImplementedError(_SDNN_MESSAGE)

    # No output/input distinction on this path: the payload takes the operands as
    # they come, outputs first then inputs.
    handles, _ = _handles(outputs, "outputs")
    in_handles, _ = _handles(inputs, "inputs")
    handles += in_handles

    callee = _callee(func)
    eager_op, deferred_op = _cluster_ops(func)

    if getattr(func, "deferred", False):
        # The arch is unknown at trace time, so only record the payload id; the
        # backend compiles it in make_llir and fills `llvm_ir` in via
        # `tritonxpu-materialize-deferred-raw`.
        source_id = func.register_pending_source()
        builder = gluon_builder(_semantic, _generator, deferred_op)
        getattr(builder, deferred_op)(callee, source_id, handles)
    else:
        llvm_ir = func.make_llvm(_generator.context if _generator is not None else None)
        builder = gluon_builder(_semantic, _generator, eager_op)
        getattr(builder, eager_op)(callee, llvm_ir, handles)
