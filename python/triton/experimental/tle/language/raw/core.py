import triton.language as tl
from triton.experimental.tle.backends import _detect_backend
from triton.language.core import builtin, tensor

if _detect_backend() == "ascend":

    # Ascend backend: unified custom-op entry with two call forms, one per
    # integration channel:
    #
    # 1. OP extension framework: string op names carry an "<backend>_<op>"
    #    prefix, which is stripped before dispatching to the custom op
    #    registered for that backend:
    #        pair = tle_raw.call("ascend_sort32", src0, src1, repeat_times, out=pair)
    #    is equivalent to tle.dsa.ascend.raw("sort32", src0, src1, repeat_times,
    #    out=pair) with identical IR; the old entry remains available.
    #
    # 2. Operator-level mixed-language programming: a @dialect(name="cann", ...)
    #    function object (see triton.experimental.tle.raw) bound to a
    #    user-supplied AscendC source/bitcode file; all operands go in one
    #    list and the output/aliased operands are marked by position:
    #        out = tle_raw.call(sort_topk, [src, tmp, out, K], output_indices=[2])
    #    When the dialect declares a source file, its bitcode is JIT-compiled
    #    with ccec and cached under the Triton cache directory on first use.
    from triton.experimental.tle.language.dsa.ascend.core import raw as _ascend_raw
    from triton.language.extra.cann.extension import builtin as _ascend_builtin

    @_ascend_builtin
    def call(op, *args, output_indices=None, out=(), _semantic=None):
        op = getattr(op, "value", op)
        if isinstance(op, str):
            backend, _, name = op.partition("_")
            if backend != "ascend":
                raise ValueError(f"unknown raw backend prefix in {op!r}")
            return _ascend_raw(name, *args, out=out, _semantic=_semantic)
        from triton.experimental.tle.raw.cann import CANNJITFunction
        if isinstance(op, CANNJITFunction):
            if len(args) == 1 and isinstance(args[0], (list, tuple, tl.tuple)):
                args = tuple(args[0])
            from triton.language.extra.cann.extension import custom_object_semantic
            return custom_object_semantic(op, args, output_indices or (), _semantic=_semantic)
        raise ValueError(f"unsupported tle_raw.call target: {op!r}")
else:

    @builtin
    def call(func, outputs, inputs, _semantic=None):
        context = _semantic.builder.get_context()
        llvm = func.make_llvm(context)
        dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm,
                                                                             [output.handle for output in outputs],
                                                                             [input.handle for input in inputs])
        tensors = [tensor(result, output.type) for result, output in zip(dsl_region_op.get_results(), outputs)]
        if len(tensors) == 1:
            return tensors[0]
        else:
            return tl.tuple(tensors)
