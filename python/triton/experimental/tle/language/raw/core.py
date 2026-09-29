import triton.language as tl
from triton.experimental.tle.backends import _detect_backend
from triton.language.core import builtin, tensor

if _detect_backend() == "ascend":

    # Ascend backend: unified custom-op entry. Op names carry an "<backend>_<op>"
    # prefix, which is stripped before dispatching to the custom op registered
    # for that backend. For example:
    #     pair = tle_raw.call("ascend_sort32", src0, src1, repeat_times, out=pair)
    # is equivalent to tle.dsa.ascend.raw("sort32", src0, src1, repeat_times,
    # out=pair) with identical IR; the old entry remains available.
    from triton.experimental.tle.language.dsa.ascend.core import raw as _ascend_raw
    from triton.language.extra.cann.extension import builtin as _ascend_builtin

    @_ascend_builtin
    def call(op_name, *args, out=(), _semantic=None):
        op_name = getattr(op_name, "value", op_name)
        backend, _, name = op_name.partition("_")
        if backend != "ascend":
            raise ValueError(f"unknown raw backend prefix in {op_name!r}")
        return _ascend_raw(name, *args, out=out, _semantic=_semantic)
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
