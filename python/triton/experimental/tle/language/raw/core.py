import triton.language as tl
from triton.language.core import builtin, tensor


@builtin
def call(func, outputs, inputs, _semantic=None):
    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(f"{func.llvm}", func.fnname,
                                                                         [output.handle for output in outputs],
                                                                         [input.handle for input in inputs])
    tensors = [tensor(result, output.type) for result, output in zip(dsl_region_op.get_results(), outputs)]
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tl.tuple(tensors)
