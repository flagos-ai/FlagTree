import triton.language as tl
from triton.language.core import builtin, tensor
from triton.experimental.tle.language.gpu import buffered_tensor


@builtin
def call(func, args, _semantic=None):
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm, [arg.handle for arg in args])
    results = dsl_region_op.get_results()
    if len(results) == 0:
        return None
    # For scalar returns, wrap each result as a tensor with its own type
    tensors = [tensor(result, result.type) for result in results]
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tl.tuple(tensors)


@builtin
def call_smem(func, outputs, inputs, _semantic=None):
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    all_args = [output.handle for output in outputs] + [input.handle for input in inputs]
    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm, all_args)
    buffer_tensors = [
        buffered_tensor(result, output.dtype, output.shape, output.type.storage, output.type.layout,
                        output.type.semantic) for result, output in zip(dsl_region_op.get_results(), outputs)
    ]
    if len(buffer_tensors) == 1:
        return buffer_tensors[0]
    else:
        return tl.tuple(buffer_tensors)
