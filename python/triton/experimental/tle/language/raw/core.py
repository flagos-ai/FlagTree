import triton.language as tl
from triton.language.core import builtin, tensor
from triton.experimental.tle.language.gpu import buffered_tensor


@builtin
def call(func, outputs, inputs, _semantic=None):
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm, [output.handle for output in outputs],
                                                                         [input.handle for input in inputs])
    tensors = [tensor(result, output.type) for result, output in zip(dsl_region_op.get_results(), outputs)]
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tl.tuple(tensors)


@builtin
def call_smem(func, outputs, inputs, _semantic=None):
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm, [output.handle for output in outputs],
                                                                         [input.handle for input in inputs])
    buffer_tensors = [
        buffered_tensor(result, output.dtype, output.shape, output.type.storage, output.type.layout,
                        output.type.semantic) for result, output in zip(dsl_region_op.get_results(), outputs)
    ]
    if len(buffer_tensors) == 1:
        return buffer_tensors[0]
    else:
        return tl.tuple(buffer_tensors)
