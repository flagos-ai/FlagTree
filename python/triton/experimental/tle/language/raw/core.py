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
    
    # Get alias information
    alias_indices = dsl_region_op.get_alias_operand_indices()
    aliased_args = [args[idx] for idx in alias_indices]

    tensors = [
        tensor(result, aliased.type)
        for result, aliased in zip(dsl_region_op.get_results(), aliased_args)
    ]
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tl.tuple(tensors)


@builtin
def call_smem(func, args, _semantic=None):
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm, [arg.handle for arg in args])
    results = dsl_region_op.get_results()
    if len(results) == 0:
        return None

    # Get alias information
    alias_indices = dsl_region_op.get_alias_operand_indices()
    aliased_args = [args[idx] for idx in alias_indices]

    buffer_tensors = [
        buffered_tensor(result, aliased.dtype, aliased.shape, aliased.type.storage, aliased.type.layout,
                        aliased.type.semantic) for result, aliased in zip(dsl_region_op.get_results(), aliased_args)
    ]
    if len(buffer_tensors) == 1:
        return buffer_tensors[0]
    else:
        return tl.tuple(buffer_tensors)
