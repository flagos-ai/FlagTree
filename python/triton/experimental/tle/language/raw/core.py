import triton.language as tl
from triton.language.core import builtin, tensor
from triton.experimental.tle.language.gpu import buffered_tensor

import importlib.util

def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))

def patch_hash_method_for_pointer_type():
    elem_dtype_list = tl.core.dtype.SINT_TYPES + tl.core.dtype.UINT_TYPES + tl.core.dtype.FP_TYPES + tl.core.dtype.OTHER_TYPES
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(tl.core.pointer_type(tl.core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash
        
def import_from_path(file_path):
    module_name = f"_imported_{abs(hash(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@builtin
def call(func, args, _semantic=None):
    if (func.compiler).lower() == "nvcc":
        func.make_cubin()
        patch_hash_method_for_pointer_type()
        module = import_from_path(func.extern)
        target_fn = getattr(module, func.extern_func_name)
        ret = target_fn(*args, _semantic=_semantic)
        return ret
    
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    handles = [arg.handle for arg in args]

    alias_indices = _semantic.builder.compute_alias_operand_indices(llvm, handles)
    aliased_args = [args[idx] for idx in alias_indices]

    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm, handles, alias_indices)
    results = dsl_region_op.get_results()
    if len(results) == 0:
        return None

    tensors = [tensor(result, aliased.type) for result, aliased in zip(results, aliased_args)]
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tl.tuple(tensors)


@builtin
def call_smem(func, args, _semantic=None):
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    handles = [arg.handle for arg in args]

    alias_indices = _semantic.builder.compute_alias_operand_indices(llvm, handles)
    aliased_args = [args[idx] for idx in alias_indices]

    dsl_region_op = _semantic.builder.create_tle_raw_region_by_llvm_func(llvm, handles, alias_indices)
    results = dsl_region_op.get_results()
    if len(results) == 0:
        return None

    buffer_tensors = [
        buffered_tensor(result, aliased.dtype, aliased.shape, aliased.type.storage, aliased.type.layout,
                        aliased.type.semantic) for result, aliased in zip(results, aliased_args)
    ]
    if len(buffer_tensors) == 1:
        return buffer_tensors[0]
    else:
        return tl.tuple(buffer_tensors)
