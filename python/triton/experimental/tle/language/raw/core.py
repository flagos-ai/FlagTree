import triton.language as tl
from triton.language.core import builtin, constexpr as tl_constexpr, tensor
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


def _resolve_alias_indices(func, llvm, handles, output_indices, _semantic):
    if output_indices is None:
        return _semantic.builder.compute_alias_operand_indices(llvm, handles)
    return output_indices


def _wrap_results(args, alias_indices, dsl_region_op, *, smem: bool):
    aliased_args = [args[idx] for idx in alias_indices]
    results = dsl_region_op.get_results()
    if len(results) == 0:
        return None
    if smem:
        buffer_tensors = [
            buffered_tensor(
                result,
                aliased.dtype,
                aliased.shape,
                aliased.type.storage,
                aliased.type.layout,
                aliased.type.semantic,
            ) for result, aliased in zip(results, aliased_args)
        ]
        if len(buffer_tensors) == 1:
            return buffer_tensors[0]
        return tl.tuple(buffer_tensors)
    tensors = [tensor(result, aliased.type) for result, aliased in zip(results, aliased_args)]
    if len(tensors) == 1:
        return tensors[0]
    return tl.tuple(tensors)


def _normalize_hint(hint):
    while isinstance(hint, tl_constexpr):
        hint = hint.value
    return str(hint) if hint else ""


def _tle_raw_call(func, args, *, output_indices, hint, smem, _semantic):
    hint = _normalize_hint(hint)
    if getattr(func, "deferred", False):
        handles = [arg.handle for arg in args]
        if output_indices is None:
            raise RuntimeError("deferred tle_raw.call requires explicit output_indices=")
        alias_indices = output_indices
        source_id = func.register_pending_source(hint=hint)
        dsl_region_op = func.create_region_deferred(_semantic.builder, source_id, handles, alias_indices, hint)
    else:
        if func.compiler.lower() == "nvcc" or (func.compiler.lower() == "clang" and func.target.lower() == "bc"):
            patch_hash_method_for_pointer_type()
            module = import_from_path(func.extern_file)
            target_fn = getattr(module, func.extern_func_name)
            ret = target_fn(*args, _semantic=_semantic)
            return ret

        handles = [arg.handle for arg in args]
        context = _semantic.builder.get_context()
        llvm = func.make_llvm(context)
        alias_indices = _resolve_alias_indices(func, llvm, handles, output_indices, _semantic)
        dsl_region_op = func.create_region_by_llvm(_semantic.builder, llvm, handles, alias_indices, hint)
    return _wrap_results(args, alias_indices, dsl_region_op, smem=smem)


@builtin
def call(func, args, output_indices=None, hint="", _semantic=None):
    return _tle_raw_call(func, args, output_indices=output_indices, hint=hint, smem=False, _semantic=_semantic)


@builtin
def call_smem(func, args, output_indices=None, hint="", _semantic=None):
    return _tle_raw_call(func, args, output_indices=output_indices, hint=hint, smem=True, _semantic=_semantic)
