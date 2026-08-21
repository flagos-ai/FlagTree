# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton.language as tl
from triton.language.core import builtin, constexpr as tl_constexpr, tensor
from triton.experimental.tle.language.gpu import buffered_tensor


def _resolve_alias_indices(func, llvm, handles, output_indices, extern_func_name, _semantic):
    if output_indices is None:
        return _semantic.builder.compute_alias_operand_indices(llvm, handles, extern_func_name)
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


def _tle_raw_call(func, args, *, output_indices, hint, smem, _semantic, _generator):
    mark_kernel_init_hook = getattr(func, "mark_kernel_init_hook", None)
    if mark_kernel_init_hook is not None:
        mark_kernel_init_hook(_semantic, _generator)
    hint = _normalize_hint(hint)
    handles = [(_semantic.to_tensor(arg) if isinstance(arg, tl_constexpr) else arg).handle for arg in args]
    if getattr(func, "deferred", False):
        if output_indices is None:
            raise RuntimeError("deferred tle_raw.call requires explicit output_indices=")
        alias_indices = output_indices
        source_id = func.register_pending_source(hint=hint)
        dsl_region_op = func.create_region_deferred(_semantic.builder, source_id, handles, alias_indices, hint)
    else:
        context = _semantic.builder.get_context()
        llvm = func.make_llvm(context)
        extern_func_name = getattr(func, "extern_func_name", None) or ""
        alias_indices = _resolve_alias_indices(func, llvm, handles, output_indices, extern_func_name, _semantic)
        dsl_region_op = func.create_region_by_llvm(_semantic.builder, llvm, handles, alias_indices, hint,
                                                   extern_func_name)
    return _wrap_results(args, alias_indices, dsl_region_op, smem=smem)


@builtin
def call(func, args, output_indices=None, hint="", _semantic=None, _generator=None):
    return _tle_raw_call(func, args, output_indices=output_indices, hint=hint, smem=False, _semantic=_semantic,
                         _generator=_generator)


@builtin
def call_smem(func, args, output_indices=None, hint="", _semantic=None, _generator=None):
    return _tle_raw_call(func, args, output_indices=output_indices, hint=hint, smem=True, _semantic=_semantic,
                         _generator=_generator)
