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

from __future__ import annotations

from typing import TYPE_CHECKING

import triton.language as tl
from triton.language.core import builtin, base_type, base_value, constexpr as tl_constexpr, tensor
from triton.experimental.tle.language.gpu import buffered_tensor

if TYPE_CHECKING:
    from .. import TLESemantic


def _tle_raw_call_eager_mode(func, args, output_indices, hint, smem, _semantic, _generator):
    context = _semantic.builder.get_context()
    llvm = func.make_llvm(context)
    extern_func_name = getattr(func, "extern_func_name", None) or ""
    handles = [(_semantic.to_tensor(arg) if isinstance(arg, tl_constexpr) else arg).handle for arg in args]
    alias_indices = output_indices
    if alias_indices is None:
        alias_indices = _semantic.builder.compute_alias_operand_indices(llvm, handles, extern_func_name)
    dsl_region_op = func.create_region_by_llvm(_semantic.builder, llvm, handles, alias_indices, hint, extern_func_name)
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


def _tle_raw_call(func, args, *, output_indices, hint, smem, _semantic: TLESemantic | None, _generator):
    func = tl.core._unwrap_if_constexpr(func)
    mark_kernel_init_hook = getattr(func, "mark_kernel_init_hook", None)
    if mark_kernel_init_hook is not None:
        mark_kernel_init_hook(_semantic, _generator)
    hint = _normalize_hint(hint)
    if output_indices is not None and any(idx < 0 or idx >= len(args) for idx in output_indices):
        raise RuntimeError("Using invalid output_indices in tle_raw.call")
    unwrapped_args = [tl.core._unwrap_if_constexpr(arg) for arg in args]
    has_fragment_args = any(isinstance(arg, tensor_fragment) for arg in unwrapped_args)
    deferred = getattr(func, "deferred", None)

    if not has_fragment_args and not deferred:
        return _tle_raw_call_eager_mode(func, args, output_indices, hint, smem, _semantic, _generator)
    elif deferred is not None and not deferred:
        raise RuntimeError("Tensor fragment args are not compatible with eager mode in tle_raw.call")
    if output_indices is None:
        raise RuntimeError("Deferred tle_raw.call requires explicit output_indices=")

    handles = []
    arg_info_list = []
    for i, arg in enumerate(args):
        ioflag = i in output_indices
        unwrapped_arg = unwrapped_args[i]
        if isinstance(unwrapped_arg, tensor_fragment):
            arg_info = "tensor_fragment"
            arg = unwrapped_arg.full_tensor
        elif isinstance(unwrapped_arg, tl.tensor) and unwrapped_arg.type.is_block():
            arg_info = "tensor"
        elif isinstance(unwrapped_arg, buffered_tensor):
            arg_info = "buffered_tensor"
        elif ioflag:
            raise TypeError(f"Argument {i} cannot be designated as an output."
                            "Only block tensors, tensor fragments, and buffered tensors are allowed.")
        else:
            arg_info = "value"
        arg_info += "_inout" if ioflag else "_input"
        arg_info_list.append(arg_info)
        handles.append((_semantic.to_tensor(arg) if isinstance(arg, tl_constexpr) else arg).handle)

    source_id = func.register_pending_source(hint=hint)
    dsl_region_op = func.create_region_deferred(_semantic.builder, source_id, handles, output_indices, hint=hint,
                                                arg_info_list=arg_info_list)
    results = dsl_region_op.get_results()

    if len(results) == 0:
        return None

    tensors = []
    for i, idx in enumerate(output_indices):
        unwrapped_arg = unwrapped_args[idx]
        result = results[i]
        if isinstance(unwrapped_arg, tensor_fragment):
            tensors.append(tl.tensor(result, unwrapped_arg.full_tensor.type))
        elif isinstance(unwrapped_arg, tl.tensor):
            tensors.append(tl.tensor(result, unwrapped_arg.type))
        elif isinstance(unwrapped_arg, buffered_tensor):
            tensors.append(
                buffered_tensor(
                    result,
                    unwrapped_arg.dtype,
                    unwrapped_arg.shape,
                    unwrapped_arg.type.storage,
                    unwrapped_arg.type.layout,
                    unwrapped_arg.type.semantic,
                    unwrapped_arg.type.alloc_shape,
                ))
    if len(tensors) == 1:
        return tensors[0]
    return tl.tuple(tensors)


class tensor_fragment_type(base_type):

    def __init__(self, tensor_type):
        self.tensor_type = tensor_type

    def __eq__(self, other):
        return (isinstance(other, tensor_fragment_type) and self.tensor_type == other.tensor_type)

    def __repr__(self):
        return f"tensor_fragment_type({self.tensor_type!r})"

    def mangle(self):
        return f"fragment_{self.tensor_type.mangle()}"

    def _flatten_ir_types(self, builder, out):
        self.tensor_type._flatten_ir_types(builder, out)

    def _unflatten_ir(self, handles, cursor):
        tensor, cursor = self.tensor_type._unflatten_ir(handles, cursor)
        return tensor_fragment(tensor), cursor


class tensor_fragment(base_value):

    def __init__(self, tensor: tl.tensor):
        if not (isinstance(tensor, tl.tensor) and tensor.type.is_block()):
            actual_type = tensor.type if isinstance(tensor, tl.tensor) else type(tensor).__name__
            raise TypeError(f"The constructor of tensor_fragment expects a block tensor, got {actual_type}")
        self.full_tensor = tensor
        self.type = tensor_fragment_type(tensor.type)

    def _flatten_ir(self, handles):
        self.full_tensor._flatten_ir(handles)

    def __repr__(self):
        return f"tensor_fragment_({self.full_tensor!r})"


@builtin
def fragment(x: tl.tensor, _semantic=None):
    return tensor_fragment(x)


@builtin
def call(func, args, output_indices=None, hint="", _semantic: TLESemantic | None = None, _generator=None):
    return _tle_raw_call(func, args, output_indices=output_indices, hint=hint, smem=False, _semantic=_semantic,
                         _generator=_generator)


@builtin
def call_smem(func, args, output_indices=None, hint="", _semantic: TLESemantic | None = None, _generator=None):
    return _tle_raw_call(func, args, output_indices=output_indices, hint=hint, smem=True, _semantic=_semantic,
                         _generator=_generator)
