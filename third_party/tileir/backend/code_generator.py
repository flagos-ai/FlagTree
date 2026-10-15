# Keep this adapter aligned with this branch's upstream AST and constexpr ABI.
import inspect
import warnings

import triton.knobs as knobs
import triton.language as language
from triton.language import constexpr, str_to_ty
from triton.language.core import base_value
from triton._utils import find_paths_if, get_iterable_path
from triton.compiler.code_generator import (
    ASTFunction,
    CodeGenerator,
    _is_constexpr,
    mangle_fn,
    flatten_values_to_ir,
    unflatten_ir_values,
)
from triton.compiler.errors import CompilationError
from triton.runtime.jit import get_jit_fn_file_line, get_full_name, JITFunction, JITCallable
from triton.backends.tileir.extend_semantic import CudaTileSemantic
from triton.backends.tileir.extend_core import as_tileir_tensor_descriptor_type


class TileIRCodeGenerator(CodeGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic = CudaTileSemantic(self.builder)

    def _get_tle_backend_name(self):
        return "tileir"

    def call_JitFunction(self, fn: JITFunction, args, kwargs, caller_context=None):
        args = inspect.getcallargs(fn.fn, *args, **kwargs)
        args = [args[name] for name in fn.arg_names]
        for i, arg in enumerate(args):
            if isinstance(arg, (language.dtype, float, int, bool, JITFunction)):
                args[i] = language.core.constexpr(arg)
        args_cst = find_paths_if(args, lambda _, x: _is_constexpr(x))
        args_cst = {path: get_iterable_path(args, path) for path in args_cst}
        args_path = find_paths_if(args, lambda _, x: not _is_constexpr(x))
        args_val = [get_iterable_path(args, path) for path in args_path]
        # mangle
        caller_context = caller_context or self.caller_context
        fn_name = mangle_fn(get_full_name(fn), [arg.type for arg in args_val], args_cst, caller_context)
        # generate function def if necessary
        if not self.module.has_function(fn_name):
            # If the callee is not set, we use the same debug setting as the caller
            file_name, begin_line = get_jit_fn_file_line(fn)
            arg_types = [
                language.core.constexpr if arg is None or isinstance(arg,
                                                                     (bool, int, language.core.dtype)) else arg.type
                for arg in args
            ]
            if fn.noinline:
                warnings.warn("TileIR inlines JIT helpers; noinline is not supported.", RuntimeWarning)
            prototype = ASTFunction([], arg_types, args_cst, dict())
            generator = TileIRCodeGenerator(self.context, prototype, fn.get_capture_scope(), module=self.module,
                                            jit_fn=fn, function_name=fn_name, function_types=self.function_ret_types,
                                            noinline=False, file_name=file_name, begin_line=begin_line,
                                            options=self.builder.options, codegen_fns=self.builder.codegen_fns,
                                            module_map=self.builder.module_map, caller_context=caller_context,
                                            is_gluon=self.is_gluon)
            try:
                generator.visit(fn.parse())
            except Exception as e:
                # Wrap the error in the callee with the location of the call.
                if knobs.compilation.front_end_debugging:
                    raise
                raise CompilationError(self.jit_fn.src, self.cur_node, None) from e

            callee_ret_type = generator.ret_type
            self.function_ret_types[fn_name] = callee_ret_type
        else:
            callee_ret_type = self.function_ret_types[fn_name]
        symbol = self.module.get_function(fn_name)
        args_val = flatten_values_to_ir(args_val)
        call_op = self.builder.call(symbol, args_val)
        if callee_ret_type == language.void:
            return None
        handles = [call_op.get_result(i) for i in range(call_op.get_num_results())]
        return next(unflatten_ir_values(handles, [callee_ret_type]))


def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    arg_types = [None] * len(fn.arg_names)

    for k, v in src.signature.items():
        idx = fn.arg_names.index(k)
        arg_types[idx] = as_tileir_tensor_descriptor_type(str_to_ty(v, None))

    def apply_constexpr_types(argument, indices, value):
        index = indices.pop()
        if len(indices) == 0:
            if isinstance(argument, list):
                argument[index] = constexpr(value).type
            else:
                argument.types[index] = constexpr(value).type
        else:
            apply_constexpr_types(argument[index], indices, value)

    for path, value in src.constants.items():
        apply_constexpr_types(arg_types, list(path)[::-1], value)

    prototype = ASTFunction([], arg_types, src.constants, src.attrs)
    file_name, begin_line = get_jit_fn_file_line(fn)
    # query function representation
    from collections import namedtuple
    leaves = filter(lambda v: len(v) == 1, src.constants)
    constants = {fn.arg_names[i[0]]: src.constants[i] for i in leaves}
    signature = src.signature
    proxy = namedtuple("SpecializationProxy", ["constants", "signature"])(constants, signature)
    generator = TileIRCodeGenerator(context, prototype, gscope=fn.get_capture_scope(), function_name=fn.repr(proxy),
                                    jit_fn=fn, is_kernel=True, file_name=file_name, begin_line=begin_line,
                                    options=options, codegen_fns=codegen_fns, module_map=module_map, module=module,
                                    is_gluon=fn.is_gluon())
    generator.visit(fn.parse())
    module = generator.module
    # module takes ownership of the context
    module.context = context
    module.name = generator.function_name
    if not module.verify():
        if not fn.is_gluon():
            print(module)
        raise RuntimeError("error encountered during parsing")
    return module
