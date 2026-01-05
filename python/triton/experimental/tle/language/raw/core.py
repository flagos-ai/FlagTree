import ast

import triton.language as tl
from triton.language.core import builtin, tensor


@builtin
def call(func, outputs, inputs, _semantic=None):
    # Extract type hints from function AST
    arg_type_hints = []
    if hasattr(func, 'ast') and func.ast.body:
        func_def = func.ast.body[0]
        if isinstance(func_def, ast.FunctionDef):
            for arg in func_def.args.args:
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Subscript):
                        # Handle type hints like Input["memref<?xi32, 3>"]
                        if isinstance(arg.annotation.slice, ast.Constant):
                            type_str = arg.annotation.slice.value
                        elif isinstance(arg.annotation.slice, ast.Str):  # Python < 3.8
                            type_str = arg.annotation.slice.s
                        else:
                            type_str = ""
                        arg_type_hints.append(type_str)
                    else:
                        arg_type_hints.append("")
                else:
                    arg_type_hints.append("")

    dsl_region_op = _semantic.builder.create_edsl_region_by_llvm_func(f"{func.llvm}", func.fnname,
                                                                      [output.handle for output in outputs],
                                                                      [input.handle
                                                                       for input in inputs], arg_type_hints)
    tensors = [tensor(result, output.type) for result, output in zip(dsl_region_op.get_results(), outputs)]
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tl.tuple(tensors)
