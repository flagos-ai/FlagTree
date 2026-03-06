# Copyright 2026- Xcoresigma Technology Co., Ltd

from triton._C.libtriton import ir
from typing import Optional, Dict
from triton.runtime import JITFunction
from .language.builder import setup_unified_builder_with_tle_builder
import importlib

try:
    from triton._C.libtriton import tle as tle_ir
except ImportError:
    raise RuntimeError("tle is not available")

triton_compiler = importlib.import_module("triton.compiler", package=__package__)
def tle_patch_for_triton_compile():
    original_compile_fn = triton_compiler.compile
    def tle_compile(src, target=None, options=None):
        # ir.context() will return a new MLIRContext each time, here should keep the same context
        cur_context = ir.context()
        tle_ir.load_dialects(cur_context)

        original_context_fn = ir.context
        def patched_context():
            return cur_context
        ir.context = patched_context

        try:
            compiled_kernel = original_compile_fn(src, target, options)
        finally:
            ir.context = original_context_fn

        return compiled_kernel
    return tle_compile

code_generator = importlib.import_module("triton.compiler.code_generator", package=__package__)

class TleCodeGenerator(code_generator.CodeGenerator):
    def __init__(self, context, prototype, gscope, attributes, constants, function_name, jit_fn: JITFunction, options,
                 codegen_fns, module_map, module=None, is_kernel=False, function_types: Optional[Dict] = None,
                 noinline=False, file_name: Optional[str] = None, begin_line=0):
        super().__init__(context, prototype, gscope, attributes, constants, function_name, jit_fn, options,
                         codegen_fns, module_map, module, is_kernel, function_types, noinline, file_name, begin_line)
        self.tle_builder = tle_ir.tle_builder(context)
        self.tle_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder_with_tle_builder(self.builder, self.tle_builder)


triton_compiler.compile = tle_patch_for_triton_compile()
code_generator.CodeGenerator = TleCodeGenerator

from .language import dsa

__all__ = [
    "dsa",
]