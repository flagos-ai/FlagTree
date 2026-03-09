# Copyright 2026- Xcoresigma Technology Co., Ltd

from triton._C.libtriton import ir
from typing import Optional, Dict
from triton.runtime import JITFunction
from .language.builder import setup_unified_builder_with_tle_builder
import importlib
import ast
from typing_extensions import override

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
        super().__init__(context, prototype, gscope, attributes, constants, function_name, jit_fn, options, codegen_fns,
                         module_map, module, is_kernel, function_types, noinline, file_name, begin_line)
        self.tle_builder = tle_ir.tle_builder(context)
        self.tle_builder.set_loc(file_name, begin_line, 0)

        # Stack to keep track of active `with`-hints (e.g., tle.hint(...))
        # Each entry is a dict mapping hint names to literal values.
        self.with_hints = []

        setup_unified_builder_with_tle_builder(self.builder, self.tle_builder)

    @override
    def visit_With(self, node):
        assert len(node.items) == 1
        context = node.items[0].context_expr

        # extract tle hints
        hints = {}
        if isinstance(context, ast.Call):
            if isinstance(context.func, ast.Attribute) and context.func.attr == "hint":
                for kw in context.keywords:
                    if not isinstance(kw.value, ast.Constant):
                        raise self._unsupported(node,
                                                "keyword arguments to hint() are only supported for constant values")
                    hints[kw.arg] = kw.value.value

        # append hints to with_hints anyway, to indicate that we're in the with scope
        self.with_hints.append(hints)

        super().visit_With(node)

        # pop hints to indicate that we're out of the with scope
        self.with_hints.pop()


def extract_tle_hints_scope(generator: TleCodeGenerator):
    """
    with tle.hints(inter_no_alias=True):
        with xxxx:
            with tle.hints(inter_no_alias=False):
                ...
                with xxx:
                    call_fn1(...)
                call_fn(...)

    when visit_Call for call_fn1, we can get the hints scope as follows:
        [{'inter_no_alias': True}, {xxx}, {'inter_no_alias': False}, {xxx}]
    should get the parent scope hints 'inter_no_alias': False for call_fn1, after visit call_fn1, pop the scope

    when visit_Call for call_fn, we can get the hints scope as follows:
        [{'inter_no_alias': True}, {xxx}, {'inter_no_alias': False}]
    and now the hint scope is 'inter_no_alias': False' for call_fn, after visit call_fn, pop the scope
    """
    if not generator.with_hints:
        return {}

    # visit with_hints backward to find inter_no_alias hint
    for i in range(len(generator.with_hints) - 1, -1, -1):
        hints = generator.with_hints[i]
        if "inter_no_alias" in hints:
            return hints

    return {}


triton_compiler.compile = tle_patch_for_triton_compile()
code_generator.CodeGenerator = TleCodeGenerator

from .language import dsa

__all__ = [
    "dsa",
]
