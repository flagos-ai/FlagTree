from . import backends

from . import language

try:
    from . import raw
except ModuleNotFoundError:
    raw = None

# Copyright 2026- Xcoresigma Technology Co., Ltd

import ast
import importlib
from typing import Dict, Optional

from triton._C.libtriton import ir
from triton.runtime import JITFunction
from typing_extensions import override

from .language.builder import setup_unified_builder_with_tle_builder

try:
    from triton._C.libtriton import tle as tle_ir
except ImportError:
    raise RuntimeError("tle is not available")

triton_compiler = importlib.import_module("triton.compiler", package=__package__)


class scope:
    """Scope marker for `with tle.scope(core_mode=...)`."""

    def __init__(self, *, core_mode):
        if core_mode not in ("cube", "vector"):
            raise ValueError(f'core_mode must be "cube" or "vector", got {core_mode!r}')
        self.core_mode = core_mode

    def __enter__(self):
        raise RuntimeError("tle.scope() can only be used inside a Triton kernel")

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _is_tle_attr_call(context, attr):
    return isinstance(context, ast.Call) and isinstance(context.func, ast.Attribute) and context.func.attr == attr


def _validate_tle_scope(context):
    if context.args:
        raise ValueError("tle.scope() only accepts keyword arguments")
    keywords = {kw.arg: kw.value for kw in context.keywords}
    if set(keywords) != {"core_mode"}:
        raise ValueError('tle.scope() requires exactly core_mode="cube" or core_mode="vector"')
    value = keywords["core_mode"]
    if not isinstance(value, ast.Constant) or value.value not in ("cube", "vector"):
        raise ValueError('tle.scope() core_mode must be the literal "cube" or "vector"')


def _emit_tle_scope(generator, node):
    from triton.language.extra.cann.extension.code_generator import handle_scope_with

    return handle_scope_with(generator, node)


def tle_patch_for_triton_compile():
    original_compile_fn = triton_compiler.compile

    def tle_compile(src, target=None, options=None):
        # ir.context() will return a new MLIRContext each time, here should keep the same context
        cur_context = ir.context()
        tle_ir.load_dialects(cur_context)
        tle_ir.dsa_ir.load_tile_dialects(cur_context)

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

    def __init__(self, context, prototype, gscope, function_name, jit_fn: JITFunction, *, options, codegen_fns,
                 module_map, is_gluon, module=None, is_kernel=False, function_types: Optional[Dict] = None,
                 noinline=False, caller_context=None, file_name: Optional[str] = None, begin_line=0):
        super().__init__(context, prototype, gscope, function_name, jit_fn, options=options, codegen_fns=codegen_fns,
                         module_map=module_map, is_gluon=is_gluon, module=module, is_kernel=is_kernel,
                         function_types=function_types, noinline=noinline, caller_context=caller_context,
                         file_name=file_name, begin_line=begin_line)

        # Stack to keep track of active `with`-hints (e.g., tle.hint(...))
        # Each entry is a dict mapping hint names to literal values.
        self.with_hints = []

        if not is_gluon:
            self.tle_builder = self.builder
        else:
            self.tle_builder = None

    @override
    def visit_With(self, node):
        assert len(node.items) == 1
        context = node.items[0].context_expr

        # extract tle hints
        hints = {}
        if _is_tle_hint := _is_tle_attr_call(context, "hint"):
            for kw in context.keywords:
                if not isinstance(kw.value, ast.Constant):
                    raise self._unsupported(node, "keyword arguments to hint() are only supported for constant values")
                hints[kw.arg] = kw.value.value

        # append hints to with_hints anyway, to indicate that we're in the with scope
        self.with_hints.append(hints)

        try:
            if _is_tle_hint:
                # tle.dsa.hint() is a marker for TLE codegen, not a runtime context
                # manager. Do not visit the context expression, otherwise the dummy
                # Python function is called and raises.
                return self.visit_compound_statement(node.body)
            if _is_tle_attr_call(context, "scope") and self.visit(context.func) is scope:
                _validate_tle_scope(context)
                return _emit_tle_scope(self, node)
            return super().visit_With(node)
        finally:
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


def __getattr__(name):
    if name == "dsa":
        from .language import dsa
        globals()[name] = dsa
        return dsa
    try:
        return getattr(backends, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "language",
    "dsa",
    "scope",
    *backends.ops(),
]

if raw is not None:
    __all__.append("raw")
