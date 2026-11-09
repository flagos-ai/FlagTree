# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Unified, non-intrusive dtype interception for unsupported Ascend operators.
#
# Two thin monkey-patch layers are installed (see ``install()``):
#
#   * The *builtin-dunder layer* wraps the operator-overload tensor dunders of
#     ``triton.language.core.tensor`` (``__add__``/``__rshift__``/``__and__``
#     ...). Operator syntax (``a + b``, ``a >> b``, ``-a``, ``a == b`` ...) is
#     turned by the AST tracer into a direct dunder builtin call and never
#     passes through ``call_Function``, so this is the only place where the
#     operator form can be intercepted. The corresponding ``TritonSemantic``
#     leaf methods are deliberately NOT patched: they are reused by internal
#     lowering (``load`` builds its mask with ``and_``/``or_``/``not_``,
#     ``sub`` lowers through ``minus``+``add``, ``atomic_min/max`` use
#     ``not_equal``/``where`` ...), so wrapping them would misfire. The dunder
#     builtin is reached only from the user's operator syntax.
#
#   * The *entry layer* wraps ``CodeGenerator.call_Function`` and validates
#     named builtins (``tl.*``), ``@jit`` composite ops (``tl.max``,
#     ``tl.sum`` ...) and cann extension builtins before they run.
#
# The rules themselves live in ``op_dtype_config.py``. Both layers are driven
# entirely by that configuration: a rule opts into the dunder layer with a
# ``"dunder"`` list and into the entry layer with ``"names"``.

import inspect
import contextvars
from functools import wraps

from .op_dtype_config import (
    OP_DTYPE_RULES,
    RULES_BY_NAME,
    ALL_SCALAR_DTYPES,
    FP8E4_DTYPES,
    FP8E5_DTYPES,
)

_LANG_MODULE_PREFIX = "triton.language"
_EXTRA_MARKER = ".extra"

# While a triton-internal composite @jit helper (e.g. the Philox code behind
# ``tl.rand`` / ``tl.randint`` in ``triton/language/random.py``) is being
# traced, this counter is > 0. Those helpers legitimately mix uint32/uint64
# arithmetic even though the standalone ``mul``/``add``/``xor``/``umulhi``
# rules document those dtypes as unsupported: the documented public contract
# is that ``rand``/``randint`` accept integer offsets, and their internals
# must not be validated as if each inner op were a user-facing call.
_INTERNAL_COMPOSITE_TRACE = contextvars.ContextVar("ascend_internal_composite_trace", default=0)


def _in_internal_composite():
    return _INTERNAL_COMPOSITE_TRACE.get() > 0


def _is_triton_internal_fn(fn):
    target = _unwrap_callable(fn)
    module = getattr(target, "__module__", "") or ""
    return module.startswith("triton.")


class UnsupportedDtypeError(TypeError):
    """Raised at the front-end when an Ascend operator gets an unsupported dtype."""


def arch_bucket(arch):
    """Classify an ``options.arch`` value into an ``"a2"`` / ``"a5"`` bucket."""
    try:
        from triton.backends.ascend.utils import is_compile_on_910_95
        return "a5" if is_compile_on_910_95(arch) else "a2"
    except Exception:
        return "a2"


def _bucket_from_semantic(semantic):
    options = getattr(getattr(semantic, "builder", None), "options", None)
    arch = getattr(options, "arch", None)
    return arch_bucket(arch)


def _normalize_dtype_name(name):
    return "int1" if name == "bool" else name


# Friendly dtype spellings shown in the front-end rejection message. Keys cover
# both the canonical rule tokens and the concrete/internal runtime names.
DTYPE_DISPLAY_NAMES = {
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "uint32": "uint32",
    "int32": "int32",
    "uint64": "uint64",
    "int64": "int64",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
    "fp8e4": "float8e4m3",
    "fp8e5": "float8e5m2",
    "bool": "bool",
    "int1": "bool",
    "fp8e4nv": "float8e4m3fn",
    "fp8e4b8": "float8e4m3fnuz",
    "fp8e4b15": "float8e4m3b11us",
    "fp8e5b16": "float8e5m2fnuz",
}


def _display_dtype(name):
    return DTYPE_DISPLAY_NAMES.get(name, name)


def _canonical_token(name):
    """Map an extracted runtime dtype name onto a rule token.

    ``bool`` arrives as ``int1`` and the concrete fp8 variants collapse onto
    their documented column (``fp8e4`` / ``fp8e5``).
    """
    if name == "int1":
        return "bool"
    if name in FP8E4_DTYPES:
        return "fp8e4"
    if name in FP8E5_DTYPES:
        return "fp8e5"
    return name


def _extract_scalar_name(value, read_pointer=False):
    """Return the scalar dtype ``name`` of a traced value, or ``None``.

    Understands:
      * a ``tl.dtype`` itself (destination dtype of ``cast`` / ``full`` ...)
      * a tensor / block (``value.type.scalar``)
      * ``constexpr`` wrapping one of the above
      * a pointer tensor: its pointee element dtype is returned only when
        ``read_pointer`` is True (i.e. the position is declared in the rule's
        ``pointer_args``). Otherwise pointers are skipped, because dunder
        builtins are reused for address computation (``ptr + offsets``) and a
        pointee dtype there does not describe the op itself.
    Python scalars, shapes, strings and ``None`` yield ``None``.
    """
    # Unwrap constexpr without importing the (compile-time) class directly.
    value = getattr(value, "value", value) if value.__class__.__name__ == "constexpr" else value

    # A bare dtype object (e.g. the ``dtype`` argument of ``cast`` / ``full``).
    name = getattr(value, "name", None)
    if isinstance(name, str) and hasattr(value, "primitive_bitwidth") and not hasattr(value, "handle"):
        return _normalize_dtype_name(name)

    type_obj = getattr(value, "type", None)
    if type_obj is None:
        return None

    # Unwrap a ``block_type`` (e.g. a vector of pointers produced by
    # ``ptr + offsets``) before the pointer test. ``block_type`` does not
    # override ``is_ptr`` and its ``scalar`` is the inner element type, so
    # without unwrapping a pointer *block* would read back as the
    # ``pointer<fp32>`` spelling instead of its pointee dtype (or instead of
    # being skipped when the position is not a declared pointer argument).
    is_block = getattr(type_obj, "is_block", None)
    if callable(is_block) and is_block():
        type_obj = getattr(type_obj, "element_ty", type_obj)

    is_ptr = getattr(type_obj, "is_ptr", None)
    if callable(is_ptr) and is_ptr():
        if not read_pointer:
            return None
        element_ty = getattr(type_obj, "element_ty", None)
        element_is_block = getattr(element_ty, "is_block", None)
        if callable(element_is_block) and element_is_block():
            element_ty = getattr(element_ty, "element_ty", element_ty)
        return _normalize_dtype_name(getattr(element_ty, "name", None))

    scalar = type_obj
    scalar_prop = getattr(type_obj, "scalar", None)
    if scalar_prop is not None and not callable(scalar_prop):
        scalar = scalar_prop
    return _normalize_dtype_name(getattr(scalar, "name", None))


def _unsupported_tokens(rule, bucket):
    """Return the unsupported-dtype blacklist for a bucket, or ``None``.

    ``None`` means "no dtype restriction on that architecture" (a missing
    bucket, see the rule schema in op_dtype_config.py); it must NOT fall back
    to the other bucket's list. An empty list means nothing is rejected.
    """
    unsupported = rule.get("unsupported")
    if not isinstance(unsupported, dict) or bucket not in unsupported:
        return None
    return unsupported[bucket]


def _supported_tokens(rule, bucket):
    """Return the accepted-dtype list for a bucket as rule tokens.

    Used only to build the user-facing error message. By default the list is
    NOT stored: it is derived as the complement of the bucket's ``unsupported``
    blacklist over ``ALL_SCALAR_DTYPES``, preserving universe order, which is
    exactly the whitelist the configuration used to carry by hand.

    A rule may still spell out an explicit ``supported`` list when its
    advertised dtypes are intentionally narrower than the blacklist complement
    (currently only ``dot_scaled``); that explicit value takes precedence.

    A missing bucket in ``unsupported`` means "no dtype restriction on that
    architecture", so every dtype is accepted there.
    """
    unsupported = _unsupported_tokens(rule, bucket)
    if unsupported is None:
        return list(ALL_SCALAR_DTYPES)
    explicit = rule.get("supported")
    if isinstance(explicit, dict) and bucket in explicit:
        return explicit[bucket]
    blocked = set(unsupported)
    return [token for token in ALL_SCALAR_DTYPES if token not in blocked]


def _format_supported_list(tokens):
    if not tokens:
        return "none"
    names = list(dict.fromkeys(_display_dtype(t) for t in tokens))
    return ", ".join(names)


def _align_operands(fn, args, kws):
    """Map runtime args/kws onto the public builtin positional parameters.

    This returns a list indexed exactly like the documented builtin signature
    (self of a member-style call included), so that ``tensor_args`` positions
    are stable regardless of whether an operand was passed positionally or by
    keyword (e.g. ``cast(x, dtype=...)`` / ``zeros(shape, dtype=...)``).
    Falls back to the raw args when introspection is unavailable.
    """
    try:
        signature_target = _unwrap_callable(fn)
        try:
            sig = inspect.signature(signature_target)
        except (TypeError, ValueError):
            return list(args)
        try:
            bound = sig.bind_partial(*args, **dict(kws or ()))
        except TypeError:
            return list(args)
        arguments = bound.arguments
        ordered = []
        for name, param in sig.parameters.items():
            if name in ("_semantic", "_generator"):
                continue
            if name in arguments:
                value = arguments[name]
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    ordered.extend(value)
                else:
                    ordered.append(value)
        return ordered
    except Exception:
        return list(args)


def validate(rule_id, args, bucket, op_label=None, kws=None, fn=None):
    """Validate selected operands against one rule.

    ``args`` are the raw runtime positional arguments. When ``fn``/``kws`` are
    provided (the entry layer), operands are aligned onto the builtin
    signature so keyword dst-dtypes land on their configured position.
    """
    rule = OP_DTYPE_RULES[rule_id]
    tokens = _unsupported_tokens(rule, bucket)
    # A missing bucket means "no dtype restriction" for this architecture.
    if tokens is None:
        return
    blocked = set(tokens)

    operands = _align_operands(fn, args, kws) if fn is not None else list(args)

    positions = rule.get("tensor_args")
    if positions is None:
        positions = range(len(operands))
    pointer_positions = set(rule.get("pointer_args") or ())

    # Operator symbol (``>>`` for a dunder) or the concise rule/function name.
    label = op_label or rule_id
    arch_name = "Ascend 950PR&950DT" if bucket == "a5" else "Atlas A2 products/Atlas A3 products"
    # The interception decision uses the blacklist above; the message tells
    # the user which dtypes ARE supported, so it is built from the complement
    # of that blacklist derived on the fly.
    supported_list = _format_supported_list(_supported_tokens(rule, bucket))

    for index in positions:
        if index >= len(operands):
            continue
        name = _extract_scalar_name(operands[index], read_pointer=index in pointer_positions)
        if name is None:
            continue
        token = _canonical_token(name)
        if token in blocked:
            raise UnsupportedDtypeError(
                f"operator '{label}' does not support dtype '{_display_dtype(name)}'"
                f" (operand #{index + 1}); supported dtypes on {arch_name}: {supported_list}."
                f" Please cast the operand to a supported dtype via tl.cast before calling '{label}'.")


# ---------------------------------------------------------------------------
# Builtin-dunder layer
# ---------------------------------------------------------------------------
#
# A rule opts into this layer with a ``"dunder"`` list (the tensor dunder
# names, reflected variants included) and a ``"dunder_label"`` (the operator
# shown in error messages), e.g.::
#
#     "rshift": {
#         "dunder": ["__rshift__", "__rrshift__"], "dunder_label": ">>",
#         ...
#     }
#
# Wrapping the builtin dunder itself intercepts only the user operator;
# internal lowering that calls a ``TritonSemantic`` leaf directly never
# touches it.
def _configured_tensor_dunders():
    mapping = {}
    for rule_id, rule in OP_DTYPE_RULES.items():
        label = rule.get("dunder_label", rule_id)
        for dunder_name in rule.get("dunder", ()):
            # A dunder belongs to exactly one operator; guard against
            # accidental double registration in the configuration.
            assert dunder_name not in mapping, f"dunder {dunder_name} registered twice"
            mapping[dunder_name] = (rule_id, label)
    return mapping


def _install_builtin_dunder_patches():
    from triton.language.core import TRITON_BUILTIN, tensor

    if getattr(tensor, "_ascend_dtype_dunder_patch", False):
        return

    for dunder_name, (rule_id, label) in _configured_tensor_dunders().items():
        original = getattr(tensor, dunder_name, None)
        if original is None or getattr(original, "_ascend_guard_wrapped", False):
            continue

        def make_wrapper(orig, rid, op_label):

            @wraps(orig)
            def wrapper(self, *args, **kwargs):
                semantic = kwargs.get("_semantic")
                # Only run while tracing (the builtin wrapper is what outside
                # users hit; inside a kernel ``_semantic`` is always supplied).
                # Skip while tracing a triton-internal composite helper (see
                # ``_INTERNAL_COMPOSITE_TRACE``).
                if semantic is not None and not _in_internal_composite():
                    try:
                        bucket = _bucket_from_semantic(semantic)
                        validate(rid, [self, *args], bucket, op_label=op_label)
                    except UnsupportedDtypeError:
                        raise
                    except Exception:
                        # Never let guard bookkeeping break compilation.
                        pass
                return orig(self, *args, **kwargs)

            wrapper._ascend_guard_wrapped = True
            # Keep the builtin marker so the tracer still treats it as such.
            setattr(wrapper, TRITON_BUILTIN, True)
            return wrapper

        setattr(tensor, dunder_name, make_wrapper(original, rule_id, label))

    tensor._ascend_dtype_dunder_patch = True


# ---------------------------------------------------------------------------
# Entry layer
# ---------------------------------------------------------------------------


def _build_short_name_index():
    lang_index = {}
    ext_index = {}
    for full_name in RULES_BY_NAME:
        short = full_name.rsplit(".", 1)[-1]
        if _EXTRA_MARKER in full_name:
            ext_index.setdefault(short, RULES_BY_NAME[full_name])
        else:
            lang_index.setdefault(short, RULES_BY_NAME[full_name])
    return lang_index, ext_index


_LANG_SHORT_INDEX, _EXT_SHORT_INDEX = _build_short_name_index()


def _unwrap_callable(fn):
    seen = set()
    for _ in range(8):
        if id(fn) in seen:
            break
        seen.add(id(fn))
        func = getattr(fn, "__func__", None)
        if func is not None:
            fn = func
            continue
        # JITFunction / ConstexprFunction wrap the raw Python function.
        inner = getattr(fn, "fn", None)
        if callable(inner):
            fn = inner
            continue
        wrapped = getattr(fn, "__wrapped__", None)
        if callable(wrapped):
            fn = wrapped
            continue
        # ``_tensor_member_fn`` registers a closure wrapper named "wrapper".
        if getattr(fn, "__name__", None) == "wrapper" and getattr(fn, "__closure__", None):
            candidates = []
            for cell in fn.__closure__:
                try:
                    contents = cell.cell_contents
                except ValueError:
                    continue
                if callable(contents) and getattr(contents, "__name__", None) not in (None, "wrapper"):
                    candidates.append(contents)
            if candidates:
                fn = candidates[0]
                continue
        break
    return fn


def _resolve_entry_rule(fn):
    target = _unwrap_callable(fn)
    module = getattr(target, "__module__", "") or ""
    qualname = getattr(target, "__qualname__", "") or getattr(target, "__name__", "") or ""
    if not qualname:
        return None

    basename = qualname.rsplit(".", 1)[-1]

    # Exact match, progressively dropping trailing module components:
    #   triton.language.standard.max -> triton.language.max
    #   triton.language.extra.cann.extension.vec_ops.insert_slice
    #       -> triton.language.extra.cann.extension.insert_slice
    parts = module.split(".")
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut]) + "." + basename
        rule_id = RULES_BY_NAME.get(candidate)
        if rule_id is not None:
            return rule_id

    # Short-name fallback, constrained to the right operator family so that
    # e.g. the extension cube ``dot`` can never shadow the regular ``tl.dot``.
    if module.startswith(_LANG_MODULE_PREFIX):
        if _EXTRA_MARKER in module:
            return _EXT_SHORT_INDEX.get(basename)
        return _LANG_SHORT_INDEX.get(basename)
    return None


def _install_entry_patch():
    from triton.compiler.code_generator import CodeGenerator, _is_triton_value

    if getattr(CodeGenerator, "_ascend_dtype_entry_patch", False):
        return

    # These bound-call wrappers live at module scope and have moved between
    # modules across Triton versions (BoundConstexprFunction is re-exported
    # from triton.runtime.jit). Resolve them opportunistically; a missing
    # wrapper must not abort the whole entry layer and silently disable dtype
    # interception for every builtin.
    import triton.compiler.code_generator as _cg

    bound_method_types = []
    for symbol in ("BoundJITMethod", "BoundConstexprFunction"):
        obj = getattr(_cg, symbol, None)
        if obj is None:
            try:
                import triton.runtime.jit as _jit

                obj = getattr(_jit, symbol, None)
            except Exception:
                obj = None
        if obj is not None and obj not in bound_method_types:
            bound_method_types.append(obj)
    bound_method_types = tuple(bound_method_types)

    original_call_function = CodeGenerator.call_Function

    def call_function(self, node, fn, args, kws):
        # Skip while tracing a triton-internal composite helper (see
        # ``_INTERNAL_COMPOSITE_TRACE``): its inner ops are library internals,
        # not user-facing calls.
        if not _in_internal_composite():
            try:
                check_fn = fn
                check_args = args
                # Member-style calls (``x.sum()``, ``x.exp()``) bind the tensor as
                # the first operand outside of ``args``; restore it logically for
                # validation without mutating the arguments handed to the original
                # implementation (which performs the same insertion itself).
                if bound_method_types and isinstance(fn, bound_method_types):
                    check_fn = fn.__func__
                    check_args = [fn.__self__, *args]
                else:
                    self_obj = getattr(fn, "__self__", None)
                    if self_obj is not None and _is_triton_value(self_obj):
                        check_fn = getattr(fn, "__func__", fn)
                        check_args = [self_obj, *args]

                rule_id = _resolve_entry_rule(check_fn)
                if rule_id is not None:
                    bucket = arch_bucket(getattr(getattr(self.builder, "options", None), "arch", None))
                    validate(rule_id, list(check_args), bucket, fn=check_fn, kws=kws)
            except UnsupportedDtypeError as exc:
                # Re-raise the actual rejection attached to the user source node,
                # mirroring how ``call_Function`` wraps builtin failures so the
                # message points at the offending call rather than the compiler.
                from triton import knobs

                if knobs.compilation.front_end_debugging:
                    raise
                from triton.compiler.errors import CompilationError

                raise CompilationError(self.jit_fn.src, node, str(exc)) from None
            except Exception:
                # Validation must never break compilation; only explicit dtype
                # rejections are allowed to propagate.
                pass
        return original_call_function(self, node, fn, args, kws)

    CodeGenerator.call_Function = call_function
    CodeGenerator._ascend_dtype_entry_patch = True

    original_call_jit_function = CodeGenerator.call_JitFunction

    def call_jit_function(self, fn, args, kwargs, caller_context=None):
        # Nested @jit helpers that ship inside triton itself (the Philox
        # implementation behind tl.rand/tl.randint lives in
        # triton/language/random.py) are library internals. Raise the internal
        # flag for the duration of their trace so the uint32/uint64 arithmetic
        # they are built from is not rejected as a standalone user op. The
        # user-facing call itself (``tl.rand(...)`` in user code) is validated
        # by the entry layer before this flag is raised.
        if _is_triton_internal_fn(fn):
            token = _INTERNAL_COMPOSITE_TRACE.set(_INTERNAL_COMPOSITE_TRACE.get() + 1)
            try:
                return original_call_jit_function(self, fn, args, kwargs, caller_context)
            finally:
                _INTERNAL_COMPOSITE_TRACE.reset(token)
        return original_call_jit_function(self, fn, args, kwargs, caller_context)

    CodeGenerator.call_JitFunction = call_jit_function


def install_dtype_guard():
    """Install all interception layers. Idempotent.

    This is the single public entry point the Ascend backend calls while
    patching ``CodeGenerator`` (see ``third_party/ascend/backend/__init__.py``).
    It is a real, importable symbol (not a lazy alias) so it can be resolved
    statically and located via ``from ... import install_dtype_guard``.
    """
    _install_builtin_dunder_patches()
    _install_entry_patch()
