"""Per-kernel routing between the NVIDIA and TileIR backends.

Activation:
    FLAGTREE_USE_TILEIR=1   — opt non-TLE kernels into the TileIR compile path.
                              Kernels that only use the supported top-level TLE
                              subset also use TileIR; other TLE kernels fall back
                              to the nvidia backend.
    FLAGTREE_TILEIR_VERBOSE=1 — print routing decisions to stderr for debugging.

When FLAGTREE_USE_TILEIR is unset (the default), the current target is returned
unchanged and flagtree behaves exactly as before.
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
from enum import Enum
from types import ModuleType

from triton.backends.compiler import GPUTarget

# ``nvidia`` is the installed backend name; its GPUTarget label is ``cuda``.
# Routing operates on target labels, so only ``cuda`` is valid here.
_CUDA_TARGET = "cuda"
_TILEIR_EXT_MODULE = "triton.backends.tileir.extend_core"
_TLE_LANGUAGE_MODULE = "triton.experimental.tle.language"
_TILEIR_SUPPORTED_TLE_ATTRS = frozenset({
    "make_tensor_view",
    "make_partition_view",
    "make_view",
    "dim",
    "load_view_tko",
    "store_view_tko",
    "create_mem_token",
    "join_mem_tokens",
})


class TLEUsage(Enum):
    NONE = "none"
    TILEIR_SUBSET = "tileir_subset"
    OTHER = "other"
    UNKNOWN = "unknown"


def _env_use_tileir() -> bool:
    return os.environ.get("FLAGTREE_USE_TILEIR", "0") == "1"


def _env_verbose() -> bool:
    return os.environ.get("FLAGTREE_TILEIR_VERBOSE", "0") == "1"


def _log(msg: str) -> None:
    if _env_verbose():
        print(f"[flagtree] {msg}", file=sys.stderr, flush=True)


def _kernel_name(jit_fn) -> str:
    fn = getattr(jit_fn, "fn", jit_fn)
    return getattr(fn, "__name__", repr(fn))


def kernel_tle_usage(jit_fn) -> TLEUsage:
    """Heuristic detection of TLE usage in a JIT'd function's source.

    ``TLEUsage.TILEIR_SUBSET`` means every detected TLE operation belongs to
    the TileIR-supported top-level TKO view/token subset.
    Any other TLE usage keeps the kernel on the native NVIDIA path.

    Result is cached on the JITFunction instance via ``_flagtree_tle_usage``.
    """
    cached = getattr(jit_fn, "_flagtree_tle_usage", None)
    if cached is not None:
        return cached

    fn = getattr(jit_fn, "fn", jit_fn)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        # Can't prove this kernel is TLE-free.
        result = TLEUsage.UNKNOWN
    else:
        result = _scan_source_for_tle(src, getattr(fn, "__globals__", None))

    jit_fn._flagtree_tle_usage = result
    return result


def kernel_uses_tle(jit_fn) -> bool:
    """Backward-compatible boolean predicate for callers/tests."""
    return kernel_tle_usage(jit_fn) in (TLEUsage.TILEIR_SUBSET, TLEUsage.OTHER)


def _global_obj_kind(obj) -> str | None:
    if isinstance(obj, ModuleType):
        module_name = getattr(obj, "__name__", "")
        if module_name == _TLE_LANGUAGE_MODULE:
            return "tle_language_module"
        if module_name.startswith("triton.experimental.tle."):
            return "tle_other_module"
        return None

    obj_module = getattr(obj, "__module__", "")
    obj_name = getattr(obj, "__name__", "")
    if obj_module == _TILEIR_EXT_MODULE and obj_name in _TILEIR_SUPPORTED_TLE_ATTRS:
        return "supported_function"
    if obj_module.startswith("triton.experimental.tle."):
        return "unsupported_function"
    return None


def _scan_source_for_tle(src: str, global_ns=None) -> TLEUsage:
    if "tle" not in src and not global_ns:
        return TLEUsage.NONE
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # `inspect.getsource` of a decorated function may include the decorator;
        # if ast can't parse, fall back to a substring check.
        if any(f"tle.{name}" in src for name in _TILEIR_SUPPORTED_TLE_ATTRS):
            return TLEUsage.TILEIR_SUBSET
        return TLEUsage.OTHER if "tle." in src or "triton.experimental.tle" in src else TLEUsage.NONE

    tle_aliases = {"tle"}
    imported_supported = False
    imported_other = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "triton.experimental" and any(n.name == "tle" for n in node.names):
                for n in node.names:
                    if n.name == "tle":
                        tle_aliases.add(n.asname or n.name)
            elif mod == "triton.experimental.tle" and any(n.name == "language" for n in node.names):
                for n in node.names:
                    if n.name == "language":
                        tle_aliases.add(n.asname or n.name)
                    elif n.name == "*":
                        imported_other = True
                    else:
                        imported_other = True
            elif mod == _TLE_LANGUAGE_MODULE:
                for n in node.names:
                    if n.name == "*":
                        imported_other = True
                    elif n.name in _TILEIR_SUPPORTED_TLE_ATTRS:
                        imported_supported = True
                    else:
                        imported_other = True
            elif mod.startswith("triton.experimental.tle"):
                imported_other = True
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name == _TLE_LANGUAGE_MODULE:
                    if n.asname:
                        tle_aliases.add(n.asname)
                    else:
                        imported_other = True
                elif n.name and n.name.startswith("triton.experimental.tle"):
                    imported_other = True

    saw_supported = imported_supported
    saw_other = imported_other

    def classify_attribute(func):
        attrs = []
        base = func
        while isinstance(base, ast.Attribute):
            attrs.append(base.attr)
            base = base.value
        if not isinstance(base, ast.Name):
            return None
        attrs.reverse()
        kind = _global_obj_kind(global_ns.get(base.id)) if global_ns else None
        is_tle_language = base.id in tle_aliases or kind == "tle_language_module"
        is_tle_other_module = kind == "tle_other_module"
        if is_tle_language and len(attrs) == 1:
            return "supported" if attrs[0] in _TILEIR_SUPPORTED_TLE_ATTRS else "other"
        if is_tle_language or is_tle_other_module:
            return "other"
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            kind = _global_obj_kind(global_ns.get(node.func.id)) if global_ns else None
            if kind == "supported_function":
                saw_supported = True
            elif kind in ("unsupported_function", "tle_other_module"):
                saw_other = True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            kind = classify_attribute(node.func)
            if kind == "supported":
                saw_supported = True
            elif kind == "other":
                saw_other = True

    if saw_other:
        return TLEUsage.OTHER
    if saw_supported:
        return TLEUsage.TILEIR_SUBSET
    return TLEUsage.NONE


def pick_target_for_kernel(target: GPUTarget, jit_fn) -> GPUTarget:
    """Return either ``target`` unchanged or a TileIR-flavored variant.

    Routing matrix:
      backend is not cuda           → unchanged (e.g. AMD)
      FLAGTREE_USE_TILEIR not set   → unchanged (no behavior change vs today)
      env set + unsupported TLE     → unchanged + warning
      env set + TileIR TLE subset   → swap backend to "tileir"
      env set + unknown source      → unchanged + warning
      env set + no TLE              → swap backend to "tileir"
    """
    if target.backend != _CUDA_TARGET:
        return target
    if not _env_use_tileir():
        return target

    name = _kernel_name(jit_fn)
    tle_usage = kernel_tle_usage(jit_fn)
    if tle_usage is TLEUsage.OTHER:
        _log(f"kernel {name}: TLE detected, falling back to nvidia "
             f"(FLAGTREE_USE_TILEIR=1 is set but this TLE subset is incompatible with TileIR)")
        return target
    if tle_usage is TLEUsage.UNKNOWN:
        _log(f"kernel {name}: source unavailable, falling back to nvidia "
             f"(cannot prove TLE usage is TileIR-compatible)")
        return target

    if tle_usage is TLEUsage.TILEIR_SUBSET:
        _log(f"kernel {name}: routed via tileir (TileIR-supported TLE subset, FLAGTREE_USE_TILEIR=1)")
    else:
        _log(f"kernel {name}: routed via tileir (no TLE, FLAGTREE_USE_TILEIR=1)")
    return GPUTarget(backend="tileir", arch=target.arch, warp_size=target.warp_size)
