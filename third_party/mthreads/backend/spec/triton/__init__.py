"""Mthreads backend spec module.

This module is loaded by flagtree_spec.py via
importlib.import_module("triton.backends.mthreads.spec") when
FLAGTREE_BACKEND=mthreads. Functions defined here are callable
via flagtree_spec.spec("function_name", ...).
"""

# Imported at module level so the JIT code generator can resolve
# the annotation name from fn.__globals__.  The module is loaded
# dynamically by flagtree_spec AFTER triton.language is ready, so
# this is safe against circular imports.
from triton.language.core import constexpr


def init_language():
    """Add mthreads-specific symbols to triton.language.

    Called explicitly from the main triton/__init__.py after
    triton.language is fully initialized, via:
        spec("init_language")
    """
    from triton.flagtree_spec import bind_language_extension_symbols_to_tl
    from triton.runtime.jit import jit as _jit
    from triton.language.core import (
        builtin as _builtin,
        static_assert as _static_assert,
        _unwrap_if_constexpr,
    )

    class _Ext:
        __all__ = [
            "squeeze",
            "unsqueeze",
            "_experimental_descriptor_load",
            "_experimental_descriptor_store",
        ]

    _ext = _Ext()

    @_jit
    def squeeze(x, dim: constexpr):
        _static_assert(x.shape[dim] == 1)
        return x.reshape(x.shape[:dim] + x.shape[dim + 1:])

    @_jit
    def unsqueeze(x, dim: constexpr):
        return x.reshape(x.shape[:dim] + (1, ) + x.shape[dim:])

    @_builtin
    def _experimental_descriptor_load(desc_pointer, offsets, shape, dtype, _semantic=None):
        """Legacy compatibility API for descriptor load.

        New code should prefer ``load_tensor_descriptor``. We keep this symbol
        so migrated tests can exercise the same backend descriptor path without
        monkeypatching triton.language internals in conftest.
        """
        _ = shape
        dtype = _unwrap_if_constexpr(dtype)
        value = desc_pointer.load(offsets, _semantic=_semantic)
        if value.dtype == dtype:
            return value
        if value.dtype.primitive_bitwidth == dtype.primitive_bitwidth:
            return value.to(dtype, bitcast=True, _semantic=_semantic)
        return value.to(dtype, _semantic=_semantic)

    @_builtin
    def _experimental_descriptor_store(desc_pointer, value, offsets, _semantic=None):
        """Legacy compatibility API for descriptor store.

        New code should prefer ``store_tensor_descriptor``.
        """
        value = _semantic.to_tensor(value)
        desc_dtype = desc_pointer.dtype
        if value.dtype != desc_dtype and value.dtype.primitive_bitwidth == desc_dtype.primitive_bitwidth:
            value = value.to(desc_dtype, bitcast=True, _semantic=_semantic)
        return desc_pointer.store(offsets, value, _semantic=_semantic)

    _ext.squeeze = squeeze
    _ext.unsqueeze = unsqueeze
    _ext._experimental_descriptor_load = _experimental_descriptor_load
    _ext._experimental_descriptor_store = _experimental_descriptor_store

    bind_language_extension_symbols_to_tl(_ext)


def spec_get_stub_target() -> GPUTarget:
    arch = os.environ.get("TRITON_OVERRIDE_ARCH") or os.environ.get("TRITON_MUSA_ARCH") or "ph1"
    return GPUTarget("musa", arch, 32)
