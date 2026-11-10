from types import SimpleNamespace

from triton._flagtree_spec import bind_language_extension_symbols_to_tl

from . import libdevice

from .utils import fp_to_fp4, fp4_to_fp_scaled, memrealtime

__all__ = ["libdevice", "memrealtime"]

bind_language_extension_symbols_to_tl(
    SimpleNamespace(__all__=("fp_to_fp4", "fp4_to_fp_scaled"), fp_to_fp4=fp_to_fp4, fp4_to_fp_scaled=fp4_to_fp_scaled))
