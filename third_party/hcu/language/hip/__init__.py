from types import SimpleNamespace

from triton._flagtree_spec import bind_language_extension_symbols_to_tl
from triton.language import core

from . import libdevice

from .utils import memrealtime

__all__ = ["libdevice", "memrealtime"]

bind_language_extension_symbols_to_tl(
    SimpleNamespace(__all__=("fp_to_fp4", "fp4_to_fp_scaled"), fp_to_fp4=core.fp_to_fp4,
                    fp4_to_fp_scaled=core.fp4_to_fp_scaled))
