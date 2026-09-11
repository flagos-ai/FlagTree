# Triton 3.6 XPU TLE - Triton Language Extension (FlagTree XPU overlay)
#
# This overlay replaces the main tree's `triton.experimental.tle` for XPU builds.
# The two are same-origin forks with incompatible mechanisms: the main tree is
# multi-backend and region-based (`create_region_by_llvm` + output_indices), this
# one is XPU typed-raw-op based (`triton_xpu.raw` + an xpu-clang payload
# pipeline). An XPU build cannot serve cuda/tops raw payloads, so it ships this
# version instead of merging the two.
#
# APIs:
#   tle.raw  -- inject a hand-written device payload (tle.raw.dialect / .call)
#   tle.gpu  -- on-chip buffers and continuous DMA copy (alloc / copy / local_ptr)
#   tle.dsa, tle.pipe -- stubs; see language/dsa/__init__.py
from . import language
from . import raw
from .language import dsa, gpu, pipe

__all__ = ["language", "gpu", "dsa", "pipe", "raw"]
