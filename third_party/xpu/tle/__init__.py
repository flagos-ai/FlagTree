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
#   tle.dsa  -- allocate buffers in a named on-chip address space (tle.dsa.alloc)
#   tle.pipe -- an explicit producer/consumer edge over tle.dsa buffers
# Phase 1: contiguous DMA copy
# APIs: tle.gpu.alloc, tle.gpu.local_ptr, tle.gpu.copy, TensorDescriptor
# tle.raw: inject a hand-written payload (tle.raw.dialect / tle.raw.call)
# tle.dsa: allocate buffers in a named on-chip address space (tle.dsa.alloc)
# tle.pipe: an explicit producer/consumer edge over tle.dsa buffers
from . import language
from . import raw
from .language import dsa, gpu, pipe

__all__ = ["language", "gpu", "dsa", "pipe", "raw"]
