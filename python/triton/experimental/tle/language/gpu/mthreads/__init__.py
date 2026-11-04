# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from . import buffer, common, copy, pipe, warp_specialize, wgmma
from .types import MusaDotOperandEncoding, MusaSqmmaEncoding, MusaWmmaEncoding, TMECachePolicy

from triton._C import libtriton

# Public capability markers for clients that need to retain a safe fallback
# when running against an older FlagTree wheel.  Consumer-sync version 1
# covers physical-CTA barrier initialization, cyclic multi-epoch named-reader
# rendezvous, and mixed-warp rectangular shared-memory slice operands.
MTHREADS_GEMM_COMPILER_OPTIONS_VERSION = 2
MTHREADS_TLE_CONSUMER_SYNC_VERSION = 1
MTHREADS_TLE_PIPE_SQMMA_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_pipe_sqmma_version", 0)
MTHREADS_TLE_MULTIFIELD_PIPE_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_multifield_pipe_version", 0)
MTHREADS_TLE_ONE_SHOT_PIPE_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_one_shot_pipe_version", 0)
MTHREADS_TLE_SPMC_PIPE_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_spmc_pipe_version", 0)
MTHREADS_TLE_DYNAMIC_LOOPS_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_dynamic_loops_version", 0)
MTHREADS_TLE_SPLIT_M_SQMMA_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_split_m_sqmma_version", 0)
MTHREADS_TLE_SPLIT128_SQMMA_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_split128_sqmma_version", 0)
MTHREADS_TLE_LOCAL_BARRIER_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_local_barrier_version", 0)
MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA_VERSION = getattr(libtriton.ir.builder,
                                                        "mthreads_tle_persistent_ordered_sqmma_version", 0)
MTHREADS_TLE_DYNAMIC_PARTITION_SYNC_VERSION = getattr(libtriton.ir.builder,
                                                      "mthreads_tle_dynamic_partition_sync_version", 0)
MTHREADS_TLE_GROUPED_COMPLETION_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_grouped_completion_version", 0)
MTHREADS_TLE_16_WARP_PERSISTENT_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_16_warp_persistent_version", 0)
MTHREADS_TLE_FUSED_PIPE_CONSUMER_VERSION = getattr(libtriton.ir.builder, "mthreads_tle_fused_pipe_consumer_version", 0)
# Python-only backend option capability. It lets persistent TLE kernels retain
# the SQMMA optimizations while avoiding the unsafe max-ILP scheduler.
MTHREADS_TLE_DISABLE_MAX_ILP_SCHEDULER_VERSION = 1

__all__ = [
    "buffer",
    "common",
    "copy",
    "pipe",
    "warp_specialize",
    "wgmma",
    "TMECachePolicy",
    "MusaDotOperandEncoding",
    "MusaSqmmaEncoding",
    "MusaWmmaEncoding",
    "MTHREADS_GEMM_COMPILER_OPTIONS_VERSION",
    "MTHREADS_TLE_CONSUMER_SYNC_VERSION",
    "MTHREADS_TLE_PIPE_SQMMA_VERSION",
    "MTHREADS_TLE_MULTIFIELD_PIPE_VERSION",
    "MTHREADS_TLE_ONE_SHOT_PIPE_VERSION",
    "MTHREADS_TLE_SPMC_PIPE_VERSION",
    "MTHREADS_TLE_DYNAMIC_LOOPS_VERSION",
    "MTHREADS_TLE_SPLIT_M_SQMMA_VERSION",
    "MTHREADS_TLE_SPLIT128_SQMMA_VERSION",
    "MTHREADS_TLE_LOCAL_BARRIER_VERSION",
    "MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA_VERSION",
    "MTHREADS_TLE_DYNAMIC_PARTITION_SYNC_VERSION",
    "MTHREADS_TLE_GROUPED_COMPLETION_VERSION",
    "MTHREADS_TLE_16_WARP_PERSISTENT_VERSION",
    "MTHREADS_TLE_FUSED_PIPE_CONSUMER_VERSION",
    "MTHREADS_TLE_DISABLE_MAX_ILP_SCHEDULER_VERSION",
]
