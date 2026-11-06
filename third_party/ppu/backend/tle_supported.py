# TLE primitives the PPU backend accepts, aligned with the per-backend lists of
# #1054 (tle.is_tle_supported reads it).
#
# gpu.alloc_barrier(s) / gpu.barrier_wait / gpu.barrier_arrive are their named
# forms only (no phaseIdx, init=PENDING, no expect_bytes): PPU has no mbarrier.
# compiler.reject_unsupported_tle names the API of a phased wait or arrive and
# of a copy with a completion barrier; the mbarrier ops every other form lowers
# to are illegal in the LLVM conversion (BarrierOpToLLVM.cpp). A named barrier
# on PPU orders shared memory between the warp groups, not global memory
# (bar.sync on NVIDIA orders both): a kernel that hands data to another group
# through global memory orders it itself, see the docstrings in
# python/triton/experimental/tle/language/gpu/core.py.
#
# distributed_barrier stays listed to keep the list aligned across backends;
# compiler.reject_unsupported_tle rejects it with a clear error, which
# test_tle_smoke.py asserts.
TLE_SUPPORTED_PRIMITIVES = [
    "load",
    "extract_tile",
    "insert_tile",
    "distributed_barrier",
    "cumsum",
    "pipe",
    "pipe.reader",
    "pipe.reader.wait",
    "pipe.reader.release",
    "pipe.writer",
    "pipe.writer.acquire",
    "pipe.writer.commit",
    "pipe.writer.close",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.warp_specialize",
    "gpu.alloc_barrier",
    "gpu.alloc_barriers",
    "gpu.barrier_wait",
    "gpu.barrier_arrive",
    "gpu.buffered_tensor.slot",
    "gpu.buffered_tensor.subslice",
    "gpu.pipeline",
]
