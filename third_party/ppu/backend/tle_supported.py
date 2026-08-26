TLE_SUPPORTED_PRIMITIVES = [
    "load",
    "extract_tile",
    "insert_tile",
    "distributed_barrier",
    "cumsum",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.buffered_tensor.slot",
    "gpu.pipeline",  # TODO: del
    "gpu.warp_specialize",
]
