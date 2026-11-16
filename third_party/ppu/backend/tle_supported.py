TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Lite
    "load",
    "extract_tile",
    "insert_tile",
    "distributed_barrier",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.buffered_tensor.slot",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.warp_specialize",
]