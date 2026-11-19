TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "range",
    "load",
    "extract_tile",
    "insert_tile",
    "distributed_barrier",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.range",
    "gpu.buffered_tensor.slot",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.warp_specialize",
]
