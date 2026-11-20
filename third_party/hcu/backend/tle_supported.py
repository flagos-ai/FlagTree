TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "extract_tile",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.alloc",
    "gpu.copy",
    "gpu.buffered_tensor.load",
    "gpu.buffered_tensor.store",
    "gpu.local_ptr",
    "gpu.set_layout",
]
