TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "load",
    "extract_tile",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.alloc",
    "gpu.buffered_tensor.load",
    "gpu.buffered_tensor.store",
    "gpu.buffered_tensor.slot",
    "gpu.buffered_tensor.subslice",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.warp_specialize",
    "gpu.set_layout",
]
