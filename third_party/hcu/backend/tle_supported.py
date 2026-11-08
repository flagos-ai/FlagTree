TLE_SUPPORTED_PRIMITIVES = [
    "load",
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "extract_tile",
    "fp_to_fp4",
    "fp4_to_fp_scaled",
    "pack_int4",
    "unpack_int4",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.set_layout",
]
