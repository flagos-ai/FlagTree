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
    "gpu.local_ptr",
    "gpu.set_layout",

    ## TLE distributed
    "device_mesh",
    "distributed_barrier",
    "sharding",
    "shard_id",
    "n_pes",
    "remote",
    "signal",
    "signal_wait",
]
