TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "load",
    "extract_tile",
    "insert_tile",
    "device_mesh",
    "sharding",
    "shard_id",
    "distributed_barrier",
    "remote",
    "cumsum",
    "pipe",
    "pipe.reader",
    "pipe.reader.wait",
    "pipe.reader.release",
    "pipe.writer",
    "pipe.writer.acquire",
    "pipe.writer.commit",
    "pipe.writer.close",

    ## TLE-Struct GPU
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.memory_space",
    "gpu.warp_specialize",
]
