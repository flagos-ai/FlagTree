TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Lite
    "extract_tile",
    "insert_tile",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.buffered_tensor.slot",
    "gpu.alloc",
    "gpu.local_ptr",

    "device_mesh",
    "shard_id",
    "distributed_barrier",
    "pipe",
    "pipe.reader",
    "pipe.reader.wait",
    "pipe.reader.release",
    "pipe.writer",
    "pipe.writer.acquire",
    "pipe.writer.commit",
    "pipe.writer.close",
    "gpu.warp_specialize",
]
