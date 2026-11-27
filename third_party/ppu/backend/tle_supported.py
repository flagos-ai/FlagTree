TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "load",
    "extract_tile",
    "insert_tile",
    "distributed_barrier",  # listed, but rejected at compile time by reject_unsupported_tle
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
    "gpu.buffered_tensor.slot",
    "gpu.buffered_tensor.subslice",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.warp_specialize",
    "gpu.alloc_barrier",
    "gpu.alloc_barriers",
    "gpu.barrier_wait",
    "gpu.barrier_arrive",
]
