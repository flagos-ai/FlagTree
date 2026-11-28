TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "load",
    "extract_tile",
    "insert_tile",
    "range",
    "device_mesh",
    "sharding",
    "shard_id",
    "n_pes",
    "distributed_barrier",
    "remote",
    "signal",
    "signal_wait",
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
    "gpu.buffered_tensor.load",  # CommonIR
    "gpu.buffered_tensor.store",  # CommonIR
    "gpu.buffered_tensor.slot",
    "gpu.buffered_tensor.reshape",
    "gpu.buffered_tensor.subslice",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.memory_space",
    "gpu.set_layout",
    "gpu.tensor_map_fenceproxy_acquire",
    "gpu.reinterpret_tensor_map",
    "gpu.warp_specialize",
    "gpu.alloc_barrier",
    "gpu.alloc_barriers",
    "gpu.barrier_wait",
    "gpu.barrier_arrive",
    "gpu.wgmma",
    "gpu.wgmma_wait",
]
