TLE_SUPPORTED_PRIMITIVES = [
    "extract_tile",
    "insert_tile",
    "cumsum",
    "gpu.alloc",
    "gpu.local_ptr",
    # FlagMega-on-Radeon primitives with RDNA lowerings:
    "device_mesh",
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
