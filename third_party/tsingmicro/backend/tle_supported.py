TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Lite
    "extract_tile",
    "insert_tile",
    "device_mesh",
    "sharding",
    "shard_id",
    "distributed_barrier",
    "remote",
    "cumsum",

    ## TLE-Struct DSA
    "dsa.alloc",
    "dsa.copy",
    "dsa.local_ptr",
    "dsa.memory_space",
    "dsa.to_tensor",
    "dsa.to_buffer",
    "dsa.add",
    "dsa.sub",
    "dsa.mul",
    "dsa.div",
    "dsa.max",
    "dsa.min",
    "dsa.pipeline",
    "dsa.extract_slice",
    "dsa.insert_slice",

    ## TLE-Struct DSA (tsingmicro)
    "dsa.tsingmicro.randgen",
    "dsa.tsingmicro.rand",
    "dsa.tsingmicro.randn",
]