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

    ## TLE-Struct DSA (thrive): inter-die RMA and synchronization
    "dsa.thrive.putmem",
    "dsa.thrive.getmem",
    "dsa.thrive.wait",
    "dsa.thrive.notify",
    "dsa.thrive.fence",
    "dsa.thrive.sync",
]
