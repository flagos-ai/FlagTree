from triton._common_ir import ENABLED as COMMON_IR_ENABLED

TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Lite
    "extract_tile",
    "insert_tile",

    ## TLE-Struct GPU
    "gpu.alloc",
    "gpu.copy",
    "gpu.metax.async_copy_global_to_local",
    "gpu.metax.gvm_arrive",
    "gpu.metax.barrier",
    "gpu.metax.barrier_shared",
    "gpu.metax.sched_bound",
    "gpu.metax.iglp",
    "gpu.buffered_tensor.slot",
    "gpu.local_ptr",
    "gpu.buffered_tensor.load",
    "gpu.buffered_tensor.store",
]

if COMMON_IR_ENABLED:
    TLE_SUPPORTED_PRIMITIVES.extend(["gpu.set_layout", "gpu.buffered_tensor.subslice"])
