import os


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
]

# The HCU compiler verifies the supported subset before lowering these opt-ins.
if os.environ.get("FLAGTREE_HCU_EXPERIMENTAL_SOFTWARE_WS") == "1":
    TLE_SUPPORTED_PRIMITIVES.append("gpu.warp_specialize")

if os.environ.get("FLAGTREE_HCU_EXPERIMENTAL_STATIC_PIPE") == "1":
    TLE_SUPPORTED_PRIMITIVES.append("pipe")
