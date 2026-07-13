import json
import os

import flag_gems
import torch
import triton
from triton.runtime import debugger

try:
    import torch_npu
except ImportError:
    torch_npu = None


def sync_device():
    if torch_npu is not None and hasattr(torch_npu, "npu"):
        torch_npu.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


output_dir = os.environ.get("FLAGTREE_DEBUGGER_BATCH_OUTPUT_DIR", "/tmp/flagtree_debugger_samples/softmax")
debugger.configure(
    output_dir=output_dir,
    record_capacity=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_RECORD_CAPACITY", "4096")),
    export_raw_records=os.environ.get("FLAGTREE_DEBUGGER_BATCH_EXPORT_RAW", "0") == "1",
)
triton.enable_debug(
    level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_LEVEL", "1")),
    addr_level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_ADDR_LEVEL", "1")),
)

torch.manual_seed(0)
if torch_npu is not None and hasattr(torch_npu, "npu"):
    torch_npu.npu.manual_seed_all(0)

with flag_gems.use_gems(include=["softmax"]):
    x = torch.randn((4, 8), dtype=torch.float32, device=flag_gems.device)
    result = torch.nn.functional.softmax(x, dim=1)

sync_device()
print(
    json.dumps(
        {
            "op": "softmax",
            "case_id": "dim1",
            "shape": list(result.shape),
            "dtype": str(result.dtype),
            "device": str(result.device),
        },
        sort_keys=True,
    ))
