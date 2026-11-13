"""P800 SDNN modulo reproduction; exit 1 on incorrect output."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton._C import libtriton


@triton.jit
def wrapped_copy(src, dst, PERIOD: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + x % PERIOD, x < LENGTH)
    tl.store(dst + x, value, x < LENGTH)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--period", type=int, choices=[7, 17, 32], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
assert mask.isdigit() and os.environ.get("XPU_VISIBLE_DEVICES") == mask, "Select one idle P800 with both masks"
assert not args.output.exists(), "Output already exists"
torch.cuda.set_device(0)
assert torch.cuda.device_count() == 1

# Padding detects an unwrapped read without exceeding the physical allocation.
host = -1000 - torch.arange(64, dtype=torch.float32)
host[:args.period] = 100 + torch.arange(args.period)
src = host.to("cuda")
dst = torch.full((64, ), -777.0, device="cuda")
kernel = wrapped_copy[(2, )](src[:args.period], dst[:19], PERIOD=args.period, LENGTH=19, BLOCK=16, is_sdnn=True)
torch.cuda.synchronize()
actual, expected = dst.cpu(), host[torch.arange(19) % args.period]
bad = (actual[:19].view(torch.int32) != expected.view(torch.int32)).nonzero().flatten().tolist()
source_ok = torch.equal(src.cpu().view(torch.int32), host.view(torch.int32))
guard_ok = bool(torch.all(actual[19:] == -777))
check = dict(mismatch_count=len(bad), mismatch_indices=bad, source_unchanged=source_ok, guard_unchanged=guard_ok,
             passed=not bad and source_ok and guard_ok, actual_values=actual[:19].tolist(),
             expected_values=expected.tolist())
result = dict(period=args.period, length=19, block=16, physical_gpu=int(mask), checks=[check],
              gate_passed=check["passed"],
              core_sha256=hashlib.sha256(Path(libtriton.__file__).read_bytes()).hexdigest(),
              xpubin_sha256=hashlib.sha256(kernel.asm["xpubin"]).hexdigest())
with args.output.open("x") as output:
    json.dump(result, output, indent=2)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["gate_passed"] else 1)
