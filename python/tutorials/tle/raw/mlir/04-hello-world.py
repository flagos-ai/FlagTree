import os
import re
import subprocess
import sys

import torch
import triton
import triton.experimental.tle.language.raw as tle_raw
from mlir import ir
from mlir.dialects import nvvm
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.mlir import vprintf

HELLO_RE = re.compile(r"Hello from bidx \d+, tidx \d+")


@dialect(name="mlir")
def edsl():
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    vprintf("Hello from bidx %d, tidx %d\n", bidx, tidx)


@triton.jit
def hello_kernel():
    tle_raw.call(edsl, [])


def hello():
    hello_kernel[(1024, )]()
    torch.cuda.synchronize()


def _self_check():
    env = os.environ.copy()
    env["TLE_HELLO_WORLD_CHILD"] = "1"
    result = subprocess.run([sys.executable, os.path.abspath(__file__)], capture_output=True, text=True, env=env)

    sys.stderr.write(result.stderr)
    if result.returncode or not HELLO_RE.findall(result.stdout):
        print("❌ self-check failed")
        return 1

    print("✅ Hello world test passed!")
    return 0


def main():
    if os.environ.get("TLE_HELLO_WORLD_CHILD") == "1":
        hello()
        return 0

    return _self_check()


if __name__ == "__main__":
    sys.exit(main())
