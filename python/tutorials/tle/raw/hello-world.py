import triton
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.mlir import vprintf
import triton.experimental.tle.language.raw as tle_raw
import torch

from mlir.dialects import nvvm
from mlir import ir

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="mlir")
def edsl():
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    vprintf("Hello from bidx %d, tidx %d\n", bidx, tidx)


@triton.jit
def hello_kernel():
    tle_raw.call(edsl, [], [])
    tle_raw.call(edsl, [], [])


def hello():
    hello_kernel[(1024, )]()
    torch.cuda.synchronize()


if __name__ == "__main__":
    hello()
