import triton
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.mlir import vprintf, vassert
import triton.experimental.tle.language.raw as tle_raw
import torch

from mlir.dialects import nvvm, arith
from mlir import ir

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="mlir")
def edsl():
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    # A. 创建比较常数 (32)
    limit = arith.constant(ir.IntegerType.get_signless(32), 32)
    # B. 生成布尔条件 (i1 类型)
    # cmpi 谓词: slt (signed less than), eq (equal), sgt (signed greater than) 等
    is_valid_tidx = arith.cmpi(arith.CmpIPredicate.slt, tidx, limit)
    # ---------------------------------------------------------
    # 3. 调用 Assert
    # 语义：如果 is_valid_tidx 为 True，程序继续执行；
    #       如果 is_valid_tidx 为 False，打印信息并终止 Kernel。
    # ---------------------------------------------------------
    vassert(is_valid_tidx, "Assertion Failed: tidx %d is too large (bidx=%d)\n", tidx, bidx)
    vprintf("Hello from bidx %d, tidx %d\n", bidx, tidx)


@triton.jit
def hello_kernel():
    tle_raw.call(edsl, [], [])


def hello():
    hello_kernel[(1024, )]()
    torch.cuda.synchronize()


if __name__ == "__main__":
    hello()
