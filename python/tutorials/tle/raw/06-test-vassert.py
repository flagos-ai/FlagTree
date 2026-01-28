import triton
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.mlir import vprintf, vassert
import triton.experimental.tle.language.raw as tle_raw
import torch
import sys

from mlir.dialects import nvvm, arith
from mlir import ir

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@dialect(name="mlir")
def edsl_assert_test():
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))

    c0 = arith.constant(ir.IntegerType.get_signless(32), 0)
    c1 = arith.constant(ir.IntegerType.get_signless(32), 1)
    cond_false = arith.cmpi(arith.CmpIPredicate.eq, c0, c1)

    vassert(cond_false, "TEST ASSERT: Block %d, Thread %d should fail!\n", bidx, tidx)

    vprintf("ERROR: This line should NOT be reached! bidx=%d\n", bidx)


@triton.jit
def assert_kernel():
    tle_raw.call(edsl_assert_test, [], [])


def run_test():
    print(">>> Starting Assert Test (Expect Crash)...")
    
    try:
        assert_kernel[(1, )]()
        torch.cuda.synchronize()
        
    except RuntimeError as e:
        msg = str(e)
        if "device-side assert triggered" in msg or "unspecified launch failure" in msg:
            print("\n✅ [SUCCESS] Assert triggered successfully!")
            print(f"   Captured Error: {msg}")
            return True
        else:
            print(f"\n❌ [FAIL] Caught unexpected RuntimeError: {msg}")
            return False
            
    except Exception as e:
        print(f"\n❌ [FAIL] Caught unexpected exception: {type(e)}")
        print(e)
        return False
        
    else:
        print("\n❌ [FAIL] Kernel finished without error (Assert did NOT trigger)")
        return False


if __name__ == "__main__":
    success = run_test()
    if not success:
        sys.exit(1)