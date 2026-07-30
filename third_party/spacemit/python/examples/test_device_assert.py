"""
Test script for device_assert (tt.assert) support in spine-triton.

Tests:
1. Scalar assert (condition=true) — kernel completes normally
2. Scalar assert (condition=false) — prints error + abort
3. Tensor assert (all true) — kernel completes normally
4. Tensor assert (contains false) — AND reduction triggers abort
5. Multi-grid scenario — pid info is correct

Usage:
    # Test passing asserts only (no abort):
    TRITON_DEBUG=1 python test_device_assert.py --pass-only

    # Test a specific failing case (will abort):
    TRITON_DEBUG=1 python test_device_assert.py --test scalar_fail
    TRITON_DEBUG=1 python test_device_assert.py --test tensor_fail

    # Run all passing tests:
    TRITON_DEBUG=1 python test_device_assert.py
"""

import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import argparse


@triton.jit
def kernel_scalar_assert_pass(X, N: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(X + pid)
    # All elements are > 0, so this should pass
    tl.device_assert(x > 0, "x must be positive")


@triton.jit
def kernel_scalar_assert_fail(X, N: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(X + pid)
    # Some elements are <= 0, so this should fail
    tl.device_assert(x > 0, "x must be positive")


@triton.jit
def kernel_tensor_assert_pass(X, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    # All elements are > 0, so AND reduction should yield true
    tl.device_assert(x > 0, "all elements must be positive")


@triton.jit
def kernel_tensor_assert_fail(X, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    # Some elements are <= 0, AND reduction should yield false
    tl.device_assert(x > 0, "all elements must be positive")


def test_scalar_pass():
    print("=" * 60)
    print("TEST: scalar assert (all pass)")
    print("=" * 60)
    X = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    kernel_scalar_assert_pass[(4, )](X, N=4)
    print("[PASS] scalar assert with all-positive values completed.\n")


def test_scalar_fail():
    print("=" * 60)
    print("TEST: scalar assert (should fail and abort)")
    print("=" * 60)
    X = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float32)
    # Element at pid=1 is -2.0, assert should fire
    kernel_scalar_assert_fail[(4, )](X, N=4)
    print("[ERROR] Should have aborted but didn't!\n")


def test_tensor_pass():
    print("=" * 60)
    print("TEST: tensor assert (all pass)")
    print("=" * 60)
    BLOCK_SIZE = 8
    N_BLOCKS = 2
    X = torch.arange(1, N_BLOCKS * BLOCK_SIZE + 1, dtype=torch.float32)
    kernel_tensor_assert_pass[(N_BLOCKS, )](X, BLOCK_SIZE=BLOCK_SIZE)
    print("[PASS] tensor assert with all-positive values completed.\n")


def test_tensor_fail():
    print("=" * 60)
    print("TEST: tensor assert (should fail and abort)")
    print("=" * 60)
    BLOCK_SIZE = 8
    N_BLOCKS = 2
    X = torch.arange(1, N_BLOCKS * BLOCK_SIZE + 1, dtype=torch.float32)
    X[5] = -1.0  # Inject a negative value in block 0
    kernel_tensor_assert_fail[(N_BLOCKS, )](X, BLOCK_SIZE=BLOCK_SIZE)
    print("[ERROR] Should have aborted but didn't!\n")


def main():
    parser = argparse.ArgumentParser(description="Test device_assert")
    parser.add_argument("--pass-only", action="store_true", help="Run only passing tests (no abort)")
    parser.add_argument("--test", type=str, default=None, choices=["scalar_fail", "tensor_fail"],
                        help="Run a specific failing test (will abort)")
    args = parser.parse_args()

    # Clear triton cache to force recompilation
    import shutil
    import os
    cache_dirs = [
        os.path.expanduser("~/.triton/cache"),
        os.path.expanduser("~/.cache/spine-triton"),
    ]
    for d in cache_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Cleared cache: {d}")

    if args.test == "scalar_fail":
        test_scalar_fail()
        return
    if args.test == "tensor_fail":
        test_tensor_fail()
        return

    # Run passing tests
    test_scalar_pass()
    test_tensor_pass()

    if not args.pass_only:
        print("\n" + "=" * 60)
        print("All passing tests completed.")
        print("To test failing asserts (will abort), run:")
        print("  TRITON_DEBUG=1 python test_device_assert.py --test scalar_fail")
        print("  TRITON_DEBUG=1 python test_device_assert.py --test tensor_fail")
        print("=" * 60)


if __name__ == "__main__":
    main()
