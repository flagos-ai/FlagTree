"""
Test suite for tt.print / tl.device_print support in spine-triton.

Tests cover:
- String-only print
- Scalar print (int, float)
- Tensor print (1D, 2D)
- Multi-grid print (verify PID output)
- Hex format
- Signed/unsigned integers
"""

import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def print_string_only_kernel():
    tl.device_print("Hello from spine-triton!")


@triton.jit
def print_scalar_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tl.device_print("pid", pid)

    # Load scalar
    x = tl.load(x_ptr + pid)
    tl.device_print("x_scalar", x)


@triton.jit
def print_tensor_kernel(x_ptr, N: tl.constexpr):
    _ = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.device_print("x_tensor", x)


@triton.jit
def print_2d_tensor_kernel(x_ptr, M: tl.constexpr, N: tl.constexpr):
    _ = tl.program_id(0)
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    offs = offs_m * N + offs_n
    x = tl.load(x_ptr + offs)
    tl.device_print("x_2d", x)


@triton.jit
def print_hex_kernel(x_ptr, N: tl.constexpr):
    _ = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.device_print("x_hex", x, hex=True)


@triton.jit
def print_multi_operand_kernel(a_ptr, b_ptr, N: tl.constexpr):
    _ = tl.program_id(0)
    offs = tl.arange(0, N)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.device_print("a", a)
    tl.device_print("b", b)
    tl.device_print("sum", a + b)


def test_print_string_only():
    """Test string-only print (no operands)."""
    print("\n=== Test: String-only print ===")
    print_string_only_kernel[(1, )]()
    print("Expected output: (0, 0, 0) Hello from spine-triton!")


def test_print_scalar_int():
    """Test scalar integer print."""
    print("\n=== Test: Scalar integer print ===")
    x = torch.tensor([42, 100, 255], dtype=torch.int32)
    print_scalar_kernel[(3, )](x, BLOCK_SIZE=1)
    print("Expected: 3 lines with pid=0/1/2 and x_scalar=42/100/255")


def test_print_scalar_float():
    """Test scalar float print."""
    print("\n=== Test: Scalar float print ===")
    x = torch.tensor([3.14, 2.718, 1.414], dtype=torch.float32)
    print_scalar_kernel[(3, )](x, BLOCK_SIZE=1)
    print("Expected: 3 lines with pid=0/1/2 and x_scalar=3.14/2.718/1.414")


def test_print_tensor_1d():
    """Test 1D tensor print."""
    print("\n=== Test: 1D tensor print ===")
    x = torch.arange(8, dtype=torch.float32)
    print_tensor_kernel[(1, )](x, N=8)
    print("Expected: (0, 0, 0) x_tensor: [0, 1, 2, 3, 4, 5, 6, 7]")


def test_print_tensor_2d():
    """Test 2D tensor print."""
    print("\n=== Test: 2D tensor print ===")
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print_2d_tensor_kernel[(1, )](x, M=3, N=4)
    print("Expected: (0, 0, 0) x_2d: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]")


def test_print_hex():
    """Test hex format print."""
    print("\n=== Test: Hex format print ===")
    x = torch.tensor([0, 15, 255, 4096], dtype=torch.int32)
    print_hex_kernel[(1, )](x, N=4)
    print("Expected: (0, 0, 0) x_hex: [0x00000000, 0x0000000f, 0x000000ff, 0x00001000]")


def test_print_multi_grid():
    """Test multi-grid print (verify PID output)."""
    print("\n=== Test: Multi-grid print ===")
    x = torch.arange(16, dtype=torch.float32)
    print_tensor_kernel[(4, )](x, N=16)
    print("Expected: 4 lines with pid_x=0/1/2/3, all printing same tensor")


def test_print_multi_operand():
    """Test multiple operands in same kernel."""
    print("\n=== Test: Multiple operands ===")
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    b = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    print_multi_operand_kernel[(1, )](a, b, N=4)
    print("Expected: 3 prints - a, b, and sum")


def test_print_signed_unsigned():
    """Test signed vs unsigned integer print."""
    print("\n=== Test: Signed/unsigned integers ===")

    @triton.jit
    def print_signed_kernel(x_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(x_ptr + offs)
        tl.device_print("signed", x)

    # Negative numbers should print correctly with signed interpretation
    x = torch.tensor([-1, -100, 127, -128], dtype=torch.int32)
    print_signed_kernel[(1, )](x, N=4)
    print("Expected: signed: [-1, -100, 127, -128]")


if __name__ == "__main__":
    print("=" * 60)
    print("spine-triton tt.print / tl.device_print Test Suite")
    print("=" * 60)

    test_print_string_only()
    test_print_scalar_int()
    test_print_scalar_float()
    test_print_tensor_1d()
    test_print_tensor_2d()
    test_print_hex()
    test_print_multi_grid()
    test_print_multi_operand()
    test_print_signed_unsigned()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
