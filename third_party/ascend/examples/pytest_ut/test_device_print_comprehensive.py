import os
import sys
import subprocess
import tempfile
import textwrap

import pytest
import torch
import torch_npu
import triton
import triton.language as tl


expected_prints = [
    "Offsets:",
    "Mask:",
    "Pointer offsets:",
    "Loaded x:",
    "Scalar factor:",
    "Temp result (x * 2):",
    "Final y (x * 2 + 1):",
    "Positive mask:",
    "Block ID:",
    "Block start:",
    "Valid elements in this block:",
]


def test_comprehensive_print():

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_script = f.name

        f.write(textwrap.dedent(f"""
import os
import sys
import subprocess
import tempfile
import textwrap

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

os.environ["TRITON_DEVICE_PRINT"] = "1"

@triton.jit
def comprehensive_print_kernel(
    x_ptr,
    y_ptr,
    mask_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    tl.device_print("Offsets: ", offsets)

    mask = offsets < n_elements
    tl.device_print("Mask: ", mask)

    x_ptrs = x_ptr + offsets
    tl.device_print("Pointer offsets: ", offsets)

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.device_print("Loaded x: ", x)

    scalar_factor = 2.0
    tl.device_print("Scalar factor: ", scalar_factor)

    y_temp = x * scalar_factor
    tl.device_print("Temp result (x * 2): ", y_temp)

    y = y_temp + 1.0
    tl.device_print("Final y (x * 2 + 1): ", y)

    positive_mask = y > 0.0
    tl.device_print("Positive mask: ", positive_mask)

    tl.device_print("Block ID: ", pid)
    tl.device_print("Block start: ", block_start)

    y_ptrs = y_ptr + offsets
    tl.store(y_ptrs, y, mask=mask)
    mask_count = tl.sum(mask.to(tl.int32))
    tl.device_print("Valid elements in this block: ", mask_count)


def test_comprehensive_print():
    size = 16
    x = torch.randn(size).npu()
    y = torch.zeros(size).npu()
    mask = torch.ones(size, dtype=torch.bool).npu()
    BLOCK_SIZE = 32

    h = comprehensive_print_kernel[1,](x, y, mask, size, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = x * 2.0 + 1.0
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)

    for i in range(11):
        opStr = "call @triton_print_" + str(i)
        assert opStr in h.asm["ttadapter"]

    print("passed!")


if __name__ == "__main__":
    test_comprehensive_print()
        """))

    result = subprocess.run(
        [sys.executable, temp_script],
        capture_output=True,
        text=True,
        env=os.environ.copy()
    )
    
    captured_output = result.stdout + "\n=== STDERR ===\n" + result.stderr

    assert "passed!" in captured_output
    for prefix in expected_prints:
        assert prefix in captured_output
