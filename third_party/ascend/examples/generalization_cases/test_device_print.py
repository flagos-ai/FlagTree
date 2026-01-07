import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import sys
import os
import subprocess
import tempfile
import textwrap

os.environ["TRITON_DEVICE_PRINT"] = "1"
os.environ["TRITON_ENABLE_TASKQUEUE"] = "0"

shape = (8,)
XS = 8
XVALS_INT = [0,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
            torch.iinfo(torch.int16).min,
            torch.iinfo(torch.int16).max,
            torch.iinfo(torch.int32).min,
            torch.iinfo(torch.int32).max,
            torch.iinfo(torch.int32).max+1]

@pytest.mark.parametrize('sigtype', ['int32', 'int64', 'int16', 'int8',
                                     'float32', 'float16','bfloat16'])
def test_device_print_int32(sigtype):

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_script = f.name
        
        f.write(textwrap.dedent(f"""
import torch
import torch_npu
import triton
import triton.language as tl
import os
import sys

os.environ["TRITON_DEVICE_PRINT"] = "1"
os.environ["TRITON_ENABLE_TASKQUEUE"] = "0"

@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, XBLOCK: tl.constexpr):
    idx = tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 + tmp1
    tl.device_print("OUTPUT = ", tmp2)
    tl.store(out_ptr0 + idx, tmp2)

def main():
    shape = (8,)
    XS = 8
    dtype = torch.{sigtype}
    
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    
    XVALS_INT = [0, -128, 127, -32768, 32767, -2147483648, 2147483647, 2147483648]
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    
    out = torch.empty_like(x0)
    
    triton_kernel[1,](out, x0, x1, XS)
    
    print("Kernel execution completed")
    
    return out

if __name__ == "__main__":
    result = main()
    print(f"Result shape: {{result.shape}}")
        """))
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    
    torch_ref = x0 + x1
    if 'int' in sigtype:
        torch_ref_str = ','.join([str(int(val)) for val in torch_ref.cpu().numpy()])
    else:
        values = torch_ref.cpu()
        if values.dtype == torch.bfloat16:
            values = values.float()
        torch_ref_str = ','.join([f"{float(val):.6f}" for val in values.numpy()])

    result = subprocess.run(
        [sys.executable, temp_script],
        capture_output=True,
        text=True,
        env=os.environ.copy()
    )
        
    captured_output = result.stdout + "\n=== STDERR ===\n" + result.stderr
    
    ##with open(f"manual_capture_{sigtype}.txt", "w") as f:
        ##f.write(captured_output)
        ##f.write(f"torch_ref:{torch_ref_str}")

    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    assert torch_ref_str in captured_output