# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def nvsmi(attrs):
    import subprocess
    import sys

    attrs = ",".join(attrs)
    cmd = ["ixsmi", "-i", "0", "--query-gpu=" + attrs, "--format=csv,noheader,nounits"]
    out = subprocess.check_output(cmd)
    return [int(value) for value in out.decode(sys.stdout.encoding).split(",")]


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    import torch
    import triton.language as tl
    from triton.runtime import driver

    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    if dtype in [torch.float32, torch.int32]:
        ops_per_sub_core = 256
    elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
        ops_per_sub_core = 512
    elif dtype in [torch.int8, tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
        ops_per_sub_core = 1024
    else:
        raise RuntimeError("dtype not supported")
    return num_subcores * clock_rate * ops_per_sub_core * 1e-9
