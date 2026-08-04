import subprocess
import sys

import triton.language as tl


def nvsmi(attrs):
    from triton._internal_testing import is_corex
    attrs = ','.join(attrs)
    if is_corex():
        cmd = ['ixsmi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    else:
        cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    import torch

    from triton.runtime import driver
    from triton._internal_testing import is_corex
    if not device:
        device = torch.cuda.current_device()

    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = torch.cuda.get_device_capability(device)
    if not is_corex() and capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype in [torch.float32, torch.int32]:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            ops_per_sub_core = 512
        elif dtype in [torch.int8, tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
            ops_per_sub_core = 1024
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops
