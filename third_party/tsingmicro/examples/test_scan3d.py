import textwrap

import numpy as np
import pytest
import torch

import triton
import triton.language as tl

import inspect
from numpy.random import RandomState
import benchmark

from triton._internal_testing import (
    int_dtypes,
    is_interpreter,
    numpy_random,
    to_triton,
    to_numpy,
)

def patch_kernel(template, to_replace):
    if is_interpreter():
        local_namespace = {}
        src = textwrap.dedent(inspect.getsource(template.fn))
        for k, v in to_replace.items():
            src = src.replace(k, v)
        exec(src, globals(), local_namespace)
        return local_namespace[template.fn.__name__]
    else:
        kernel = triton.JITFunction(template.fn)
        for key, value in to_replace.items():
            kernel._unsafe_update_src(kernel.src.replace(key, value))
        return kernel

def check_type_supported(dtype, device):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
        if cc[0] < 9 and dtype in {tl.float8e4nv, "float8e4nv", "float8_e4m3fn"}:
            pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")

# scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]
scan3d_shapes = [((2, 2, 32))]

scan_configs = [(op, type, shape, axis, reverse)
                # for type in ['int32', 'float32', 'bfloat16']
                for type in ['float32']
                # for axis in [1, 0]
                for axis in [0, 1, 2]
                for reverse in [True, False]
                # for reverse in [True]
                for shape in scan3d_shapes
                # for op in ['cumsum', 'cumprod', 'cummax', 'linear_recurrence']
                for op in ['cumsum']]
# negative_config = [('cumsum', 'float32', (32, 32), -1, False)]
negative_config = []

@pytest.mark.interpreter
@pytest.mark.parametrize("op, dtype_str, shape, axis, reverse", scan_configs + negative_config)
def test_scan3d(op, dtype_str, shape, axis, reverse, device):
    check_type_supported(dtype_str, device)

    numpy_dtype_str = 'float32' if dtype_str == 'bfloat16' else dtype_str

    # triton kernel
    @triton.jit
    def kernel(X, Y, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        range_k = tl.arange(0, BLOCK_K)
        x = tl.load(X + range_m[:, None, None] * BLOCK_N * BLOCK_K + range_n[None, :,  None] * BLOCK_K + range_k[None, None, :])
        y = tl.load(Y + range_m[:, None, None] * BLOCK_N * BLOCK_K + range_n[None, :,  None] * BLOCK_K + range_k[None, None, :])
        GENERATE_TEST_HERE
        tl.store(Z + range_m[:, None, None] * BLOCK_N * BLOCK_K + range_n[None, :,  None] * BLOCK_K + range_k[None, None, :], z)

    if op == 'cumsum' or op == 'cumprod':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'z = tl.{op}(x, axis={axis}, reverse={reverse})'})
    elif op == 'cummax':
        rg = "range_m[:, None]" if axis == 0 else "range_n[None, :]"
        rg = f"tl.broadcast_to({rg}.to(tl.int64), [BLOCK_M, BLOCK_N])"
        kernel = patch_kernel(kernel, {
            'GENERATE_TEST_HERE':
            f'_, z = tl.associative_scan((x, {rg}), axis={axis}, combine_fn={op}, reverse={reverse})'
        })
    else:
        assert 1 == 0

    # input
    rs = RandomState(17)

    # x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    x = np.arange(0, np.prod(shape), dtype=np.float32, device=device).reshape(shape)
    print(x)
    # y is just used in linear_recurrence
    y = numpy_random(shape, dtype_str=dtype_str, rs=rs)

    x_in = x
    if reverse:
        x_in = np.flip(x, axis)
    z = np.empty_like(x)
    x_tri = to_triton(x, device=device, dst_type=dtype_str)
    y_tri = to_triton(y, device=device, dst_type=dtype_str)
    if op == 'cumsum' or op == 'cumprod':
        numpy_op = {'cumsum': np.cumsum, 'cumprod': np.cumprod}[op]
        z_ref = numpy_op(x_in, axis=axis).astype(getattr(np, numpy_dtype_str))
        if reverse:
            z_ref = np.flip(z_ref, axis)

    elif op == 'cummax':
        # NumPy does not have cummax
        z = z.astype(np.int64)
        z_ref = torch.cummax(torch.from_numpy(x_in.copy()), axis=axis).indices.numpy()
        if reverse:
            z_ref = x_in.shape[axis] - np.flip(z_ref, axis) - 1


    else:
        assert 1 == 0


    # triton result
    # we don't cast the `fp32 = bf16 op bf16` result to bfloat16 to alleviate accuracy issues
    z_tri = to_triton(z, device=device)
    kernel[(1, )](x_tri, y_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], BLOCK_K=shape[2], AXIS=axis)

    z_tri = to_numpy(z_tri)

    print(
        f"The maximum difference between torch and triton is "
        f"{np.max(np.abs(z_ref - z_tri))}"
    )
    # compare
    if dtype_str not in int_dtypes:
        if op == 'cumprod':
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01, atol=1e-3)
        else:
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        np.testing.assert_equal(z_ref, z_tri)

if __name__ == "__main__":
    benchmark.select_cpu_backend()
    test_scan3d('cumsum', 'float32', (2, 2, 32), 2, False, 'cpu')
    test_scan3d('cumsum', 'float32', (2, 2, 32), 0, False, 'cpu')
    test_scan3d('cumsum', 'float32', (2, 2, 32), 1, False, 'cpu')
    # test_scan2d('cumprod', 'float32', (8, 32), 1, False, 'cpu')
    # test_scan2d('get_first_element', 'float32', (8, 32), 1, False, 'cpu')
    # test_scan2d('linear_recurrence', 'float32', (8, 32), 1, False, 'cpu')
    # test_scan2d('cummax', 'float32', (8, 32), 1, False, 'cpu')
    # test_scan2d('roll', 'float32', (8, 32), 1, False, 'cpu')