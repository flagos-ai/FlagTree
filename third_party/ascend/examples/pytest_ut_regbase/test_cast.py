import triton
import triton.language as tl
import torch
import pytest
import atexit
from test_common import (
    generate_tensor,
    validate_cmp,
    fill_negative_with_one,
    _float_dtypes_without_bf16 as _float_dtypes,
    _int_dtypes,
    _shape_1d,
)

#################################################
FUNCTIONS_TO_TEST = {
    'cast': ("x0.to(dst_dtype) + x1", "x0.to(dst_dtype) + x1"),
}
#################################################

# Global dictionary to keep track of temporary files
_temp_kernel_files = {}

def _cleanup_temp_files():
    import os
    for file_path in _temp_kernel_files.values():
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass

atexit.register(_cleanup_temp_files)

def create_triton_kernel(func_name, func_pattern):
    import tempfile
    import os
    kernel_source = f"""
import triton
import triton.language as tl

@triton.jit
def triton_kernel(in_ptr0, in_ptr1, out_ptr0, numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, dst_dtype: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    num_loop: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop in range(num_loop):
        idx = offset + loop * XBLOCK_SUB + base
        msk = idx < numel
        x0 = tl.load(in_ptr0 + idx, mask=msk)
        x1 = tl.load(in_ptr1 + idx, mask=msk)
        y0 = {func_pattern}
        tl.store(out_ptr0 + idx, y0, mask=msk)
"""

    # Create a temporary file with a unique name based on the function name
    if func_name in _temp_kernel_files:
        temp_file_path = _temp_kernel_files[func_name]
    else:
        fd, temp_file_path = tempfile.mkstemp(suffix='.py', prefix=f'triton_kernel_{func_name}_')
        os.close(fd)  # We don't need the file descriptor
        _temp_kernel_files[func_name] = temp_file_path

        # Write the kernel source to the file
        with open(temp_file_path, 'w') as f:
            f.write(kernel_source)

    # Import the kernel from the temporary file
    import importlib.util
    module_name = f"triton_kernel_{func_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.triton_kernel


@pytest.mark.parametrize('src_dtype', ['float32'])
@pytest.mark.parametrize('dst_dtype', _float_dtypes + _int_dtypes)
@pytest.mark.parametrize('xblock_sub', _shape_1d)
@pytest.mark.parametrize('func_name', FUNCTIONS_TO_TEST.keys())
def test_cast(src_dtype, dst_dtype, xblock_sub, func_name):

    if (src_dtype == dst_dtype):
        return

    # if dtype in _int_dtypes:
    #     skip_int_dtype_ops = [
    #         'exp', 'exp2', 'log', 'log2', 'sin', 'cos', 'sqrt', 'rsqrt', 'sqrt_rn',
    #         'sigmoid', 'erf', 'ceil', 'floor',
    #         ]
    #     if func_name in skip_int_dtype_ops:
    #         pytest.skip(f"{func_name} only tested with float dtypes")
    # if dtype in _float_dtypes:
    #     skip_float_dtype_ops = [
    #         'not', 'invert', 'lshift', 'rshift',
    #         ]
    #     if func_name in skip_float_dtype_ops:
    #         pytest.skip(f"{func_name} only tested with int dtypes")

    xblock = triton.next_power_of_2(xblock_sub)
    shape = (xblock,)
    triton_func_op, torch_func_op = FUNCTIONS_TO_TEST[func_name]

    def torch_func(x0, dst_dtype):
        x1 = torch.ones((x0.numel(),), dtype=dst_dtype, device=x0.device)
        return eval(torch_func_op)

    def get_autotune_config():
        return [
            triton.Config({'XBLOCK': xblock, 'XBLOCK_SUB': xblock_sub, 'dst_dtype': eval(f"tl.{dst_dtype}")}),
        ]

    triton_kernel = create_triton_kernel(func_name, triton_func_op)
    triton_kernel = triton.autotune(
            configs=get_autotune_config(),
            key=['numel'],
        )(triton_kernel)

    def triton_func(x0, dst_dtype):
        numel = x0.numel()
        y0 = torch.empty((numel,), dtype=dst_dtype, device=x0.device)
        y1 = torch.ones((numel,), dtype=dst_dtype, device=x0.device)
        grid = lambda meta: (triton.cdiv(numel, meta['XBLOCK']), )
        triton_kernel[grid](x0, y1, y0, numel)
        return y0

    x0 = generate_tensor(shape, src_dtype).npu()

    # if func_name in ['sqrt', 'rsqrt', 'sqrt_rn', 'log', 'log2']:
    #     x0 = fill_negative_with_one(x0)

    torch_ref = torch_func(x0, eval(f"torch.{dst_dtype}"))
    triton_cal = triton_func(x0, eval(f"torch.{dst_dtype}"))
    validate_cmp(dst_dtype, triton_cal, torch_ref)
