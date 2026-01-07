import triton
import triton.language as tl
import torch
import pytest
import atexit
from test_common import (
    generate_tensor,
    validate_cmp,
    fill_zero_with_one,
    _float_dtypes_without_bf16 as _float_dtypes,
    _int_dtypes,
    _shape_1d,
)

#################################################
FUNCTIONS_TO_TEST = {
    'add': ("x0 + x1", "x0 + x1"),
    'sub': ("x0 - x1", "x0 - x1"),
    'mul': ("x0 * x1", "x0 * x1"),
    'div': ("x0 / x1", "x0 / x1 if x0.dtype in [torch.float32, torch.float16] else (x0.to(torch.float32) / x1.to(torch.float32)).to(x0.dtype)"),
    'floordiv': ("x0 // x1", "x0 // x1"),
    'mod': ("x0 % x1", "(x0.cpu() % x1.cpu()).npu() if x0.dtype in [torch.int8, torch.int16] else x0 % x1"),
    'and': ("x0 & x1", "x0 & x1"),
    'or': ("x0 | x1", "x0 | x1"),
    'xor': ("x0 ^ x1", "x0 ^ x1"),
    'gt': ("x0 > x1", "x0 > x1"),
    'ge': ("x0 >= x1", "x0 >= x1"),
    'lt': ("x0 < x1", "x0 < x1"),
    'le': ("x0 <= x1", "x0 <= x1"),
    'eq': ("x0 == x1", "x0 == x1"),
    'ne': ("x0 != x1", "x0 != x1"),
    'cdiv': ("tl.cdiv(x0, x1)", "( (x0.cpu() + x1.cpu() - 1) // x1.cpu() ).npu()"),
    'fdiv': ("tl.fdiv(x0, x1)", "x0 / x1"),
    'div_rn': ("tl.div_rn(x0, x1)", "x0 / x1"),
    'logical_and': ("tl.logical_and(x0, x1)", "torch.logical_and(x0, x1)"),
    'logical_or': ("tl.logical_or(x0, x1)", "torch.logical_or(x0, x1)"),
    'maximum': ("tl.maximum(x0, x1)", "torch.maximum(x0, x1)"),
    'minimum': ("tl.minimum(x0, x1)", "torch.minimum(x0, x1)"),
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
def triton_kernel(in_ptr0, in_ptr1, out_ptr0, numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
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


@pytest.mark.parametrize('dtype', _float_dtypes + _int_dtypes)
@pytest.mark.parametrize('xblock_sub', _shape_1d)
@pytest.mark.parametrize('func_name', FUNCTIONS_TO_TEST.keys())
def test_binary(dtype, xblock_sub, func_name):

    if dtype in _int_dtypes:
        skip_int_dtype_ops = [
            'div_rn', 'fdiv',
            ]
        if func_name in skip_int_dtype_ops:
            pytest.skip(f"{func_name} only tested with float dtypes")
    if dtype in _float_dtypes:
        skip_float_dtype_ops = [
            'mod', 'floordiv', 'cdiv', 'and', 'or', 'xor',
            ]
        if func_name in skip_float_dtype_ops:
            pytest.skip(f"{func_name} only tested with int dtypes")
    if func_name == 'mod' and dtype == 'int64':
        pytest.skip(f"{func_name} skips int64")
    if func_name in ['logical_and', 'logical_or'] and dtype != 'bool':
        pytest.skip(f"{func_name} tests only bool")

    xblock = triton.next_power_of_2(xblock_sub)
    shape = (xblock,)
    triton_func_op, torch_func_op = FUNCTIONS_TO_TEST[func_name]

    def torch_func(x0, x1):
        return eval(torch_func_op)

    def get_autotune_config():
        return [
            triton.Config({'XBLOCK': xblock, 'XBLOCK_SUB': xblock_sub}),
        ]

    triton_kernel = create_triton_kernel(func_name, triton_func_op)
    triton_kernel = triton.autotune(
            configs=get_autotune_config(),
            key=['numel'],
        )(triton_kernel)

    def triton_func(x0, x1, out_dtype):
        y0 = torch.empty_like(x0).to(out_dtype)
        numel = x0.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['XBLOCK']), )
        triton_kernel[grid](x0, x1, y0, numel)
        return y0

    x0 = generate_tensor(shape, dtype).npu()
    x1 = generate_tensor(shape, dtype).npu()

    if func_name in ['div', 'fdiv', 'div_rn', 'cdiv', 'floordiv', 'mod']:
        x1 = fill_zero_with_one(x1)
    out_dtype = x0.dtype
    if func_name in [
        'gt', 'gt', 'lt', 'le', 'eq', 'ne'
        ]:
        out_dtype = torch.bool

    triton_cal = triton_func(x0, x1, out_dtype)
    torch_ref = torch_func(x0, x1)
    validate_cmp(dtype, triton_cal, torch_ref)
