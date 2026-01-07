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
    'sum': ("tl.sum(x0, 1) + tl.sum(x1, 1)", "torch.sum(x0, 1) + torch.sum(x1, 1)"),
    'max': ("tl.max(x0, 1) + tl.max(x1, 1)", "( torch.max(x0.cpu(), 1)[0] + torch.max(x1.cpu(), 1)[0] ).npu()"),
    'min': ("tl.min(x0, 1) + tl.min(x1, 1)", "( torch.min(x0.cpu(), 1)[0] + torch.min(x1.cpu(), 1)[0] ).npu()"),
    'argmax': ("tl.argmax(x0, 1) + tl.argmax(x1, 1)", "torch.argmax(x0, 1) + torch.argmax(x1, 1)"),
    'argmin': ("tl.argmin(x0, 1) + tl.argmin(x1, 1)", "torch.argmin(x0, 1) + torch.argmin(x1, 1)"),
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
def triton_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    xbase = tl.arange(0, XBLOCK_SUB)
    ridx = tl.arange(0, RBLOCK)
    num_loop: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop in range(num_loop):
        xidx = offset + loop * XBLOCK_SUB + xbase
        idx = xidx[:, None] * rnumel + ridx[None, :]
        xmsk = xidx < xnumel
        rmsk = ridx < xnumel
        msk = xmsk[:, None] & rmsk[None, :]
        x0 = tl.load(in_ptr0 + idx, mask=msk)
        x1 = tl.load(in_ptr1 + idx, mask=msk)
        y0 = {func_pattern}
        tl.store(out_ptr0 + xidx, y0, mask=xmsk)
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
@pytest.mark.parametrize('xblock_sub', [5])
@pytest.mark.parametrize('rblock', [128])
@pytest.mark.parametrize('func_name', FUNCTIONS_TO_TEST.keys())
def test_reduce(dtype, xblock_sub, rblock, func_name):

    # if dtype in _int_dtypes:
    #     skip_int_dtype_ops = [
    #         'div_rn', 'fdiv',
    #         ]
    #     if func_name in skip_int_dtype_ops:
    #         pytest.skip(f"{func_name} only tested with float dtypes")
    # if dtype in _float_dtypes:
    #     skip_float_dtype_ops = [
    #         'mod', 'floordiv', 'cdiv', 'and', 'or', 'xor',
    #         ]
    #     if func_name in skip_float_dtype_ops:
    #         pytest.skip(f"{func_name} only tested with int dtypes")
    # if func_name == 'mod' and dtype == 'int64':
    #     pytest.skip(f"{func_name} skips int64")
    # if func_name in ['logical_and', 'logical_or'] and dtype != 'bool':
    #     pytest.skip(f"{func_name} tests only bool")

    xblock = triton.next_power_of_2(xblock_sub)
    shape = (xblock, rblock)
    triton_func_op, torch_func_op = FUNCTIONS_TO_TEST[func_name]

    def torch_func(x0, x1):
        return eval(torch_func_op)

    def get_autotune_config():
        return [
            triton.Config({'XBLOCK': xblock, 'XBLOCK_SUB': xblock_sub, 'RBLOCK': rblock}),
        ]

    triton_kernel = create_triton_kernel(func_name, triton_func_op)
    triton_kernel = triton.autotune(
            configs=get_autotune_config(),
            key=['xnumel'],
        )(triton_kernel)

    def triton_func(x0, x1, out_dtype):
        xnumel, rnumel = x0.shape
        y0 = torch.empty((xnumel,), dtype=x0.dtype).npu()
        grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']), )
        triton_kernel[grid](x0, x1, y0, xnumel, rnumel)
        return y0

    x0 = generate_tensor(shape, dtype).npu()
    x1 = generate_tensor(shape, dtype).npu()

    # if func_name in ['div', 'fdiv', 'div_rn', 'cdiv', 'floordiv', 'mod']:
    #     x1 = fill_zero_with_one(x1)

    out_dtype = x0.dtype
    if func_name in [
        'argmax', 'argmin'
        ]:
        out_dtype = torch.int32

    triton_cal = triton_func(x0, x1, out_dtype)
    torch_ref = torch_func(x0, x1)
    validate_cmp(out_dtype.__str__().split('.')[1], triton_cal, torch_ref)
