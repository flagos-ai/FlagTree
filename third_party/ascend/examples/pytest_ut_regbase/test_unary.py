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
    'abs': ("tl.abs(x0)", "torch.abs(x0)"),
    'exp': ("tl.exp(x0)", "torch.exp(x0)"),
    'exp2': ("tl.exp2(x0)", "torch.exp2(x0)"),
    'log': ("tl.log(x0)", "torch.log(x0)"),
    'log2': ("tl.log2(x0)", "torch.log2(x0)"),
    'sin': ("tl.sin(x0)", "torch.sin(x0)"),
    'cos': ("tl.cos(x0)", "torch.cos(x0)"),
    'sqrt': ("tl.sqrt(x0)", "torch.sqrt(x0)"),
    'rsqrt': ("tl.rsqrt(x0)", "torch.rsqrt(x0)"),
    'sigmoid': ("tl.sigmoid(x0)", "torch.sigmoid(x0)"),
    'sqrt_rn': ("tl.sqrt_rn(x0)", "torch.sqrt(x0)"),
    'erf': ("tl.erf(x0)", "torch.erf(x0)"),
    'neg': ("-x0", "-x0"),
    'not': ("not(x0)", "torch.bitwise_not(x0)"),
    'invert': ("~x0", "( ~( x0.to(torch.int32) ) ).to(x0.dtype) if x0.dtype in [torch.float32, torch.float16] else ~x0"),
    'ceil': ("tl.ceil(x0)", "torch.ceil(x0)"),
    'floor': ("tl.floor(x0)", "torch.floor(x0)"),
    'lshift': ("x0 << 2", "x0 << 2"),
    'rshift': ("x0 >> 2", "x0 >> 2"),
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
def triton_kernel(in_ptr0, out_ptr0, numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    num_loop: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop in range(num_loop):
        idx = offset + loop * XBLOCK_SUB + base
        msk = idx < numel
        x0 = tl.load(in_ptr0 + idx, mask=msk)
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
def test_unary(dtype, xblock_sub, func_name):

    if dtype in _int_dtypes:
        skip_int_dtype_ops = [
            'exp', 'exp2', 'log', 'log2', 'sin', 'cos', 'sqrt', 'rsqrt', 'sqrt_rn',
            'sigmoid', 'erf', 'ceil', 'floor',
            ]
        if func_name in skip_int_dtype_ops:
            pytest.skip(f"{func_name} only tested with float dtypes")
    if dtype in _float_dtypes:
        skip_float_dtype_ops = [
            'not', 'invert', 'lshift', 'rshift',
            ]
        if func_name in skip_float_dtype_ops:
            pytest.skip(f"{func_name} only tested with int dtypes")

    xblock = triton.next_power_of_2(xblock_sub)
    shape = (xblock,)
    triton_func_op, torch_func_op = FUNCTIONS_TO_TEST[func_name]

    def torch_func(x0):
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

    def triton_func(x0):
        y0 = torch.empty_like(x0)
        numel = x0.numel()
        grid = lambda meta: (triton.cdiv(numel, meta['XBLOCK']), )
        triton_kernel[grid](x0, y0, numel)
        return y0

    x0 = generate_tensor(shape, dtype).npu()

    if func_name in ['sqrt', 'rsqrt', 'sqrt_rn', 'log', 'log2']:
        x0 = fill_negative_with_one(x0)

    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0)
    validate_cmp(dtype, triton_cal, torch_ref)
