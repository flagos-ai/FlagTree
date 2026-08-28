import pytest
import torch

import triton
import triton.language as tl

from triton.language.extra import libdevice


@pytest.mark.parametrize("dtype_str", ["float32"])
@pytest.mark.parametrize("size", [128, 4])
@pytest.mark.parametrize(
    "libdevice_fn, torch_special_fn",
    [("tanh", "tanh"), ("pow", "pow"), ("fmod", "fmod"), ("isnan", "isnan"),
     ("isinf", "isinf"), ("finitef", "isfinite"), ("ceil", "ceil"), ("floor", "floor"), ("rint", "round"),
     ("trunc", "trunc"),  # Add the new function test
     ],
)
def test_special(dtype_str, size, libdevice_fn, torch_special_fn, device):
    SIZE = size
    dtype = getattr(torch, dtype_str)

    if torch_special_fn in ["pow", "fmod"]:
        x = torch.randn((SIZE, ), dtype=dtype, device=device)
        y = torch.randn((SIZE, ), dtype=dtype, device=device)
        y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
        if torch_special_fn == "pow":
            y_ref = torch.pow(x, y)
        elif torch_special_fn == "fmod":
            y_ref = torch.fmod(x, y)
    elif torch_special_fn in ["isnan", "isinf", "isfinite"]:  # Add isfinite to this condition
        x = torch.randn((SIZE, ), dtype=dtype, device=device)
        # Set some element as nan&-nan
        x[SIZE // 4] = float('inf')
        x[SIZE // 2] = float('-inf')
        x[3 * SIZE // 4] = float('nan')

        # Use bool as return value
        y_exp = torch.empty((SIZE, ), dtype=torch.bool, device=device)
        if torch_special_fn == "isnan":
            y_ref = torch.isnan(x)
        elif torch_special_fn == "isinf":
            y_ref = torch.isinf(x)
        else:  # isfinite
            y_ref = torch.isfinite(x)
    elif torch_special_fn in ["ceil", "floor", "trunc", "round"]:
        # For ceil, floor, and trunc, we can use the same input
        # as they are unary operations.
        x = torch.randn((SIZE, ), dtype=dtype, device=device)
        y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
        if torch_special_fn == "ceil":
            y_ref = torch.ceil(x)
        elif torch_special_fn == "floor":
            y_ref = torch.floor(x)
        elif torch_special_fn == "trunc":
            y_ref = torch.trunc(x)
        elif torch_special_fn == "round":
            y_ref = torch.round(x)
    else:
        x = torch.randn((SIZE, ), dtype=dtype, device=device)
        y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
        if torch_special_fn == "tanh":
            y_ref = torch.tanh(x)
        else:
            y_ref = getattr(torch.special, torch_special_fn)(x)

    @triton.jit
    def kernel_pow(x_ptr, y_ptr, out_ptr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        res = libdevice.pow(x, y)
        tl.store(out_ptr + off, res)

    @triton.jit
    def kernel_fmod(x_ptr, y_ptr, out_ptr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        res = libdevice.fmod(x, y)
        tl.store(out_ptr + off, res)

    @triton.jit
    def kernel_rint(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get rounded result
        res = libdevice.rint(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_unary(in_p, out_p, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        res = getattr(libdevice, fn)(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_isnan(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get bool result
        res = libdevice.isnan(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_isinf(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get bool result
        res = libdevice.isinf(x)
        tl.store(out_p + off, res)

    @triton.jit
    def kernel_finitef(in_p, out_p, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        # Get bool result
        res = libdevice.finitef(x)
        tl.store(out_p + off, res)

    if torch_special_fn == "pow":
        kernel_pow[(1, )](x, y, y_exp, SIZE=SIZE, num_warps=4, num_ctas=1)
    elif torch_special_fn == "round":
        kernel_rint[(1, )](x, y_exp, SIZE=SIZE, num_warps=4, num_ctas=1)
    elif torch_special_fn == "fmod":
        kernel_fmod[(1, )](x, y, y_exp, SIZE=SIZE, num_warps=4, num_ctas=1)
    elif torch_special_fn == "isnan":
        kernel_isnan[(1, )](x, y_exp, SIZE=SIZE, num_warps=4, num_ctas=1)
    elif torch_special_fn == "isinf":
        kernel_isinf[(1, )](x, y_exp, SIZE=SIZE, num_warps=4, num_ctas=1)
    elif torch_special_fn == "isfinite":
        kernel_finitef[(1, )](x, y_exp, SIZE=SIZE, num_warps=4, num_ctas=1)
    else:
        kernel_unary[(1, )](x, y_exp, fn=libdevice_fn, SIZE=SIZE, num_warps=4, num_ctas=1)

    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)


def test_libdevice_rename(device):

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE, device=device)
    out = torch.empty_like(inp)

    triton_copy[(1, )](inp, out, BLOCK_SIZE)
