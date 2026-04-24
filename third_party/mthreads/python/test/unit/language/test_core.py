# flake8: noqa: F821,F841
import contextlib
import itertools
import re
from typing import Optional
import math
import textwrap
import tempfile

import numpy as np
import pytest
import torch
import inspect
from numpy.random import RandomState

import triton
import triton.language as tl
from triton.runtime.jit import TensorWrapper
from triton.language.extra import libdevice

from triton._internal_testing import (
    integral_dtypes,
    int_dtypes,
    uint_dtypes,
    float_dtypes,
    dtypes,
    dtypes_with_bfloat16,
    is_cuda,
    is_interpreter,
    is_hip,
    is_musa,
    get_arch,
    torch_float8_dtypes,
    torch_dtypes,
    numpy_random,
    to_triton,
    torch_dtype_name,
    to_numpy,
)


@contextlib.contextmanager
def promotion_numpy_2_0():
    state = np._get_promotion_state()
    np._set_promotion_state("weak")
    try:
        yield
    finally:
        np._set_promotion_state(state)


# TODO: enable multiple cta cluster testing.
# num_ctas_list = [1, 4] if torch.cuda.get_device_capability()[0] == 9 else [1]
num_ctas_list = [1]

GPU_DIALECT = "triton_gpu"
if is_interpreter():
    THREADS_PER_WARP = 1
elif is_hip():
    THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
else:
    THREADS_PER_WARP = 32


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


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
            kernel.src = kernel.src.replace(key, value)
        return kernel


def check_cuda_or_hip(device):
    # CUDA and HIP both use pytorch device 'cuda'.  Other backends like Intel
    # GPU do not.
    if device not in ['cuda']:
        pytest.skip("Only for cuda")


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
    if is_musa() and dtype in {tl.float8e4b15, "float8e4b15"}:
        pytest.skip("float8e4b15 is not supported on MUSA")
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")


class MfmaLayout:

    def __init__(self, version, warps_per_cta, instr_shape, is_transposed):
        self.version = version
        self.warps_per_cta = warps_per_cta
        self.instr_shape = instr_shape
        self.is_transposed = is_transposed

    def __str__(self):
        return f"#{GPU_DIALECT}.amd_mfma<{{versionMajor={self.version[0]}, versionMinor={self.version[1]}, warpsPerCTA = {self.warps_per_cta}, instrShape={self.instr_shape}, isTransposed = {str(self.is_transposed).lower()}}}>"


class WmmaLayout:

    def __init__(self, version, warps_per_cta):
        self.version = version
        self.warps_per_cta = warps_per_cta

    def __str__(self):
        return f"#{GPU_DIALECT}.amd_wmma<{{version = {self.version}, warpsPerCTA = {self.warps_per_cta}}}>"


class MmaLayout:

    def __init__(self, version, warps_per_cta, ctas_per_cga, cta_split_num, cta_order, instr_shape):
        self.version = version
        self.warps_per_cta = warps_per_cta
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order
        self.instr_shape = instr_shape

    def __str__(self):
        return f"#{GPU_DIALECT}.nvidia_mma<{{versionMajor={self.version[0]}, versionMinor={self.version[1]}, warpsPerCTA={self.warps_per_cta}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}, instrShape={self.instr_shape}}}>"


class BlockedLayout:

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order, ctas_per_cga, cta_split_num, cta_order):
        self.sz_per_thread = size_per_thread
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class SharedLayout:

    def __init__(self, vec, per_phase, max_phase, order, ctas_per_cga, cta_split_num, cta_order):
        self.vec = vec
        self.per_phase = per_phase
        self.max_phase = max_phase
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#{GPU_DIALECT}.shared<{{vec={self.vec}, perPhase={self.per_phase}, maxPhase={self.max_phase}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


def is_layout_applicable(layout) -> bool:
    common_layouts = [BlockedLayout, SharedLayout]
    if layout in common_layouts:
        return True
    elif is_cuda():
        return isinstance(layout, MmaLayout)
    elif is_hip():
        target_arch = triton.runtime.driver.active.get_current_target().arch
        if "gfx11" in target_arch:
            # RDNA 3
            return isinstance(layout, WmmaLayout)
        elif any(arch for arch in ["gfx8", "gfx9"] if arch in target_arch):
            # CDNA 1, 2, 3
            return isinstance(layout, MfmaLayout)
        else:
            return False
    else:
        return True


def filter_layouts(layouts):
    return [l for l in layouts if is_layout_applicable(l)]


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x", list(dtypes) + ["bfloat16"])
def test_empty_kernel(dtype_x, device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass

    check_type_supported(dtype_x, device)
    x = to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device, dst_type=dtype_x)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)


# generic test functions
def _test_unary(dtype_x, expr, numpy_expr=None, device='cuda', num_ctas=1):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    if 'log' in expr:
        x = np.abs(x) + 0.01
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(x), device=device, dst_type=dtype_x)
    kernel[(1, )](Z=z_tri, X=x_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
    # compare
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


def _binary_op_dtype_override(a: str, b: str) -> Optional[np.dtype]:
    """
    Given two dtype strings, returns the numpy dtype Triton thinks binary
    operations on the two types should return. Returns None if the return value
    matches numpy. This is generally needed because Triton and pytorch return
    narrower floating point types than numpy in mixed operations, and because
    Triton follows C/C++ semantics around mixed signed/unsigned operations, and
    numpy/pytorch do not.
    """
    overrides = {
        ('float16', 'int16'): np.float16,
        ('float16', 'int32'): np.float16,
        ('float16', 'int64'): np.float16,
        ('float16', 'uint16'): np.float16,
        ('float16', 'uint32'): np.float16,
        ('float16', 'uint64'): np.float16,
        ('int8', 'uint8'): np.uint8,
        ('int8', 'uint16'): np.uint16,
        ('int8', 'uint32'): np.uint32,
        ('int8', 'uint64'): np.uint64,
        ('int16', 'uint16'): np.uint16,
        ('int16', 'uint32'): np.uint32,
        ('int16', 'uint64'): np.uint64,
        ('int32', 'uint32'): np.uint32,
        ('int32', 'uint64'): np.uint64,
        ('int64', 'uint64'): np.uint64,
    }
    key = (a, b) if a < b else (b, a)
    return overrides.get(key)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, mode_x='real', mode_y='real', device='cuda', num_ctas=1,
                 y_low=None, y_high=None, filter_y=None, test_broadcast=True, test_scalar=True):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    check_type_supported(dtype_y, device)
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_lhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_broadcast_rhs(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    @triton.jit
    def kernel_scalar_rhs(Z, X, y: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    replacements = {'GENERATE_TEST_HERE': expr}
    kernel = patch_kernel(kernel, replacements)
    kernel_broadcast_lhs = patch_kernel(kernel_broadcast_lhs, replacements)
    kernel_broadcast_rhs = patch_kernel(kernel_broadcast_rhs, replacements)
    kernel_scalar_rhs = patch_kernel(kernel_scalar_rhs, replacements)

    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs, low=y_low, high=y_high)
    if filter_y:
        y[filter_y(y)] = 1
    if mode_x == 'nan':
        x[:] = float('nan')
    if mode_y == 'nan':
        y[:] = float('nan')

    def do_test(x, y, kernel_fn):
        x_is_scalar = isinstance(x, (bool, int, float))
        y_is_scalar = isinstance(y, (bool, int, float))
        scalar_test = x_is_scalar or y_is_scalar

        # For scalars, we follow the NumPy 2.0 (and JAX/PyTorch pretty much) casting rules.
        if scalar_test:
            # We remove any explicit casting
            pattern = r'\.astype\(np\.\w+\)'
            scalar_expr = expr if numpy_expr is None else re.sub(pattern, '', numpy_expr)
            with promotion_numpy_2_0():
                z_ref = eval(scalar_expr)
        else:
            z_ref = eval(expr if numpy_expr is None else numpy_expr)

        dtype_z = _binary_op_dtype_override(dtype_x, dtype_y)
        if not scalar_test and dtype_z is not None:
            z_ref = z_ref.astype(dtype_z)

        # triton result
        x_tri = x if x_is_scalar else to_triton(x, device=device, dst_type=dtype_x)
        y_tri = y if y_is_scalar else to_triton(y, device=device, dst_type=dtype_y)
        z_tri = to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
        kernel_fn[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
        err_msg = f"{expr}, {kernel_fn.__name__}"
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), err_msg=err_msg, atol=3e-3, rtol=0.01)

    def get_scalar(x, dtype, low, high, filter):
        # If dtype is int, don't choose a huge number for the scalar
        # as it'll overflow easily when converted to the other dtype
        if dtype in integral_dtypes:
            # Choose in range [-7, 7] ([0, 7] for uints)
            low_x = 0 if dtype in uint_dtypes else -7
            if low is not None:
                low_x = max(low_x, low)
            high_x = 7
            if high is not None:
                high_x = min(high_x, high)
            scalar = numpy_random((), dtype_str=dtype, rs=rs, low=low_x, high=high_x).item()
            if filter and filter(scalar):
                #  https://xkcd.com/221/
                scalar = 4
        else:
            scalar = x.flat[0].item()
        return scalar

    do_test(x, y, kernel)
    if mode_y != 'nan' and test_scalar:
        if dtype_x in uint_dtypes:
            low = 0 if y_low is None else max(y_low, 0)
        else:
            low = y_low
        y_scalar = get_scalar(y, dtype_y, low, y_high, filter_y)
        do_test(x, y_scalar, kernel_scalar_rhs)
    if test_broadcast:
        do_test(x[:1].reshape(()), y, kernel_broadcast_lhs)
        do_test(x, y[:1].reshape(()), kernel_broadcast_rhs)


def _mod_operation_ill_conditioned(dtype_x, dtype_y) -> bool:
    # FIXME For large x, we are casting x to a floating point where it does not fit
    #       For small y, we are computing floor(div(float(x), y)) which may not fit
    return (dtype_x, dtype_y) in [
        ('int32', 'bfloat16'),
        ('int32', 'float16'),
        ('int32', 'float32'),
        ('int64', 'bfloat16'),
        ('int64', 'float16'),
        ('int64', 'float32'),
        ('int64', 'float64'),
        ('uint16', 'bfloat16'),
        ('uint16', 'float16'),
        ('uint16', 'float32'),
        ('uint32', 'bfloat16'),
        ('uint32', 'float16'),
        ('uint32', 'float32'),
        ('uint64', 'bfloat16'),
        ('uint64', 'float16'),
        ('uint64', 'float32'),
        ('uint64', 'float64'),
    ]


def test_dtype_codegen():
    for dtype in dtypes_with_bfloat16:
        full_name = f"triton.language.{dtype}"
        assert repr(eval(full_name)) == full_name


# ---------------
# test binary ops
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op)
    for op in ['+', '-', '*', '/', '%']
    for dtype_x in dtypes_with_bfloat16
    for dtype_y in dtypes_with_bfloat16
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bin_op(dtype_x, dtype_y, op, num_ctas, device):
    if dtype_x.startswith("uint") and dtype_y.startswith("uint") and op in ["/", "%"]:
        pytest.skip("skip unsupport dtype")
    expr = f'x {op} y'
    if op == '%' and dtype_x in int_dtypes + uint_dtypes and dtype_y in int_dtypes + uint_dtypes:
        # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
        numpy_expr = 'np.fmod(x, y)'
    elif op in ('/', '%') and dtype_x in ('int16', 'float16', 'bfloat16') and dtype_y in ('int16', 'float16',
                                                                                          'bfloat16'):
        # Triton promotes 16-bit floating-point / and % to 32-bit because there
        # are no native div or FRem operations on float16. Since we have to
        # convert anyway, we may as well take the accuracy bump.
        numpy_expr = f'x.astype(np.float32) {op} y.astype(np.float32)'
    elif (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if op == '%' and _mod_operation_ill_conditioned(dtype_x, dtype_y):
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)
    elif (op in ('%', '/') and ((dtype_x in int_dtypes and dtype_y in uint_dtypes) or
                                (dtype_x in uint_dtypes and dtype_y in int_dtypes))):
        with pytest.raises(triton.TritonError, match='Cannot use .* because they have different signedness'):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)
    else:
        # skip when bfloat16, as NumPy's ref performs the computation in float32
        # while Triton performs it in bfloat16
        # We also skip mod when it is ill-conditioned
        skip_scalar_test = ((dtype_x == "bfloat16" and "float" in dtype_y)
                            or (expr == "x % y" and dtype_x in int_dtypes + uint_dtypes and dtype_y in float_dtypes
                                and _mod_operation_ill_conditioned(dtype_x, "float32")))
        # can't divide by zero
        not_zero = op in ('/', '%') and dtype_x in integral_dtypes and dtype_y in integral_dtypes
        # can't represent -int(max)
        not_minus_one = op in ('*', '/') and dtype_x in int_dtypes and dtype_y in int_dtypes
        if not_zero or not_minus_one:
            filter_y = lambda y: not_zero * (y == 0) | not_minus_one * (y == -1)
        else:
            filter_y = None
        _test_binary(
            dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas,
            # fails with values where fmod(x, y) is roughly zero, but happens to
            # pass with the random values chosen for non-broadcast tests
            test_broadcast=(op != "%"), filter_y=filter_y, test_scalar=not skip_scalar_test)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype, order", [(dtype, order) for dtype in dtypes_with_bfloat16 for order in [0, 1]])
def test_addptr(dtype, order, device):
    check_type_supported(dtype, device)

    @triton.jit
    def kernel(x, y, ORDER: tl.constexpr, SIZE: tl.constexpr):
        offs = tl.arange(0, SIZE)
        if ORDER == 0:
            tl.store(y + offs, tl.load(x + offs))
        else:
            tl.store(offs + y, tl.load(offs + x))

    SIZE = 1024
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    x_tri = to_triton(x, dst_type=dtype, device=device)
    y_tri = to_triton(y, dst_type=dtype, device=device)
    y = x
    kernel[
        1,
    ](x_tri, y_tri, order, SIZE)
    np.testing.assert_allclose(y, to_numpy(y_tri))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y", [  #
    (dtype_x, dtype_y) for dtype_x in int_dtypes for dtype_y in int_dtypes
] + [(dtype_x, dtype_y) for dtype_x in uint_dtypes for dtype_y in uint_dtypes])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_floordiv(dtype_x, dtype_y, num_ctas, device):
    # Triton has IEEE, not numpy/torch, semantics for %, and those carry
    # through to //, so we have to use a nonstandard expression to get a
    # reference result for //.
    if dtype_x.startswith("uint") or dtype_y.startswith("uint"):
        pytest.skip("skip unsupport data type")
    expr = 'x // y'
    numpy_expr = '((x - np.fmod(x, y)) / y)'
    # can't represent -int(max)
    not_minus_one = dtype_x in int_dtypes and dtype_y in int_dtypes
    if not_minus_one:
        filter_y = lambda y: y == -1
    else:
        filter_y = None
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, filter_y=filter_y, device=device, num_ctas=num_ctas)


def test_unsigned_name_mangling(device):
    # Test that uint32 and int32 are mangled differently by the compiler
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(O1, O2, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        out1 = tl.abs(x)  # uint32 -> nop
        out2 = tl.abs(-y)  # int32 -> should have an effect
        tl.store(O1 + off, out1)
        tl.store(O2 + off, out2)

    dtype_x = 'uint32'
    dtype_y = 'int32'
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs)
    # reference result
    expect = (np.abs(x), np.abs(-y))
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    actual = tuple(to_triton(np.empty_like(e), device=device) for e in expect)
    kernel[(1, )](actual[0], actual[1], x_tri, y_tri, SIZE=SIZE, num_warps=4)

    # Bitwise op, so expect exact equality
    assert (expect[0] == to_numpy(actual[0])).all()
    assert (expect[1] == to_numpy(actual[1])).all()


# test bitwise ops
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op)
    for op in ['&', '|', '^']
    for dtype_x in dtypes + dtypes_with_bfloat16
    for dtype_y in dtypes + dtypes_with_bfloat16
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bitwise_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if 'float' in dtype_x + dtype_y:
        # The CompilationError must have been caused by a C++ exception with this text.
        with pytest.raises(triton.TritonError, match='invalid operands of type'):
            _test_binary(dtype_x, dtype_y, expr, numpy_expr='np.array([])', device=device, num_ctas=num_ctas)
    else:
        _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_y, op", [  #
    (dtype_x, dtype_y, op) for op in ['<<', '>>'] for dtype_x in int_dtypes + uint_dtypes for dtype_y in uint_dtypes
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_shift_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f'x {op} y'
    bw = max(_bitwidth(dtype_x), _bitwidth(dtype_y))
    if dtype_x.startswith('int'):
        dtype_z = f'int{bw}'
    else:
        dtype_z = f'uint{bw}'
    numpy_expr = f'x.astype(np.{dtype_z}) {op} y.astype(np.{dtype_z})'
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas, y_low=0, y_high=bw)


# ---------------
# test compare ops
# ---------------
ops = ['==', '!=', '>', '<', '>=', '<=']


@pytest.mark.interpreter
@pytest.mark.parametrize(
    "dtype_x, dtype_y, op, mode_x, mode_y",
    # real
    [(dtype_x, dtype_y, op, 'real', 'real') for op in ops for dtype_x in dtypes for dtype_y in dtypes]
    # NaNs
    + [('float32', 'float32', op, mode_x, mode_y)
       for op in ops
       for mode_x, mode_y in [('nan', 'real'), ('real', 'nan'), ('nan', 'nan')]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_compare_op(dtype_x, dtype_y, op, mode_x, mode_y, num_ctas, device):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, mode_x=mode_x, mode_y=mode_y, device=device, num_ctas=num_ctas)


# ---------------
# test broadcast
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", dtypes_with_bfloat16)
def test_broadcast(dtype, device):
    check_type_supported(dtype, device)

    @triton.jit
    def broadcast_kernel(x_ptr, y_ptr, y_broadcasted_ptr, M: tl.constexpr, N: tl.constexpr):
        offset1 = tl.arange(0, M)
        offset2 = tl.arange(0, N)
        x = tl.load(x_ptr + N * offset1[:, None] + offset2[None, :])
        y = tl.load(y_ptr + offset2)
        _, y_broadcasted = tl.broadcast(x, y)
        tl.store(y_broadcasted_ptr + N * offset1[:, None] + offset2[None, :], y_broadcasted)

    M = 32
    N = 64
    rs = RandomState(17)
    x = numpy_random((M, N), dtype_str=dtype, rs=rs)
    y = numpy_random(N, dtype_str=dtype, rs=rs)
    _, y_broadcasted_np = np.broadcast_arrays(x, y)

    x_tri = to_triton(x, device=device, dst_type=dtype)
    y_tri = to_triton(y, device=device, dst_type=dtype)
    y_broadcasted_tri = to_triton(np.empty((M, N), dtype=y_broadcasted_np.dtype), device=device, dst_type=dtype)

    broadcast_kernel[(1, )](x_tri, y_tri, y_broadcasted_tri, M=M, N=N)
    assert (y_broadcasted_np == to_numpy(y_broadcasted_tri)).all()


# ----------
# test slice
# ----------


@pytest.mark.interpreter
def test_slice(device):

    @triton.jit
    def slice_kernel(XBLOCK: tl.constexpr):
        data = tl.arange(0, XBLOCK)
        tl.static_assert(data.shape == [XBLOCK])

        t = data[None, :]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, :, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        scalar = tl.full([], 1, tl.int32)
        tl.static_assert(scalar.shape == [])

        t = scalar[None]
        tl.static_assert(t.shape == [1])

        t = scalar[None, None]
        tl.static_assert(t.shape == [1, 1])

    slice_kernel[(1, )](XBLOCK=32)


# ------------------
# test invalid slice
# ------------------


@pytest.mark.interpreter
def test_invalid_slice(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        dst[10:]

    with pytest.raises(triton.TritonError, match='unsupported tensor index'):
        _kernel[(1, )](dst=dst)


# ----------------
# test expand_dims
# ----------------
@pytest.mark.interpreter
def test_expand_dims(device):

    @triton.jit
    def expand_dims_kernel(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 0)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, 1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -2)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, (0, -1))
        tl.static_assert(t.shape == [1, N, 1])

        t = tl.expand_dims(offset1, (0, 1, 3))
        tl.static_assert(t.shape == [1, 1, N, 1])

        t = tl.expand_dims(offset1, (-4, 2, -1))
        tl.static_assert(t.shape == [1, N, 1, 1])

        t = tl.expand_dims(offset1, (3, 1, 2))
        tl.static_assert(t.shape == [N, 1, 1, 1])

        scalar = tl.sum(offset1)
        tl.static_assert(scalar.shape == [])
        t = tl.expand_dims(scalar, 0)
        tl.static_assert(t.shape == [1])

        t = tl.expand_dims(scalar, -1)
        tl.static_assert(t.shape == [1])

        # N is a scalar that's not even a tl.tensor -- this should work too.
        t = tl.expand_dims(N, -1)
        tl.static_assert(t.shape == [1])

    N = 32
    dummy_tensor = torch.empty((), device=device)
    expand_dims_kernel[(1, )](dummy_tensor, N)


@pytest.mark.interpreter
def test_expand_dims_error_cases(device):

    @triton.jit
    def dim_out_of_range1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, -2)
        t = tl.expand_dims(offset1, -3)

    @triton.jit
    def dim_out_of_range2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 1)
        t = tl.expand_dims(offset1, 2)

    @triton.jit
    def dim_out_of_range3(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, 1)
        scalar = tl.sum(offset1)

        t = tl.expand_dims(scalar, 1)

    @triton.jit
    def duplicate_dim1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, 0))

    @triton.jit
    def duplicate_dim2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, -3))

    N = 32
    dummy_tensor = torch.empty((), device=device)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range1[(1, )](dummy_tensor, N)
    assert "invalid axis -3" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range2[(1, )](dummy_tensor, N)
    assert "invalid axis 2" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        dim_out_of_range3[(1, )](dummy_tensor, N)
    assert "invalid axis 1" in str(exc_info.value.__cause__)

    with pytest.raises(triton.TritonError) as exc_info:
        duplicate_dim1[(1, )](dummy_tensor, N)
    assert re.search(r"duplicate axes, normalized axes = \[0, 0\]", str(exc_info.value.__cause__))

    with pytest.raises(triton.TritonError) as exc_info:
        duplicate_dim2[(1, )](dummy_tensor, N)
    assert re.search(r"duplicate axes, normalized axes = \[0, 0\]", str(exc_info.value.__cause__))


# ----------------------------
# test invalid program id axis
# ----------------------------
@pytest.mark.interpreter
def test_invalid_pid_axis(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        pid = tl.program_id(20)

    with pytest.raises(triton.TritonError) as exc_info:
        _kernel[(1, )](dst)
    assert re.search(r"program_id axis must be 0, 1, or 2 but got 20", str(exc_info.value.__cause__))


# ---------------
# test where
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", dtypes_with_bfloat16 + ["*int32"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_where(dtype, num_ctas, device):
    select_ptrs = False
    if dtype == "*int32":
        dtype = "int64"
        select_ptrs = True
    check_type_supported(dtype, device)

    @triton.jit
    def where_kernel(cond_ptr, a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                     TEST_POINTERS: tl.constexpr, TEST_SCALAR_POINTERS: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        decide = tl.load(cond_ptr + offsets, mask=mask)
        if TEST_SCALAR_POINTERS:
            ptr = tl.where(tl.load(cond_ptr), a_ptr, b_ptr)
            output = tl.load(ptr + offsets, mask=mask)
        else:
            if TEST_POINTERS:
                a = tl.load(a_ptr + offsets, mask=mask).to(tl.pi32_t)
                b = tl.load(b_ptr + offsets, mask=mask).to(tl.pi32_t)
            else:
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
            output = tl.where(decide, a, b)
        tl.store(output_ptr + offsets, output, mask=mask)

    SIZE = 1_000
    rs = RandomState(17)
    cond = numpy_random(SIZE, 'bool', rs)
    x = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    z = np.where(cond, x, y)

    cond_tri = to_triton(cond, device=device)
    x_tri = to_triton(x, device=device, dst_type=dtype)
    y_tri = to_triton(y, device=device, dst_type=dtype)
    z_tri = to_triton(np.empty(SIZE, dtype=z.dtype), device=device, dst_type=dtype)

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )
    where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs,
                       TEST_SCALAR_POINTERS=False, num_ctas=num_ctas)
    assert (z == to_numpy(z_tri)).all()
    if select_ptrs:
        where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs,
                           TEST_SCALAR_POINTERS=True)
        z = np.where(cond[0], x, y)
        assert (z == to_numpy(z_tri)).all()


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_where_broadcast(num_ctas, device):

    @triton.jit
    def where_kernel(cond_ptr, a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]

        mask = tl.load(cond_ptr + yoffsets)
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    @triton.jit
    def where_scalar_condition(a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]
        mask = False
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    SIZE = 32
    dtype = 'float32'
    rs = RandomState(17)
    x = numpy_random((SIZE, SIZE), dtype_str=dtype, rs=rs)
    mask = numpy_random(SIZE, 'bool', rs=rs)
    z = np.where(mask, x, 0)
    cond_tri = to_triton(mask, device=device)
    x_tri = to_triton(x, device=device, dst_type=dtype)
    z_tri = to_triton(np.empty((SIZE, SIZE), dtype=z.dtype), device=device, dst_type=dtype)
    where_kernel[(1, )](cond_tri, x_tri, z_tri, SIZE)
    assert (z == to_numpy(z_tri)).all()
    where_scalar_condition[(1, )](x_tri, z_tri, SIZE, num_ctas=num_ctas)
    z = np.where(0, x, 0)
    assert (z == to_numpy(z_tri)).all()


# ---------------
# test unary ops
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, expr",
                         [(dtype_x, ' -x') for dtype_x in dtypes_with_bfloat16] + [(dtype_x, ' ~x')
                                                                                   for dtype_x in int_dtypes])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_unary_op(dtype_x, expr, num_ctas, device):
    _test_unary(dtype_x, expr, device=device, num_ctas=num_ctas)


# ----------------
# test math ops
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, expr, x",
                         [(dtype_x, expr, x)
                          for dtype_x in ["float32", "float64"]
                          for expr in ['exp', 'log', 'cos', 'sin', 'exp2', 'log2', 'sqrt', 'floor', 'ceil']
                          for x in ['x', '3.0']])
def test_math_op(dtype_x, expr, x, device):
    _test_unary(dtype_x, f'tl.{expr}({x})', f'np.{expr}({x}) ', device=device)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [dtype for dtype in ["float32", "float64"]])
def test_math_erf_op(dtype, device):
    check_type_supported(dtype, device)
    SIZE = 128

    if dtype == "float64":
        device = "cpu"

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.math.erf(x)
        tl.store(Z + off, z)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    x = torch.randn(SIZE, dtype=torch_dtype, device=device)
    z_ref = torch.erf(x)
    z_tri = torch.zeros_like(x)
    if dtype == "float64":
        x = x.musa()
        z_ref = z_ref.musa()
        z_tri = z_tri.musa()
    kernel[(1, )](z_tri, x, SIZE=SIZE, num_warps=4)
    torch.testing.assert_close(z_tri, z_ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [dtype for dtype in ["float32", "float64"]])
def test_math_fma_op(dtype, device):
    check_type_supported(dtype, device)
    SIZE = 128

    if dtype == "float64":
        device = "cpu"

    @triton.jit
    def kernel(Z, X, Y, W, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        w = tl.load(W + off)
        z = tl.math.fma(x, y, w)
        tl.store(Z + off, z)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    x = torch.randn(SIZE, dtype=torch_dtype, device=device)
    y = torch.randn(SIZE, dtype=torch_dtype, device=device)
    w = torch.randn(SIZE, dtype=torch_dtype, device=device)
    z_ref = x * y + w
    z_tri = torch.zeros_like(x)
    if dtype == "float64":
        x = x.musa()
        y = y.musa()
        w = w.musa()
        z_ref = z_ref.musa()
        z_tri = z_tri.musa()
    kernel[(1, )](z_tri, x, y, w, SIZE=SIZE, num_warps=4)
    torch.testing.assert_close(z_tri, z_ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("expr", ["tl.math.fdiv(x, y)", "tl.math.div_rn(x, y)"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_math_divide_op(expr, num_ctas, device):
    numpy_expr = "x / y"
    dtype = "float32"
    _test_binary(dtype, dtype, expr, numpy_expr, device=device, num_ctas=num_ctas)


# ----------------
# test abs
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x", [(dtype_x) for dtype_x in dtypes_with_bfloat16])
def test_abs(dtype_x, device):
    _test_unary(dtype_x, 'tl.abs(x)', 'np.abs(x) ', device=device)


# ----------------
# test passing shapes as individual params rather than tuples
# ----------------


@pytest.mark.interpreter
def test_shapes_as_params(device):

    @triton.jit
    def kernel():
        a = tl.arange(0, 32).expand_dims(-1).broadcast_to(32, 32)
        tl.static_assert(a.shape == [tl.constexpr(32), tl.constexpr(32)])

        a = tl.arange(0, 32).reshape(4, 8).permute(1, 0)
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4)])

        a = tl.arange(0, 32).reshape(4, 8).trans()
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4)])

        a = tl.arange(0, 32).reshape(4, 8).reshape(32)
        tl.static_assert(a.shape == [tl.constexpr(32)])

        a = tl.arange(0, 64).reshape(2, 4, 8).trans(2, 1, 0)
        tl.static_assert(a.shape == [tl.constexpr(8), tl.constexpr(4), tl.constexpr(2)])

        a = tl.arange(0, 64).view(2, 4, 8)
        tl.static_assert(a.shape == [tl.constexpr(2), tl.constexpr(4), tl.constexpr(8)])

    kernel[(1, )]()


# ----------------
# test transpose
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x", [(dtype_x) for dtype_x in dtypes_with_bfloat16])
def test_transpose(dtype_x, device):
    check_type_supported(dtype_x, device)
    SIZE = 128

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        off2d = off[None, :] + (tl.arange(0, 2) * SIZE)[:, None]
        x = tl.load(X + off2d)
        z = x.T
        tl.store(Z + off2d.T, z)

    x = numpy_random([SIZE, 2], dtype_str=dtype_x)
    z_ref = x.T
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(z_ref), device=device, dst_type=dtype_x)
    kernel[(1, )](z_tri, x_tri, SIZE=SIZE)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri))


# ----------------
# test indexing
# ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


# TODO: handle `%4 = triton_gpu.convert_layout %3 : tensor<32xi32, #blocked0> -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>``
@pytest.mark.parametrize("expr, dtype_str", [(f'x[{s}]', d)
                                             for s in ['None, :', ':, None', 'None, :, :', ':, :, None']
                                             for d in ['int32', 'uint32', 'uint16']])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_index1d(expr, dtype_str, num_ctas, device):
    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]
    shape_z_rank_mismatch = [32 for _ in range(rank_y + 1)]
    shape_z_dim_mismatch = [64 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
            'GENERATE_TEST_HERE': expr,
        }
        return patch_kernel(kernel, to_replace)

    kernel_match = generate_kernel(shape_x, shape_z)
    kernel_dim_mismatch = generate_kernel(shape_x, shape_z_dim_mismatch)
    kernel_rank_mismatch = generate_kernel(shape_x, shape_z_rank_mismatch)

    # torch result
    x = numpy_random(shape_x, dtype_str=dtype_str)
    y = np.zeros(shape_z, dtype=getattr(np, dtype_str))
    z_ref = eval(expr) + y
    # triton result
    z_tri = to_triton(np.empty_like(z_ref), device=device)
    x_tri = to_triton(x, device=device)
    kernel_match[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
    # compare
    assert (z_ref == to_numpy(z_tri)).all()

    def catch_compilation_error(kernel):
        try:
            kernel[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0], num_ctas=num_ctas)
        except triton.CompilationError as e:
            np.testing.assert_(True)
        except BaseException:
            np.testing.assert_(False)

    catch_compilation_error(kernel_dim_mismatch)
    catch_compilation_error(kernel_rank_mismatch)


# ---------------
# test tuples
# ---------------


@triton.jit
def tuples_fn(a, b):
    return a + b, \
        a - b, \
        a * b


@pytest.mark.interpreter
def test_tuples(device):

    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = tuples_fn(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


# ---------------
# test atomics
# ---------------
@pytest.mark.interpreter
@pytest.mark.parametrize(
    "op, dtype_x_str, mode, sem",
    itertools.chain.from_iterable([[
        ('add', 'float16', mode, sem),
        ('add', 'uint32', mode, sem),
        ('add', 'int32', mode, sem),
        ('add', 'float32', mode, sem),
        ('add', 'uint64', mode, sem),
        ('add', 'int64', mode, sem),
        ('add', 'float64', mode, sem),
        ('max', 'uint32', mode, sem),
        ('max', 'int32', mode, sem),
        ('max', 'float32', mode, sem),
        ('max', 'uint64', mode, sem),
        ('max', 'int64', mode, sem),
        ('max', 'float64', mode, sem),
        ('min', 'uint32', mode, sem),
        ('min', 'int32', mode, sem),
        ('min', 'float32', mode, sem),
        ('min', 'uint64', mode, sem),
        ('min', 'int64', mode, sem),
        ('min', 'float64', mode, sem),
    ]
                                   for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']
                                   for sem in [None, 'acquire', 'release', 'acq_rel', 'relaxed']]))
def test_atomic_rmw(op, dtype_x_str, mode, sem, device):
    if is_interpreter():
        if dtype_x_str == 'float16':
            pytest.skip("Only test atomic float16 ops on GPU")

    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE
        tl.static_assert(old.dtype == x.dtype)

    sem_arg = sem if sem is None else f'"{sem}"'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x, sem={sem_arg})'})
    numpy_op = {'add': np.sum, 'max': np.max, 'min': np.min}[op]
    max_neutral = float('-inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).min
    min_neutral = float('inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    rs = RandomState(17)
    x = np.array([2**i for i in range(n_programs)], dtype=getattr(np, dtype_x_str))
    if mode == 'all_neg':
        x = -np.abs(x)
    if mode == 'all_pos':
        x = np.abs(x)
    if mode == 'min_neg':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = -np.max(np.abs(x)) - 1
    if mode == 'max_pos':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = np.max(np.abs(x)) + 1
    x_tri = to_triton(x, device=device)

    z_tri = to_triton(np.array([neutral], dtype=getattr(np, dtype_x_str)), device=device)
    h = kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = numpy_op(x).astype(getattr(np, dtype_x_str))
    # compare
    exact = op not in ['add']
    if exact:
        assert z_ref.item() == to_numpy(z_tri).item()
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
    sem_str = "acq_rel" if sem is None else sem
    if not is_cuda():
        return

    assert f"atom.global.gpu.{sem_str}" in h.asm["ptx"]


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_atomic_rmw_predicate(num_ctas, device):

    @triton.jit
    def kernel(X):
        val = tl.program_id(0)
        if val < 64:
            tl.atomic_max(X, val)

    x = torch.zeros((1, ), device=device, dtype=torch.int32)
    kernel[(4096, )](x, num_ctas=num_ctas)
    assert x.item() == 63


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, axis, num_ctas, dtype_x_str",
                         [(shape, axis, num_ctas, dtype_x_str)
                          for shape in [(2, 2), (2, 8), (8, 2), (8, 8), (32, 32), (64, 64)]
                          for axis in [0, 1]
                          for num_ctas in num_ctas_list
                          for dtype_x_str in ['float32', 'uint64', 'int64', 'float64']])
def test_tensor_atomic_rmw(shape, axis, num_ctas, dtype_x_str, device):
    shape0, shape1 = shape
    # triton kernel

    @triton.jit
    def kernel(Z, X, OLD, AXIS: tl.constexpr, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        x = tl.load(X + off0[:, None] * SHAPE1 + off1[None, :])
        z = tl.sum(x, axis=AXIS)
        if AXIS == 1:
            old = tl.atomic_add(Z + off0, z)
            tl.store(OLD + off0, old)
        else:
            old = tl.atomic_add(Z + off1, z)
            tl.store(OLD + off1, old)

    rs = RandomState(17)
    x = numpy_random((shape0, shape1), dtype_str=dtype_x_str, rs=rs)
    z_shape = (shape0, ) if axis == 1 else (shape1, )
    z = numpy_random(z_shape, dtype_str=dtype_x_str, rs=rs)
    old = np.zeros(z_shape, dtype=getattr(np, dtype_x_str))
    # reference results
    z_ref = z + np.sum(x, axis=axis, keepdims=False)
    old_ref = np.copy(z)
    # triton result
    x_tri = to_triton(x, device=device)
    z_tri = to_triton(z, device=device)
    old_tri = to_triton(old, device=device)
    kernel[(1, )](z_tri, x_tri, old_tri, axis, shape0, shape1, num_ctas=num_ctas)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=1e-4)
    np.testing.assert_equal(old_ref, to_numpy(old_tri))


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_tensor_atomic_rmw_block(num_ctas, device):
    shape = (8, 8)

    @triton.jit
    def kernel(X, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        offs = off0[:, None] * SHAPE1 + off1[None, :]
        val = offs.to(tl.float32)
        x = X + offs
        tl.atomic_min(x, val)

    x = torch.ones((8, 8), device=device, dtype=torch.float32)
    kernel[(2, )](x, shape[0], shape[1], num_ctas=num_ctas)
    assert torch.min(x).item() == 0.0


@pytest.mark.interpreter
@pytest.mark.parametrize("sem", [None, 'acquire', 'release', 'acq_rel', 'relaxed'])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_atomic_cas(sem, num_ctas, device):
    # 1. make sure that atomic_cas changes the original value (Lock)
    @triton.jit
    def change_value(Lock):
        tl.atomic_cas(Lock, 0, 1)

    Lock = torch.zeros((1, ), device=device, dtype=torch.int32)
    change_value[(1, )](Lock)

    assert (Lock[0] == 1)

    # 2. only one block enters the critical section
    @triton.jit
    def serialized_add(data, Lock, SEM: tl.constexpr):
        ptrs = data + tl.arange(0, 128)
        while tl.atomic_cas(Lock, 0, 1, SEM) == 1:
            pass

        tl.store(ptrs, tl.load(ptrs) + 1.0)

        # insert barrier to set a fence between tl.store and
        # tl.atomic_xchg in a block.
        tl.debug_barrier()

        # release lock
        tl.atomic_xchg(Lock, 0)

    Lock = torch.zeros((1, ), device=device, dtype=torch.int32)
    data = torch.zeros((128, ), device=device, dtype=torch.float32)
    ref = torch.full((128, ), 2000.0)
    h = serialized_add[(2000, )](data, Lock, SEM=sem, num_ctas=num_ctas)
    sem_str = "acq_rel" if sem is None else sem
    np.testing.assert_allclose(to_numpy(data), to_numpy(ref))
    if not is_cuda():
        return
    assert f"atom.global.{sem_str}" in h.asm["ptx"]


@pytest.mark.interpreter
@pytest.mark.parametrize("sem", [None, 'acquire', 'release', 'acq_rel', 'relaxed'])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_tensor_atomic_cas(sem, num_ctas, device):

    @triton.jit
    def change_value(X, BLOCK_SIZE: tl.constexpr, sem: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        t1 = tl.full((BLOCK_SIZE, ), 0, dtype=tl.int64)
        t2 = tl.full((BLOCK_SIZE, ), 2, dtype=tl.int64)
        tl.atomic_cas(X + offsets, t1, t2, sem=sem)

    X = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device, dtype=torch.int64)
    Y = torch.tensor([2, 1, 2, 1, 2, 1, 2, 1], device=device, dtype=torch.int64)

    change_value[(2, )](X, 4, sem)
    assert (torch.equal(X, Y))


# ---------------
# test cast
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_x, dtype_z, bitcast, size",
                         [(dtype_x, dtype_z, False, 1024) for dtype_x in dtypes for dtype_z in dtypes] + [
                             ('float32', 'bfloat16', False, 1024),
                             ('bfloat16', 'float32', False, 1024),
                             ('float32', 'int32', True, 1024),
                             ('float32', 'int1', False, 1024),
                             ('int8', 'bfloat16', False, 1024),
                         ] + [(f'uint{x}', f'int{x}', True, 1024)
                              for x in [8, 16, 32, 64]] + [(f'int{x}', f'uint{x}', True, 1024)
                                                           for x in [8, 16, 32, 64]] +
                         (([(dtype_x, dtype_z, False, size)
                            for dtype_x in torch_float8_dtypes
                            for dtype_z in ["float16", "float32", "bfloat16"]
                            for size in [1024, 32]]  #
                           + [(dtype_x, dtype_z, False, size)
                              for dtype_z in torch_float8_dtypes
                              for dtype_x in ["float16", "float32", "bfloat16"]
                              for size in [1024, 32]]) if torch.__version__ >= "2.1" else []))
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_cast(dtype_x, dtype_z, bitcast, size, num_ctas, device):
    # CUDA: bfloat16 on cc < 80 will not be tested
    # Interpreter: Only bfloat16 <-> float32 is supported
    if not is_interpreter() or \
        (is_interpreter() and not ((dtype_z == 'bfloat16' and dtype_x == 'float32')
                                   or (dtype_z == 'float32' and dtype_x == 'bfloat16'))):
        check_type_supported(dtype_x, device)
        check_type_supported(dtype_z, device)

    if is_hip() and (dtype_z in ("bfloat16", "float8_e4m3fn") or dtype_x == "float8_e4m3fn"):
        pytest.skip(f'test_cast{(dtype_x, dtype_z)} cast to bfloat16 not supported on HIP.')

    torch.manual_seed(0)
    # This is tricky because numpy doesn't have bfloat, and torch doesn't have uints.
    if dtype_x.startswith('bfloat'):
        x_tri = torch.randn(size, dtype=getattr(torch, dtype_x), device=device)
    elif dtype_x.startswith('float8'):
        x_tri = torch.randn(size, dtype=torch.half, device=device).to(dtype=getattr(torch, dtype_x))
    else:
        x = numpy_random(size, dtype_str=dtype_x, low=-10, high=10) * 10
        # Triton clamps negative values to zero, while numpy wraps around
        # intmax, so avoid negatives for now.
        # TODO: figure out which one should actually be happening, and test it
        if dtype_z in uint_dtypes:
            x = np.absolute(x)
        x_tri = to_triton(x, device=device)
    if 'float' in dtype_z and 'float' in dtype_x:
        # make sure we use values that can be represented in both types
        x_tri = x_tri.to(getattr(torch, dtype_z)).to(getattr(torch, dtype_x))
    # triton kernel

    @triton.jit
    def kernel(X, Z, BITCAST: tl.constexpr, SIZE: tl.constexpr, ARG_HASH: tl.constexpr):
        x_ptr = X + tl.arange(0, SIZE)
        z_ptr = Z + tl.arange(0, SIZE)
        x = tl.load(x_ptr)

        # Depending on the value of ARG_HASH (a "random" number determined by
        # the test parameters), spell the cast one of three different ways.
        if ARG_HASH % 3 == 0:
            z = x.to(Z.dtype.element_ty, bitcast=BITCAST)
        elif ARG_HASH % 3 == 1:
            z = x.cast(Z.dtype.element_ty, bitcast=BITCAST)
        else:
            z = tl.cast(x, Z.dtype.element_ty, bitcast=BITCAST)

        tl.store(z_ptr, z)

    # "Random" number used inside the kernel to determine how we spell the cast.
    # This way we don't have to increase the number of tests.
    arg_hash = hash((dtype_x, dtype_z, bitcast, size, num_ctas))

    dtype_z_np = dtype_z if dtype_z != 'int1' else 'bool_'
    # triton result
    if dtype_z.startswith('bfloat'):
        z_tri = torch.empty((size, ), dtype=getattr(torch, dtype_z), device=device)
    elif dtype_z.startswith('float8'):
        z_tri = torch.empty((size, ), dtype=torch.half, device=device).to(dtype=getattr(torch, dtype_z))
    else:
        z_tri = to_triton(np.empty((size, ), dtype=getattr(np, dtype_z_np)), device=device)
    kernel[(1, )](x_tri, z_tri, BITCAST=bitcast, SIZE=size, ARG_HASH=arg_hash, num_warps=1, num_ctas=num_ctas)
    # torch result
    if dtype_z.startswith('bfloat') or dtype_x.startswith('bfloat') or dtype_z.startswith(
            'float8') or dtype_x.startswith('float8'):
        assert bitcast is False
        z_ref = x_tri.to(z_tri.dtype)
        if dtype_z.startswith('float8') and device not in ['cuda']:
            t = z_ref.byte() ^ z_tri.byte()
            torch.testing.assert_close(torch.zeros_like(t, dtype=torch.uint8), t)
        else:
            torch.testing.assert_close(z_ref, z_tri, rtol=0, atol=0)
    else:
        if bitcast:
            z_ref = x.view(getattr(np, dtype_z_np))
        else:
            z_ref = x.astype(getattr(np, dtype_z_np))
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0, atol=0)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str, num_warps",
                         [(dtype_str, num_warps) for dtype_str in int_dtypes + float_dtypes for num_warps in [4, 8]])
def test_cat(dtype_str, num_warps, device):
    check_type_supported(dtype_str, device)

    device = "cpu"

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.cat(x, y, can_reorder=True)
        tl.store(Z + tl.arange(0, 2 * N), z)

    x = torch.arange(0, 128, device=device).to(getattr(torch, dtype_str))
    y = torch.arange(-128, 0, device=device).to(getattr(torch, dtype_str))
    z_ref = torch.cat([x, y], dim=0).sum()
    z = torch.zeros((256, ), dtype=getattr(torch, dtype_str), device=device)

    x = x.musa()
    y = y.musa()
    z = z.musa()
    kernel[(1, )](x, y, z, N=128, num_warps=num_warps)
    z = z.to('cpu')
    assert z.sum() == z_ref
    # check if there's no duplicate value in z
    assert z.unique().size(0) == z.size(0)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", list(torch_dtypes))
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_store_constant(dtype_str, num_ctas, device):
    check_type_supported(dtype_str, device)
    """Tests that boolean True is stored as 1"""

    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        output = GENERATE_TEST_HERE
        tl.store(output_ptr + offsets, output, mask=mask)

    triton_dtype_str = 'uint8' if dtype_str == 'bool' else dtype_str
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.zeros([BLOCK_SIZE], dtype=tl.{triton_dtype_str}) + 1'})
    block_size = 128
    ref = torch.ones([block_size], dtype=getattr(torch, dtype_str), device=device)
    output = torch.zeros([block_size], dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](output, block_size, BLOCK_SIZE=block_size, num_ctas=num_ctas)

    assert torch.all(output == ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_store_constant_default_dtype(num_ctas, device):
    """Tests that boolean True is stored as 1"""

    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = 1
        output = tl.full([BLOCK_SIZE], value=value, dtype=value.dtype)
        tl.store(output_ptr + offsets, output, mask=mask)

    block_size = 128
    ref = torch.ones([block_size], dtype=getattr(torch, 'int32'), device=device)
    output = torch.zeros([block_size], dtype=getattr(torch, 'int32'), device=device)
    kernel[(1, )](output, block_size, BLOCK_SIZE=block_size, num_ctas=num_ctas)

    assert torch.all(output == ref)


def test_load_store_same_ptr(device):

    @triton.jit()
    def kernel(in_out_ptr):
        pid = tl.program_id(axis=0)
        x = tl.load(in_out_ptr + pid)
        out = x * 2
        tl.store(in_out_ptr + pid, out)

    for _ in range(1000):
        x = torch.ones((65536, ), device=device, dtype=torch.float32)
        if is_hip():
            kernel[(65536, )](x, num_warps=16)  # threads per Warp for ROCM is 64
        else:
            kernel[(65536, )](x, num_warps=32)
        assert torch.all(x == 2)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ['int32'])
def test_umulhi(dtype_str, device):

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.umulhi(x, y)
        tl.store(Z + tl.arange(0, N), z)

    def umulhi32(a, b):
        # Convert to 64-bit unsigned integers to prevent overflow
        a_64 = a.astype(np.int64)
        b_64 = b.astype(np.int64)

        # Perform the multiplication in 64-bit
        product_64 = a_64 * b_64

        # Shift right by 32 bits to get the high part of the product
        result_high_32 = product_64 >> 32
        return result_high_32

    rs = RandomState(17)
    N = 128
    x = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    x_tri = to_triton(x, device=device)
    y = numpy_random((N, ), dtype_str=dtype_str, rs=rs, low=0)
    y_tri = to_triton(y, device=device)
    z_tri = torch.zeros_like(x_tri)
    kernel[(1, )](x_tri, y_tri, z_tri, N=N)

    z_ref = umulhi32(x, y)
    np.testing.assert_equal(z_ref, to_numpy(z_tri))


@pytest.mark.interpreter
def test_join(device):

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.join(x, y)
        tl.store(Z + tl.arange(0, N)[:, None] * 2 + tl.arange(0, 2)[None, :], z)

    x = torch.arange(0, 128, device=device).to(torch.int32)
    y = torch.arange(-128, 0, device=device).to(torch.int32)
    z_ref = torch.stack([x, y], dim=-1)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](x, y, z, N=128)

    np.testing.assert_equal(to_numpy(z_ref), to_numpy(z))


@pytest.mark.interpreter
def test_join_scalars(device):

    @triton.jit
    def kernel(X, Y, Z):
        x = tl.load(X)
        y = tl.load(Y)
        z = tl.join(x, y)
        tl.static_assert(z.shape == [2])
        tl.store(Z + tl.arange(0, 2), z)

    x = torch.full([1], 42, device=device).to(torch.int32)
    y = torch.full([1], 100, device=device).to(torch.int32)
    z = torch.zeros([2], device=device)
    kernel[(1, )](x, y, z)

    np.testing.assert_equal([42, 100], to_numpy(z))


@pytest.mark.interpreter
def test_join_with_mma(device):

    @triton.jit
    def kernel(X, Z):
        x = tl.load(X + 16 * tl.arange(0, 32)[:, None] + tl.arange(0, 16)[None, :])  # (32,16)
        x2 = tl.join(x, 2 * x)  # (32,16,2)
        x3 = tl.reshape(x2, (32, 32))
        z = tl.dot(x3, x3)  # (32,32)
        tl.store(Z + 32 * tl.arange(0, 32)[:, None] + tl.arange(0, 32)[None, :], z)

    x = torch.arange(0, 32 * 16, device=device, dtype=torch.float32).reshape((32, 16))
    r = torch.stack([x, 2 * x], dim=-1).reshape((32, 32))
    z_ref = torch.matmul(r, r)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](x, z)

    torch.testing.assert_close(z, z_ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("debug", [False, True])
def test_interleave(device, debug):

    @triton.jit(debug=debug)
    def kernel(Z, N: tl.constexpr):
        z = tl.interleave(tl.arange(0, N), tl.arange(N, 2 * N))
        tl.store(Z + tl.arange(0, 2 * N), z)

    x = torch.arange(0, 128, device=device).to(torch.int32)
    y = torch.arange(128, 256, device=device).to(torch.int32)
    z_ref = torch.stack([x, y], dim=-1).reshape(256)
    z = torch.zeros_like(z_ref)
    kernel[(1, )](z, N=128)

    np.testing.assert_equal(to_numpy(z_ref), to_numpy(z))


@pytest.mark.interpreter
def test_interleave_scalars(device):

    @triton.jit
    def kernel(X, Y, Z):
        z = tl.interleave(X, Y)
        tl.static_assert(z.shape == [tl.constexpr(2)])
        tl.store(Z + tl.arange(0, 2), z)

    z = torch.zeros(2, device=device)
    kernel[(1, )](10, 20, z)

    np.testing.assert_equal([10, 20], to_numpy(z))


@pytest.mark.interpreter
def test_split(device):

    @triton.jit
    def kernel(X, Z1, Z2, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        x1 = tl.reshape(x, (N // 2, 2))
        z1, z2 = tl.split(x1)
        tl.store(Z1 + tl.arange(0, N // 2), z1)
        tl.store(Z2 + tl.arange(0, N // 2), z2)

    x = torch.arange(0, 256, device=device).to(torch.int32).reshape((128, 2))
    z1_ref, z2_ref = (x[:, 0], x[:, 1])
    z1 = torch.zeros_like(z1_ref)
    z2 = torch.zeros_like(z2_ref)
    kernel[(1, )](x, z1, z2, N=256)

    np.testing.assert_equal(to_numpy(z1_ref), to_numpy(z1))
    np.testing.assert_equal(to_numpy(z2_ref), to_numpy(z2))


@pytest.mark.interpreter
def test_split_to_scalar(device):

    @triton.jit
    def kernel(X, Z1, Z2):
        offs = tl.arange(0, 2)
        x = tl.load(X + offs)
        z1, z2 = tl.split(x)
        tl.static_assert(isinstance(z1, tl.tensor))
        tl.static_assert(isinstance(z2, tl.tensor))
        tl.static_assert(z1.shape == [])
        tl.static_assert(z2.shape == [])
        tl.store(Z1, z1)
        tl.store(Z2, z2)

    N = 2
    x = torch.arange(0, N, device=device).reshape(N // 2, 2)
    z1_ref, z2_ref = (x[:, 0], x[:, 1])
    z1 = torch.zeros_like(z1_ref)
    z2 = torch.zeros_like(z2_ref)
    kernel[(1, )](x, z1, z2)

    np.testing.assert_equal(to_numpy(z1_ref), to_numpy(z1))
    np.testing.assert_equal(to_numpy(z2_ref), to_numpy(z2))


def convert_float_to_float32(fp: torch.tensor, dtype=None):
    if not dtype:
        dtype = getattr(tl, torch_dtype_name(fp.dtype))

    fp = fp.view(getattr(torch, f"int{dtype.primitive_bitwidth}"))
    exp_width = dtype.primitive_bitwidth - dtype.fp_mantissa_width - 1
    exp_bias = dtype.exponent_bias
    sign = ((fp >> (dtype.primitive_bitwidth - 1)) & 0x01).int()
    exp = ((fp >> dtype.fp_mantissa_width) & ((1 << exp_width) - 1)).int()
    frac = (fp & ((1 << dtype.fp_mantissa_width) - 1)).int()

    output = torch.where(
        exp == 0,
        # subnormal
        ((-1.0)**sign) * (2.0**(1 - exp_bias)) * (frac / (2.0**dtype.fp_mantissa_width)),
        # normal
        ((-1.0)**sign) * (2.0**(exp - exp_bias)) * (1.0 + frac / (2.0**dtype.fp_mantissa_width))).float()

    extended_exp = (
        (1 << (tl.float32.primitive_bitwidth - tl.float32.fp_mantissa_width - 1)) - 1) << tl.float32.fp_mantissa_width
    # special cases, exp is 0b11..1
    if dtype in [tl.float8e4nv, tl.float8e4b15]:
        # float8e4m3nv does not have infinities
        output[fp == 0b01111111] = torch.nan
        output[fp == 0b11111111] = torch.nan
    else:
        output = torch.where(exp == (1 << exp_width) - 1,
                             ((sign << (tl.float32.primitive_bitwidth - 1)) | extended_exp
                              | (frac << (tl.float32.fp_mantissa_width - dtype.fp_mantissa_width)))  #
                             .view(torch.float32), output)
    return output


@pytest.mark.interpreter
@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
def test_convert_float16_to_float32(in_dtype, device):
    """Tests that check convert_float_to_float32 function"""
    check_type_supported(in_dtype, device)

    f16_input = torch.tensor(range(-int(2**(16 - 1)), int(2**(16 - 1))), dtype=torch.int16).view(in_dtype)
    f32_output = convert_float_to_float32(f16_input)

    nan = f16_input.isnan()
    assert torch.all(f32_output[nan].isnan())
    inf = f16_input.isinf()
    assert torch.all(f32_output[inf].isinf())
    other = torch.logical_not(torch.logical_or(nan, inf))
    assert torch.all(f16_input[other] == f32_output[other])


def serialize_fp8(np_data, in_dtype):
    return np_data


# inverse of `serialize_fp8`


def deserialize_fp8(np_data, in_dtype):
    return np_data


# ---------------
# test reduce
# ---------------


@pytest.mark.interpreter
def test_max_returns_zero(device):
    # Simple test with a tl.max call that returns 0.  The interpreter had a bug
    # where it didn't handle this correctly.
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        z = tl.max(x)
        tl.store(Z, z)

    BLOCK = 128
    x = torch.zeros((BLOCK, ), device=device)
    z = torch.ones((1, ), device=device)

    kernel[(1, )](x, z, BLOCK=BLOCK)
    assert z[0] == 0


def get_reduced_dtype(dtype_str, op):
    if op in ('argmin', 'argmax'):
        return 'int32'
    if dtype_str == 'bfloat16':
        return 'float32'
    return dtype_str


@pytest.mark.interpreter
@pytest.mark.parametrize("op, dtype_str, shape", [(op, dtype, shape) for op in [
    'min',
    'max',
    'min-with-indices',
    'max-with-indices',
    'argmin-tie-break-left',
    'argmax-tie-break-left',
    'sum',
] for dtype in dtypes_with_bfloat16 for shape in [32, 64, 128, 512]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_reduce1d(op, dtype_str, shape, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        GENERATE_TEST_HERE
        tl.store(Z, z)

    if 'with-indices' in op:
        patch = f'z, _ = tl.{op.split("-")[0]}(x, axis=0, return_indices=True)'
    elif 'arg' in op:
        tie_break_left = 'tie-break-left' in op
        patch = f'z = tl.{op.split("-")[0]}(x, axis=0, tie_break_left={tie_break_left})'
    else:
        patch = f'z = tl.{op}(x, axis=0)'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': patch})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((shape, ), dtype_str=dtype_str, rs=rs)
    numpy_op = {
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
        'max-with-indices': np.max,
        'min-with-indices': np.min,
        'argmin-tie-break-fast': np.argmin,
        'argmin-tie-break-left': np.argmin,
        'argmax-tie-break-fast': np.argmax,
        'argmax-tie-break-left': np.argmax,
    }[op]
    if 'tie-break-left' in op:
        x[3:10] = x[numpy_op(x)]
    x_tri = to_triton(x, device=device)
    # numpy result
    z_dtype_str = 'int32' if op in ('argmin', 'argmax') else dtype_str
    z_tri_dtype_str = z_dtype_str
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
        z_tri_dtype_str = 'bfloat16'
    else:
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(numpy_random((1, ), dtype_str=z_dtype_str, rs=rs), device=device, dst_type=z_tri_dtype_str)
    kernel[(1, )](x_tri, z_tri, BLOCK=shape, num_ctas=num_ctas)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op in ('argmin', 'argmax'):
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            np.testing.assert_equal(x[z_ref], x[z_tri])
        else:
            np.testing.assert_equal(z_ref, z_tri)


# TODO: [Qingyi] Fix argmin / argmax
reduce_configs1 = [(op, dtype, (1, 1024), axis, False)
                   for dtype in dtypes_with_bfloat16
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for axis in [1]]

# shape (128, 256) and (32, 1024) are not enabled on sm86 because the required shared memory
# exceeds the limit of 99KB
reduce2d_shapes = [(2, 32), (4, 32), (4, 128)]
# TODO: fix and uncomment
# , (32, 64), (64, 128)]
if is_musa() and 'MTT S5000' in torch.musa.get_device_name(0):
    reduce2d_shapes += [(128, 256) and (32, 1024)]

reduce_configs2 = [(op, 'float32', shape, axis, False)
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for shape in reduce2d_shapes
                   for axis in [0, 1]] + [(op, 'float32', [16, 32], None, False) for op in ['min', 'max', 'sum']]

reduce3d_shapes = [(2, 32, 16), (32, 2, 16), (32, 16, 2)]
reduce_configs3 = [(op, 'float32', shape, axis, False)
                   for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                   for shape in reduce3d_shapes
                   for axis in [0, 1, 2]]
invalid_config = [('sum', 'float32', (32, 32), axis, False) for axis in [2, 3]]
negative_config = [('sum', 'float32', (32, 32), -1, False)]
keep_dims_2d_configs = [(op, 'float32', (32, 32), axis, True)
                        for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                        for axis in [0, 1]] + [(op, 'float32', (32, 32), None, True) for op in ['min', 'max', 'sum']]
keep_dims_3d_configs = [(op, 'float32', (32, 2, 16), axis, True)
                        for op in ['min', 'max', 'sum', 'argmin', 'argmax']
                        for axis in [0, 1, 2]] + [(op, 'float32', (32, 2, 16), None, True)
                                                  for op in ['min', 'max', 'sum']]
reduce_bool = [(op, 'bool', shape, axis, False) for op in ['xor_sum'] for shape in reduce2d_shapes for axis in [0, 1]]


@pytest.mark.interpreter
@pytest.mark.parametrize(
    "op, dtype_str, shape, axis, keep_dims", reduce_configs1 + reduce_configs2 + reduce_configs3 + invalid_config +
    negative_config + keep_dims_2d_configs + keep_dims_3d_configs + reduce_bool)
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_reduce(op, dtype_str, shape, axis, keep_dims, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested

    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, IS_3D: tl.constexpr,
               AXIS: tl.constexpr, KEEP_DIMS: tl.constexpr, USE_I1: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        range_k = tl.arange(0, BLOCK_K)
        if IS_3D:
            x = tl.load(X + range_m[:, None, None] * BLOCK_N * BLOCK_K + range_n[None, :, None] * BLOCK_K +
                        range_k[None, None, :])
        else:
            x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        if USE_I1:
            x = tl.cast(x, tl.int1)
        z = GENERATE_TEST_HERE
        z_ptr = Z
        if KEEP_DIMS and AXIS is None:
            if IS_3D:
                z_ptr = z_ptr[None, None, None, :]
            else:
                z_ptr = z_ptr[None, None, :]
        if IS_3D:
            if AXIS == 0:
                z_ptr = Z + range_n[:, None] * BLOCK_K + range_k[None, :]
            elif AXIS == 1 or AXIS == -2:
                z_ptr = Z + range_m[:, None] * BLOCK_K + range_k[None, :]
            elif AXIS == 2 or AXIS == -1:
                z_ptr = Z + range_m[:, None] * BLOCK_N + range_n[None, :]
        else:
            if AXIS == 0:
                z_ptr = Z + range_n
            elif AXIS == 1 or AXIS == -1:
                z_ptr = Z + range_m
        if KEEP_DIMS and AXIS is not None:
            z_ptr = tl.expand_dims(z_ptr, axis=AXIS)
        tl.store(z_ptr, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis=AXIS, keep_dims=KEEP_DIMS)'})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x, device=device)
    numpy_op = {
        'sum': np.sum, 'max': np.max, 'min': np.min, 'argmin': np.argmin, 'argmax': np.argmax, 'xor_sum':
        np.bitwise_xor.reduce
    }[op]
    z_dtype_str = get_reduced_dtype(dtype_str, op)
    z_tri_dtype_str = z_dtype_str
    if z_dtype_str == 'bool':
        z_dtype_str = 'int8'

    # numpy result
    # Silence numpy error on axis out of bounds, to give triton a chance to fail
    np_axis = axis if axis is not None and axis < len(shape) else None
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_tri_dtype_str = 'bfloat16'
        z_ref = numpy_op(x, axis=np_axis, keepdims=keep_dims).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
    else:
        z_ref = numpy_op(x, axis=np_axis, keepdims=keep_dims).astype(getattr(np, z_dtype_str))

    # triton result
    z_shape = z_ref.shape
    z_tri = to_triton(numpy_random(z_shape, dtype_str=z_dtype_str, rs=rs), device=device, dst_type=z_tri_dtype_str)
    BLOCK_K = 1 if len(shape) == 2 else shape[2]
    IS_3D = bool(len(shape) == 3)
    USE_I1 = dtype_str == 'bool'
    if axis is not None and axis >= len(shape):
        with pytest.raises(triton.TritonError):
            kernel[(1, )](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], BLOCK_K=BLOCK_K, IS_3D=IS_3D, AXIS=axis,
                          KEEP_DIMS=keep_dims, USE_I1=USE_I1, num_ctas=num_ctas)
        return
    else:
        kernel[(1, )](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], BLOCK_K=BLOCK_K, IS_3D=IS_3D, AXIS=axis,
                      KEEP_DIMS=keep_dims, USE_I1=USE_I1, num_ctas=num_ctas)

    z_tri = to_numpy(z_tri)

    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op in ('argmin', 'argmax'):
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            z_ref_index = z_ref
            z_tri_index = z_tri
            if not keep_dims:
                z_ref_index = np.expand_dims(z_ref, axis=axis)
                z_tri_index = np.expand_dims(z_tri, axis=axis)
            z_ref_value = np.take_along_axis(x, z_ref_index, axis=axis)
            z_tri_value = np.take_along_axis(x, z_tri_index, axis=axis)
            np.testing.assert_equal(z_ref_value, z_tri_value)
        else:
            np.testing.assert_equal(z_ref, z_tri)


scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]

scan_configs = [(op, type, shape, axis, reverse, num_warps)
                for num_warps in [4, 16]
                for type in ['int32', 'float32', 'bfloat16']
                for axis in [1, 0]
                for reverse in [True, False]
                for shape in scan2d_shapes
                for op in ['cumsum', 'cumprod', 'get_first_element', 'linear_recurrence', 'cummax', 'roll']]
negative_config = [('cumsum', 'float32', (32, 32), -1, False, 4)]


@triton.jit
# trivial associative but not commutative function
def get_first_element(a, b):
    return a


# Compute x_i = a_i * x_{i-1} + b_i
@triton.jit
def linear_recurrence(a1, b1, a2, b2):
    return a1 * a2, b1 * a2 + b2


@triton.jit
def cummax(v0, i0, v1, i1):
    gt = v0 > v1
    return tl.where(gt, v0, v1), tl.where(gt, i0, i1)


@triton.jit
def roll(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur


@pytest.mark.interpreter
@pytest.mark.parametrize("op, dtype_str, shape, axis, reverse, num_warps", scan_configs + negative_config)
def test_scan2d(op, dtype_str, shape, axis, reverse, num_warps, device):
    check_type_supported(dtype_str, device)
    if dtype_str == 'bfloat16':
        if op == 'cummax':
            pytest.skip("bfloat16 compare not suppoted before sm90")
        if op == 'linear_recurrence':
            pytest.skip("Skipping linear_recurrence scan on bfloat16 due to accuracy issues")
    numpy_dtype_str = 'float32' if dtype_str == 'bfloat16' else dtype_str

    # triton kernel
    @triton.jit
    def kernel(X, Y, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        y = tl.load(Y + range_m[:, None] * BLOCK_N + range_n[None, :])
        GENERATE_TEST_HERE
        tl.store(Z + range_m[:, None] * BLOCK_N + range_n[None, :], z)

    if op == 'cumsum' or op == 'cumprod':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'z = tl.{op}(x, axis={axis}, reverse={reverse})'})
    elif op == 'get_first_element':
        kernel = patch_kernel(
            kernel,
            {'GENERATE_TEST_HERE': f'z = tl.associative_scan(x, axis={axis}, combine_fn={op}, reverse={reverse})'})
    elif op == 'cummax':
        rg = "range_m[:, None]" if axis == 0 else "range_n[None, :]"
        rg = f"tl.broadcast_to({rg}.to(tl.int64), [BLOCK_M, BLOCK_N])"
        kernel = patch_kernel(kernel, {
            'GENERATE_TEST_HERE':
            f'_, z = tl.associative_scan((x, {rg}), axis={axis}, combine_fn={op}, reverse={reverse})'
        })
    elif op == 'roll':
        assert op == 'roll'
        kernel = patch_kernel(
            kernel, {
                'GENERATE_TEST_HERE':
                f'_, z, _ = tl.associative_scan((1 + 0* x, 0 * x, x), axis={axis}, combine_fn={op}, reverse={reverse})'
            })
    else:
        assert op == 'linear_recurrence'
        kernel = patch_kernel(kernel, {
            'GENERATE_TEST_HERE':
            f'_, z = tl.associative_scan((x, y), axis={axis}, combine_fn={op}, reverse={reverse})'
        })
    # input
    rs = RandomState(17)
    if op == 'linear_recurrence' and dtype_str in int_dtypes:
        # If the numbers are too large the op will overflow
        # We sample numbers in -1, 0, 1
        x = rs.randint(-1, 2, shape, dtype=dtype_str)
        y = rs.randint(-1, 2, shape, dtype=dtype_str)
    else:
        x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
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
    elif op == 'roll':
        ROLL = 1
        z_ref = np.roll(x_in.copy(), ROLL, axis=axis)
        if axis == 0:
            z_ref[:ROLL] = 0
        else:
            z_ref[:, :ROLL] = 0

        if reverse:
            z_ref = np.flip(z_ref, axis)
    elif op == 'linear_recurrence':
        # Simplify to the axis=1 case
        x_ref = x.T if axis == 0 else x
        y_ref = y.T if axis == 0 else y
        if reverse:
            x_ref = np.flip(x_ref, 1)
            y_ref = np.flip(y_ref, 1)

        result = []
        for x_refi, y_refi in zip(x_ref, y_ref):
            li = []
            acc = 0
            for xi, yi in zip(x_refi, y_refi):
                acc = xi * acc + yi
                li.append(acc)
            result.append(li)
        z_ref = np.array(result)
        if reverse:
            z_ref = np.flip(z_ref, 1)

        if axis == 0:
            z_ref = z_ref.T
    else:
        assert op == 'get_first_element'
        z_ref = x
        if axis == 0:
            if reverse:
                z_ref[:-1] = x[-1]
            else:
                z_ref[1:] = x[0]
        else:
            if reverse:
                z_ref[:, :-1] = x[:, -1:]
            else:
                z_ref[:, 1:] = x[:, 0:1]

    # triton result
    # we don't cast the `fp32 = bf16 op bf16` result to bfloat16 to alleviate accuracy issues
    z_tri = to_triton(z, device=device)
    kernel[(1, )](x_tri, y_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis, num_warps=num_warps)

    z_tri = to_numpy(z_tri)
    # compare
    if dtype_str not in int_dtypes:
        if op == 'cumprod':
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01, atol=1e-3)
        else:
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        np.testing.assert_equal(z_ref, z_tri)


scan_layouts = [
    BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 2], [1, THREADS_PER_WARP // 1], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N", [[32, 16], [32, 32], [32, 64], [64, 32]])
@pytest.mark.parametrize("src_layout", scan_layouts)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("add_overflow_check", [False, True])
def test_scan_layouts(M, N, src_layout, axis, add_overflow_check, device):

    overflow_check = """
        %17 = arith.extsi %arg2 : i32 to i64
        %18 = arith.extsi %arg3 : i32 to i64
        %19 = arith.addi %17, %18 : i64
        %i32.min = arith.constant -2147483648: i64
        %i32.max = arith.constant 2147483647: i64
        %20 = arith.cmpi slt, %19, %i32.max : i64
        %21 = arith.cmpi sge, %19, %i32.min : i64
        %22 = arith.andi %20, %21 : i1
        tt.assert %22, "overflow detected" : i1
    """

    ir = f"""
    #blocked = {src_layout}
    module attributes {{"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
      %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
      %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>>
      %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
      %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #blocked>
      %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %4 = tt.addptr %3, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>>
      %6 = tt.expand_dims %5 {{axis = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
      %7 = tt.broadcast %4 : tensor<{M}x1x!tt.ptr<i32>, #blocked> -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %8 = tt.broadcast %6 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
      %9 = tt.addptr %7, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %11 = "tt.scan"(%10) <{{axis = {axis} : i32, reverse = false}}> ({{
      ^bb0(%arg2: i32, %arg3: i32):
        %16 = arith.addi %arg2, %arg3 : i32{overflow_check if add_overflow_check else ""}
        tt.scan.return %16 : i32
      }}) : (tensor<{M}x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
      %12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %13 = tt.addptr %12, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %14 = tt.broadcast %13 : tensor<{M}x1x!tt.ptr<i32>, #blocked> -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %15 = tt.addptr %14, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      tt.store %15, %11 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      tt.return
    }}
    }}
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    rs = RandomState(17)
    x = rs.randint(-100, 100, (M, N)).astype('int32')

    z = np.zeros((M, N)).astype('int32')
    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, z_tri)

    z_ref = np.cumsum(x, axis=axis)

    np.testing.assert_equal(z_ref, z_tri.cpu().numpy())


layouts = [
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1],
              instr_shape=[16, 8])
]


@pytest.mark.parametrize("M", [32, 64, 128, 256])
@pytest.mark.parametrize("src_layout", layouts)
def test_store_op(M, src_layout, device):

    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "{GPU_DIALECT}.num-ctas" = 1 : i32, "{GPU_DIALECT}.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %2 = tt.addptr %1, %0 : tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %4 = tt.expand_dims %3 {{axis = 1 : i32}} : tensor<{M}xf32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xf32, #src>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %6 = tt.expand_dims %5 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
            %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<{M}x1x!tt.ptr<f32>, #src>
            %8 = tt.addptr %7, %6 : tensor<{M}x1x!tt.ptr<f32>, #src>, tensor<{M}x1xi32, #src>
            tt.store %8, %4 : tensor<{M}x1x!tt.ptr<f32>, #src>
            tt.return
        }}
    }}
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        store_kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, 1)).astype('float32')
    y = np.zeros((M, 1), dtype='float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)

    pgm = store_kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


layouts = [
    # TODO (lixun): Add MfmaLayout
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1],
              instr_shape=[16, 8])
]


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("dst_layout", filter_layouts(layouts))
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
def test_convert1d(M, src_layout, dst_layout, src_dim, dst_dim, device):

    ir = f"""
    #dst = {dst_layout}
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %2 = tt.addptr %0, %1 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %6 = tt.addptr %4, %5 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %7 = {GPU_DIALECT}.convert_layout %3 : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>> -> tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.store %6, %7 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.return
        }}
    }}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, )).astype('int32')
    y = np.zeros((M, ), dtype='int32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    pgm = kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = weight_2 / new_weight
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


layouts = [
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    # [HIP] TO DO: some tests are flaky with the layout, so turn off them for now.
    # BlockedLayout([1, 4], [1, THREADS_PER_WARP], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 32, 32], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1])
]


@pytest.mark.parametrize("M, N", [[128, 128], [256, 128], [256, 256], [128, 256]])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("first_axis", [0, 1])
def test_chain_reduce(M, N, src_layout, op, device, first_axis):

    op_str = ""
    if op == "sum":
        op_str = """
        %13 = arith.addi %arg2, %arg3 : i32
        tt.reduce.return %13 : i32"""
    elif op == "max":
        op_str = """
        %13 = arith.cmpi "sgt", %arg2, %arg3 : i32
        %14 = arith.select %13, %arg2, %arg3 : i32
        tt.reduce.return %14 : i32"""
    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
        %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #src>
        %3 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>>
        %4 = tt.expand_dims %3 {{axis = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
        %5 = tt.broadcast %2 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %6 = tt.broadcast %4 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %7 = arith.addi %5, %6 : tensor<{M}x{N}xi32, #src>
        %8 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %9 = tt.addptr %8, %7 : tensor<{M}x{N}x!tt.ptr<i32>, #src>, tensor<{M}x{N}xi32, #src>
        %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %11 = "tt.reduce"(%10) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = {first_axis} : i32}} : (tensor<{M}x{N}xi32, #src>) -> tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>
        %12 = "tt.reduce"(%11) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = 0 : i32}} : (tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>) -> i32
        tt.store %arg1, %12 : !tt.ptr<i32>
        tt.return
    }}
    }}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, N)).astype('int32')

    z = np.zeros((1, )).astype('int32')

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    pgm = kernel[(1, 1, 1)](x_tri, z_tri)
    if op == "sum":
        z_ref = np.sum(x)
    elif op == "max":
        z_ref = np.max(x)

    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@pytest.mark.interpreter
def test_generic_reduction(device):

    @triton.jit
    def var_mean_kernel(X, out_mean, out_var, BLOCK: tl.constexpr):
        xindex = tl.arange(0, BLOCK)
        x = tl.load(X + xindex)
        mean = x
        m2 = tl.zeros_like(x)
        weight = tl.full(x.shape, 1, x.dtype)
        (mean, m2, weight) = tl.reduce((mean, m2, weight), 0, _welford_combine)
        tl.store(out_mean, mean)
        tl.store(out_var, m2 / weight)

    SIZE = 512
    x = torch.rand(SIZE, device=device)
    out_mean = torch.empty((), device=device)
    out_var = torch.empty((), device=device)

    var_mean_kernel[(1, )](x, out_mean, out_var, BLOCK=SIZE)

    expect_var, expect_mean = torch.var_mean(x, dim=0, correction=0)
    torch.testing.assert_close(out_mean, expect_mean)
    torch.testing.assert_close(out_var, expect_var)


# ---------------
# test permute
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str, shape, perm", [(dtype, shape, perm)
                                                    # TODO: bfloat16
                                                    for dtype in ['float8e4b15', 'float16', 'float32']
                                                    for shape in [(64, 64), (128, 128)]
                                                    for perm in [(1, 0)]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_permute(dtype_str, shape, perm, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested
    if dtype_str == "float8e4b15" and (is_hip() or (is_cuda() and torch.cuda.get_device_capability() >= (9, 0))):
        pytest.skip("float8e4b15 not supported on ROCm or CUDA >= 9.0")
    if is_hip():
        if shape == (128, 128) and dtype_str == 'float32':
            pytest.skip("TODO Out of LDS for float32 with shape 128x128")

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xn, Z, stride_zm, stride_zn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        tl.store(Zs, tl.load(Xs))

    # input
    x = numpy_random(shape, dtype_str=dtype_str)
    # triton result
    z_tri = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    z_tri_contiguous = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    x_tri = to_triton(x, device=device, dst_type=dtype_str)
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1), z_tri, z_tri.stride(1), z_tri.stride(0),
                         BLOCK_M=shape[0], BLOCK_N=shape[1], num_ctas=num_ctas)
    pgm_contiguous = kernel[(1, 1)](x_tri, x_tri.stride(1),
                                    x_tri.stride(0), z_tri_contiguous, z_tri_contiguous.stride(0),
                                    z_tri_contiguous.stride(1), BLOCK_M=shape[0], BLOCK_N=shape[1], num_ctas=num_ctas)
    # numpy result
    if dtype_str == 'float8e4b15':
        ty = tl.float8e4b15
        z_ref = serialize_fp8(deserialize_fp8(x, ty).T.copy(), ty)
        z_tri = z_tri.base
        z_tri_contiguous = z_tri_contiguous.base
    else:
        z_ref = x.transpose(*perm)
    # compare
    np.testing.assert_allclose(to_numpy(z_tri), z_ref)
    np.testing.assert_allclose(to_numpy(z_tri_contiguous), z_ref)

    if not is_cuda():
        return

    # parse ptx to make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx
    ptx = pgm_contiguous.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["int32", "int8"])
@pytest.mark.parametrize("shape", [(2, 4), (16, 16)])
@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1])))
def test_trans_2d(dtype_str, shape, perm, device):

    @triton.jit
    def kernel(In, Out, in_shape1: tl.constexpr, in_shape2: tl.constexpr, ou_shape1: tl.constexpr,
               ou_shape2: tl.constexpr, trans1: tl.constexpr, trans2: tl.constexpr):
        in_offs = tl.arange(0, in_shape1)[:, None] * in_shape2 + tl.arange(0, in_shape2)[None, :]
        ou_offs = tl.arange(0, ou_shape1)[:, None] * ou_shape2 + tl.arange(0, ou_shape2)[None, :]
        tl.store(Out + ou_offs, tl.permute(tl.load(In + in_offs), (trans1, trans2)))

    input = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str), device='cpu').reshape(shape).to(device)
    expected = torch.permute(input, perm)
    # Don't do zeros_like -- that copies the layout, which we don't want.
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)

    kernel[(1, )](input, actual, *shape, *[shape[i] for i in perm], *perm)

    np.testing.assert_equal(to_numpy(expected), to_numpy(actual))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["int32", "int8"])
@pytest.mark.parametrize("shape", [(2, 2, 8, 64), (4, 4, 4, 4)])
@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1, 2, 3])))
def test_trans_4d(dtype_str, shape, perm, device):

    @triton.jit
    def kernel(In, Out,  #
               in_shape1: tl.constexpr, in_shape2: tl.constexpr, in_shape3: tl.constexpr, in_shape4: tl.constexpr,
               ou_shape1: tl.constexpr, ou_shape2: tl.constexpr, ou_shape3: tl.constexpr, ou_shape4: tl.constexpr,
               trans1: tl.constexpr, trans2: tl.constexpr, trans3: tl.constexpr, trans4: tl.constexpr):
        in_ptr = tl.make_block_ptr(
            base=In,
            shape=(in_shape1, in_shape2, in_shape3, in_shape4),
            strides=(in_shape4 * in_shape3 * in_shape2, in_shape4 * in_shape3, in_shape4, 1),
            offsets=(0, 0, 0, 0),
            block_shape=(in_shape1, in_shape2, in_shape3, in_shape4),
            order=(3, 2, 1, 0),
        )
        out_ptr = tl.make_block_ptr(
            base=Out,
            shape=(ou_shape1, ou_shape2, ou_shape3, ou_shape4),
            strides=(ou_shape4 * ou_shape3 * ou_shape2, ou_shape4 * ou_shape3, ou_shape4, 1),
            offsets=(0, 0, 0, 0),
            block_shape=(ou_shape1, ou_shape2, ou_shape3, ou_shape4),
            order=(3, 2, 1, 0),
        )
        tl.store(out_ptr, tl.load(in_ptr).permute((trans1, trans2, trans3, trans4)))

    input = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str), device="cpu").to(device).reshape(shape)
    expected = torch.permute(input, perm)
    # Don't do zeros_like -- that copies the layout, which we don't want.
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)

    kernel[(1, )](input, actual, *shape, *[shape[i] for i in perm], *perm, num_warps=8)

    np.testing.assert_equal(to_numpy(expected), to_numpy(actual))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", int_dtypes + ['uint8'] + float_dtypes + ['bfloat16'])
@pytest.mark.parametrize("shape", [(), (1, ), (128, )])
def test_full(dtype_str, shape, device):
    if dtype_str in uint_dtypes and not hasattr(torch, dtype_str):
        # PyTorch only has unsigned 8, but not 16, 32, or 64
        dtype = getattr(torch, dtype_str[1:])  # uintx -> intx
    else:
        dtype = getattr(torch, dtype_str)
    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    @triton.jit
    def kernel_static(out):
        a = GENERATE_TEST_HERE
        tl.static_assert(a.shape == SHAPE)
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    @triton.jit
    def kernel_dynamic(out, val, dtype: tl.constexpr):
        a = tl.full(SHAPE, val, dtype)
        tl.static_assert(a.shape == SHAPE)
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    kernel_static_patched = patch_kernel(kernel_static, {
        'GENERATE_TEST_HERE': f"tl.full({shape}, 2, tl.{dtype_str})",
        'SHAPE': str(list(shape)),
    })
    out_static = torch.zeros((128), dtype=dtype, device=device)
    kernel_static_patched[(1, )](out_static)
    assert torch.all(out_static == 2)

    kernel_dynamic_patched = patch_kernel(kernel_dynamic, {'SHAPE': str(list(shape))})
    out_dynamic = torch.zeros((128), dtype=dtype, device=device)
    kernel_dynamic_patched[(1, )](out_dynamic, 2, getattr(triton.language, dtype_str))
    assert torch.all(out_dynamic == 2)


@pytest.mark.parametrize("literal, dtype_str", [(1e+50, "f64"), (1e+10, "f32"), (1.0, "f32"), ('float("inf")', "f32"),
                                                ('float("-inf")', "f32"), ('float("nan")', "f32"),
                                                ('float("-nan")', "f32"), (0., "f32"), (5, "i32"), (2**40, "i64")])
def test_constexpr(literal, dtype_str, device):

    @triton.jit
    def kernel(out_ptr):
        val = GENERATE_TEST_HERE
        tl.store(out_ptr.to(tl.pointer_type(val.dtype)), val)

    kernel_patched = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{literal}"})
    out = torch.zeros((1, ), dtype=torch.float32, device=device)
    h = kernel_patched[(1, )](out)
    assert re.search(r"arith.constant .* : " + dtype_str, h.asm["ttir"]) is not None


@triton.jit
def pass_const(a, b, choose_b):
    if choose_b:
        return b
    else:
        return a


@pytest.mark.parametrize("choose_const", [True, False])
@pytest.mark.parametrize("constexpr", [True, False])
@pytest.mark.parametrize("mode", ["direct", "call", "ternary", "if"])
def test_const(device, choose_const, constexpr, mode):

    @triton.jit(do_not_specialize=["choose_const"])
    def kernel(in_ptr: tl.const, out, c_out: tl.const, choose_const, n_elems: tl.int32, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elems
        val = tl.load(in_ptr + offsets, mask=mask)
        LOSE_TAIL
        tl.store(final_out + offsets, val, mask=mask)

    @triton.jit
    def kernel_constexpr(in_ptr: tl.const, out, c_out: tl.const, choose_const: tl.constexpr, n_elems: tl.int32,
                         BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elems
        val = tl.load(in_ptr + offsets, mask=mask)
        LOSE_TAIL
        tl.store(final_out + offsets, val, mask=mask)

    if mode == "direct":
        if choose_const:
            LOSE_TAIL = "final_out = c_out"
        else:
            LOSE_TAIL = "final_out = out"
    elif mode == "call":
        LOSE_TAIL = "final_out = pass_const(out, c_out, choose_const)"
    elif mode == "ternary":
        LOSE_TAIL = "final_out = c_out if choose_const else out"
    elif mode == "if":
        LOSE_TAIL = """
    if choose_const:
        final_out = c_out
    else:
        final_out = out
"""

    SIZE = 128
    input = torch.randn((SIZE, ), dtype=torch.float32, device=device)
    output = torch.zeros((SIZE, ), dtype=torch.float32, device=device)
    patched_kernel = patch_kernel(kernel_constexpr if constexpr else kernel, {'LOSE_TAIL': LOSE_TAIL, 'CONSTEXPR': ''})

    expect_fail = (not constexpr and mode != "direct") or choose_const
    if expect_fail:
        with pytest.raises(triton.CompilationError) as exc_info:
            patched_kernel[(1, )](input, output, output, choose_const, SIZE, SIZE)
        if constexpr:
            error = "Cannot store to a constant pointer"
        else:
            if mode == "call":
                error = "Inconsistent return types"
            elif mode == "if":
                error = "Mismatched type for final_out"
            elif mode == "ternary":
                error = "Ternary expression with dynamic condition has inconsistent type"
            else:
                assert mode == "direct" and choose_const
                error = "Cannot store to a constant pointer"
        error_msg = exc_info.value.error_message or str(exc_info.value.__cause__)
        assert error in error_msg, "Wrong error message!"
    else:
        patched_kernel[(1, )](input, output, output, choose_const, SIZE, SIZE)
        assert torch.all(input == output)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ['float32', 'float16'])
def test_dot_without_load(dtype_str, device):

    @triton.jit
    def _kernel(out):
        a = GENERATE_TEST_HERE
        b = GENERATE_TEST_HERE
        c = tl.dot(a, b)
        out_ptr = out + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(out_ptr, c)

    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.full((32, 32), 1.0, tl.{dtype_str})"})
    a = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    b = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    out_ref = torch.matmul(a, b)
    out = torch.zeros((32, 32), dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](out)
    assert torch.all(out == out_ref)


# ---------------
# test arange
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("start", [0, 1, 7, 16])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_arange(start, num_ctas, device):
    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(z + off, val)

    _kernel[(1, )](z_tri, START=start, END=start + BLOCK, BLOCK=BLOCK, num_ctas=num_ctas)
    z_ref = torch.arange(start, BLOCK + start, dtype=torch.int32, device=device)
    np.testing.assert_allclose(to_numpy(z_tri), to_numpy(z_ref))


# ---------------
# test load
# ---------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str, size, size_diff, other", [(dtype_str, size, size_diff, other)
                                                               for dtype_str in torch_dtypes
                                                               for size in [128, 512]
                                                               for size_diff in [0, 1, 2, 3, 4]
                                                               for other in [0, 1]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_masked_load(dtype_str, size, size_diff, other, num_ctas, device):
    dtype = getattr(torch, dtype_str)
    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    input_size = size - size_diff
    output_size = size
    if dtype_str == 'bool':
        input = torch.randint(0, 2, (input_size, ), dtype=dtype, device=device)
    elif dtype_str in int_dtypes or dtype_str in uint_dtypes:
        input = torch.randint(0, 127, (input_size, ), dtype=dtype, device=device)
    else:
        input = torch.rand(input_size, dtype=dtype, device=device)
    output = torch.zeros((output_size, ), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in_ptr, out_ptr, in_size: tl.constexpr, out_size: tl.constexpr):
        in_offsets = tl.arange(0, out_size)
        # Load inputs.
        x = GENERATE_TEST_HERE
        # Store output
        output_offsets = tl.arange(0, out_size)
        tl.store(out_ptr + output_offsets, x)

    mask_str = f"mask=in_offsets < in_size, other={other}" if size_diff > 0 else "None"
    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.load(in_ptr + in_offsets, {mask_str})"})
    kernel[(1, )](input, output, input_size, output_size, num_ctas=num_ctas)

    reference_out = torch.cat((input, torch.full((size_diff, ), other, dtype=dtype, device=device)))
    torch.testing.assert_close(output, reference_out)


@pytest.mark.interpreter
@pytest.mark.parametrize("num_ctas", num_ctas_list)
@pytest.mark.parametrize("mask_val", [True, False])
@pytest.mark.parametrize("other_val", [0, 1])
def test_masked_load_scalar(num_ctas, mask_val, other_val, device):
    input_val = 4.0
    size = 128
    dtype = torch.float32
    input = torch.full((size, ), input_val, dtype=dtype, device=device)
    output = torch.zeros((size, ), dtype=dtype, device=device)

    @triton.jit
    def kernel(in_ptr, out_ptr, size: tl.constexpr, mask: tl.constexpr, other: tl.constexpr):
        offsets = tl.arange(0, size)
        x = tl.load(in_ptr + offsets, mask=mask, other=other)
        tl.store(out_ptr + offsets, x)

    kernel[(1, )](input, output, size, mask_val, other_val, num_ctas=num_ctas)

    if mask_val:
        reference_out = torch.full((size, ), input_val, dtype=dtype, device=device)
    else:
        reference_out = torch.full((size, ), other_val, dtype=dtype, device=device)

    torch.testing.assert_close(output, reference_out)


# Testing masked loads with a copy to shared memory.
# FIXME: Shape too small for ldmatrix when num_ctas=4
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_masked_load_shared_memory(dtype, device):

    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    M = 32
    N = 32
    K = 16

    in1 = torch.rand((M, K), dtype=dtype, device=device)
    in2 = torch.rand((K, N), dtype=dtype, device=device)
    out = torch.zeros((M, N), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in1_ptr, in2_ptr, output_ptr, in_stride, in2_stride, out_stride, in_numel, in2_numel, out_numel,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):

        M_offsets = tl.arange(0, M)
        N_offsets = tl.arange(0, N)
        K_offsets = tl.arange(0, K)

        in_offsets = M_offsets[:, None] * in_stride + K_offsets[None, :]
        in2_offsets = K_offsets[:, None] * in2_stride + N_offsets[None, :]

        # Load inputs.
        x = tl.load(in1_ptr + in_offsets, mask=in_offsets < M * K)
        w = tl.load(in2_ptr + in2_offsets, mask=in2_offsets < K * N)

        # Without a dot product the memory doesn't get promoted to shared.
        o = tl.dot(x, w, out_dtype=tl.float32)

        # Store output
        output_offsets = M_offsets[:, None] * out_stride + N_offsets[None, :]
        tl.store(output_ptr + output_offsets, o, mask=output_offsets < M * N)

    pgm = _kernel[(1, )](in1, in2, out, in1.stride()[0], in2.stride()[0], out.stride()[0], in1.numel(), in2.numel(),
                         out.numel(), M=M, N=N, K=K)

    reference_out = torch.matmul(in1, in2)
    torch.testing.assert_close(out, reference_out, atol=1e-2, rtol=0)


@pytest.mark.interpreter
def test_assume(device):

    @triton.jit
    def _kernel(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
        current_size = N - tl.program_id(0) * BLOCK_N
        tl.assume(current_size >= BLOCK_N)
        if current_size >= 128:
            tl.store(out_ptr + tl.program_id(0), current_size)
        else:
            tl.store(out_ptr + tl.program_id(0), current_size + 101024)

    output = torch.zeros(1024 // 128, device=device)
    pgm = _kernel[(1024 // 128, )](output, N=1024, BLOCK_N=128)

    if is_interpreter():
        return

    assert 'llvm.assume' in pgm.asm['llir']


# ---------------
# test default
# ---------------
# TODO: can't be local to test_default


@triton.jit
def _impl(value=10):
    return value


@pytest.mark.interpreter
def test_default(device):
    value = 5
    ret0 = torch.zeros(1, dtype=torch.int32, device=device)
    ret1 = torch.zeros(1, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(ret0, ret1, value=3):
        tl.store(ret0, _impl())
        tl.store(ret1, _impl(value))

    _kernel[(1, )](ret0, ret1, value)
    assert ret0.item() == 10
    assert ret1.item() == value

    _kernel[(1, )](ret0, ret1)
    assert ret0.item() == 10
    assert ret1.item() == 3


# ---------------
# test noop
# ----------------


@pytest.mark.interpreter
def test_noop(device):

    @triton.jit
    def kernel(x):
        pass

    x = to_triton(numpy_random((1, ), dtype_str='int32'), device=device)
    kernel[(1, )](x)


@pytest.mark.parametrize("device", ['musa', 'cpu', 'cpu_pinned'])
def test_pointer_arguments(device):

    @triton.jit
    def kernel(x):
        pass

    pin_memory = 'pinned' in device
    x = torch.empty(1024, device=device.split('_')[0], pin_memory=pin_memory)
    if device == "cpu":
        with pytest.raises(ValueError):
            kernel[(1, )](x)
    else:
        kernel[(1, )](x)


@pytest.mark.parametrize("value, value_type", [(-1, 'i32'), (0, 'i32'), (-2**31, 'i32'), (2**31 - 1, 'i32'),
                                               (2**31, 'i64'), (2**32 - 1, 'i64'), (2**32, 'i64'), (2**63 - 1, 'i64'),
                                               (-2**63, 'i64'), (2**63, 'u64'), (2**64 - 1, 'u64')])
def test_value_specialization(value: int, value_type: str, device) -> None:

    def repr(specialization):
        spec_type = specialization.signature["VALUE"]
        return f"kernel_{spec_type}"

    @triton.jit(repr=repr)
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device=device)
    h = kernel[(1, )](value, x)
    assert value_type in h.name


# --------------------
# value specialization
# --------------------


@pytest.mark.parametrize("value, overflow", [(2**64 - 1, False), (2**64, True), (-2**63, False), (-2**63 - 1, True)])
def test_value_specialization_overflow(value: int, overflow: bool, device) -> None:

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device=device)

    if overflow:
        with pytest.raises(OverflowError):
            kernel[(1, )](value, x)
    else:
        kernel[(1, )](value, x)


# ----------------
# test constexpr
# ----------------


@pytest.mark.interpreter
@pytest.mark.parametrize("op", ['+', '-', '*', '/', '%', '<', '>', '<<', '>>', '&', '^', '|'])
@pytest.mark.parametrize("is_lhs_constexpr", [False, True])
@pytest.mark.parametrize("is_rhs_constexpr", [True, False])
def test_bin_op_constexpr(op, is_lhs_constexpr, is_rhs_constexpr, device):

    @triton.jit
    def kernel(Z, X, Y):
        x = tl.load(X)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z, z)

    if op in ['<<', '>>', '&', '^', '|']:  # int op
        x_str = "3" if is_lhs_constexpr else "x"
        y_str = "4" if is_rhs_constexpr else "y"
        x = numpy_random((1, ), dtype_str="int32")

        # NOTE: bitshifting beyond bitwidth can lead to undefined behavior
        if op in ['<<', '>>']:
            y = numpy_random((1, ), dtype_str="int32", low=0, high=_bitwidth("int32"))
        else:
            y = numpy_random((1, ), dtype_str="int32")
    else:
        x_str = "3.14" if is_lhs_constexpr else "x"
        y_str = "4.13" if is_rhs_constexpr else "y"
        x = numpy_random((1, ), dtype_str="float32")
        y = numpy_random((1, ), dtype_str="float32")
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{x_str} {op} {y_str}"})
    z = np.array(eval(f"{x_str} {op} {y_str}"))
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    z_tri = to_triton(np.empty((1, ), dtype=z.dtype), device=device)
    kernel[(1, )](z_tri, x_tri, y_tri)
    np.testing.assert_allclose(z, to_numpy(z_tri), rtol=1e-3)


@pytest.mark.interpreter
def test_constexpr_shape(device):

    @triton.jit
    def kernel(X):
        off = tl.arange(0, 128 + 128)
        tl.store(X + off, off)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32), device=device)
    kernel[(1, )](x_tri)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256))


@pytest.mark.interpreter
def test_constexpr_scalar_shape(device):

    @triton.jit
    def kernel(X, s):
        off = tl.arange(0, 256)
        val = off % (256 // s)
        tl.store(X + off, val)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32), device=device)
    kernel[(1, )](x_tri, 32)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256) % 8)


reshape_list = [((64, ), (8, 8)), ((2, 32), (16, 4)), ((512, ), (2, 2, 2, 2, 2, 2, 2, 2, 2)), ((64, 32), (16, 8, 16))]


@pytest.mark.interpreter
@pytest.mark.parametrize("formats", reshape_list)
def test_reshape(formats, device):
    in_format, out_format = formats

    @triton.jit
    def kernel(Z, X, out_tuple: tl.constexpr):
        x = tl.load(X_PTR_EXPR)
        z = tl.reshape(x, out_tuple)
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
        }
        return patch_kernel(kernel, to_replace)

    x = numpy_random(in_format, dtype_str="int32")
    z = x.reshape(out_format)
    x_tri = to_triton(x, device=device)
    patched_kernel = generate_kernel(in_format, out_format)
    z_tri = to_triton(np.empty(out_format, dtype=np.int32), device=device)
    patched_kernel[(1, )](z_tri, x_tri, out_format)
    np.testing.assert_equal(z, to_numpy(z_tri))


def test_reshape_err(device):

    @triton.jit
    def kernel():
        x = tl.arange(0, 8 * 8)
        y = tl.reshape(x, (8 * 4, ))

    with pytest.raises(triton.CompilationError) as exc_info:
        kernel[(1, )]()

    assert "reshape" in str(exc_info.value)


def test_trans_reshape(device):

    @triton.jit
    def kernel(in_base_ptr, out_base_ptr, IN_SHAPE0: tl.constexpr, IN_SHAPE1: tl.constexpr):

        in_block_ptr = tl.make_block_ptr(
            base=in_base_ptr,
            shape=(IN_SHAPE0, IN_SHAPE1),
            strides=(IN_SHAPE1, 1),
            offsets=(0, 0),
            block_shape=(IN_SHAPE0, IN_SHAPE1),
            order=(1, 0),
        )
        x = tl.load(in_block_ptr)
        x = tl.reshape(x, (32, 4, 4, 2))
        x = tl.permute(x, (1, 2, 3, 0))
        x = tl.reshape(x, (IN_SHAPE0 * IN_SHAPE1, ))
        tl.store(out_base_ptr + tl.arange(0, IN_SHAPE0 * IN_SHAPE1), x)

    shape = (32, 32)
    input = torch.arange(math.prod(shape), dtype=torch.int32, device=device).reshape(shape)
    expected = torch.permute(input, (1, 0))
    # Don't do zeros_like -- that copies the layout, which we don't want.
    actual = torch.zeros(expected.shape, dtype=torch.int32, device=device)

    k = kernel[(1, )](input, actual, shape[0], shape[1])
    assert k.asm['ttgir'].count(
        'triton_gpu.convert_layout') == 1, "Expected exactly one convert_layout op in the TTGIR after optimization"

    np.testing.assert_equal(to_numpy(expected), to_numpy(actual))


# -------------
# test call
# -------------


@triton.jit
def val_multiplier(val, i):
    return val * i


@triton.jit(noinline=True)
def val_multiplier_noinline(val, i):
    return val * i


@triton.jit
def vecmul_kernel(ptr, n_elements, rep, type: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    vec = tl.load(ptr + offsets, mask=mask)
    for i in range(1, rep):
        if type == "inline":
            vec = val_multiplier(vec, i)
        else:
            vec = val_multiplier_noinline(vec, i)
    tl.store(ptr + offsets, vec, mask=mask)


@pytest.mark.interpreter
@pytest.mark.parametrize("type", ["inline", "noinline"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_call(type, num_ctas, device):

    @triton.jit
    def kernel(ptr, n_elements, num1, num2, type: tl.constexpr):
        vecmul_kernel(ptr, n_elements, num1, type)
        vecmul_kernel(ptr, n_elements, num2, type)

    size = 1024
    rand_val = numpy_random((size, ), dtype_str="float32")
    rand_val_tri = to_triton(rand_val, device=device)
    err_msg = ""
    try:
        kernel[(size // 128, )](rand_val_tri, size, 3, 5, type, num_ctas=num_ctas)
    except Exception as e:
        err_msg = str(e)

    if type == "noinline" and not is_interpreter():
        assert err_msg != ""
    else:
        ans = rand_val * 1 * 2 * 1 * 2 * 3 * 4
        np.testing.assert_equal(to_numpy(rand_val_tri), ans)


# -------------
# test if
# -------------


@pytest.mark.interpreter
@pytest.mark.parametrize("if_type", [
    "if", "if_and_dynamic", "if_exp_static", "if_exp_dynamic", "if_exp_dynamic_constexpr", "if_exp_dynamic_void",
    "if_and_static"
])
def test_if(if_type, device):

    @triton.jit
    def kernel(Cond, XTrue, XFalse, Ret, IfType: tl.constexpr, BoolVar: tl.constexpr, StaticVaue: tl.constexpr):
        pid = tl.program_id(0)
        cond = tl.load(Cond)
        if IfType == "if":
            if pid % 2 == 0:  # eq
                tl.store(Ret, tl.load(XTrue))
            elif 1 == pid % 2:  # req
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_dynamic":
            val = tl.load(XTrue) if pid % 2 == 0 else tl.load(XFalse)
            tl.store(Ret, val)
        elif IfType == "if_exp_dynamic_constexpr":
            val = 3.14 if pid % 2 == 0 else tl.load(XFalse)
            tl.store(Ret, val)
        elif IfType == "if_exp_dynamic_void":
            tl.store(Ret, tl.load(XTrue)) if pid % 2 == 0 else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_static":
            tl.store(Ret, tl.load(XTrue)) if BoolVar else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_dynamic":
            if BoolVar and (1 != pid % 2 and pid % 2 != 1):  # rne and ne
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_static":
            if StaticVaue != 0 and StaticVaue != 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))

    cond = torch.ones(1, dtype=torch.int32, device=device)
    x_true = torch.tensor([3.14], dtype=torch.float32, device=device)
    x_false = torch.tensor([1.51], dtype=torch.float32, device=device)
    ret = torch.zeros(1, dtype=torch.float32, device=device)

    kernel[(1, )](cond, x_true, x_false, ret, if_type, True, 1)
    assert torch.equal(ret, x_true)


def test_num_warps_pow2(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        pass

    with pytest.raises(AssertionError, match='must be a power of 2'):
        _kernel[(1, )](dst=dst, num_warps=3)
    _kernel[(1, )](dst=dst, num_warps=1)
    _kernel[(1, )](dst=dst, num_warps=2)
    _kernel[(1, )](dst=dst, num_warps=4)


@pytest.mark.interpreter
@pytest.mark.parametrize("func_str", ['sqrt', 'rsqrt', 'exp', 'exp2', 'log', 'log2', 'sin', 'cos'])
def test_unary_math(func_str, device):

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.FUNC_STR(x)
        tl.store(Y + tl.arange(0, BLOCK), y)

    kernel = patch_kernel(kernel, {'FUNC_STR': func_str})

    shape = (128, )
    x = torch.randn(shape, dtype=torch.float32, device=device)
    if func_str in ['sqrt', 'rsqrt']:
        x = torch.abs(x)
    if func_str in ['log', 'log2']:
        x = torch.max(x, torch.tensor(1e-6, dtype=torch.float32, device=device))
    y = torch.zeros(shape, dtype=torch.float32, device=device)

    kernel[(1, )](x, y, BLOCK=shape[0])
    torch.allclose(getattr(torch, func_str)(x), y, rtol=1e-3)


# -----------------------
# test control flow
# -----------------------


@pytest.mark.parametrize("lo, hi, iv", [(2**35, 2**35 + 20, 1), (2**35, 2**35 + 20, 2), (2**35, 2**35 + 20, 3),
                                        (15, -16, -1), (15, -16, -2), (15, -16, -3), (-18, -22, -1), (22, 18, -1)])
def test_for_iv(lo, hi, iv, device):

    @triton.jit
    def kernel(Out, lo, hi, iv: tl.constexpr):
        acc = 0
        acc = acc.to(tl.int64)
        for i in range(lo, hi, iv):
            acc += i
        tl.store(Out, acc)

    lo = 2**35
    hi = 2**35 + 20
    out = to_triton(np.zeros((1, ), dtype=np.int64), device=device)
    kernel[(1, )](out, lo, hi, iv)
    assert out[0] == sum(range(lo, hi, iv))


@pytest.mark.interpreter
def test_if_else(device):

    @triton.jit
    def kernel(Cond, TrueVal, FalseVal, Out):
        if tl.load(Cond):
            val = tl.load(TrueVal)
        else:
            val = tl.load(FalseVal)
        tl.store(Out, val)

    out = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    true_val = to_triton(np.full((1, ), 1, dtype=np.int32), device=device)
    false_val = to_triton(np.full((1, ), 2, dtype=np.int32), device=device)
    cond = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    # True
    cond[0] = True
    kernel[(1, )](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == true_val[0]
    # False
    cond[0] = False
    kernel[(1, )](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == false_val[0]


@pytest.mark.interpreter
@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_if_return(mode, device):

    @triton.jit
    def kernel(ExitEarly, Out, cond: tl.constexpr, mode: tl.constexpr):
        if mode == "dynamic":
            if tl.load(ExitEarly):
                tl.store(Out, 0)
                return
        else:
            if cond:
                tl.store(Out, 0)
                return
        tl.store(Out, 1)

    out = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    exit_early = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    # exit early path taken
    exit_early[0] = 1
    kernel[(1, )](exit_early, out, True, mode)
    assert to_numpy(out)[0] == 0
    # exit early path not taken
    exit_early[0] = 0
    kernel[(1, )](exit_early, out, False, mode)
    assert to_numpy(out)[0] == 1


@pytest.mark.interpreter
@pytest.mark.parametrize("_cond1", [True, False])
@pytest.mark.parametrize("_cond2", [True, False])
@pytest.mark.parametrize("_cond3", [True, False])
def test_nested_if_else_return(_cond1, _cond2, _cond3, device):

    @triton.jit
    def kernel(Cond1, Cond2, Cond3, Val1, Val2, Val3, Out):
        val = 0
        if tl.load(Cond1):
            if tl.load(Cond2):
                val = tl.load(Val1)
            else:
                return
        else:
            if tl.load(Cond3):
                val = tl.load(Val2)
            else:
                val = tl.load(Val3)
        tl.store(Out, val)

    out = to_triton(np.full((1, ), -1, dtype=np.int32), device=device)
    cond1 = to_triton(np.full((1, ), _cond1, dtype=np.int32), device=device)
    cond2 = to_triton(np.full((1, ), _cond2, dtype=np.int32), device=device)
    cond3 = to_triton(np.full((1, ), _cond3, dtype=np.int32), device=device)
    val1 = to_triton(np.full((1, ), 1, dtype=np.int32), device=device)
    val2 = to_triton(np.full((1, ), 2, dtype=np.int32), device=device)
    val3 = to_triton(np.full((1, ), 3, dtype=np.int32), device=device)
    kernel[(1, )](cond1, cond2, cond3, val1, val2, val3, out)
    targets = {
        (True, True, True): val1[0],
        (True, True, False): val1[0],
        (True, False, True): out[0],
        (True, False, False): out[0],
        (False, True, True): val2[0],
        (False, True, False): val3[0],
        (False, False, True): val2[0],
        (False, False, False): val3[0],
    }
    assert out[0] == targets[(_cond1, _cond2, _cond3)]


@pytest.mark.interpreter
def test_while(device):

    @triton.jit
    def kernel(InitI, Bound, CutOff, OutI, OutInitI, OutJ):
        init_i = tl.load(InitI)
        curr_i = init_i
        j = 0
        # Check that init_i is not updated by the loop
        while j < tl.load(Bound):
            curr_i = curr_i + (j == tl.load(CutOff))
            j += 1
            tl.store(OutInitI, init_i)
        tl.store(OutI, curr_i)
        tl.store(OutJ, j)

    out_i = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    out_j = to_triton(np.zeros((1, ), dtype=np.int32), device=device)
    init_i = to_triton(np.full((1, ), 1, dtype=np.int32), device=device)
    out_init_i = to_triton(np.full((1, ), 0, dtype=np.int32), device=device)
    bound = to_triton(np.full((1, ), 10, dtype=np.int32), device=device)
    cut_off = to_triton(np.full((1, ), 5, dtype=np.int32), device=device)
    kernel[(1, )](init_i, bound, cut_off, out_i, out_init_i, out_j)
    assert out_init_i[0] == init_i[0]
    assert out_i[0] == init_i[0] + 1
    assert out_j[0] == bound[0]


@pytest.mark.interpreter
def test_nested_while(device):

    @triton.jit
    def nested_while(data, countPtr):
        for i in range(10):
            count = tl.load(countPtr)
            while count > 0:
                tl.store(data, tl.load(data) + 1.0)
                count = count - 2

    counter = torch.tensor([8], dtype=torch.int32, device=device)
    data = torch.zeros((1, ), device=device, dtype=torch.float32)
    nested_while[(1, )](data, counter)
    assert data[0] == 40


def test_constexpr_if_return(device):

    @triton.jit
    def kernel(Semaphore, Out, total: tl.constexpr):
        if total == 1:
            tl.store(Out, tl.program_id(0))
            return

        prev = tl.atomic_add(Semaphore, 1)
        if prev + 1 != total:
            return

        tl.store(Out, tl.program_id(0) + prev)

    sem = torch.zeros((), device=device, dtype=torch.int32)
    out = torch.empty((), device=device, dtype=torch.int32)
    kernel[(1, )](sem, out, 1)
    assert out.item() == 0

    sem = torch.zeros((), device=device, dtype=torch.int32)
    out = torch.full((), fill_value=-1, device=device, dtype=torch.int32)
    kernel[(4, )](sem, out, 4)
    assert out.item() >= 0


@pytest.mark.interpreter
def test_load_scalar_with_mask(device):

    @triton.jit
    def kernel(Input, Index, Out, N: int):
        index = tl.load(Index)
        scalar = tl.load(Input + index, mask=index < N, other=0)
        tl.store(Out, scalar, mask=index < N)

    Index = torch.tensor([0], dtype=torch.int32, device=device)
    Input = torch.tensor([0], dtype=torch.int32, device=device)
    Out = torch.empty_like(Index, device=device)
    kernel[(1, )](Input, Index, Out, Index.numel())
    assert Out.data[0] == 0


# -----------------------
# test propagate_nan
# -----------------------


@pytest.mark.parametrize("dtype", ['float16', 'float32'])
@pytest.mark.parametrize("propagate_nan", ['NONE', 'ALL'])
@pytest.mark.parametrize("func", ['minimum', 'maximum', 'clamp'])
def test_propagate_nan(dtype, propagate_nan, func, device):

    @triton.jit
    def kernel(A, B, C, propagate_nan: tl.constexpr, func: tl.constexpr):
        if func == 'clamp':
            tl.store(
                C,
                getattr(tl, func)(tl.load(A), -tl.load(B), tl.load(B),
                                  propagate_nan=getattr(tl.PropagateNan, propagate_nan)))
        else:
            tl.store(C,
                     getattr(tl, func)(tl.load(A), tl.load(B), propagate_nan=getattr(tl.PropagateNan, propagate_nan)))

    for mode in ['A', 'B', 'both']:
        if func == 'clamp' and mode == 'B':
            # clamp does not guarantee propagation from 'min' and 'max' args
            continue
        A = torch.randn((1, ), device=device, dtype=getattr(torch, dtype))
        if mode == 'A' or mode == 'both': A[0] = torch.nan
        B = torch.randn((1, ), device=device, dtype=getattr(torch, dtype))
        if mode == 'B' or mode == 'both': B[0] = torch.nan
        C = torch.zeros_like(A, device=device, dtype=getattr(torch, dtype))
        kernel[(1, )](A, B, C, propagate_nan, func)

        if mode == 'both' or propagate_nan == 'ALL':
            assert torch.isnan(C[0])
        else:
            assert not torch.isnan(C[0])


# -----------------------
# test clamp
# -----------------------


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", ['float16', 'float32'])
def test_clamp(dtype, device):

    @triton.jit
    def kernel(x_ptr, min_ptr, max_ptr, out_ptr, ref_ptr, N, BLOCK_SIZE: tl.constexpr):

        off = tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask)
        min = tl.load(min_ptr + off, mask=mask)
        max = tl.load(max_ptr + off, mask=mask)
        out = out_ptr + off
        ref = ref_ptr + off

        tl.store(out, tl.clamp(x, min, max), mask=mask)
        ref_val = tl.minimum(tl.maximum(x, min), max)
        tl.store(ref, ref_val, mask=mask)

    size = 128

    x = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    a = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    b = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    min = torch.min(a, b)
    max = torch.max(a, b)
    out = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))
    ref = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))

    kernel[(size, )](x, min, max, out, ref, x.numel(), BLOCK_SIZE=size)

    torch.testing.assert_close(out, ref)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", ['float16', 'float32'])
def test_clamp_symmetric(dtype, device):

    @triton.jit
    def kernel(x_ptr, limit_ptr, out_ptr, ref_ptr, N, BLOCK_SIZE: tl.constexpr):

        off = tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask)
        limit = tl.load(limit_ptr + off, mask=mask)
        out = out_ptr + off
        ref = ref_ptr + off

        tl.store(out, tl.clamp(x, -limit, limit), mask=mask)
        ref_val = tl.minimum(tl.maximum(x, -limit), limit)
        tl.store(ref, ref_val, mask=mask)

    size = 128

    x = torch.randn((size, ), device=device, dtype=getattr(torch, dtype))
    limit = torch.randn((size, ), device=device, dtype=getattr(torch, dtype)).abs()
    out = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))
    ref = torch.zeros_like(x, device=device, dtype=getattr(torch, dtype))

    kernel[(size, )](x, limit, out, ref, x.numel(), BLOCK_SIZE=size)

    torch.testing.assert_close(out, ref)


# -----------------------
# test iterators
# -----------------------


@pytest.mark.interpreter
def test_static_range(device):

    @triton.jit
    def loop_kernel(Z, N: tl.constexpr, step: tl.constexpr):
        acc = 0
        for i in tl.static_range(0, N, step=step):
            acc += i
        tl.store(Z, acc)

    N = 100
    step = 7
    Out = torch.empty(1, dtype=torch.int32, device=device)
    loop_kernel[(1, )](Out, N, step)
    Acc = torch.tensor([0], dtype=torch.int32, device=device)
    for i in range(0, N, step):
        Acc += i
    assert (Out == Acc).all(), (Out, Acc)


@pytest.mark.interpreter
def test_temp_var_in_loop(device):

    @triton.jit
    def temp_in_loop(Z, N: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.full((BLOCK, ), 0, dtype=tl.int32)
        for i in range(N):
            if i == 0:
                temp = tl.full((BLOCK, ), 2, dtype=tl.int32)
                acc = temp
            else:
                acc += tl.full((BLOCK, ), 1, dtype=tl.int32)
            temp = tl.full((BLOCK, ), 1, dtype=tl.int32)
            acc += temp
        z = Z + tl.arange(0, BLOCK)
        tl.store(z, acc)

    N = 10
    BLOCK = 32
    out = torch.empty((BLOCK, ), dtype=torch.int32, device=device)
    temp_in_loop[(1, )](out, N, BLOCK)
    acc = torch.full((BLOCK, ), 0, dtype=torch.int32, device=device)
    for i in range(N):
        if i == 0:
            temp = torch.full((BLOCK, ), 2, dtype=torch.int32, device=device)
            acc = temp
        else:
            acc += torch.full((BLOCK, ), 1, dtype=torch.int32, device=device)
        temp = torch.full((BLOCK, ), 1, dtype=torch.int32, device=device)
        acc += temp
    assert (acc == out).all()


@pytest.mark.interpreter
def test_num_programs(device):
    # Assuming that the kernel is launched with a grid of (11, 21, 31)
    grid = (11, 21, 31)
    input = torch.empty((3, ), dtype=torch.int32, device=device)

    @triton.jit
    def kernel(input):
        num_programs_0 = tl.num_programs(0)
        num_programs_1 = tl.num_programs(1)
        num_programs_2 = tl.num_programs(2)
        tl.store(input, num_programs_0)
        tl.store(input + 1, num_programs_1)
        tl.store(input + 2, num_programs_2)

    kernel[grid](input)
    assert torch.all(input == torch.tensor(grid, device=device))


# -----------------------
# test extern functions
# -----------------------


@pytest.mark.parametrize("dtype_str", ['float32', 'float64'])
def test_math_extern(dtype_str, device):
    if is_interpreter():
        pytest.skip('math_extern does not work in the interpreter mode')

    @triton.jit
    def kernel(
        x_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = libdevice.tanh(x)
        tl.store(y_ptr + offsets, y, mask=mask)

    shape = (128, )
    rs = RandomState(17)

    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    y_ref = np.tanh(x)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(numpy_random(shape, dtype_str=dtype_str, rs=rs), device=device)
    kernel[(1, )](x_tri, y_tri, shape[0], BLOCK_SIZE=shape[0])
    # compare
    np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)


# -----------------------
# test loop unrolling
# -----------------------


def test_unroll_attr(device):

    @triton.jit
    def _kernel(dst, unroll_factor: tl.constexpr):
        pid = tl.program_id(axis=0)
        for i in tl.range(0, 10, loop_unroll_factor=unroll_factor):
            tl.atomic_add(dst + pid, i + pid)

    def check_loop_unroll_count(ir, opStr, loop_unroll_factor):
        for line in ir.splitlines():
            if opStr in line:
                loop_unroll_factor = loop_unroll_factor - 1
        # Sometimes we get a remainder loop
        assert loop_unroll_factor <= 0

    # Try for all different loop unroll factors:
    for unroll_factor in [1, 2, 4, 5, 8]:
        h = _kernel[(1, )](torch.empty(1, device=device), unroll_factor)
        check_loop_unroll_count(h.asm["ttir"], 'tt.atomic_rmw', unroll_factor)


@triton.jit
def sanitize_add(a, b):
    a64 = a.to(tl.int64)
    b64 = b.to(tl.int64)
    r64 = a64 + b64
    tl.device_assert((r64 >= -2**31) & (r64 <= 2**31 - 1))
    return a + b


def test_side_effectful_reduction(device):

    @triton.jit(debug=True)
    def sanitize_sum_kernel(Z, X, BLOCK: tl.constexpr):
        vals = tl.load(X + tl.arange(0, BLOCK))
        z = tl.reduce(vals, 0, sanitize_add)
        tl.store(Z, z)

    BLOCK = 512
    torch.manual_seed(42)
    X = torch.randint(0, 10, [BLOCK], device=device, dtype=torch.int32)
    X[:300] = 32
    X[300:] = 0
    Z = torch.zeros((), device=device, dtype=torch.int32)
    sanitize_sum_kernel[(1, )](Z, X, BLOCK=BLOCK)
    torch.testing.assert_close(Z, X.sum().to(torch.int32))


@pytest.mark.parametrize("reduce_dim", [0, 1])
def test_side_effectful_reduction_2d(device, reduce_dim):

    @triton.jit(debug=True)
    def sanitize_sum_2d_kernel(Z, X, BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, reduce_dim: tl.constexpr,
                               NON_REDUCE_DIM: tl.constexpr):
        offsets = tl.arange(0, BLOCK_0)[:, None] * BLOCK_1 + tl.arange(0, BLOCK_1)[None, :]
        vals = tl.load(X + offsets)
        z = tl.reduce(vals, reduce_dim, sanitize_add)
        tl.store(Z + tl.arange(0, NON_REDUCE_DIM), z)

    BLOCK_0 = 16
    BLOCK_1 = 32
    NON_REDUCE_DIM = BLOCK_1 if reduce_dim == 0 else BLOCK_0
    torch.manual_seed(42)
    X = torch.randint(0, 10, [BLOCK_0, BLOCK_1], device=device, dtype=torch.int32)
    Z = torch.zeros([NON_REDUCE_DIM], device=device, dtype=torch.int32)
    sanitize_sum_2d_kernel[(1, )](Z, X, BLOCK_0=BLOCK_0, BLOCK_1=BLOCK_1, reduce_dim=reduce_dim,
                                  NON_REDUCE_DIM=NON_REDUCE_DIM)
    torch.testing.assert_close(Z, X.sum(reduce_dim).to(torch.int32))


def test_side_effectful_scan(device):

    @triton.jit(debug=True)
    def sanitize_cumsum_kernel(Z, X, BLOCK: tl.constexpr):
        vals = tl.load(X + tl.arange(0, BLOCK))
        z = tl.associative_scan(vals, 0, sanitize_add)
        tl.store(Z + tl.arange(0, BLOCK), z)

    BLOCK = 512
    torch.manual_seed(42)
    X = torch.randint(0, 10, [BLOCK], device=device, dtype=torch.int32)
    X[:300] = 32
    X[300:] = 0
    Z = torch.zeros_like(X)
    sanitize_cumsum_kernel[(1, )](Z, X, BLOCK=BLOCK)
    torch.testing.assert_close(Z, X.cumsum(0).to(torch.int32))
