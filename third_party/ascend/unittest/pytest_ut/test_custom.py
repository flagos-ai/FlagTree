#!/usr/bin/env python3
import subprocess
import os
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg
import pytest


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions()
        ttir = ast_to_ttir(kernel, src, context, options, {}, {})
        metadata = {
            **options.__dict__,
        }
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        return str(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")
        return None


# ============== Kernel definitions ==============


@al.register_custom_op
class my_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "my_custom_func"
    # fake path, this test only check Triton successfully lowered to MLIR
    bitcode = os.path.abspath(__file__)
    iterator_types = [
        al.IteratorType.Parallel,
        al.IteratorType.Broadcast,
        al.IteratorType.Transpose,
        al.IteratorType.Reduction,
        al.IteratorType.Interleave,
        al.IteratorType.Deinterleave,
        al.IteratorType.Inverse,
        al.IteratorType.Pad,
        al.IteratorType.Concat,
        al.IteratorType.Gather,
        al.IteratorType.Cumulative,
        al.IteratorType.Opaque,
    ]

    def __init__(self, x, ptr1, ptr2, offset: tl.int64, other, out=None):
        # Add optional custom-op attribute: ArrayAttr<AffineMapAttr>.
        self.indexing_map = [al.affine_map.get_identity(1)]

        # Tag ptr2 as an argument that should be aligned at dimension 1.
        # Tag 2nd argument that should be aligned at dimension 0.
        self.align_dim = {"ptr2": 1, 1: 0}


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    result = al.custom("my_custom_op", x, x_ptr, y_ptr + i, (1, 2, 3), [4.1, 5.2], out=y)
    a = 123
    result = al.custom("my_custom_op", x, x_ptr, y_ptr, (a, n), (1.2, 3.4), out=result)
    tl.store(out_ptr + i, result, mask=i < n)


@al.register_custom_op
class my_custom_op_extra_buf:
    """Custom op declaring extra_buffers with several scalar Triton dtypes."""

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "my_extra_buf_func"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, out=None):
        self.indexing_map = [al.affine_map.get_identity(1)]
        self.extra_buffers = [
            (tl.bfloat16, 256),
            (tl.float64, 424242),
            (tl.int8, 11),
            (tl.float16, 22),
            (tl.int32, 33),
        ]


@al.register_custom_op
class my_custom_op_extra_buf_single_buf:
    """Custom op declaring extra_buffers with single buf."""

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "my_extra_buf_func_single_buf"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, out=None):
        self.indexing_map = [al.affine_map.get_identity(1)]
        self.extra_buffers = (tl.bfloat16, 256)


@triton.jit
def kernel_extra_buf(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(out_ptr + i, mask=i < n)
    r = al.custom("my_custom_op_extra_buf", x, out=y)
    tl.store(out_ptr + i, r, mask=i < n)


@triton.jit
def kernel_extra_buf_single_buf(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(out_ptr + i, mask=i < n)
    r = al.custom("my_custom_op_extra_buf_single_buf", x, out=y)
    tl.store(out_ptr + i, r, mask=i < n)


@al.register_custom_op
class my_custom_op_extra_buf_wide:
    """Cover more integer widths and unsigned dtypes in extra_buffers_types."""

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "my_extra_buf_wide_func"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, out=None):
        self.indexing_map = [al.affine_map.get_identity(1)]
        self.extra_buffers = [
            (tl.int16, 1001),
            (tl.uint16, 1002),
            (tl.int64, 1003),
            (tl.uint32, 1004),
            (tl.uint8, 1005),
        ]


@triton.jit
def kernel_extra_buf_wide(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(out_ptr + i, mask=i < n)
    r = al.custom("my_custom_op_extra_buf_wide", x, out=y)
    tl.store(out_ptr + i, r, mask=i < n)


# ============== Pytest tests ==============


def test_custom_op():
    """Test custom op compile to linalg MLIR."""
    mlir = compile_kernel(my_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    assert mlir and len(mlir) > 0
    assert "func.func @my_kernel(" in mlir
    assert "hivm.hir.custom" in mlir
    for line in mlir.splitlines():
        if "hivm.hir.custom" in line:
            # custom op name
            assert '"my_custom_op"' in line
            # All tt.ptr converted to memref.
            assert "tt.ptr" not in line
            # Required attributes are set.
            assert "hivm.pipe = #hivm.pipe" in line
            assert "hivm.tcore_type = #hivm.tcore_type" in line
            assert "hivm.vf_mode = #hivm.vf_mode" in line
            # Optional indexing map attribute should be attached.
            assert "indexing_map = [" in line
            # Tagged argument alignment info is attached as integer operand attr.
            assert "align_dim = 1" in line
            assert "align_dim = 0" in line
            # All offset converted to int64.
            assert 'i64, ' in line
            assert 'i32, ' not in line
            assert "iterator_types" in line
            for iterator_name in (
                    "parallel",
                    "broadcast",
                    "transpose",
                    "reduction",
                    "interleave",
                    "deinterleave",
                    "inverse",
                    "pad",
                    "concat",
                    "gather",
                    "cumulative",
                    "opaque",
            ):
                assert iterator_name in line


def _custom_lines(mlir: str, op_name: str):
    # Match the MLIR string attribute exactly (avoid `my_custom_op` matching
    # `my_custom_op_extra_buf`).
    quoted = f'"{op_name}"'
    return [line for line in mlir.splitlines() if "hivm.hir.custom" in line and quoted in line]


def test_custom_op_extra_buffers_mixed_scalar_types():
    """extra_buffers_types must preserve bf16/f64/i8/f16/i32 (not all lowered to f32)."""
    mlir = compile_kernel(
        kernel_extra_buf,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert mlir and len(mlir) > 0
    lines = _custom_lines(mlir, "my_custom_op_extra_buf")
    assert lines, "expected at least one hivm.hir.custom line for my_custom_op_extra_buf"
    line = lines[0]
    assert "extra_buffers_types" in line
    assert "extra_buffers_sizes" in line
    assert "bf16" in line
    assert "f64" in line
    assert "i8" in line
    assert "f16" in line
    assert "i32" in line
    assert "424242" in line


def test_custom_op_extra_buffers_single_buffer():
    mlir = compile_kernel(
        kernel_extra_buf_single_buf,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert mlir and len(mlir) > 0
    lines = _custom_lines(mlir, "my_custom_op_extra_buf_single_buf")
    assert lines, "expected at least one hivm.hir.custom line for my_custom_op_extra_buf_single_buf"
    line = lines[0]
    assert "extra_buffers_types" in line
    assert "extra_buffers_sizes" in line
    assert "f32" in line


def test_custom_op_extra_buffers_integer_variants():
    """extra_buffers accept int16/uint16/int64/uint32/uint8 (IR uses i* storage types)."""
    mlir = compile_kernel(
        kernel_extra_buf_wide,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert mlir and len(mlir) > 0
    lines = _custom_lines(mlir, "my_custom_op_extra_buf_wide")
    assert lines
    line = lines[0]
    assert "extra_buffers_types" in line
    assert "extra_buffers_sizes" in line
    assert "i16" in line
    assert "i64" in line
    assert "i32" in line
    assert "i8" in line
    assert "1001" in line and "1005" in line


def test_custom_op_without_extra_buffers_has_no_extra_buffer_attrs():
    """Ops that do not set extra_buffers should not emit extra_buffers_* attributes."""
    mlir = compile_kernel(
        my_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert mlir
    for line in _custom_lines(mlir, "my_custom_op"):
        assert "extra_buffers_types" not in line
        assert "extra_buffers_sizes" not in line


# ============== Main for manual testing ==============

if __name__ == "__main__":
    test_custom_op()
    test_custom_op_without_extra_buffers_has_no_extra_buffer_attrs()
    test_custom_op_extra_buffers_integer_variants()
    test_custom_op_extra_buffers_mixed_scalar_types()
    test_custom_op_extra_buffers_single_buffer()
    mlir = compile_kernel(my_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
