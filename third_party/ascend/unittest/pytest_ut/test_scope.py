#!/usr/bin/env python3
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


# ============== Kernel definitions ==============


@triton.jit
def kernel_nested_scope(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test nested scopes."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        with al.scope(core_mode="vector"):
            with al.scope(core_mode="cube"):
                x = tl.load(x_ptr + i, mask=i < n)
                y = tl.load(y_ptr + i, mask=i < n)
                result = x + y
                tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_escape(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test variable defined inside scope, used outside."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
    # Use x outside of the scope
    a = x + 1.0
    tl.store(out_ptr + i, a, mask=i < n)


@triton.jit
def kernel_scope_cube(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test cube core mode."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="cube"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_vector(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test vector core mode."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_disable_auto_sync(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test disable auto sync."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


# ============== Pytest tests ==============


def test_nested_scope():
    """Test nested scopes compile successfully."""
    mlir = compile_kernel(kernel_nested_scope, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    assert "scope.scope" in mlir
    assert len(mlir) > 0


def test_scope_escape():
    """Test variable escaping from scope."""
    mlir = compile_kernel(kernel_scope_escape, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
    assert "scope.scope" in mlir
    assert len(mlir) > 0


def test_scope_cube_mode():
    """Test cube core mode generates correct attributes."""
    mlir = compile_kernel(kernel_scope_cube, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    assert "scope.scope" in mlir
    # Check for cube core type attribute
    assert "hivm.tcore_type" in mlir or "CUBE" in mlir.upper()


def test_scope_vector_mode():
    """Test vector core mode generates correct attributes."""
    mlir = compile_kernel(kernel_scope_vector, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    assert "scope.scope" in mlir
    # Check for vector core type attribute
    assert "hivm.tcore_type" in mlir or "VECTOR" in mlir.upper()


def test_scope_disable_auto_sync():
    """Test disable auto sync generates correct attributes."""
    mlir = compile_kernel(
        kernel_scope_disable_auto_sync,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert "scope.scope" in mlir
    # Check for disable auto sync attribute
    assert "hivm.disable_auto_sync" in mlir


# ============== Main for manual testing ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Nested Scopes")
    print("=" * 60)
    mlir = compile_kernel(kernel_nested_scope, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                          {"BLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)

    print("\n" + "=" * 60)
    print("Test 2: Scope Escape")
    print("=" * 60)
    mlir = compile_kernel(kernel_scope_escape, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
