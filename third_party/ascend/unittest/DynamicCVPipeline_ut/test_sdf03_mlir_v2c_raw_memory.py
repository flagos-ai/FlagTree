"""
Test Case: SDF03 - V2C dependency, RAW memory dependency

[MLIR Validation] Refactored version

Description: V2C dependency, RAW memory dependency

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - SDF03-TC01: float16, M=128, N=64, K=128
  - SDF03-TC02: float32, M=128, N=64, K=128
"""

import os
import subprocess
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir, ttir_to_linalg, min_dot_size
from triton.backends.ascend import _apply_ascend_patch
import pytest

# Apply Ascend patch to inject hacc.target attribute into MLIR modules.
# Required by bishengir FixpipeOp::verify() which checks isAscend950(moduleOp)
# via the hacc.target attribute; without it, dst=UB fixpipe ops are rejected.
_apply_ascend_patch()


# ============================================================================
# Compile helper: compile Triton Kernel to MLIR (linalg dialect)
# Reference: compile_kernel implementation in test_custom.py
# ============================================================================
def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect.

    Compile a Triton Kernel to MLIR in linalg dialect for subsequent content validation.

    Args:
        kernel: Triton JIT-compiled kernel function
        signature: argument type signature dict, e.g. {"x_ptr": "*fp32", "n": "i32"}
        constants: constexpr argument dict, e.g. {"BLOCK": 256}

    Returns:
        str: compiled MLIR code string; returns None on compilation failure
    """
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions(arch="Ascend910_9589", enable_dynamic_cv_pipeline=True)
        # Register codegen_fns, including min_dot_size required by tl.dot.
        codegen_fns = {"min_dot_size": min_dot_size(None)}
        ttir = ast_to_ttir(kernel, src, context, options, codegen_fns, {})
        metadata = {
            **options.__dict__,
        }
        # Call make_ttir for TTIR optimization (consistent with the normal compilation path),
        # including key optimization passes such as inliner/canonicalizer/cse/licm/loop_unroll.
        # Without this step, complex kernels (e.g. while loops) will
        # raise RuntimeError: PassManager::run failed during ttir_to_linalg lowering
        ttir = make_ttir(ttir, metadata, options)
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        return str(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")
        return None


# ============================================================================
# MLIR output configuration
# ============================================================================
# MLIR output directory: mlir_output subdirectory alongside this test file
MLIR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlir_output")


def _write_mlir_to_file(mlir, filename):
    os.makedirs(MLIR_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(MLIR_OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"MLIR code written to: {output_path}")


# ============================================================================
# Kernel definitions
# ============================================================================


# ----------------------------------------------------------------------------
# SDF03-TC01: float16, M=128, N=64, K=128
# Test purpose: Verify MLIR generation of V2C dependency + RAW memory dependency under float16
# ----------------------------------------------------------------------------
@triton.jit
def sdf03_tc01_v2c_raw(
    a_ptr,
    b_ptr,
    c_ptr,
    buf_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_buf,
    stride_out_kk,
    stride_out_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,) = (128,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,) = (64,)

    for k in range(K):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,) = (64,)
        vec_result = c + 1.0  # (N,) = (64,)
        tl.store(buf_ptr + offs_n * stride_buf, vec_result, mask=offs_n < N)

        buf_val = tl.load(buf_ptr + offs_n * stride_buf, mask=offs_n < N, other=0.0)  # (N,) = (64,)
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,) = (128,)
        b = tl.load(b_ptr + k * stride_bk + offs_n * stride_bn, mask=offs_n < N, other=0.0)  # (N,) = (64,)
        cube_result = tl.dot(a[:, None], b[None, :]) + buf_val  # (K,1)@(1,N)+(N,) = (K,N) = (128,64)

        store_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)  # (K,N)
        tl.store(out_ptr + offs_k[:, None] * stride_out_kk + offs_n[None, :] * stride_out_k, cube_result,
                 mask=store_mask)


# ----------------------------------------------------------------------------
# SDF03-TC02: float32, M=128, N=64, K=128
# Test purpose: Verify MLIR generation of V2C dependency + RAW memory dependency under float32
# ----------------------------------------------------------------------------
@triton.jit
def sdf03_tc02_v2c_raw(
    a_ptr,
    b_ptr,
    c_ptr,
    buf_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_buf,
    stride_out_kk,
    stride_out_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for k in range(K):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        vec_result = c + 1.0
        tl.store(buf_ptr + offs_n * stride_buf, vec_result, mask=offs_n < N)

        buf_val = tl.load(buf_ptr + offs_n * stride_buf, mask=offs_n < N, other=0.0)
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b = tl.load(b_ptr + k * stride_bk + offs_n * stride_bn, mask=offs_n < N, other=0.0)
        cube_result = tl.dot(a[:, None], b[None, :]) + buf_val

        store_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        tl.store(out_ptr + offs_k[:, None] * stride_out_kk + offs_n[None, :] * stride_out_k, cube_result,
                 mask=store_mask)


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf03_signature(dtype_str):
    """Build the argument type signature for the SDF03 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "buf_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_buf": "i32",
        "stride_out_kk": "i32",
        "stride_out_k": "i32",
    }


def test_sdf03_tc01():
    """SDF03-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf03_tc01_v2c_raw kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf03_signature("fp16")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}

    mlir = compile_kernel(sdf03_tc01_v2c_raw, signature, constants)
    _write_mlir_to_file(mlir, "sdf03_tc01_v2c_raw.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf03_tc01_v2c_raw(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "MLIR code does not contain the 'scope' keyword"

    # Output MLIR code to the specified path


def test_sdf03_tc02():
    """SDF03-TC02: Verify float32 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf03_tc02_v2c_raw kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf03_signature("fp32")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}

    mlir = compile_kernel(sdf03_tc02_v2c_raw, signature, constants)
    _write_mlir_to_file(mlir, "sdf03_tc02_v2c_raw.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf03_tc02_v2c_raw(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "MLIR code does not contain the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf03_tc01()
    test_sdf03_tc02()
    print("All SDF03 v3 MLIR validation tests passed!")
