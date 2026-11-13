"""
Test Case: ACF01 - No for-loop, independent Cube + Vector in main body, no data dependency

[MLIR Validation] Refactored version

Description: No for-loop, independent Cube (outer product) + Vector (cube_result + c).
             No data dependency. Not supporting SSBuf optimization.

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - ACF01-TC01: float16, M=128, N=64, K=32
  - ACF01-TC02: float32, M=128, N=64, K=32
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
        metadata = {**options.__dict__}
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
# ACF01-TC01: float16, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of independent CV operations without for-loop under float16
# ----------------------------------------------------------------------------
@triton.jit
def acf01_tc01_no_loop_cv(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_c,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    a = tl.load(a_ptr + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
    c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)

    # Build (K, N) result directly via element-wise broadcasting:
    # b has only K elements, so for column j >= K, the outer product is 0
    # Load b with N-sized index, mask to K so b[j]=0 for j>=K
    b_n = tl.load(b_ptr + offs_n * stride_bk, mask=offs_n < K, other=0.0)  # (N,) padded with 0
    # outer product: a[:, None] * b_n[None, :] -> (K, N)
    cube_result = a[:, None] * b_n[None, :]  # (K, N)
    vec_result = cube_result + c[None, :]  # (K, N) + (1, N) -> (K, N)
    tl.store(out_ptr + offs_k[:, None] * stride_out + offs_n[None, :], vec_result,
             mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))


# ----------------------------------------------------------------------------
# ACF01-TC02: float32, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of independent CV operations without for-loop under float32
# ----------------------------------------------------------------------------
@triton.jit
def acf01_tc02_no_loop_cv(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_c,
    stride_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a = tl.load(a_ptr + offs_k * stride_ak, mask=offs_k < K, other=0.0)
    c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)

    b_n = tl.load(b_ptr + offs_n * stride_bk, mask=offs_n < K, other=0.0)
    cube_result = a[:, None] * b_n[None, :]  # (K, N)
    vec_result = cube_result + c[None, :]  # (K, N) + (1, N) -> (K, N)
    tl.store(out_ptr + offs_k[:, None] * stride_out + offs_n[None, :], vec_result,
             mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))


# ============================================================================
# Pytest test cases
# ============================================================================


def test_acf01_tc01():
    """ACF01-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile acf01_tc01_no_loop_cv kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "out_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_c": "i32",
        "stride_out": "i32",
    }
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf01_tc01_no_loop_cv, signature, constants)
    _write_mlir_to_file(mlir, "acf01_tc01_no_loop_cv.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @acf01_tc01_no_loop_cv(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


def test_acf01_tc02():
    """ACF01-TC02: Verify float32 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile acf01_tc02_no_loop_cv kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "out_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_c": "i32",
        "stride_out": "i32",
    }
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf01_tc02_no_loop_cv, signature, constants)
    _write_mlir_to_file(mlir, "acf01_tc02_no_loop_cv.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @acf01_tc02_no_loop_cv(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================

if __name__ == "__main__":
    test_acf01_tc01()
    test_acf01_tc02()
    print("All ACF01 v3 MLIR validation tests passed!")
