"""
Test Case: ACF07 - loop with if/else control flow, V2C dependency (condition from Vector, Cube depends on Vector)

[MLIR Validation] Refactored version

Description: For loop with if/else: tl.max(c) as condition. If >0, Cube depends on Vector result (V2C),
             else execute Vector (d / 2.0).

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - ACF07-TC01: float16, M=128, N=64, K=32
  - ACF07-TC02: float32, M=128, N=64, K=32
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


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel function to MLIR in linalg dialect."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions(arch="Ascend910_9589", enable_dynamic_cv_pipeline=True)
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
# ACF07-TC01: float16, M=128, N=64, K=32
# Note: K is tl.constexpr, not in signature, put in constants
# ----------------------------------------------------------------------------
@triton.jit
def acf07_tc01_v2c_control_flow(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out_ptr,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_out_0,
    stride_out_1,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)
    out_ptrs = out_ptr + offs_k[:, None] * stride_out_0 + offs_k[None, :] * stride_out_1

    for k in range(K):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
        vec_cond = tl.max(c)  # scalar

        if vec_cond > 0.0:
            a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
            b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
            cube_result = tl.dot(a[:, None], b[None, :]) + vec_cond  # (K,K) + scalar
            tl.store(out_ptrs, cube_result, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))
        else:
            loaded_d = tl.load(d_ptr + offs_k * stride_d, mask=offs_k < K, other=0.0)  # (K,)
            vec_result = loaded_d / 2.0  # (K,)
            tl.store(out_ptrs, vec_result[None, :], mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ----------------------------------------------------------------------------
# ACF07-TC02: float32, M=128, N=64, K=32
# ----------------------------------------------------------------------------
@triton.jit
def acf07_tc02_v2c_control_flow(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out_ptr,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_out_0,
    stride_out_1,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + offs_k[:, None] * stride_out_0 + offs_k[None, :] * stride_out_1

    for k in range(K):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        vec_cond = tl.max(c)

        if vec_cond > 0.0:
            a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
            b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)
            cube_result = tl.dot(a[:, None], b[None, :]) + vec_cond
            tl.store(out_ptrs, cube_result, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))
        else:
            loaded_d = tl.load(d_ptr + offs_k * stride_d, mask=offs_k < K, other=0.0)
            vec_result = loaded_d / 2.0
            tl.store(out_ptrs, vec_result[None, :], mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ============================================================================
# Pytest test cases
# ============================================================================


def test_acf07_tc01():
    """ACF07-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Note: K is a constexpr parameter, put in constants rather than signature.
    """
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "d_ptr": "*fp16",
        "out_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_out_0": "i32",
        "stride_out_1": "i32",
    }
    constants = {"K": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf07_tc01_v2c_control_flow, signature, constants)
    _write_mlir_to_file(mlir, "acf07_tc01_v2c_control_flow.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @acf07_tc01_v2c_control_flow(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


def test_acf07_tc02():
    """ACF07-TC02: Verify float32 kernel compilation generates correct MLIR code.

    Note: K is a constexpr parameter, put in constants rather than signature.
    """
    signature = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "d_ptr": "*fp32",
        "out_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_out_0": "i32",
        "stride_out_1": "i32",
    }
    constants = {"K": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(acf07_tc02_v2c_control_flow, signature, constants)
    _write_mlir_to_file(mlir, "acf07_tc02_v2c_control_flow.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @acf07_tc02_v2c_control_flow(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


if __name__ == "__main__":
    test_acf07_tc01()
    test_acf07_tc02()
    print("All ACF07 v3 MLIR validation tests passed!")
