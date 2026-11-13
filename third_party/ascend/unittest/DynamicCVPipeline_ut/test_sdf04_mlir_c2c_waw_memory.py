"""
Test Case: SDF04 - C2C dependency, WAW memory dependency

[MLIR Validation] Refactored version

Description: C2C dependency, WAW memory dependency

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - SDF04-TC01: float16, M=128, N=64, K=128
  - SDF04-TC02: float32, M=128, N=64, K=128
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
# SDF04-TC01: float16, M=128, N=64, K=128
# Test purpose: Verify MLIR generation of C2C dependency + WAW memory dependency under float16
# ----------------------------------------------------------------------------
@triton.jit
def sdf04_tc01_c2c_waw(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    buf_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_ck,
    stride_dk,
    stride_dn,
    stride_buf_kk,
    stride_buf_k,
    stride_out_kk,
    stride_out_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,) = (128,)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,) = (128,)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,) = (128,)
        cube1_result = tl.dot(a[:, None], b[None, :])  # (K,K) = (128,128)

        store_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(buf_ptr + offs_k[:, None] * stride_buf_kk + offs_k[None, :] * stride_buf_k, cube1_result,
                 mask=store_mask)

        c = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)  # (K,) = (128,)
        d = tl.load(d_ptr + k * stride_dk + offs_k * stride_dn, mask=offs_k < K, other=0.0)  # (K,) = (128,)
        cube2_result = tl.dot(c[:, None], d[None, :])  # (K,K) = (128,128)

        tl.store(buf_ptr + offs_k[:, None] * stride_buf_kk + offs_k[None, :] * stride_buf_k, cube2_result,
                 mask=store_mask)
        tl.store(out_ptr + offs_k[:, None] * stride_out_kk + offs_k[None, :] * stride_out_k, cube2_result,
                 mask=store_mask)


# ----------------------------------------------------------------------------
# SDF04-TC02: float32, M=128, N=64, K=128
# Test purpose: Verify MLIR generation of C2C dependency + WAW memory dependency under float32
# ----------------------------------------------------------------------------
@triton.jit
def sdf04_tc02_c2c_waw(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    buf_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_ck,
    stride_dk,
    stride_dn,
    stride_buf_kk,
    stride_buf_k,
    stride_out_kk,
    stride_out_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)
        cube1_result = tl.dot(a[:, None], b[None, :])

        store_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(buf_ptr + offs_k[:, None] * stride_buf_kk + offs_k[None, :] * stride_buf_k, cube1_result,
                 mask=store_mask)

        c = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)
        d = tl.load(d_ptr + k * stride_dk + offs_k * stride_dn, mask=offs_k < K, other=0.0)
        cube2_result = tl.dot(c[:, None], d[None, :])

        tl.store(buf_ptr + offs_k[:, None] * stride_buf_kk + offs_k[None, :] * stride_buf_k, cube2_result,
                 mask=store_mask)
        tl.store(out_ptr + offs_k[:, None] * stride_out_kk + offs_k[None, :] * stride_out_k, cube2_result,
                 mask=store_mask)


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf04_signature(dtype_str):
    """Build the argument type signature for the SDF04 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "buf_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_ck": "i32",
        "stride_dk": "i32",
        "stride_dn": "i32",
        "stride_buf_kk": "i32",
        "stride_buf_k": "i32",
        "stride_out_kk": "i32",
        "stride_out_k": "i32",
    }


def test_sdf04_tc01():
    """SDF04-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf04_tc01_c2c_waw kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf04_signature("fp16")
    constants = {"BLOCK_SIZE_K": 128}

    mlir = compile_kernel(sdf04_tc01_c2c_waw, signature, constants)
    _write_mlir_to_file(mlir, "sdf04_tc01_c2c_waw.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf04_tc01_c2c_waw(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" in mlir, "MLIR code does not contain the 'scope' keyword"

    # Output MLIR code to the specified path


def test_sdf04_tc02():
    """SDF04-TC02: Verify float32 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf04_tc02_c2c_waw kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf04_signature("fp32")
    constants = {"BLOCK_SIZE_K": 128}

    mlir = compile_kernel(sdf04_tc02_c2c_waw, signature, constants)
    _write_mlir_to_file(mlir, "sdf04_tc02_c2c_waw.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf04_tc02_c2c_waw(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" in mlir, "MLIR code does not contain the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf04_tc01()
    test_sdf04_tc02()
    print("All SDF04 v3 MLIR validation tests passed!")
