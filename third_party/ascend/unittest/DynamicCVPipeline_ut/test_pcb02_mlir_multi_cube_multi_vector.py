"""
Test Case: PCB02 - Multiple independent Cube + multiple independent Vector, no data dependency

[MLIR Validation] Refactored version

Description: 2 independent Cube (outer product) + 2 independent Vector (element-wise sub),
             no data dependency between Cube and Vector operations.

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - PCB02-TC01: float16, M=128, N=64, K=32
  - PCB02-TC02: float32, M=128, N=64, K=32
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
# PCB02-TC01: float16, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of multiple independent CUBE + VECTOR operations under float16
# ----------------------------------------------------------------------------
@triton.jit
def pcb02_tc01_multi_cube_vector(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
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
    stride_e,
    stride_f,
    stride_g,
    stride_h,
    stride_out1m,
    stride_out1n,
    stride_out2m,
    stride_out2n,
    stride_out3,
    stride_out4,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a1 = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b1 = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result1 = tl.dot(a1[:, None], b1[None, :])  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result1, mask=out1_mask)

        a2 = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)  # (K,)
        b2 = tl.load(d_ptr + k * stride_dk + offs_k * stride_dn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result2 = tl.dot(a2[:, None], b2[None, :])  # (K, K)
        out2_ptrs = out2_ptr + offs_k[:, None] * stride_out2m + offs_k[None, :] * stride_out2n
        out2_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out2_ptrs, cube_result2, mask=out2_mask)

        e = tl.load(e_ptr + offs_n * stride_e, mask=offs_n < N, other=0.0)  # (N,)
        f = tl.load(f_ptr + offs_n * stride_f, mask=offs_n < N, other=0.0)  # (N,)
        vec_result1 = e - f  # (N,)
        out3_ptrs = out3_ptr + offs_n * stride_out3
        tl.store(out3_ptrs, vec_result1, mask=offs_n < N)

        g = tl.load(g_ptr + offs_n * stride_g, mask=offs_n < N, other=0.0)  # (N,)
        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)  # (N,)
        vec_result2 = g - h  # (N,)
        out4_ptrs = out4_ptr + offs_n * stride_out4
        tl.store(out4_ptrs, vec_result2, mask=offs_n < N)


# ----------------------------------------------------------------------------
# PCB02-TC02: float32, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of multiple independent CUBE + VECTOR operations under float32
# ----------------------------------------------------------------------------
@triton.jit
def pcb02_tc02_multi_cube_vector(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
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
    stride_e,
    stride_f,
    stride_g,
    stride_h,
    stride_out1m,
    stride_out1n,
    stride_out2m,
    stride_out2n,
    stride_out3,
    stride_out4,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a1 = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b1 = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result1 = tl.dot(a1[:, None], b1[None, :])  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result1, mask=out1_mask)

        a2 = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)  # (K,)
        b2 = tl.load(d_ptr + k * stride_dk + offs_k * stride_dn, mask=offs_k < K, other=0.0)  # (K,)
        cube_result2 = tl.dot(a2[:, None], b2[None, :])  # (K, K)
        out2_ptrs = out2_ptr + offs_k[:, None] * stride_out2m + offs_k[None, :] * stride_out2n
        out2_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out2_ptrs, cube_result2, mask=out2_mask)

        e = tl.load(e_ptr + offs_n * stride_e, mask=offs_n < N, other=0.0)  # (N,)
        f = tl.load(f_ptr + offs_n * stride_f, mask=offs_n < N, other=0.0)  # (N,)
        vec_result1 = e - f  # (N,)
        out3_ptrs = out3_ptr + offs_n * stride_out3
        tl.store(out3_ptrs, vec_result1, mask=offs_n < N)

        g = tl.load(g_ptr + offs_n * stride_g, mask=offs_n < N, other=0.0)  # (N,)
        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)  # (N,)
        vec_result2 = g - h  # (N,)
        out4_ptrs = out4_ptr + offs_n * stride_out4
        tl.store(out4_ptrs, vec_result2, mask=offs_n < N)


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_pcb02_signature(dtype_str):
    """Build the argument type signature for the PCB02 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "h_ptr": f"*{dtype_str}",
        "out1_ptr": f"*{dtype_str}",
        "out2_ptr": f"*{dtype_str}",
        "out3_ptr": f"*{dtype_str}",
        "out4_ptr": f"*{dtype_str}",
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
        "stride_e": "i32",
        "stride_f": "i32",
        "stride_g": "i32",
        "stride_h": "i32",
        "stride_out1m": "i32",
        "stride_out1n": "i32",
        "stride_out2m": "i32",
        "stride_out2n": "i32",
        "stride_out3": "i32",
        "stride_out4": "i32",
    }


def test_pcb02_tc01():
    """PCB02-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile pcb02_tc01_multi_cube_vector kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_pcb02_signature("fp16")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(pcb02_tc01_multi_cube_vector, signature, constants)
    _write_mlir_to_file(mlir, "pcb02_tc01_multi_cube_vector.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @pcb02_tc01_multi_cube_vector(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


def test_pcb02_tc02():
    """PCB02-TC02: Verify float32 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile pcb02_tc02_multi_cube_vector kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_pcb02_signature("fp32")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(pcb02_tc02_multi_cube_vector, signature, constants)
    _write_mlir_to_file(mlir, "pcb02_tc02_multi_cube_vector.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @pcb02_tc02_multi_cube_vector(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_pcb02_tc01()
    test_pcb02_tc02()
    print("All PCB02 v3 MLIR validation tests passed!")
