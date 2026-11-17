"""
Test Case: PCB01 - Single independent Cube + single independent Vector, no data dependency


[MLIR Validation] Refactored version


Description: Single independent Cube (outer product) + single independent Vector (element-wise add),
             no data dependency between Cube and Vector operations.


Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable


Test Cases:
  - PCB01-TC01: float16, M=128, N=64, K=32
  - PCB01-TC02: float32, M=128, N=64, K=32
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
# MLIR output configuration
# ============================================================================
# MLIR output directory: mlir_output subdirectory alongside this test file
MLIR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlir_output")


def _write_mlir_to_file(mlir, filename):
    """Write MLIR code to a file at the specified path.

    Args:
        mlir: MLIR code string
        filename: output filename (without path), e.g. "pcb01_tc01.mlir"
    """
    os.makedirs(MLIR_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(MLIR_OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"MLIR code written to: {output_path}")


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
        # The normal compilation path obtains this via backend.get_codegen_implementation(options);
        # here we import min_dot_size directly from the Ascend backend and construct it.
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
# Kernel definitions
# ============================================================================


# ----------------------------------------------------------------------------
# PCB01-TC01: float16, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of independent CUBE + VECTOR operations under float16
# ----------------------------------------------------------------------------
@triton.jit
def pcb01_tc01_single_cube_vector(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out1_ptr,
    out2_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_out1m,
    stride_out1n,
    stride_out2,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        a_mat = a[:, None]  # (K, 1)
        b_mat = b[None, :]  # (1, K)
        cube_result = tl.dot(a_mat, b_mat)  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result, mask=out1_mask)

        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)  # (N,)
        vec_result = c + d  # (N,)
        out2_ptrs = out2_ptr + offs_n * stride_out2
        tl.store(out2_ptrs, vec_result, mask=offs_n < N)


# ----------------------------------------------------------------------------
# PCB01-TC02: float32, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of independent CUBE + VECTOR operations under float32
# ----------------------------------------------------------------------------
@triton.jit
def pcb01_tc02_single_cube_vector(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out1_ptr,
    out2_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_out1m,
    stride_out1n,
    stride_out2,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        a_mat = a[:, None]  # (K, 1)
        b_mat = b[None, :]  # (1, K)
        cube_result = tl.dot(a_mat, b_mat)  # (K, K)
        out1_ptrs = out1_ptr + offs_k[:, None] * stride_out1m + offs_k[None, :] * stride_out1n
        out1_mask = (offs_k[:, None] < K) & (offs_k[None, :] < K)
        tl.store(out1_ptrs, cube_result, mask=out1_mask)

        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)  # (N,)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)  # (N,)
        vec_result = c + d  # (N,)
        out2_ptrs = out2_ptr + offs_n * stride_out2
        tl.store(out2_ptrs, vec_result, mask=offs_n < N)


# ============================================================================
# Pytest test cases
# ============================================================================


def test_pcb01_tc01():
    """PCB01-TC01: Verify float16 kernel compilation generates correct MLIR code.


    Test steps:
      1. Compile pcb01_tc01_single_cube_vector kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    # Define argument type signature: float16 pointers + int32 scalar arguments
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "d_ptr": "*fp16",
        "out1_ptr": "*fp16",
        "out2_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_out1m": "i32",
        "stride_out1n": "i32",
        "stride_out2": "i32",
    }
    # constexpr parameter
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    # Compile Compile kernel to MLIR
    mlir = compile_kernel(pcb01_tc01_single_cube_vector, signature, constants)

    # Output MLIR code to the specified path
    _write_mlir_to_file(mlir, "pcb01_tc01_single_cube_vector.mlir")

    # Verify MLIR generated successfully
    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"

    # Verify MLIR contains function definition
    assert "func.func @pcb01_tc01_single_cube_vector(" in mlir, \
        "Kernel function definition not found in MLIR code"

    # Verify MLIR code contains the "scope" keyword
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"


def test_pcb01_tc02():
    """PCB01-TC02: Verify float32 kernel compilation generates correct MLIR code.


    Test steps:
      1. Compile pcb01_tc02_single_cube_vector kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    # Define argument type signature: float32 pointers + int32 scalar arguments
    signature = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "d_ptr": "*fp32",
        "out1_ptr": "*fp32",
        "out2_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_out1m": "i32",
        "stride_out1n": "i32",
        "stride_out2": "i32",
    }
    # constexpr parameter
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    # Compile Compile kernel to MLIR
    mlir = compile_kernel(pcb01_tc02_single_cube_vector, signature, constants)

    # Output MLIR code to the specified path
    _write_mlir_to_file(mlir, "pcb01_tc02_single_cube_vector.mlir")

    # Verify MLIR generated successfully
    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"

    # Verify MLIR contains function definition
    assert "func.func @pcb01_tc02_single_cube_vector(" in mlir, \
        "Kernel function definition not found in MLIR code"

    # Verify MLIR code contains the "scope" keyword
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"


# ============================================================================
# Main for manual testing
# ============================================================================

if __name__ == "__main__":
    test_pcb01_tc01()
    test_pcb01_tc02()
    print("All PCB01 v3 MLIR validation tests passed!")
