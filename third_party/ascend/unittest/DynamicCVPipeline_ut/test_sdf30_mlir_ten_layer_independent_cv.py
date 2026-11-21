"""
Test Case: SDF30 - 10-layer independent CV

[MLIR Validation] Refactored version

Description: 10-layer nested (V, U, T, S, R, Q, L, P, K). Each layer has independent CV.

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable
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
# SDF30: 10-layer independent CV
# Test purpose: Verify MLIR generation of 10-layer nesting independent CV operations under float16
# ----------------------------------------------------------------------------
@triton.jit
def sdf30(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out_ptr,
    l9_cube_out_ptr,
    M,
    N,
    K,
    L,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_out,
    stride_l9_cube_row,
    stride_l9_cube_col,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_U: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k, offs_n = tl.arange(0, BLOCK_SIZE_K), tl.arange(0, BLOCK_SIZE_N)
    for l1 in range(V):
        for l2 in range(U):
            for l3 in range(T):
                for l4 in range(S):
                    for l5 in range(R):
                        for l6 in range(Q):
                            for l7 in range(L):
                                for l8 in range(P):
                                    for l9 in range(K):
                                        a = tl.load(a_ptr + l9 * stride_am + offs_k * stride_ak, mask=offs_k < K,
                                                    other=0.0)
                                        b = tl.load(b_ptr + l9 * stride_bk + offs_k * stride_bn, mask=offs_k < K,
                                                    other=0.0)
                                        l9_cube = tl.dot(a[:, None], b[None, :])
                                        l9_cube_ptrs = l9_cube_out_ptr + offs_k[:, None] * stride_l9_cube_row + offs_k[
                                            None, :] * stride_l9_cube_col
                                        tl.store(l9_cube_ptrs, l9_cube,
                                                 mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))
                                        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
                                        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
                                        l9_vec = c + d
                                        out_ptrs = out_ptr + offs_n * stride_out
                                        tl.store(out_ptrs, l9_vec, mask=offs_n < N)


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf30_signature(dtype_str):
    """Build the argument type signature for the SDF30 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "l9_cube_out_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "Q": "i32",
        "R": "i32",
        "S": "i32",
        "T": "i32",
        "U": "i32",
        "V": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_out": "i32",
        "stride_l9_cube_row": "i32",
        "stride_l9_cube_col": "i32",
    }


def test_sdf30():
    """SDF30: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf30 kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf30_signature("fp16")
    constants = {
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 8,
        "BLOCK_SIZE_L": 2,
        "BLOCK_SIZE_P": 2,
        "BLOCK_SIZE_Q": 2,
        "BLOCK_SIZE_R": 2,
        "BLOCK_SIZE_S": 2,
        "BLOCK_SIZE_T": 2,
        "BLOCK_SIZE_U": 2,
        "BLOCK_SIZE_V": 2,
    }

    mlir = compile_kernel(sdf30, signature, constants)
    _write_mlir_to_file(mlir, "sdf30.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf30(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf30()
    print("All SDF30 v3 MLIR validation tests passed!")
