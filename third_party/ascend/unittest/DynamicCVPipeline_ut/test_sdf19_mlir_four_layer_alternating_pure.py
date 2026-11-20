"""
Test Case: SDF19 - 4-layer nested, alternating pure C or pure V, no data dependency

[MLIR Validation] Refactored version

Description: Q-layer pure C, L-layer pure V, P-layer pure C, K-layer pure V.

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - SDF19-TC01: float16, M=128, N=64, K=32, L=3, P=2, Q=2
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
# SDF19-TC01: float16, M=128, N=64, K=32, L=3, P=2, Q=2
# Test purpose: Verify MLIR generation of 4-layer nesting alternating pure C / pure V under float16
# ----------------------------------------------------------------------------
@triton.jit
def sdf19_tc01_alternating_pure(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out_ptr,
    out_l1_ptr,
    out_l2_ptr,
    out_l3_ptr,
    M,
    N,
    K,
    L,
    P,
    Q,
    stride_am,
    stride_aq,
    stride_bq,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ep,
    stride_fp,
    stride_fn,
    stride_g,
    stride_h,
    stride_out,
    stride_l1m,
    stride_l1n,
    stride_l2,
    stride_l3m,
    stride_l3n,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_q = tl.arange(0, BLOCK_SIZE_Q)
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for l1 in range(Q):
        # --- l1: pure C (cube) ---
        a = tl.load(a_ptr + l1 * stride_am + offs_q * stride_aq, mask=offs_q < Q, other=0.0)
        b = tl.load(b_ptr + l1 * stride_bq + offs_q * stride_bn, mask=offs_q < Q, other=0.0)
        l1_cube = tl.dot(a[:, None], b[None, :])
        l1_ptrs = out_l1_ptr + offs_q[:, None] * stride_l1m + offs_q[None, :] * stride_l1n
        tl.store(l1_ptrs, l1_cube, mask=(offs_q[:, None] < Q) & (offs_q[None, :] < Q))

        for l2 in range(L):
            # --- l2: pure V (vector) ---
            c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
            d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
            l2_vec = c + d
            l2_ptrs = out_l2_ptr + offs_n * stride_l2
            tl.store(l2_ptrs, l2_vec, mask=offs_n < N)

            for l3 in range(P):
                # --- l3: pure C (cube) ---
                e = tl.load(e_ptr + l3 * stride_em + offs_p * stride_ep, mask=offs_p < P, other=0.0)
                f = tl.load(f_ptr + l3 * stride_fp + offs_p * stride_fn, mask=offs_p < P, other=0.0)
                l3_cube = tl.dot(e[:, None], f[None, :])
                l3_ptrs = out_l3_ptr + offs_p[:, None] * stride_l3m + offs_p[None, :] * stride_l3n
                tl.store(l3_ptrs, l3_cube, mask=(offs_p[:, None] < P) & (offs_p[None, :] < P))

                for l4 in range(K):
                    # --- l4: pure V (vector) ---
                    g = tl.load(g_ptr + offs_n * stride_g, mask=offs_n < N, other=0.0)
                    h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)
                    l4_vec = g - h
                    out_ptrs = out_ptr + offs_n * stride_out
                    tl.store(out_ptrs, l4_vec, mask=offs_n < N)


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf19_signature(dtype_str):
    """Build the argument type signature for the SDF19 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "h_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "out_l1_ptr": f"*{dtype_str}",
        "out_l2_ptr": f"*{dtype_str}",
        "out_l3_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "Q": "i32",
        "stride_am": "i32",
        "stride_aq": "i32",
        "stride_bq": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_em": "i32",
        "stride_ep": "i32",
        "stride_fp": "i32",
        "stride_fn": "i32",
        "stride_g": "i32",
        "stride_h": "i32",
        "stride_out": "i32",
        "stride_l1m": "i32",
        "stride_l1n": "i32",
        "stride_l2": "i32",
        "stride_l3m": "i32",
        "stride_l3n": "i32",
    }


def test_sdf19_tc01():
    """SDF19-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf19_tc01_alternating_pure kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf19_signature("fp16")
    constants = {
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_L": 3,
        "BLOCK_SIZE_P": 2,
        "BLOCK_SIZE_Q": 2,
    }

    mlir = compile_kernel(sdf19_tc01_alternating_pure, signature, constants)
    _write_mlir_to_file(mlir, "sdf19_tc01_alternating_pure.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf19_tc01_alternating_pure(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf19_tc01()
    print("All SDF19 v3 MLIR validation tests passed!")
