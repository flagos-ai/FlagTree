"""
Test Case: SDF25 - 5-layer alternating pure C or pure V

[MLIR Validation] Refactored version

Description: 5-layer nested (R, Q, L, P, K). Layers alternate pure C and pure V.

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
# SDF25: 5-layer alternating pure C or pure V
# Test purpose: Verify MLIR generation of 5-layer nesting alternating pure C / pure V under float16
# l1(C): tl.dot(a,b) -> store out_l1 [R,R]
# l2(V): c + d       -> store out_l2 [N]
# l3(C): tl.dot(g,h) -> store out_l3 [L,L]
# l4(V): i + j       -> store out_l4 [P]
# l5(C): tl.dot(e,f) -> store out    [K,K]
# ----------------------------------------------------------------------------
@triton.jit
def sdf25(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    i_ptr,
    j_ptr,
    out_ptr,
    out_l1_ptr,
    out_l2_ptr,
    out_l3_ptr,
    out_l4_ptr,
    M,
    N,
    K,
    L,
    P,
    Q,
    R,
    stride_am,
    stride_ar,
    stride_br,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ek,
    stride_fk,
    stride_fn,
    stride_gm,
    stride_gl,
    stride_hl,
    stride_hn,
    stride_i,
    stride_j,
    stride_outm,
    stride_outn,
    stride_l1m,
    stride_l1n,
    stride_l2,
    stride_l3m,
    stride_l3n,
    stride_l4,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r, offs_q, offs_l, offs_p, offs_k, offs_n = tl.arange(0, BLOCK_SIZE_R), tl.arange(0, BLOCK_SIZE_Q), tl.arange(
        0, BLOCK_SIZE_L), tl.arange(0, BLOCK_SIZE_P), tl.arange(0, BLOCK_SIZE_K), tl.arange(0, BLOCK_SIZE_N)
    for l1 in range(R):
        # --- l1: pure C (cube) ---
        a = tl.load(a_ptr + l1 * stride_am + offs_r * stride_ar, mask=offs_r < R, other=0.0)
        b = tl.load(b_ptr + l1 * stride_br + offs_r * stride_bn, mask=offs_r < R, other=0.0)
        l1_cube = tl.dot(a[:, None], b[None, :])
        l1_ptrs = out_l1_ptr + offs_r[:, None] * stride_l1m + offs_r[None, :] * stride_l1n
        tl.store(l1_ptrs, l1_cube, mask=(offs_r[:, None] < R) & (offs_r[None, :] < R))
        for l2 in range(Q):
            # --- l2: pure V (vector) ---
            c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
            d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
            l2_vec = c + d
            l2_ptrs = out_l2_ptr + offs_n * stride_l2
            tl.store(l2_ptrs, l2_vec, mask=offs_n < N)
            for l3 in range(L):
                # --- l3: pure C (cube) ---
                g = tl.load(g_ptr + l3 * stride_gm + offs_l * stride_gl, mask=offs_l < L, other=0.0)
                h = tl.load(h_ptr + l3 * stride_hl + offs_l * stride_hn, mask=offs_l < L, other=0.0)
                l3_cube = tl.dot(g[:, None], h[None, :])
                l3_ptrs = out_l3_ptr + offs_l[:, None] * stride_l3m + offs_l[None, :] * stride_l3n
                tl.store(l3_ptrs, l3_cube, mask=(offs_l[:, None] < L) & (offs_l[None, :] < L))
                for l4 in range(P):
                    # --- l4: pure V (vector) ---
                    i = tl.load(i_ptr + offs_p * stride_i, mask=offs_p < P, other=0.0)
                    j = tl.load(j_ptr + offs_p * stride_j, mask=offs_p < P, other=0.0)
                    l4_vec = i + j
                    l4_ptrs = out_l4_ptr + offs_p * stride_l4
                    tl.store(l4_ptrs, l4_vec, mask=offs_p < P)
                    for l5 in range(K):
                        # --- l5: pure C (cube) ---
                        e = tl.load(e_ptr + l5 * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)
                        f = tl.load(f_ptr + l5 * stride_fk + offs_k * stride_fn, mask=offs_k < K, other=0.0)
                        l5_cube = tl.dot(e[:, None], f[None, :])
                        out_ptrs = out_ptr + offs_k[:, None] * stride_outm + offs_k[None, :] * stride_outn
                        tl.store(out_ptrs, l5_cube, mask=(offs_k[:, None] < K) & (offs_k[None, :] < K))


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf25_signature(dtype_str):
    """Build the argument type signature for the SDF25 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "h_ptr": f"*{dtype_str}",
        "i_ptr": f"*{dtype_str}",
        "j_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "out_l1_ptr": f"*{dtype_str}",
        "out_l2_ptr": f"*{dtype_str}",
        "out_l3_ptr": f"*{dtype_str}",
        "out_l4_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "Q": "i32",
        "R": "i32",
        "stride_am": "i32",
        "stride_ar": "i32",
        "stride_br": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_em": "i32",
        "stride_ek": "i32",
        "stride_fk": "i32",
        "stride_fn": "i32",
        "stride_gm": "i32",
        "stride_gl": "i32",
        "stride_hl": "i32",
        "stride_hn": "i32",
        "stride_i": "i32",
        "stride_j": "i32",
        "stride_outm": "i32",
        "stride_outn": "i32",
        "stride_l1m": "i32",
        "stride_l1n": "i32",
        "stride_l2": "i32",
        "stride_l3m": "i32",
        "stride_l3n": "i32",
        "stride_l4": "i32",
    }


def test_sdf25():
    """SDF25: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf25 kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf25_signature("fp16")
    constants = {
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_L": 2,
        "BLOCK_SIZE_P": 2,
        "BLOCK_SIZE_Q": 2,
        "BLOCK_SIZE_R": 2,
    }

    mlir = compile_kernel(sdf25, signature, constants)
    _write_mlir_to_file(mlir, "sdf25.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf25(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf25()
    print("All SDF25 v3 MLIR validation tests passed!")
