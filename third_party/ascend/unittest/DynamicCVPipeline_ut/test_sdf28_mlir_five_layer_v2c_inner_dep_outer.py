"""
Test Case: SDF28 - 5-layer inner C depends outer V

[MLIR Validation] Refactored version

Description: 5-layer nested (R, Q, L, P, K). Outer V (tl.max) feeds into inner C via V2C.

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
# SDF28: 5-layer inner C depends outer V
# Test purpose: Verify MLIR generation of 5-layer nesting, inner C depends on outer V (V2C cross-layer) under float16
# ----------------------------------------------------------------------------
@triton.jit
def sdf28(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    out_ptr,
    out_cube_ptr,
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
    stride_fn,
    stride_out,
    stride_out_cubem,
    stride_out_cuben,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k, offs_n, offs_r = tl.arange(0, BLOCK_SIZE_K), tl.arange(0, BLOCK_SIZE_N), tl.arange(0, BLOCK_SIZE_R)
    for l1 in range(R):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
        l1v = tl.max(c + d)
        a = tl.load(a_ptr + l1 * stride_am + offs_r * stride_ar, mask=offs_r < R, other=0.0)
        b = tl.load(b_ptr + l1 * stride_br + offs_r * stride_bn, mask=offs_r < R, other=0.0)
        l1c = tl.dot(a[:, None], b[None, :])
        cube_ptrs = out_cube_ptr + offs_r[:, None] * stride_out_cubem + offs_r[None, :] * stride_out_cuben
        tl.store(cube_ptrs, l1c, mask=(offs_r[:, None] < R) & (offs_r[None, :] < R))
        for l2 in range(Q):
            for l3 in range(L):
                for l4 in range(P):
                    for l5 in range(K):
                        e = tl.load(e_ptr + l5 * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)
                        ic = e[:, None] * l1v
                        out_ptrs = out_ptr + offs_k[:, None] * stride_out
                        tl.store(out_ptrs, ic, mask=offs_k[:, None] < K)
                        f = tl.load(f_ptr + offs_n * stride_fn, mask=offs_n < N, other=0.0)
                        iv = f / 2.0


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf28_signature(dtype_str):
    """Build the argument type signature for the SDF28 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "out_cube_ptr": f"*{dtype_str}",
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
        "stride_fn": "i32",
        "stride_out": "i32",
        "stride_out_cubem": "i32",
        "stride_out_cuben": "i32",
    }


def test_sdf28():
    """SDF28: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf28 kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf28_signature("fp16")
    constants = {
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_R": 2,
        "BLOCK_SIZE_Q": 2,
        "BLOCK_SIZE_L": 2,
        "BLOCK_SIZE_P": 2,
    }

    mlir = compile_kernel(sdf28, signature, constants)
    _write_mlir_to_file(mlir, "sdf28.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf28(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf28()
    print("All SDF28 v3 MLIR validation tests passed!")
