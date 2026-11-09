"""
Test Case: SDF21 - 4-layer nested, inner C depends on outer C (C2C cross-layer)

[MLIR Validation] Refactored version

Description: 4-layer nested (Q, L, P, K). Inner loop uses l1_cube from outer in tl.dot.

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
# SDF21: 4-layer nested, inner C depends on outer C
# Test purpose: Verify MLIR generation of 4-layer nesting, inner C depends on outer C under float16
# ----------------------------------------------------------------------------
@triton.jit
def sdf21(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    out_ptr,
    M,
    N,
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
    stride_ek,
    stride_fn,
    stride_out,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_q, offs_k, offs_n = tl.arange(0, BLOCK_SIZE_Q), tl.arange(0, BLOCK_SIZE_K), tl.arange(0, BLOCK_SIZE_N)
    for l1 in range(Q):
        a = tl.load(a_ptr + l1 * stride_am + offs_k * stride_aq, mask=offs_k < Q, other=0.0)
        b = tl.load(b_ptr + l1 * stride_bq + offs_k * stride_bn, mask=offs_k < Q, other=0.0)
        l1_cube = tl.dot(a[:, None], b[None, :])
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
        l1_vec = c - d
        for l2 in range(L):
            for l3 in range(P):
                for l4 in range(K):
                    e = tl.load(e_ptr + l4 * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)
                    inner_cube = tl.dot(l1_cube.to(tl.float16), e[:, None])
                    out_ptrs = out_ptr + offs_k[:, None] * stride_out
                    tl.store(out_ptrs, inner_cube, mask=offs_k[:, None] < K)
                    f = tl.load(f_ptr + offs_n * stride_fn, mask=offs_n < N, other=0.0)
                    inner_vec = f * 2.0


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf21_signature(dtype_str):
    """Build the argument type signature for the SDF21 kernel.

    Note: In SDF21, K is a constexpr parameter and does not appear in signature.
    """
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
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
        "stride_ek": "i32",
        "stride_fn": "i32",
        "stride_out": "i32",
    }


def test_sdf21():
    """SDF21: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf21 kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf21_signature("fp16")
    constants = {
        "K": 32,
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_L": 3,
        "BLOCK_SIZE_P": 2,
        "BLOCK_SIZE_Q": 2,
        "BLOCK_SIZE_N": 64,
    }

    mlir = compile_kernel(sdf21, signature, constants)
    _write_mlir_to_file(mlir, "sdf21.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf21(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" in mlir, "MLIR code does not contain the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf21()
    print("All SDF21 v3 MLIR validation tests passed!")
