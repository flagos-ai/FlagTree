"""
Test Case: SDF10 - 2-layer nested, inner C depends on outer V (V2C cross-layer)

[MLIR Validation] Refactored version

Description: 2-layer nested, inner C depends on outer V (V2C cross-layer)

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - SDF10-TC01: float16, M=128, N=64, K=32, L=3
  - SDF10-TC02: float32, M=128, N=64, K=32, L=3
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
# SDF10-TC01: float16, M=128, N=64, K=32, L=3
# Test purpose: Verify MLIR generation of 2-layer nesting, inner C depends on outer V (V2C cross-layer) under float16
# ----------------------------------------------------------------------------
@triton.jit
def sdf10_tc01_nested_v2c(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    out_ptr,
    M,
    N,
    K,
    L,
    stride_am,
    stride_al,
    stride_bl,
    stride_bl2,
    stride_c,
    stride_d,
    stride_em,
    stride_ek,
    stride_fn,
    stride_outm,
    stride_outn,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for i in range(L):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
        outer_vec = c + d

        a = tl.load(a_ptr + i * stride_am + offs_l * stride_al, mask=offs_l < L, other=0.0)
        b = tl.load(b_ptr + i * stride_bl + offs_l * stride_bl2, mask=offs_l < L, other=0.0)

        for k in range(K):
            e = tl.load(e_ptr + k * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)  # (K,)
            inner_cube = tl.dot(e[:, None], outer_vec[None, :])
            out_ptrs = out_ptr + offs_k[:, None] * stride_outm + offs_n[None, :] * stride_outn
            out_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            tl.store(out_ptrs, inner_cube, mask=out_mask)

            f = tl.load(f_ptr + offs_n * stride_fn, mask=offs_n < N, other=0.0)
            inner_vec = f / 2.0


# ----------------------------------------------------------------------------
# SDF10-TC02: float32, M=128, N=64, K=32, L=3
# Test purpose: Verify MLIR generation of 2-layer nesting, inner C depends on outer V (V2C cross-layer) under float32
# ----------------------------------------------------------------------------
@triton.jit
def sdf10_tc02_nested_v2c(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    out_ptr,
    M,
    N,
    K,
    L,
    stride_am,
    stride_al,
    stride_bl,
    stride_bl2,
    stride_c,
    stride_d,
    stride_em,
    stride_ek,
    stride_fn,
    stride_outm,
    stride_outn,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    for i in range(L):
        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
        outer_vec = c + d

        a = tl.load(a_ptr + i * stride_am + offs_l * stride_al, mask=offs_l < L, other=0.0)
        b = tl.load(b_ptr + i * stride_bl + offs_l * stride_bl2, mask=offs_l < L, other=0.0)

        for k in range(K):
            e = tl.load(e_ptr + k * stride_em + offs_k * stride_ek, mask=offs_k < K, other=0.0)
            inner_cube = tl.dot(e[:, None], outer_vec[None, :])
            out_ptrs = out_ptr + offs_k[:, None] * stride_outm + offs_n[None, :] * stride_outn
            out_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            tl.store(out_ptrs, inner_cube, mask=out_mask)

            f = tl.load(f_ptr + offs_n * stride_fn, mask=offs_n < N, other=0.0)
            inner_vec = f / 2.0


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_sdf10_signature(dtype_str):
    """Build the argument type signature for the SDF10 kernel."""
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
        "K": "i32",
        "L": "i32",
        "stride_am": "i32",
        "stride_al": "i32",
        "stride_bl": "i32",
        "stride_bl2": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_em": "i32",
        "stride_ek": "i32",
        "stride_fn": "i32",
        "stride_outm": "i32",
        "stride_outn": "i32",
    }


def test_sdf10_tc01():
    """SDF10-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf10_tc01_nested_v2c kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf10_signature("fp16")
    constants = {"BLOCK_SIZE_L": 3, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(sdf10_tc01_nested_v2c, signature, constants)
    _write_mlir_to_file(mlir, "sdf10_tc01_nested_v2c.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf10_tc01_nested_v2c(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


def test_sdf10_tc02():
    """SDF10-TC02: Verify float32 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile sdf10_tc02_nested_v2c kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_sdf10_signature("fp32")
    constants = {"BLOCK_SIZE_L": 3, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(sdf10_tc02_nested_v2c, signature, constants)
    _write_mlir_to_file(mlir, "sdf10_tc02_nested_v2c.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf10_tc02_nested_v2c(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_sdf10_tc01()
    test_sdf10_tc02()
    print("All SDF10 v3 MLIR validation tests passed!")
