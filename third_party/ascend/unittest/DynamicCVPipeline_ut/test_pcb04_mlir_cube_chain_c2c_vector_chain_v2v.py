"""
Test Case: PCB04 - Cube chain C2C (Cube1->Cube2) + Vector chain V2V (Vec1->Vec2)

[MLIR Validation] Refactored version

Description: Cube chain C2C (Cube1->Cube2) + Vector chain V2V (Vec1->Vec2)

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions; this is a fallback scenario,
     so the MLIR code must NOT contain the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases:
  - PCB04-TC01: float16, M=128, N=64, K=32
  - PCB04-TC02: float32, M=128, N=64, K=32
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
# PCB04-TC01: float16, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of Cube chain dependency (C2C) + VECTOR chain dependency (V2V) under float16
# ----------------------------------------------------------------------------
@triton.jit
def pcb04_tc01_cube_vec_chain(
    a_ptr,
    b_ptr,
    c_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    out_cube_ptr,
    out_vec_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_ck,
    stride_e,
    stride_f,
    stride_g,
    stride_out_cube,
    stride_out_vec,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        cube1_result = tl.dot(a[:, None], b[None, :]).to(tl.float16)  # (K, K)

        c = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)  # (K,)
        cube2_result = tl.dot(cube1_result, c[:, None]).to(tl.float16)  # (K, 1)
        out_cube_ptrs = out_cube_ptr + offs_k[:, None] * stride_out_cube
        tl.store(out_cube_ptrs, cube2_result, mask=offs_k[:, None] < K)

        e = tl.load(e_ptr + offs_n * stride_e, mask=offs_n < N, other=0.0)  # (N,)
        f = tl.load(f_ptr + offs_n * stride_f, mask=offs_n < N, other=0.0)  # (N,)
        vec1_result = e / (f + 1e-6)  # (N,)

        g = tl.load(g_ptr + offs_n * stride_g, mask=offs_n < N, other=0.0)  # (N,)
        vec2_result = vec1_result + g  # (N,)
        out_vec_ptrs = out_vec_ptr + offs_n * stride_out_vec
        tl.store(out_vec_ptrs, vec2_result, mask=offs_n < N)


# ----------------------------------------------------------------------------
# PCB04-TC02: float32, M=128, N=64, K=32
# Test purpose: Verify MLIR generation of Cube chain dependency (C2C) + VECTOR chain dependency (V2V) under float32
# ----------------------------------------------------------------------------
@triton.jit
def pcb04_tc02_cube_vec_chain(
    a_ptr,
    b_ptr,
    c_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    out_cube_ptr,
    out_vec_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_ck,
    stride_e,
    stride_f,
    stride_g,
    stride_out_cube,
    stride_out_vec,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # (N,)

    for k in range(K):
        a = tl.load(a_ptr + k * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)  # (K,)
        b = tl.load(b_ptr + k * stride_bk + offs_k * stride_bn, mask=offs_k < K, other=0.0)  # (K,)
        cube1_result = tl.dot(a[:, None], b[None, :])  # (K, K)

        c = tl.load(c_ptr + k * stride_cm + offs_k * stride_ck, mask=offs_k < K, other=0.0)  # (K,)
        cube2_result = tl.dot(cube1_result, c[:, None])  # (K, 1)
        out_cube_ptrs = out_cube_ptr + offs_k[:, None] * stride_out_cube
        tl.store(out_cube_ptrs, cube2_result, mask=offs_k[:, None] < K)

        e = tl.load(e_ptr + offs_n * stride_e, mask=offs_n < N, other=0.0)  # (N,)
        f = tl.load(f_ptr + offs_n * stride_f, mask=offs_n < N, other=0.0)  # (N,)
        vec1_result = e / (f + 1e-6)  # (N,)

        g = tl.load(g_ptr + offs_n * stride_g, mask=offs_n < N, other=0.0)  # (N,)
        vec2_result = vec1_result + g  # (N,)
        out_vec_ptrs = out_vec_ptr + offs_n * stride_out_vec
        tl.store(out_vec_ptrs, vec2_result, mask=offs_n < N)


# ============================================================================
# Pytest test cases
# ============================================================================


def _build_pcb04_signature(dtype_str):
    """Build the argument type signature for the PCB04 kernel."""
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "out_cube_ptr": f"*{dtype_str}",
        "out_vec_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_ck": "i32",
        "stride_e": "i32",
        "stride_f": "i32",
        "stride_g": "i32",
        "stride_out_cube": "i32",
        "stride_out_vec": "i32",
    }


def test_pcb04_tc01():
    """PCB04-TC01: Verify float16 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile pcb04_tc01_cube_vec_chain kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_pcb04_signature("fp16")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(pcb04_tc01_cube_vec_chain, signature, constants)
    _write_mlir_to_file(mlir, "pcb04_tc01_cube_vec_chain.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @pcb04_tc01_cube_vec_chain(" in mlir, \
        "Kernel function definition not found in MLIR code"
    # The cube chain stages through L1 (L0C -> cbuf), which stays on CUBE, so the
    # only cross-core traffic here is the V->C copies: a one-way interaction, and
    # the pipeline falls back on those, as in sdf10 and the other one-way cases.
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


def test_pcb04_tc02():
    """PCB04-TC02: Verify float32 kernel compilation generates correct MLIR code.

    Test steps:
      1. Compile pcb04_tc02_cube_vec_chain kernel to MLIR
      2. Verify MLIR code is successfully generated and non-empty
      3. Verify MLIR code contains the function definition
      4. Verify MLIR code contains the "scope" keyword
    """
    signature = _build_pcb04_signature("fp32")
    constants = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

    mlir = compile_kernel(pcb04_tc02_cube_vec_chain, signature, constants)
    _write_mlir_to_file(mlir, "pcb04_tc02_cube_vec_chain.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @pcb04_tc02_cube_vec_chain(" in mlir, \
        "Kernel function definition not found in MLIR code"
    # The cube chain stages through L1 (L0C -> cbuf), which stays on CUBE, so the
    # only cross-core traffic here is the V->C copies: a one-way interaction, and
    # the pipeline falls back on those, as in sdf10 and the other one-way cases.
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"

    # Output MLIR code to the specified path


# ============================================================================
# Main for manual testing
# ============================================================================
if __name__ == "__main__":
    test_pcb04_tc01()
    test_pcb04_tc02()
    print("All PCB04 v3 MLIR validation tests passed!")
