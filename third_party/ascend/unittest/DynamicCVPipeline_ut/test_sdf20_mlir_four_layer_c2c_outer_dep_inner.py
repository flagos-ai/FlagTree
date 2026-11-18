"""
Test Case: SDF20 - 4-layer nested, outer C depends on inner C (C2C cross-layer)

[MLIR Validation] Refactored version

Description: 4-layer nested (Q, L, P, K). Inner loop accumulates, outer uses.

Refactoring approach (refer to test_custom.py):
  1. Remove precision comparison logic between Kernel and reference implementation
  2. Refer to compile_kernel in test_custom.py to implement MLIR code generation for each Triton Kernel
  3. Add MLIR content validation in test functions to ensure the MLIR code contains the "scope" keyword
  4. Keep the test framework complete and maintainable

Test Cases: SDF20-TC01: float16, M=128, N=64, K=32, L=3, P=2, Q=2
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


def compile_kernel(kernel, signature, constants):
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    try:
        options = NPUOptions(arch="Ascend910_9589", enable_dynamic_cv_pipeline=True)
        codegen_fns = {"min_dot_size": min_dot_size(None)}
        ttir = ast_to_ttir(kernel, src, context, options, codegen_fns, {})
        metadata = {**options.__dict__}
        ttir = make_ttir(ttir, metadata, options)
        linalg = ttir_to_linalg(ttir, metadata, options, named_ops=True)
        return str(linalg)
    except subprocess.CalledProcessError as ex:
        print(ex.stdout.decode())
        print(ex.stderr.decode())
        print("failed")
        return None


MLIR_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlir_output")


def _write_mlir_to_file(mlir, filename):
    os.makedirs(MLIR_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(MLIR_OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"MLIR code written to: {output_path}")


@triton.jit
def sdf20_tc01(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    l_ptr,
    m_ptr,
    g_ptr,
    h_ptr,
    out_ptr,
    M,
    N,
    K,
    L,
    P,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c,
    stride_d,
    stride_em,
    stride_ep,
    stride_fp,
    stride_fn,
    stride_lm,
    stride_ll,
    stride_mm,
    stride_ml,
    stride_gm,
    stride_gq,
    stride_h,
    stride_out,
    Q: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_q = tl.arange(0, BLOCK_SIZE_Q)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    inner_cube_acc = tl.zeros([BLOCK_SIZE_Q, BLOCK_SIZE_Q], tl.float32)
    for l1 in range(Q):
        for l2 in range(L):
            for l3 in range(P):
                for l4 in range(K):
                    a = tl.load(a_ptr + l4 * stride_am + offs_q * stride_ak, mask=offs_q < Q, other=0.0)
                    b = tl.load(b_ptr + l4 * stride_bk + offs_q * stride_bn, mask=offs_q < Q, other=0.0)
                    inner_cube = tl.dot(a[:, None], b[None, :])
                    inner_cube_acc = inner_cube_acc + inner_cube
                    c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
                    d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
                    inner_vec = c * d
                e = tl.load(e_ptr + l3 * stride_em + offs_p * stride_ep, mask=offs_p < P, other=0.0)
                f = tl.load(f_ptr + l3 * stride_fp + offs_p * stride_fn, mask=offs_p < P, other=0.0)
                mid_cube_p = tl.dot(e[:, None], f[None, :])
            l_ptr_val = tl.load(l_ptr + l2 * stride_lm + offs_l * stride_ll, mask=offs_l < L, other=0.0)
            m_ptr_val = tl.load(m_ptr + l2 * stride_mm + offs_l * stride_ml, mask=offs_l < L, other=0.0)
            mid_cube_l = tl.dot(l_ptr_val[:, None], m_ptr_val[None, :])
        g = tl.load(g_ptr + l1 * stride_gm + offs_q * stride_gq, mask=offs_q < Q, other=0.0).to(tl.float32)
        l1_cube = tl.dot(inner_cube_acc, g[:, None])
        out_ptrs = out_ptr + offs_q[:, None] * stride_out
        tl.store(out_ptrs, l1_cube, mask=offs_q[:, None] < Q)
        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)
        l1_vec = h + 1.0


def _build_sdf20_signature(dtype_str):
    return {
        "a_ptr": f"*{dtype_str}",
        "b_ptr": f"*{dtype_str}",
        "c_ptr": f"*{dtype_str}",
        "d_ptr": f"*{dtype_str}",
        "e_ptr": f"*{dtype_str}",
        "f_ptr": f"*{dtype_str}",
        "l_ptr": f"*{dtype_str}",
        "m_ptr": f"*{dtype_str}",
        "g_ptr": f"*{dtype_str}",
        "h_ptr": f"*{dtype_str}",
        "out_ptr": f"*{dtype_str}",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_c": "i32",
        "stride_d": "i32",
        "stride_em": "i32",
        "stride_ep": "i32",
        "stride_fp": "i32",
        "stride_fn": "i32",
        "stride_lm": "i32",
        "stride_ll": "i32",
        "stride_mm": "i32",
        "stride_ml": "i32",
        "stride_gm": "i32",
        "stride_gq": "i32",
        "stride_h": "i32",
        "stride_out": "i32",
    }


def test_sdf20_tc01():
    signature = _build_sdf20_signature("fp16")
    constants = {
        "Q": 2,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_L": 3,
        "BLOCK_SIZE_P": 2,
        "BLOCK_SIZE_Q": 2,
    }

    mlir = compile_kernel(sdf20_tc01, signature, constants)
    _write_mlir_to_file(mlir, "sdf20_tc01.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf20_tc01(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"


if __name__ == "__main__":
    test_sdf20_tc01()
    print("All SDF20 v3 MLIR validation tests passed!")
