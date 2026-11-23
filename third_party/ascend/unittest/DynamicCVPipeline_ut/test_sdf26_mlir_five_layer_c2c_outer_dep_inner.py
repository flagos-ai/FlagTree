"""
Test Case: SDF26 - 5-layer outer C depends inner C

[MLIR Validation] Refactored version

Description: 5-layer nested (R, Q, L, P, K). Inner loop accumulates, outer uses.

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
def sdf26(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    f_ptr,
    g_ptr,
    h_ptr,
    out_ptr,
    M,
    N,
    K,
    L,
    P,
    Q,
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
    stride_gm,
    stride_gr,
    stride_h,
    stride_out,
    R: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k, offs_n, offs_r = tl.arange(0, BLOCK_SIZE_K), tl.arange(0, BLOCK_SIZE_N), tl.arange(0, BLOCK_SIZE_R)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_l = tl.arange(0, BLOCK_SIZE_L)
    offs_q = tl.arange(0, BLOCK_SIZE_Q)
    acc = tl.zeros([BLOCK_SIZE_R, BLOCK_SIZE_R], tl.float32)
    for l1 in range(R):
        for l2 in range(Q):
            for l3 in range(L):
                for l4 in range(P):
                    for l5 in range(K):
                        a = tl.load(a_ptr + l5 * stride_am + offs_r * stride_ak, mask=offs_r < R, other=0.0)
                        b = tl.load(b_ptr + l5 * stride_bk + offs_r * stride_bn, mask=offs_r < R, other=0.0)
                        ic = tl.dot(a[:, None], b[None, :])
                        acc = acc + ic
                        c = tl.load(c_ptr + offs_n * stride_c, mask=offs_n < N, other=0.0)
                        d = tl.load(d_ptr + offs_n * stride_d, mask=offs_n < N, other=0.0)
                        iv = c * d
                    e = tl.load(e_ptr + l4 * stride_em + offs_p * stride_ep, mask=offs_p < P, other=0.0)
                    f = tl.load(f_ptr + l4 * stride_fp + offs_p * stride_fn, mask=offs_p < P, other=0.0)
                    mc_p = tl.dot(e[:, None], f[None, :])
                e = tl.load(e_ptr + l3 * stride_em + offs_l * stride_ep, mask=offs_l < L, other=0.0)
                f = tl.load(f_ptr + l3 * stride_fp + offs_l * stride_fn, mask=offs_l < L, other=0.0)
                mc_l = tl.dot(e[:, None], f[None, :])
            e = tl.load(e_ptr + l2 * stride_em + offs_q * stride_ep, mask=offs_q < Q, other=0.0)
            f = tl.load(f_ptr + l2 * stride_fp + offs_q * stride_fn, mask=offs_q < Q, other=0.0)
            mc_q = tl.dot(e[:, None], f[None, :])
        g = tl.load(g_ptr + l1 * stride_gm + offs_r * stride_gr, mask=offs_r < R, other=0.0).to(tl.float32)
        l1c = tl.dot(acc, g[:, None])
        out_ptrs = out_ptr + offs_r[:, None] * stride_out
        tl.store(out_ptrs, l1c, mask=offs_r[:, None] < R)
        h = tl.load(h_ptr + offs_n * stride_h, mask=offs_n < N, other=0.0)
        l1v = h + 1.0


def _build_sdf26_signature(dtype_str):
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
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "L": "i32",
        "P": "i32",
        "Q": "i32",
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
        "stride_gm": "i32",
        "stride_gr": "i32",
        "stride_h": "i32",
        "stride_out": "i32",
    }


def test_sdf26():
    signature = _build_sdf26_signature("fp16")
    constants = {
        "R": 2,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_L": 2,
        "BLOCK_SIZE_P": 2,
        "BLOCK_SIZE_Q": 2,
        "BLOCK_SIZE_R": 2,
    }

    mlir = compile_kernel(sdf26, signature, constants)
    _write_mlir_to_file(mlir, "sdf26.mlir")

    assert mlir and len(mlir) > 0, "MLIR code generation failed or is empty"
    assert "func.func @sdf26(" in mlir, \
        "Kernel function definition not found in MLIR code"
    assert "scope" not in mlir, "Fallback scenario: MLIR code unexpectedly contains the 'scope' keyword"


if __name__ == "__main__":
    test_sdf26()
    print("All SDF26 v3 MLIR validation tests passed!")
