"""FlagGems-style matmul kernel, standalone (no flag_gems runtime dependency).

Ported from flag_gems/runtime/backend/_ascend/ops/mm.py.

Changes vs. the original:
- Removed @libentry / @libtuner / @triton.heuristics decorators.
- All kernel parameters (M, N, K, strides, BLOCK_*, SPLIT_K, EVEN_K) are
  regular (non-constexpr) arguments so the kernel compiles without the
  FlagGems tuning infrastructure.
- torch_device_fn replaced with a plain torch.npu / torch.cuda context.
- A standalone call() entry-point is provided for use in benchmarks.
"""

import argparse
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Fixed tile sizes – match the second config returned by get_tuned_config("mm")
# on the Ascend backend: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, SPLIT_K=1.
_DEFAULT_M = 1024
_DEFAULT_N = 1024
_DEFAULT_K = 1024

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 128
SPLIT_K = 1
GROUP_M = 8


@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A, mask=(ram < M)[:, None], other=0.0)
            b = tl.load(B, mask=(rbn < N)[None, :], other=0.0)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(
                A,
                mask=(rk[None, :] < k_remaining) & (ram < M)[:, None],
                other=0.0,
            )
            b = tl.load(
                B,
                mask=(rk[:, None] < k_remaining) & (rbn < N)[None, :],
                other=0.0,
            )
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a
    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm(a, b):
    """Compute a @ b using the FlagGems-style Triton kernel."""
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=a.device, dtype=c_dtype)
    even_k = (K % (BLOCK_K * SPLIT_K)) == 0
    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        SPLIT_K,
    )
    mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        dot_out_dtype=tl.float32,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, SPLIT_K=SPLIT_K, EVEN_K=even_k,
    )
    return c


def call(mat_a, mat_b, _num_cores=None):
    """Benchmark-compatible entry point matching native_matmul.call signature."""
    return mm(mat_a, mat_b)


# =============================================================================
#  Intermediate TTIR dump (no device required)
# =============================================================================
class _DumpOptions:
    num_warps = 4
    num_stages = 1
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    allowed_dot_input_precisions = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default = 0
    default_dot_input_precision = "ieee"
    sanitize_overflow = False


def _matmul_signature():
    """Static signature for ast_to_ttir — non-constexpr args only."""
    return {
        "A": "*fp16",
        "B": "*fp16",
        "C": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }


def dump_ttir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K, return_module=False):
    """Compile matmul_kernel to TTIR and write to *path*."""
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    try:
        from triton._C.libtriton import tle as tle_ir
    except ImportError:
        tle_ir = None

    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "flaggems_matmul_triton.mlir")

    signature = _matmul_signature()
    even_k = (K % (BLOCK_K * SPLIT_K)) == 0
    constants = {
        "dot_out_dtype": tl.float32,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_M": GROUP_M,
        "SPLIT_K": SPLIT_K,
        "EVEN_K": even_k,
    }

    src = ASTSource(mm_kernel.fn, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    if tle_ir is not None:
        tle_ir.load_dialects(context)
        tle_ir.load_tile_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(mm_kernel, src, context, _DumpOptions(), codegen_fns, {})

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_ttir: module.verify() failed")

    mlir = str(module)
    if path:
        with open(path, "w") as f:
            f.write(mlir)
        print(f"[dump_ttir] wrote TTIR ({len(mlir)} chars) to {path}")

    if return_module:
        return module
    return mlir


# =============================================================================
#  Full Linalg IR dump (TTIR → TileIR → Linalg lowering)
# =============================================================================
def dump_linalg(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K):
    """Compile matmul_kernel through full TileIR → Linalg lowering pipeline.
    """
    from triton._C.libtriton import ir, passes, ascend

    # Step 1: compile to TTIR — get module directly (avoids reparse/dialect issues)
    module = dump_ttir(path=None, M=M, N=N, K=K, return_module=True)
    context = module.context

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "flaggems_matmul_triton_linalg.mlir")

    # ── ① TileIR → HIVM ──────────────────────────────────────────────────
    pm = ir.pass_manager(context)
    passes.common.add_inliner(pm)
    ascend.passes.ttir.add_tileir_to_hivm(pm)
    pm.run(module)
    print(f"[dump_linalg] ① tileir_to_hivm: verify={module.verify()}", flush=True)

    # ── ①b Erase unrealized_conversion_cast ops ──────────────────────────
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_erase_linalg_casts(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module)
    print(f"[dump_linalg] ①b erase_linalg_casts: verify={module.verify()}", flush=True)

    # ── ② Structured (r1) + discrete mask ────────────────────────────────
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, False, False)
    pm.run(module)
    print(f"[dump_linalg] ② structure(r1)+discrete_mask: verify={module.verify()}", flush=True)

    # ── ③ Unstructured + HIVM + HFusion + LLVM ──────────────────────────
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_triton_to_unstructure_incubated(pm, False, False)
    ascend.passes.ttir.add_triton_to_hivm(pm)
    ascend.passes.ttir.add_triton_to_hfusion(pm)
    ascend.passes.ttir.add_triton_to_llvm(pm)
    pm.run(module)
    print(f"[dump_linalg] ③ unstructure+hivm+hfusion+llvm: verify={module.verify()}", flush=True)

    # ── ④ Bubble-up + structured (r2) ────────────────────────────────────
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_bubble_up_operation(pm)
    ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ④ bubble_up+structure(r2): verify={module.verify()}", flush=True)

    # ── ④b Inline + canonicalize ─────────────────────────────────────────
    pm = ir.pass_manager(context)
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module)
    print(f"[dump_linalg] ④b inline+canonicalize: verify={module.verify()}", flush=True)

    # ── ⑤ Triton → Linalg ───────────────────────────────────────────────
    try:
        pm = ir.pass_manager(context)
        ascend.passes.ttir.add_triton_to_linalg_incubated(pm, False, True, False, False, False)
        pm.run(module)
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: verify={module.verify()}", flush=True)
    except RuntimeError as e:
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: partial ({e})", flush=True)

    # ── ⑤c Fold staging copies (eliminate redundant default-space allocs) ─
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_fold_staging_copy(pm)
    pm.run(module)
    print(f"[dump_linalg] ⑤c fold_staging_copy: verify={module.verify()}", flush=True)

    # ── ⑤b Erase linalg casts (post) ────────────────────────────────────
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_erase_linalg_casts(pm)
    pm.run(module)
    print(f"[dump_linalg] ⑤b erase_linalg_casts (post): verify={module.verify()}", flush=True)

    # ── ⑥ Final canonicalize + CSE + DCE ─────────────────────────────────
    pm = ir.pass_manager(context)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(module)
    print(f"[dump_linalg] ⑥ final cleanup: verify={module.verify()}", flush=True)

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_linalg: module.verify() failed after pipeline")

    mlir = str(module)
    if path:
        with open(path, "w") as f:
            f.write(mlir)
        print(f"[dump_linalg] wrote Linalg IR ({len(mlir)} chars) to {path}")
    return mlir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlagGems standalone matmul")
    parser.add_argument("--M", type=int, default=_DEFAULT_M)
    parser.add_argument("--N", type=int, default=_DEFAULT_N)
    parser.add_argument("--K", type=int, default=_DEFAULT_K)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--dump-ttir", nargs="?", const="", default=None,
                        help="Dump TTIR to PATH and exit; no device needed.")
    parser.add_argument("--dump-linalg", nargs="?", const="", default=None,
                        help="Dump Linalg IR to PATH and exit; no device needed.")
    args = parser.parse_args()
    
    if args.dump_ttir is not None:
        dump_ttir(path=(args.dump_ttir or None), M=args.M, N=args.N, K=args.K)
        raise SystemExit(0)

    if args.dump_linalg is not None:
        dump_linalg(path=(args.dump_linalg or None), M=args.M, N=args.N, K=args.K)
        raise SystemExit(0)

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    a = torch.randn((args.M, args.K), dtype=torch.float16, device=device)
    b = torch.randn((args.K, args.N), dtype=torch.float16, device=device)
    c = mm(a, b)

    if not args.no_check:
        ref = torch.matmul(a.float(), b.float()).to(torch.float16)
        torch.testing.assert_close(ref, c, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
