import argparse
import os

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
_DEFAULT_M = 1024
_DEFAULT_N = 1024
_DEFAULT_K = 1024
_DEFAULT_NUM_CORES = 24

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 128


def get_number_cores():
    """Return the number of AI cores to use as the launch grid size."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.get_device_properties(0).ai_core_num
    except Exception:
        return _DEFAULT_NUM_CORES


# =============================================================================
#  Simple matmul kernel: single loop with tle.dsa.copy + tl.dot
#
#  grid = (NUM_CORES,). Each core handles multiple (M_tile, N_tile) output
#  blocks in round-robin fashion, iterating over K dimension.
# =============================================================================
@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    NUM_K_BLOCKS = tl.cdiv(K, BLOCK_K)

    # On-chip buffers: A/B in L1.
    mat_a_l1 = tle.dsa.alloc(
        [BLOCK_M, BLOCK_K], dtype=mat_a.dtype.element_ty, mem_addr_space=tle.dsa.ascend.L1)
    mat_b_l1 = tle.dsa.alloc(
        [BLOCK_K, BLOCK_N], dtype=mat_b.dtype.element_ty, mem_addr_space=tle.dsa.ascend.L1)

    # Each core processes output blocks in round-robin
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        # Compute M/N tile indices
        pid_m = block_idx // NUM_BLOCKS_N
        pid_n = block_idx % NUM_BLOCKS_N
        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N

        # Create block pointers for A and B
        a_block_ptr = tl.make_block_ptr(
            mat_a, (M, K), (K, 1),
            (m_start, 0),
            (BLOCK_M, BLOCK_K), (1, 0))
        b_block_ptr = tl.make_block_ptr(
            mat_b, (K, N), (N, 1),
            (0, n_start),
            (BLOCK_K, BLOCK_N), (1, 0))

        mat_c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # K-loop: load A/B tiles from GM to L1, then dot
        for k_idx in range(0, NUM_K_BLOCKS):
            # Copy A tile
            tle.dsa.copy(a_block_ptr, mat_a_l1, [BLOCK_M, BLOCK_K])
            # Copy B tile
            tle.dsa.copy(b_block_ptr, mat_b_l1, [BLOCK_K, BLOCK_N])

            # Accumulate: C += A @ B
            mat_c_acc = tl.dot(
                tle.dsa.to_tensor(mat_a_l1, writable=False),
                tle.dsa.to_tensor(mat_b_l1, writable=False),
                mat_c_acc, out_dtype=tl.float32)

            # Advance block pointers along K
            a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_K])
            b_block_ptr = tl.advance(b_block_ptr, [BLOCK_K, 0])

        # Store result back to GM
        tl.store(
            tl.make_block_ptr(
                mat_c, (M, N), (N, 1),
                (m_start, n_start),
                (BLOCK_M, BLOCK_N), (1, 0)),
            mat_c_acc.to(mat_c.dtype.element_ty))


# =============================================================================
#  Host-side launch
# =============================================================================
def call(mat_a, mat_b):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    num_cores = get_number_cores()
    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores,
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return mat_c


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
        "mat_a": "*fp16",
        "mat_b": "*fp16",
        "mat_c": "*fp16",
        "M": "i32",
    }


def _compile_matmul_module(M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K,
                           NUM_CORES=_DEFAULT_NUM_CORES):
    """Compile matmul_kernel to the frontend module that contains tile.* ops."""
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    from triton._C.libtriton import tle as tle_ir

    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    signature = _matmul_signature()
    constants = {
        "N": N,
        "K": K,
        "NUM_CORES": NUM_CORES,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
    }

    src = ASTSource(matmul_kernel.fn, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(matmul_kernel, src, context, _DumpOptions(), codegen_fns, {})

    ok = module.verify()
    if not ok:
        raise RuntimeError("_compile_matmul_module: module.verify() failed")
    return module


def dump_ttir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K,
              NUM_CORES=_DEFAULT_NUM_CORES, return_module=False):
    """Compile matmul_kernel to TTIR and write to *path*."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "native_matmul_dsa_ttir.mlir")

    module = _compile_matmul_module(M=M, N=N, K=K, NUM_CORES=NUM_CORES)
    mlir = str(module)
    if path:
        with open(path, "w") as f:
            f.write(mlir)
        print(f"[dump_ttir] module.verify() = True; wrote TTIR ({len(mlir)} chars) to {path}")

    if return_module:
        return module
    return mlir


def dump_tileir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K,
                NUM_CORES=_DEFAULT_NUM_CORES, return_module=False):
    """Dump the frontend TileIR stage before TileIR-to-HIVM lowering."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "native_matmul_dsa_tileir.mlir")

    module = _compile_matmul_module(M=M, N=N, K=K, NUM_CORES=NUM_CORES)
    mlir = str(module)
    if "tile." not in mlir:
        raise RuntimeError("dump_tileir: expected tile.* ops in frontend IR")

    if path:
        with open(path, "w") as f:
            f.write(mlir)
        print(f"[dump_tileir] module.verify() = True; wrote TileIR ({len(mlir)} chars) to {path}")

    if return_module:
        return module
    return mlir


# =============================================================================
#  Full Linalg IR dump (TTIR → TileIR → Linalg lowering)
# =============================================================================
def dump_linalg(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K,
                NUM_CORES=_DEFAULT_NUM_CORES):
    """Compile matmul_kernel through full TileIR → Linalg lowering pipeline.

    Pipeline:
      ① tileir_to_hivm            — tile.* → memref/hivm
      ①b erase_linalg_casts       — eliminate unrealized casts
      ② structure(r1) + discrete mask
      ③ unstructure + hivm + hfusion + llvm
      ④ bubble_up + structure(r2)
      ④b inline + canonicalize
      ⑤ triton_to_linalg_incubated
      ⑤b erase_linalg_casts (post)
      ⑥ final canonicalize + CSE + DCE

    Returns the final Linalg MLIR string.
    """
    from triton._C.libtriton import ir, passes, ascend

    # Step 1: compile to TTIR — get module directly (avoids reparse/dialect issues)
    module = _compile_matmul_module(M=M, N=N, K=K, NUM_CORES=NUM_CORES)
    context = module.context

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "native_matmul_dsa_linalg.mlir")

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
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_linalg] wrote Linalg IR ({len(mlir)} chars) to {path}")
    return mlir


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matmul kernel (tle.dsa.alloc/copy/to_tensor + tl.dot)")
    parser.add_argument("--M", type=int, default=_DEFAULT_M)
    parser.add_argument("--N", type=int, default=_DEFAULT_N)
    parser.add_argument("--K", type=int, default=_DEFAULT_K)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--dump-ttir", nargs="?", const="", default=None,
                        help="Dump TTIR to PATH and exit; no device needed.")
    parser.add_argument("--dump-tileir", nargs="?", const="", default=None,
                        help="Dump frontend TileIR with tile.* ops to PATH and exit; no device needed.")
    parser.add_argument("--dump-linalg", nargs="?", const="", default=None,
                        help="Dump Linalg IR to PATH and exit; no device needed.")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    num_cores = args.num_cores or get_number_cores()

    if args.dump_ttir is not None:
        dump_ttir(path=(args.dump_ttir or None), M=M, N=N, K=K, NUM_CORES=num_cores)
        raise SystemExit(0)

    if args.dump_tileir is not None:
        dump_tileir(path=(args.dump_tileir or None), M=M, N=N, K=K, NUM_CORES=num_cores)
        raise SystemExit(0)

    if args.dump_linalg is not None:
        dump_linalg(path=(args.dump_linalg or None), M=M, N=N, K=K, NUM_CORES=num_cores)
        raise SystemExit(0)

    # ---- functional test on device ------------------------------------------
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    mat_a = torch.randn((M, K), dtype=torch.float16, device=device)
    mat_b = torch.randn((K, N), dtype=torch.float16, device=device)

    mat_c = call(mat_a, mat_b)

    if not args.no_check:
        ref = torch.matmul(mat_a.float(), mat_b.float()).to(torch.float16)
        torch.testing.assert_close(ref, mat_c, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
