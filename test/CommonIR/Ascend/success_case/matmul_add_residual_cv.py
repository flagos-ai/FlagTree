import argparse
import os

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle  # noqa: F401  (registers tile/tle dialects)
from triton.experimental.tle.language.dsa.ascend import (  # noqa: F401
    L1, L0C, UB, PIPE, sync_block_set, sync_block_wait,
)
from triton.experimental.tle.language.dsa import tile_alloc, tile_copy, tile_to_tensor  # noqa: F401

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

# =============================================================================
#  Cross-core semaphore IDs
#
#  SEM_MM_READY : cube  → vector : workspace slot has matmul result
#  SEM_WS_FREE  : vector → cube  : workspace slot is free for next matmul
# =============================================================================
SEM_MM_READY: tl.constexpr = tl.constexpr(0)
SEM_WS_FREE: tl.constexpr = tl.constexpr(1)


def get_number_cores():
    """Return the number of AI cores to use as the launch grid size."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.get_device_properties(0).ai_core_num
    except Exception:
        return _DEFAULT_NUM_CORES


# =============================================================================
#  Cube sub-function: matmul one (BLOCK_M, BLOCK_N) output tile
#
#  Waits for the workspace slot to be free (SEM_WS_FREE), executes the
#  K-loop tl.dot, stores the fp16 result into workspace[cid, :, :] in GM,
#  then signals SEM_MM_READY so the vector core can proceed.
# =============================================================================
@triton.jit
def _cube_matmul(
    mat_a,
    mat_b,
    a_l1,
    b_l1,
    workspace,
    cid,
    m_start,
    n_start,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Cube core: compute one (BLOCK_M, BLOCK_N) matmul tile and store to GM workspace."""
    # wait workspace slot free (vector released it; or init token on first tile)
    sync_block_wait("vector", "cube", SEM_WS_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    NUM_K_BLOCKS = tl.cdiv(K, BLOCK_K)

    a_block_ptr = tl.make_block_ptr(mat_a, (M, K), (K, 1), (m_start, 0), (BLOCK_M, BLOCK_K), (1, 0))
    b_block_ptr = tl.make_block_ptr(mat_b, (K, N), (N, 1), (0, n_start), (BLOCK_K, BLOCK_N), (1, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, NUM_K_BLOCKS):
        tile_copy(a_block_ptr, a_l1, [BLOCK_M, BLOCK_K])
        tile_copy(b_block_ptr, b_l1, [BLOCK_K, BLOCK_N])
        acc = tl.dot(tile_to_tensor(a_l1, writable=False), tile_to_tensor(b_l1, writable=False), acc,
                     out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_K])
        b_block_ptr = tl.advance(b_block_ptr, [BLOCK_K, 0])

    # store matmul result into per-core GM workspace slot
    ws_ptr = tl.make_block_ptr(workspace + cid * BLOCK_M * BLOCK_N, (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                               (BLOCK_M, BLOCK_N), (1, 0))
    tl.store(ws_ptr, acc.to(workspace.dtype.element_ty))

    # notify vector core: workspace has matmul result
    sync_block_set("cube", "vector", SEM_MM_READY, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


# =============================================================================
#  Vector sub-function: add residual to the matmul result in GM workspace
#
#  Waits for SEM_MM_READY, loads workspace tile, adds the corresponding
#  residual tile, writes output to mat_c, then signals SEM_WS_FREE.
# =============================================================================
@triton.jit
def _vector_add_residual(
    residual,
    mat_c,
    workspace,
    cid,
    m_start,
    n_start,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Vector core: load workspace + residual, add element-wise, store to output."""
    # wait matmul result ready from cube core
    sync_block_wait("cube", "vector", SEM_MM_READY, PIPE.PIPE_MTE2, PIPE.PIPE_V)

    # load matmul result from GM workspace
    ws_ptr = tl.make_block_ptr(workspace + cid * BLOCK_M * BLOCK_N, (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                               (BLOCK_M, BLOCK_N), (1, 0))
    mm_tile = tl.load(ws_ptr).to(tl.float32)

    # load residual tile
    res_ptr = tl.make_block_ptr(residual, (M, N), (N, 1), (m_start, n_start), (BLOCK_M, BLOCK_N), (1, 0))
    res_tile = tl.load(res_ptr).to(tl.float32)

    # element-wise add and store to output
    out_tile = mm_tile + res_tile
    out_ptr = tl.make_block_ptr(mat_c, (M, N), (N, 1), (m_start, n_start), (BLOCK_M, BLOCK_N), (1, 0))
    tl.store(out_ptr, out_tile.to(mat_c.dtype.element_ty))

    # notify cube core: workspace slot is free
    sync_block_set("vector", "cube", SEM_WS_FREE, PIPE.PIPE_V, PIPE.PIPE_MTE2)


# =============================================================================
#  Main kernel: matmul + residual with cube/vector core separation
#
#  C = A @ B + residual
#
#  grid = (NUM_CORES,). Each program drives one Cube engine and one Vector
#  engine that work sequentially:
#    1. Cube  computes A @ B for each (BLOCK_M, BLOCK_N) tile and stores the
#             fp16 result into a per-core GM workspace slot.
#    2. Vector reads the workspace tile, adds the residual tile, and writes
#             the final result to mat_c.
#  The two engines are ordered by a pair of cross-core semaphores passed
#  through GM (sync_block_set / sync_block_wait on SEM_MM_READY / SEM_WS_FREE).
# =============================================================================
@triton.jit
def matmul_add_residual_cv_kernel(
    mat_a,
    mat_b,
    mat_c,
    residual,
    workspace,  # GM scratch: [NUM_CORES, BLOCK_M, BLOCK_N] fp16
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    cid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # On-chip buffers for cube engine
    a_l1 = tile_alloc([BLOCK_M, BLOCK_K], mat_a.dtype.element_ty, tle.language.dsa.ascend.L1)
    b_l1 = tile_alloc([BLOCK_K, BLOCK_N], mat_b.dtype.element_ty, tle.language.dsa.ascend.L1)

    # =========================================================================
    #  init: pre-arm one SEM_WS_FREE token so the cube engine can enter its
    #  first tile without stalling (mirrors fa_triton_arch.py init pattern).
    # =========================================================================
    with tle.scope(core_mode="vector"):
        sync_block_set("vector", "cube", SEM_WS_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    # =========================================================================
    #  Main loop: each core handles output tiles in round-robin order.
    #  Within each tile: cube first, then vector (sequential via semaphores).
    # =========================================================================
    for block_idx in range(cid, NUM_BLOCKS, NUM_CORES):
        pid_m = block_idx // NUM_BLOCKS_N
        pid_n = block_idx % NUM_BLOCKS_N
        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N

        # -----------------------------------------------------------------
        #  Cube scope: matmul → GM workspace
        # -----------------------------------------------------------------
        with tle.scope(core_mode="cube"):
            _cube_matmul(
                mat_a,
                mat_b,
                a_l1,
                b_l1,
                workspace,
                cid,
                m_start,
                n_start,
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )

        # -----------------------------------------------------------------
        #  Vector scope: add residual, write output
        # -----------------------------------------------------------------
        with tle.scope(core_mode="vector"):
            _vector_add_residual(
                residual,
                mat_c,
                workspace,
                cid,
                m_start,
                n_start,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
            )

    # =========================================================================
    #  destroy: consume the outstanding SEM_WS_FREE token that would be left
    #  over after the last tile's vector scope re-signals it (mirrors fa init
    #  token teardown pattern).
    # =========================================================================
    with tle.scope(core_mode="cube"):
        sync_block_wait("vector", "cube", SEM_WS_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)


# =============================================================================
#  Host-side launch
# =============================================================================
def call(mat_a, mat_b, residual, num_cores=_DEFAULT_NUM_CORES):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    # GM workspace: one [BLOCK_M, BLOCK_N] fp16 slot per core
    workspace = torch.empty(num_cores, BLOCK_M, BLOCK_N, dtype=mat_a.dtype, device=mat_a.device)
    matmul_add_residual_cv_kernel[(num_cores, )](mat_a, mat_b, mat_c, residual, workspace, m, n, k, num_cores,
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


def _signature():
    """Static signature for ast_to_ttir — non-constexpr args only."""
    return {
        "mat_a": "*fp16",
        "mat_b": "*fp16",
        "mat_c": "*fp16",
        "residual": "*fp16",
        "workspace": "*fp16",
        "M": "i32",
    }


def dump_ttir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K, NUM_CORES=_DEFAULT_NUM_CORES, return_module=False):
    """Compile matmul_add_residual_cv_kernel to TTIR and write to *path*."""
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    from triton._C.libtriton import tle as tle_ir

    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matmul_add_residual_cv.mlir")

    signature = _signature()
    constants = {
        "N": N,
        "K": K,
        "NUM_CORES": NUM_CORES,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
    }

    src = ASTSource(matmul_add_residual_cv_kernel.fn, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.dsa_ir.load_tile_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(matmul_add_residual_cv_kernel, src, context, _DumpOptions(), codegen_fns, {})

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
def dump_linalg(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K, NUM_CORES=_DEFAULT_NUM_CORES):
    """Compile matmul_add_residual_cv_kernel through full TileIR → Linalg lowering pipeline.

    Pipeline:
      ① tileir_to_hivm            — tile.* → memref/hivm
      ①b erase_linalg_casts       — eliminate unrealized casts
      ② structure(r1) + discrete mask
      ③ unstructure + hivm + hfusion + llvm
      ④ bubble_up + structure(r2)
      ④b inline + canonicalize
      ⑤ triton_to_linalg_incubated
      ⑤c fold_staging_copy
      ⑤b erase_linalg_casts (post)
      ⑥ final canonicalize + CSE + DCE

    Returns the final Linalg MLIR string.
    """
    from triton._C.libtriton import ir, passes, ascend

    # Step 1: compile to TTIR — get module directly (avoids reparse/dialect issues)
    module = dump_ttir(path=None, M=M, N=N, K=K, NUM_CORES=NUM_CORES, return_module=True)
    context = module.context

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matmul_add_residual_cv_linalg.mlir")

    # ── ① TileIR → HIVM ──────────────────────────────────────────────────
    pm = ir.pass_manager(context)
    passes.common.add_inliner(pm)
    ascend.passes.ttir.add_tileir_to_hivm(pm)
    pm.run(module)
    print(f"[dump_linalg] ① tileir_to_hivm: verify={module.verify()}", flush=True)

    # ── ①b Erase unrealized_conversion_cast ops ──────────────────────────
    # pm = ir.pass_manager(context)
    # ascend.passes.ttir.add_erase_linalg_casts(pm)
    # passes.common.add_canonicalizer(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ①b erase_linalg_casts: verify={module.verify()}", flush=True)

    # ── ② Structured (r1) + discrete mask ────────────────────────────────
    pm = ir.pass_manager(context)
    # ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ② structure(r1)+discrete_mask: verify={module.verify()}", flush=True)

    # ── ③ Unstructured + HIVM + HFusion + LLVM ──────────────────────────
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_triton_to_unstructure(pm, False, False)
    ascend.passes.ttir.add_triton_to_hivm(pm)
    ascend.passes.ttir.add_triton_to_hfusion(pm)
    ascend.passes.ttir.add_triton_to_llvm(pm)
    pm.run(module)
    print(f"[dump_linalg] ③ unstructure+hivm+hfusion+llvm: verify={module.verify()}", flush=True)

    # ── ④ Bubble-up + structured (r2) ────────────────────────────────────
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_bubble_up_operation(pm)
    # ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
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
        ascend.passes.ttir.add_triton_to_linalg(pm, False, True, False, False, False)
        pm.run(module)
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: verify={module.verify()}", flush=True)
    except RuntimeError as e:
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: partial ({e})", flush=True)

    # ── ⑤c Fold staging copies (eliminate redundant default-space allocs) ─
    # pm = ir.pass_manager(context)
    # ascend.passes.ttir.add_fold_staging_copy(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ⑤c fold_staging_copy: verify={module.verify()}", flush=True)

    # ── ⑤b Erase linalg casts (post) ────────────────────────────────────
    # pm = ir.pass_manager(context)
    # ascend.passes.ttir.add_erase_linalg_casts(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ⑤b erase_linalg_casts (post): verify={module.verify()}", flush=True)

    # ── ⑥ Final canonicalize + CSE + DCE ─────────────────────────────────
    # pm = ir.pass_manager(context)
    # passes.common.add_canonicalizer(pm)
    # passes.common.add_cse(pm)
    # passes.common.add_symbol_dce(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ⑥ final cleanup: verify={module.verify()}", flush=True)

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
    parser = argparse.ArgumentParser(
        description="Matmul + residual kernel with cube/vector core split (C = A @ B + residual)")
    parser.add_argument("--M", type=int, default=_DEFAULT_M)
    parser.add_argument("--N", type=int, default=_DEFAULT_N)
    parser.add_argument("--K", type=int, default=_DEFAULT_K)
    parser.add_argument("--num-cores", type=int, default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--dump-ttir", nargs="?", const="", default=None,
                        help="Dump TTIR to PATH and exit; no device needed.")
    parser.add_argument("--dump-linalg", nargs="?", const="", default=None,
                        help="Dump Linalg IR to PATH and exit; no device needed.")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    num_cores = args.num_cores or get_number_cores()

    if args.dump_ttir is not None:
        dump_ttir(path=(args.dump_ttir or None), M=M, N=N, K=K, NUM_CORES=num_cores)
        raise SystemExit(0)

    if args.dump_linalg is not None:
        dump_linalg(path=(args.dump_linalg or None), M=M, N=N, K=K, NUM_CORES=num_cores)
        raise SystemExit(0)

    # ---- functional test on device ------------------------------------------
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    mat_a = torch.randn((M, K), dtype=torch.float16, device=device)
    mat_b = torch.randn((K, N), dtype=torch.float16, device=device)
    residual = torch.randn((M, N), dtype=torch.float16, device=device)

    mat_c = call(mat_a, mat_b, residual, num_cores)

    if not args.no_check:
        ref = (torch.matmul(mat_a.float(), mat_b.float()) + residual.float()).to(torch.float16)
        torch.testing.assert_close(ref, mat_c, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
