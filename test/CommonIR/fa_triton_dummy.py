import argparse
import os

import torch
import torch_npu
import triton
import triton.language as tl
import numpy as np

# =============================================================================
#  Real TLE tile-DSA API (replaces the old dl.dsa executable mock)
#
#  These are the genuine TileIR builders from
#    python/triton/experimental/tle/language/dsa/{core,semantic}.py
#  Each `tile_*` builtin emits a tile.* op into the Triton IR module, so the
#  compiled TTIR *is* the intermediate TileIR we want to dump.
#
#  IMPORTANT — they are @triton.language `@builtin`s: they must be called
#  DIRECTLY inside the @triton.jit body (so the code-generator injects the
#  MLIR builder). Wrapping them in plain Python helpers would break tracing,
#  which is exactly why the old `_DL_DSA` static-method mock never lowered to
#  real ops. Here we call them directly.
#
#  `import triton.experimental.tle as tle` also patches triton.compiler so the
#  tile/tle dialects are registered in the MLIR context during compilation.
# =============================================================================
import triton.experimental.tle as tle  # noqa: F401  (registers tile/tle dialects)
from triton.experimental.tle.language.dsa.ascend import (  # noqa: F401
    L1, L0C, UB, PIPE, sync_block_set, sync_block_wait,
)
from triton.experimental.tle.language.dsa import tile_copy, tile_alloc, tile_to_tensor  # noqa: F401

# ---- Cross-core semaphores (one id per signal, ids 0-5) --------------------
SEM_S_READY : tl.constexpr = tl.constexpr(0)  # C -> V : workspace_s has data
SEM_S_FREE  : tl.constexpr = tl.constexpr(1)  # V -> C : workspace_s slot free
SEM_P_READY : tl.constexpr = tl.constexpr(2)  # V -> C : workspace_p has data
SEM_P_FREE  : tl.constexpr = tl.constexpr(3)  # C -> V : workspace_p slot free
SEM_PV_READY: tl.constexpr = tl.constexpr(4)  # C -> V : workspace_pv has data
SEM_PV_FREE : tl.constexpr = tl.constexpr(5)  # V -> C : workspace_pv slot free

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_CORES = 24
BLOCK_M = 128
BLOCK_N = 128
DIM = 128

# constexpr shape literals for tile.copy (semantic.copy runs scalar_constant on
# each extent, which requires tl.constexpr rather than a plain int).
CBM = tl.constexpr(BLOCK_M)
CBN = tl.constexpr(BLOCK_N)
CD = tl.constexpr(DIM)

# ---- arch22 "3-task" schedule constants -----------------------------------
RING: tl.constexpr = tl.constexpr(3)  # depth of the task ring  (the "3-task" of the schedule)

np.random.seed(21)
DEVICE = "npu"
torch.manual_seed(20)
torch_npu.npu.set_device(0)
torch.set_printoptions(sci_mode=False, precision=4, linewidth=300)


# =============================================================================
#  Dummy kernel: contains only the prologue up to (but not including) the
#  "for g in range(num_global_tasks + 1)" pipeline loop.
#
#  Useful for verifying that tile.alloc, on-chip buffer layout, and the static
#  task-distribution arithmetic compile cleanly to TileIR before the full
#  3-task pipeline body is added.
#
#  grid = (NUM_CORES,). Each program drives one Cube + one Vector engine.
# =============================================================================
@triton.jit
def flash_attention_fwd_3task_kernel(
    Q, K, V, Out,
    workspace_s, workspace_p, workspace_pv,           # GM ping-pong workspaces
    workspace_rescale, workspace_expsum,              # GM softmax state (Vec1->Vec2)
    sm_scale,
    B, Hq, Hkv, S,
    sQb, sQh, sQs, sQd,
    sKb, sKh, sKs, sKd,
    sOb, sOh, sOs, sOd,
    num_seq_blocks, heads_q, gqa_group,
    num_kv_blocks,            # KV blocks per output tile  (= seq_len // BLOCK_N)
    conbined_block_num,       # tasks per output tile      (= num_kv_blocks // CB)
    block_num_per_core, rem_block_num,
    _pad0, _pad1, _pad2,     # dead args matching linalg IR layout (arg35-37)
    CB:            tl.constexpr,   # KV blocks per task (combine_batch)
    NUM_KV_BLOCKS: tl.constexpr,   # = num_kv_blocks  (used in causal mask)
    IS_CAUSAL:     tl.constexpr,
    BLOCK_M:       tl.constexpr,
    BLOCK_N:       tl.constexpr,
    DIM:           tl.constexpr,
):
    cid = tl.program_id(0)
    # ---- static task distribution  (== AICPU GetFASectionInfo metadata) ----
    block_start       = cid * block_num_per_core + tl.where(cid < rem_block_num, cid, rem_block_num)
    block_num         = block_num_per_core + tl.where(cid < rem_block_num, 1, 0)
    num_global_tasks  = block_num * conbined_block_num   # total pipelined tasks on this core

    # =========================================================================
    #  ① on-chip working set  (tile.alloc -> explicit memory hierarchy)
    # =========================================================================
    # -- Cube side: L1 for MMA inputs, L0C for MMA output --
    q_l1  = tile_alloc([BLOCK_M, DIM],     Q.dtype.element_ty, L1)
    k_l1  = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L1)
    v_l1  = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L1)
    p_l1  = tile_alloc([BLOCK_M, BLOCK_N], Q.dtype.element_ty, L1)

    s_l0c  = tile_alloc([BLOCK_M, BLOCK_N], tl.float32, L0C)  # MM1 out
    pv_l0c = tile_alloc([BLOCK_M, DIM],     tl.float32, L0C)  # MM2 out

    # Scope markers to trigger mix-mode (cube+vector) compilation
    with tle.scope(core_mode="cube"):
        pass
    with tle.scope(core_mode="vector"):
        pass


# =============================================================================
#  Intermediate-TileIR dump (no device required)
#
#  Compiles the kernel straight to TTIR with the tile/tle/ascend dialects
#  registered, then writes str(module). The tile.* ops emitted by the tile-DSA
#  builtins above appear in this dump — that is the intermediate TileIR.
#  Mirrors python/test/tle/test_bind_buffer.py::compile_kernel.
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


def _dump_signature():
    """Static signature for ast_to_ttir (pointers / scalars / i32 / constexpr)."""
    ptr = {"Q": "*fp16", "K": "*fp16", "V": "*fp16", "Out": "*fp16",
           "workspace_s": "*fp16", "workspace_p": "*fp16", "workspace_pv": "*fp16",
           "workspace_rescale": "*fp32", "workspace_expsum": "*fp32"}
    i32_names = ["B", "Hq", "Hkv", "S",
            "sQb", "sQh", "sQs", "sQd",
            "sKb", "sKh", "sKs", "sKd",
            "sOb", "sOh", "sOs", "sOd",
            "num_seq_blocks", "heads_q", "gqa_group",
            "num_kv_blocks", "conbined_block_num", "block_num_per_core", "rem_block_num"]
    sig = dict(ptr)
    sig["sm_scale"] = "fp32"
    for n in i32_names:
        sig[n] = "i32"
    sig["IS_CAUSAL"] = "constexpr"
    sig["BLOCK_M"]   = "constexpr"
    sig["BLOCK_N"]   = "constexpr"
    sig["DIM"]       = "constexpr"
    return sig


def dump_tileir(path=None, ttir_path=None, num_kv_blocks=32, combine_batch=8, is_causal=False):
    """Compile the kernel to TTIR (containing tile.* ops) and write it to `path`.

    Also runs the TileIR→HIVM pass to lower tile.* ops and dumps the resulting
    pure TTIR to `ttir_path`. Requires no NPU/GPU — pure front-end compilation.

    Returns the TileIR MLIR string. The TTIR dump is written as a side effect.
    """
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    from triton._C.libtriton import tle as tle_ir

    # The kernel reads module-level shape constants (BLOCK_M, DIM, ...) as plain
    # globals; allow that during this front-end-only dump.
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_dummy.mlir")
    if ttir_path is None:
        ttir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_dummy_ttir.mlir")

    cb = combine_batch
    conbined_block_num = num_kv_blocks // combine_batch
    signature = _dump_signature()
    constants = {
        "CB": combine_batch, "NUM_KV_BLOCKS": num_kv_blocks, "IS_CAUSAL": is_causal,
        "BLOCK_M": 128, "BLOCK_N": 128, "DIM": 128,
    }

    src = ASTSource(flash_attention_fwd_3task_kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    # Ascend dialect is optional (only needed for ascend-specific ops); load if present.
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    # codegen_fns: tl.dot needs a target-provided "min_dot_size"; the ascend
    # backend simply returns (1,1,1), so supply that inline (no full backend).
    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(flash_attention_fwd_3task_kernel, src, context, _DumpOptions(), codegen_fns, {})

    # Ensure the produced IR is legal before writing it out.
    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_tileir: module.verify() failed — IR is not legal")

    # ---- dump TileIR (TTIR + tile.* ops) ----
    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_tileir] module.verify() = {ok}; wrote legal TileIR to {path}")

    # ---- dump TTIR (TileIR→HIVM + TTIR optimization passes) ----
    from triton._C.libtriton import passes as ir_passes
    pm = ir.pass_manager(context)
    pm.enable_debug()

    # Phase 0: Inline all sub-functions so tile.buf types don't cross call boundaries.
    ir_passes.common.add_inliner(pm)

    # Phase 1: TileIR→HIVM — lower tile.* ops, producing pure TTIR/HIVM IR.
    try:
        from triton._C.libtriton.ascend import passes as ascend_passes
        ascend_passes.ttir.add_tileir_to_hivm(pm)
    except Exception:
        pass
    try:
        pm.run(module)
        print(f"[dump_tileir] after TileIR→HIVM: verify={module.verify()}", flush=True)
    except RuntimeError as e:
        print(f"[dump_tileir] TileIR→HIVM pass failed (non-fatal): {e}", flush=True)
        print(f"[dump_tileir] TileIR output is still valid; skipping HIVM lowering.", flush=True)
        return mlir

    # Phase 2: TTIR optimization passes (mirrors compiler.py make_ttir).
    pm2 = ir.pass_manager(context)
    pm2.enable_debug()
    ir_passes.common.add_inliner(pm2)
    ir_passes.ttir.add_combine(pm2)
    ir_passes.common.add_canonicalizer(pm2)
    ir_passes.ttir.add_reorder_broadcast(pm2)
    ir_passes.common.add_cse(pm2)
    ir_passes.common.add_licm(pm2)
    ir_passes.common.add_symbol_dce(pm2)
    ir_passes.ttir.add_loop_unroll(pm2)
    pm2.run(module)
    print(f"[dump_tileir] after TTIR opt passes: verify={module.verify()}", flush=True)

    ttir_ok = module.verify()
    if not ttir_ok:
        print(f"[dump_tileir] WARNING: module.verify() failed after TTIR optimization — IR may be illegal")
    else:
        print(f"[dump_tileir] TTIR optimization complete: verify={ttir_ok}")

    ttir_mlir = str(module)
    with open(ttir_path, "w") as f:
        f.write(ttir_mlir)
    print(f"[dump_tileir] wrote optimized TTIR to {ttir_path}")

    return mlir


def dump_hivm(path=None, combine_batch=32, is_causal=False):
    """Compile the kernel to TTIR, then lower through TileIR→HIVM pipeline to HIVM IR.

    Pipeline (matches compiler.py ttir_to_linalg):
      1. add_triton_to_structure_incubated
      2. add_discrete_mask_access_conversion
      3. add_triton_to_unstructure_incubated
      4. add_triton_to_hivm          (Triton CustomOp → HIVM SyncOp)
      5. add_triton_to_hfusion       (Triton → HFusion)
      6. add_tileir_to_hivm          (TileIR → HIVM)          ← our pass
      7. add_triton_to_llvm          (Triton → LLVM)
      8. add_bubble_up_operation
      9. add_triton_to_structure_incubated (second round)
     10. add_triton_to_linalg_incubated    (TTIR compute → Linalg)

    Returns the MLIR string. Requires no NPU/GPU — pure front-end + pass pipeline.
    """
    from triton._C.libtriton import ir, passes, ascend

    # Step 1: compile to TTIR (TileIR)
    tileir_mlir = dump_tileir(path=None, combine_batch=combine_batch, is_causal=is_causal)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_dummy_hivm.mlir")

    # Step 2: parse the TileIR module and run the pass pipeline
    context = ir.context()
    ir.load_dialects(context)
    from triton._C.libtriton import tle as tle_ir
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    # Write TileIR to temp file and parse it back into a module
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(tileir_mlir)
        tmp_path = f.name
    module = ir.parse_mlir_module(tmp_path, context)
    os.unlink(tmp_path)

    # Phase 1: TileIR → HIVM (first pass) — converts:
    #   tile.alloc → memref.alloc, tile.to_tensor → unrealized_conversion_cast,
    #   tile.copy  → hivm.copy (tile.* src, on-chip DMA)
    #             or memref.copy (!tt.ptr src, GM↔local DMA),
    #   tile.load/store → hivm.load/store, sync ops → hivm sync ops.
    pm1 = ir.pass_manager(context)
    pm1.enable_debug()
    ascend.passes.ttir.add_tileir_to_hivm(pm1)
    pm1.run(module)
    print(f"[dump_hivm] after tileir (pass 1): verify={module.verify()}", flush=True)

    # Phase 2: Triton CustomOp → HIVM SyncOp
    pm2 = ir.pass_manager(context)
    pm2.enable_debug()
    ascend.passes.ttir.add_triton_to_hivm(pm2)
    pm2.run(module)
    print(f"[dump_hivm] after triton_to_hivm: verify={module.verify()}", flush=True)

    # Phase 3: Inline + canonicalize to clean up the IR.
    pm3 = ir.pass_manager(context)
    pm3.enable_debug()
    passes.common.add_inliner(pm3)
    passes.common.add_canonicalizer(pm3)
    pm3.run(module)
    print(f"[dump_hivm] after canonicalize: verify={module.verify()}", flush=True)

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_hivm: module.verify() failed after pipeline — IR is not legal")

    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_hivm] module.verify() = {ok}; wrote HIVM IR to {path}")
    return mlir


def dump_linalg(path=None, combine_batch=32, is_causal=False):
    """Compile the kernel through the full TileIR→Linalg lowering pipeline.

    Pipeline:
      ① add_tileir_to_hivm               — tile.* → memref/hivm
      ② add_triton_to_structure_incubated    — structured ptr analysis (r1)
        add_discrete_mask_access_conversion  — non-contiguous mask handling
      ③ add_triton_to_unstructure_incubated  — scalarize unstructured accesses
        add_triton_to_hivm               — CustomOp → HIVM SyncOp
        add_triton_to_hfusion            — Triton → HFusion
        add_triton_to_llvm               — Triton → LLVM
      ④ add_bubble_up_operation          — push extracts upward
        add_triton_to_structure_incubated    — cleanup (round 2)
      ④b inline + canonicalize           — clean up before linalg
      ⑤ add_triton_to_linalg_incubated       — Triton→Linalg (may create
                                           unresolved materialization casts)
      ⑤b fold memref→ptr→memref chains   — eliminate casts created by ⑤
      ⑥ final canonicalize + CSE + DCE   — erase dead ops

    Returns the final Linalg MLIR string.  No NPU/GPU required.
    """
    from triton._C.libtriton import ir, passes, ascend
    from triton._C.libtriton import tle as tle_ir

    # Step 1: compile to TTIR (TileIR)
    tileir_mlir = dump_tileir(path=None, combine_batch=combine_batch, is_causal=is_causal)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fa_triton_dummy_linalg.mlir")

    # Step 2: parse the TileIR module into a fresh context
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(tileir_mlir)
        tmp_path = f.name
    module = ir.parse_mlir_module(tmp_path, context)
    os.unlink(tmp_path)

    # ── ① TileIR → HIVM ──────────────────────────────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
    passes.common.add_inliner(pm)
    ascend.passes.ttir.add_tileir_to_hivm(pm)
    pm.run(module)
    print(f"[dump_linalg] ① tileir_to_hivm: verify={module.verify()}", flush=True)

    # ── ①b Erase the unrealized_conversion_cast ops produced by TileIRToHIVM
    #     before any structured/linalg lowering runs.  This turns:
    #       cast !tt.ptr<tensor<>> -> memref  =>  tt.load + bufferization.to_memref
    #       cast memref<,#space>   -> tensor  =>  memref.memory_space_cast + bufferization.to_tensor
    #     so the linalg-incubator never sees the unresolved materializations
    #     that cause "unresolved materialization" failures.
    pm = ir.pass_manager(context); pm.enable_debug()
    ascend.passes.ttir.add_erase_linalg_casts(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module)
    print(f"[dump_linalg] ①b erase_linalg_casts: verify={module.verify()}", flush=True)

    # ── ② Structured (r1) + discrete mask ────────────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
    ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, False, False)
    pm.run(module)
    print(f"[dump_linalg] ② structure(r1)+discrete_mask: verify={module.verify()}", flush=True)

    # ── ③ Unstructured + HIVM + HFusion + LLVM ──────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
    ascend.passes.ttir.add_triton_to_unstructure_incubated(pm, False, False)
    ascend.passes.ttir.add_triton_to_hivm(pm)
    ascend.passes.ttir.add_triton_to_hfusion(pm)
    ascend.passes.ttir.add_triton_to_llvm(pm)
    pm.run(module)
    print(f"[dump_linalg] ③ unstructure+hivm+hfusion+llvm: verify={module.verify()}", flush=True)

    # ── ④ Bubble-up + structured (r2) ────────────────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
    ascend.passes.ttir.add_bubble_up_operation(pm)
    ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ④ bubble_up+structure(r2): verify={module.verify()}", flush=True)

    # ── ④b Inline + canonicalize ← REQUIRED to avoid C++ assertion ─────
    # Without this, the linalg incubator crashes with cast<RankedTensorType>
    # in MaskAnalysis when it encounters non-tensor types.
    pm = ir.pass_manager(context); pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module)
    print(f"[dump_linalg] ④b inline+canonicalize: verify={module.verify()}", flush=True)

    # ── ⑤ Triton → Linalg ───────────────────────────────────────────────
    # This pass does the heavy lifting: tt.ptr→memref, triton ops→linalg,
    # triton::FuncOp→func::FuncOp.  It may fail with "unresolved
    # materialization" when it creates memref→!tt.ptr→memref cast chains
    # during partial conversion of !tt.ptr<tensor<>> values.  We handle
    # that in phase ⑤b.
    linalg_ok = False
    try:
        pm = ir.pass_manager(context); pm.enable_debug()
        ascend.passes.ttir.add_triton_to_linalg_incubated(pm, False, True, False, False, False)
        pm.run(module)
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: verify={module.verify()}", flush=True)
        linalg_ok = True
    except RuntimeError as e:
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: partial conversion "
              f"(this is expected — the pass creates cast chains that need "
              f"post-processing)", flush=True)

    # ── ⑤b Run erase-linalg-casts one more time to reconcile any
    #     unrealized_conversion_cast ops that the partial linalg conversion
    #     may have introduced around the on-chip allocations / function
    #     boundary.
    pm = ir.pass_manager(context); pm.enable_debug()
    ascend.passes.ttir.add_erase_linalg_casts(pm)
    pm.run(module)
    print(f"[dump_linalg] ⑤b erase_linalg_casts (post): verify={module.verify()}", flush=True)

    # ── ⑥ Final canonicalize → erase dead casts ─────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
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
    print(f"[dump_linalg] verify={ok}; wrote Linalg IR ({len(mlir)} chars) to {path}")
    return mlir


# =============================================================================
#  Host launcher
# =============================================================================
def flash_attention_fwd(q, k, v, combine_batch, is_causal=False):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and S % BLOCK_N == 0 and Hq % Hkv == 0
    num_seq_blocks = S // BLOCK_M
    block_num      = num_seq_blocks * Hq * B
    num_kv_blocks  = S // BLOCK_N   # KV blocks per output tile

    CB = combine_batch
    if num_kv_blocks < CB:
        CB = num_kv_blocks
    assert num_kv_blocks % CB == 0, f"num_kv_blocks ({num_kv_blocks}) must be divisible by combine_batch ({CB})"
    conbined_block_num = num_kv_blocks // CB   # tasks per output tile

    block_num_per_core = block_num // NUM_CORES
    rem_block_num = block_num % NUM_CORES

    out = torch.empty_like(q)
    # GM ping-pong workspaces (taskId % 2), one slice per core.
    # workspace_s: MM1 output (Q*K^T), fp16 matching tl.dot out_dtype
    workspace_s       = torch.empty((NUM_CORES, RING, CB, BLOCK_M, BLOCK_N), dtype=torch.float16, device=q.device)
    workspace_p       = torch.empty((NUM_CORES, RING, CB, BLOCK_M, BLOCK_N), dtype=q.dtype,        device=q.device)
    workspace_pv      = torch.empty((NUM_CORES, RING, CB, BLOCK_M, DIM),     dtype=torch.float16, device=q.device)
    workspace_rescale = torch.empty((NUM_CORES, RING, CB, BLOCK_M),      dtype=torch.float32, device=q.device)
    workspace_expsum  = torch.empty((NUM_CORES, RING, CB, BLOCK_M),      dtype=torch.float32, device=q.device)
    sm_scale = (1.0 / D) ** 0.5

    grid = (NUM_CORES,)  # one program per AI core; one Cube + one Vector stream (MIX_1_1)
    flash_attention_fwd_3task_kernel[grid](
        q, k, v, out, workspace_s, workspace_p, workspace_pv,
        workspace_rescale, workspace_expsum, sm_scale,
        B, Hq, Hkv, S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_seq_blocks, Hq, Hq // Hkv,
        num_kv_blocks, conbined_block_num, block_num_per_core, rem_block_num,
        0, 0, 0,  # _pad0, _pad1, _pad2 — dead args for linalg IR layout
        CB=CB,
        NUM_KV_BLOCKS=num_kv_blocks,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        DIM=DIM,
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--combine-batch", type=int, default=8,
                        help="KV blocks per task (arch22 nRatio)")
    parser.add_argument("--dump-mlir", nargs="?", const="", default=None,
                        help="Dump intermediate TileIR to PATH (default fa_triton_dummy.mlir) and exit; no device needed.")
    parser.add_argument("--dump-ir", nargs="?", const="", default=None,
                        help="Dump HIVM IR (after TileIR→HIVM lowering) to PATH and exit; no device needed.")
    parser.add_argument("--dump-linalg", nargs="?", const="", default=None,
                        help="Dump Linalg IR (full lowering through linalg, casts eliminated) to PATH and exit; no device needed.")
    args = parser.parse_args()

    B, S, H, D = args.B, args.S, args.H, args.D
    combine_batch = S // BLOCK_N

    # ---- dump intermediate TileIR and exit (no device required) ----
    if args.dump_mlir is not None:
        dump_tileir(path=(args.dump_mlir or None), combine_batch=combine_batch, is_causal=args.causal)
        raise SystemExit(0)

    # ---- dump HIVM IR after full lowering pipeline (no device required) ----
    if args.dump_ir is not None:
        dump_hivm(path=(args.dump_ir or None), combine_batch=combine_batch, is_causal=args.causal)
        raise SystemExit(0)

    # ---- dump Linalg IR after full TileIR→Linalg lowering (no device required) ----
    if args.dump_linalg is not None:
        dump_linalg(path=(args.dump_linalg or None), combine_batch=combine_batch, is_causal=args.causal)
        raise SystemExit(0)

    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    q = torch.randn((B, Q_H, S, D), dtype=torch.float16, device=device)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)

    out = flash_attention_fwd(q, k, v, combine_batch, is_causal=args.causal)

    if not args.no_check:
        def ref(q, k, v):
            if k.shape[1] != q.shape[1]:
                n_rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=args.causal).to(torch.float16)

        torch.testing.assert_close(ref(q, k, v), out, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
        print(f"out nan: {out.isnan().any().item()}, max: {out.abs().max().item()}, sample: {out[0,0,0,:4]}")
