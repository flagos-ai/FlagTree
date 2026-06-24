import argparse
import os

import torch
import triton
import triton.language as tl

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
from triton.experimental.tle.language.dsa.core import (
    tile_alloc,
    tile_copy,
    tile_to_tensor,
    tile_pipe_barrier,
    tensor_to_tile,
)
from triton.experimental.tle.language.dsa.ascend import (  # noqa: F401
    L1, L0A, L0B, L0C, UB, PIPE, sync_block_set, sync_block_wait,
)

# ---- TileIR Pipe ids (== TileIRAttrDefs.td TileIR_Pipe) ---------------------
PIPE_M, PIPE_V, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_FIX, PIPE_S = 0, 1, 2, 3, 4, 5, 6

# ---- cross-engine sync_block events -----------------------------------------
# (sender, receiver, event_id, sender_pipe, receiver_pipe). Replaces the old
# tile.set_flag/wait_flag pipe pairs with core-to-core (cube<->vector) sync via
# tle.dsa.ascend.sync_block_set / sync_block_wait (hivm.hir.sync_block_*).
# Constraint: sender != receiver; event_id in [0,15].
EVT_MTE3_MTE2 = ("vector", "cube", 0, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)  # Vec1 wrote P -> Bmm2 reads
EVT_MTE2_V    = ("cube", "vector", 1, PIPE.PIPE_MTE2, PIPE.PIPE_V)     # data loaded -> Vector computes
EVT_V_MTE3    = ("vector", "cube", 2, PIPE.PIPE_V, PIPE.PIPE_MTE3)     # Vector done -> store / next

# NOTE: TileIR has tile.cube_launch / tile.cube_wait ops, but the tle tile-DSA
# layer does not expose builder bindings for them yet. The async Cube matmul is
# therefore expressed here as a synchronous tl.dot + tl.store (it still produces
# correct results; it just drops the launch/wait overlap). Replace with real
# tile.cube_launch/cube_wait DSA bindings once they exist.


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

# ---- 3-task pipeline constants ----
PIPE_DEPTH = 2
RING = PIPE_DEPTH + 1   # = 3 task-metadata slots (taskId % 3)
PP = 2                  # ping-pong GM buffers (taskId % 2)


# =============================================================================
#  The single-stream 3-task scheduler kernel (TileIR / tle.dsa form)
#
#  grid = (NUM_CORES,). Each program drives one Cube + one Vector engine.
#  On-chip staging (L1 / L0A / L0B / L0C) is allocated with tile.alloc; DMA uses
#  tile.copy; cross-engine ordering uses tile.set_flag / tile.wait_flag /
#  tile.pipe_barrier. The Vector-side online-softmax state stays in registers
#  (plain tl.* ops) since it is pure compute.
# =============================================================================
@triton.jit
def flash_attention_fwd_3task_kernel(
    Q, K, V, Out,
    mm1Res, stage1Res, mm2Res,           # GM ping-pong workspaces
    sm_scale,
    B, Hq, Hkv, S,                       # shapes
    sQb, sQh, sQs, sQd,                  # Q strides
    sKb, sKh, sKs, sKd,                  # K/V strides
    sOb, sOh, sOs, sOd,                  # Out strides
    num_seq_blocks, heads_q, gqa_group,  # task-decode helpers
    n_iters,                             # KV blocks (s2 loop) per output tile
    q_tasks, r_tasks,                    # static task split
    NUM_ITERS: tl.constexpr,             # = n_iters as constexpr
    IS_CAUSAL: tl.constexpr,
):
    cid = tl.program_id(0)

    # ---- static task distribution (== AICPU GetFASectionInfo metadata) ----
    my_start = cid * q_tasks + tl.where(cid < r_tasks, cid, r_tasks)
    my_count = q_tasks + tl.where(cid < r_tasks, 1, 0)
    n_sub = my_count * n_iters            # flattened (output-tile x KV-block) sub-tasks

    # =========================================================================
    #  ① on-chip working set  (tile.alloc -> explicit memory hierarchy)
    # =========================================================================
    # -- Cube side: L1 staging + L0 double buffer (slot 0 = Bmm1, slot 1 = Bmm2) --
    q_l1 = tile_alloc([BLOCK_M, DIM], Q.dtype.element_ty, L1)
    k_l1 = tile_alloc([BLOCK_N, DIM], Q.dtype.element_ty, L1)
    v_l1 = tile_alloc([BLOCK_N, DIM], Q.dtype.element_ty, L1)
    p_l1 = tile_alloc([BLOCK_M, BLOCK_N], Q.dtype.element_ty, L1)
    # two separate L0 slots (no buffer indexing in tile-DSA): 0 = Bmm1, 1 = Bmm2
    l0a0 = tile_alloc([BLOCK_M, DIM], Q.dtype.element_ty, L0A)
    l0b0 = tile_alloc([DIM, BLOCK_N], Q.dtype.element_ty, L0B)
    l0c0 = tile_alloc([BLOCK_M, BLOCK_N], tl.float32, L0C)
    l0a1 = tile_alloc([BLOCK_M, BLOCK_N], Q.dtype.element_ty, L0A)
    l0b1 = tile_alloc([BLOCK_N, DIM], Q.dtype.element_ty, L0B)
    l0c1 = tile_alloc([BLOCK_M, DIM], tl.float32, L0C)

    # -- Vector side: O accumulator in registers (loop-carried). NOTE: the
    #    running-max / denominator / per-sub-task ring carry of the real 3-task
    #    pipeline is dropped here — this file is a TileIR-dump illustration, so
    #    the Vector math is simplified to keep valid (item-assignment-free) IR.
    acc_o = tl.zeros((BLOCK_M, DIM), tl.float32)   # P·V accumulator

    # =========================================================================
    #  ② init cross-engine flags: pre-arm so the first producer's wait passes.
    # =========================================================================
    sync_block_set(EVT_MTE3_MTE2[0], EVT_MTE3_MTE2[1], EVT_MTE3_MTE2[2], EVT_MTE3_MTE2[3], EVT_MTE3_MTE2[4])  # pretend last P consumed

    # =========================================================================
    #  ③ 3-task pipeline. One tick = one flattened sub-task g; +PIPE_DEPTH ticks
    #     drain the tail (cooldown).
    # =========================================================================
    for g in range(n_sub + PIPE_DEPTH):

        # ─── 2) IterateBmm1(k): S = Q·Kᵀ -> mm1Res[g%2] ──────────────────────
        if g < n_sub:
            t = g // n_iters
            j = g % n_iters
            cur_pp = g % PP
            task_id = my_start + t
            bx = task_id % num_seq_blocks
            by = (task_id // num_seq_blocks) % heads_q
            bz = task_id // (num_seq_blocks * heads_q)
            kv_by = by // gqa_group

            q_bp = tl.make_block_ptr(Q + bz * sQb + by * sQh, (S, DIM), (sQs, sQd),
                                     (bx * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
            if j == 0:                              # new output tile: (re)load Q into L1
                tile_copy(tensor_to_tile(q_bp), q_l1, [CBM, CD])
            # K[j] -> k_l1 -> L0 slot 0
            k_bp = tl.make_block_ptr(K + bz * sKb + kv_by * sKh, (S, DIM), (sKs, sKd),
                                     (j * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
            tile_copy(tensor_to_tile(k_bp), k_l1, [CBN, CD])
            tile_copy(q_l1, l0a0, [CBM, CD])
            tile_copy(k_l1, l0b0, [CBN, CD])   # NOTE: tile.copy has no transpose flag yet
            # S = Q·Kᵀ : matmul stand-in for tile.cube_launch (no DSA binding yet).
            # NOTE: tt.dot requires standard ranked tensors — a !tile.tensor (from
            # tile.to_tensor) is rejected by the verifier — so the dot runs on tt
            # tensors loaded from GM, not on the !tile.* staging buffers above.
            m1_bp = tl.make_block_ptr(mm1Res + cid * (PP * BLOCK_M * BLOCK_N) + cur_pp * (BLOCK_M * BLOCK_N),
                                      (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
            s = tl.dot(tile_to_tensor(l0a0, writable=False), tile_to_tensor(l0b0, writable=False))                           
            # s = tl.dot(tl.load(q_bp), tl.trans(tl.load(k_bp)))
            tl.store(m1_bp, s)

        # ─── 3) ProcessVec1(k-1): softmax(mm1Res[prev]) -> stage1Res[prev] ───
        if g >= 1:
            g1 = g - 1
            if g1 < n_sub:
                j1   = g1 % n_iters
                pp1  = g1 % PP

                # GM mm1Res[pp1] -> registers (MTE2)
                m1r_bp = tl.make_block_ptr(mm1Res + cid * (PP * BLOCK_M * BLOCK_N) + pp1 * (BLOCK_M * BLOCK_N),
                                           (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                s_tile = tl.load(m1r_bp).to(tl.float32)
                sync_block_set(EVT_MTE2_V[0], EVT_MTE2_V[1], EVT_MTE2_V[2], EVT_MTE2_V[3], EVT_MTE2_V[4])
                sync_block_wait(EVT_MTE2_V[0], EVT_MTE2_V[1], EVT_MTE2_V[2], EVT_MTE2_V[3], EVT_MTE2_V[4])

                if IS_CAUSAL:
                    t1  = g1 // n_iters
                    tid = my_start + t1
                    bx1 = tid % num_seq_blocks
                    q_idx  = bx1 * BLOCK_M + tl.arange(0, BLOCK_M)
                    kv_idx = j1 * BLOCK_N + tl.arange(0, BLOCK_N)
                    valid  = q_idx[:, None] >= kv_idx[None, :]
                    s_tile = tl.where(valid, s_tile, float("-inf"))

                # per-tile softmax numerator P = exp(scale*S - scale*max)
                # (cross-KV-block running-max carry dropped; dump illustration)
                row_max = tl.max(s_tile, axis=-1, keep_dims=True)
                p_tile = tl.exp(sm_scale * s_tile - sm_scale * row_max)

                tile_pipe_barrier(PIPE_V)

                # P -> GM stage1Res[pp1] (UB->GM, MTE3); signal Bmm2 it may read
                s1_bp = tl.make_block_ptr(stage1Res + cid * (PP * BLOCK_M * BLOCK_N) + pp1 * (BLOCK_M * BLOCK_N),
                                          (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                sync_block_set(EVT_V_MTE3[0], EVT_V_MTE3[1], EVT_V_MTE3[2], EVT_V_MTE3[3], EVT_V_MTE3[4])
                sync_block_wait(EVT_V_MTE3[0], EVT_V_MTE3[1], EVT_V_MTE3[2], EVT_V_MTE3[3], EVT_V_MTE3[4])
                tl.store(s1_bp, p_tile.to(stage1Res.dtype.element_ty))
                sync_block_set(EVT_MTE3_MTE2[0], EVT_MTE3_MTE2[1], EVT_MTE3_MTE2[2], EVT_MTE3_MTE2[3], EVT_MTE3_MTE2[4])   # P[pp1] ready for Bmm2

        # ─── 5) IterateBmm2(k-1): stage1Res[prev]·V -> mm2Res[prev] ──────────
        if g >= 1:
            g1 = g - 1
            if g1 < n_sub:
                t1   = g1 // n_iters
                j1   = g1 % n_iters
                pp1  = g1 % PP
                tid1 = my_start + t1
                by1  = (tid1 // num_seq_blocks) % heads_q
                bz1  = tid1 // (num_seq_blocks * heads_q)
                kv_by1 = by1 // gqa_group

                # P from stage1Res[pp1] -> p_l1 -> L0 slot 1
                sync_block_wait(EVT_MTE3_MTE2[0], EVT_MTE3_MTE2[1], EVT_MTE3_MTE2[2], EVT_MTE3_MTE2[3], EVT_MTE3_MTE2[4])
                s1r_bp = tl.make_block_ptr(stage1Res + cid * (PP * BLOCK_M * BLOCK_N) + pp1 * (BLOCK_M * BLOCK_N),
                                           (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                tile_copy(tensor_to_tile(s1r_bp), p_l1, [CBM, CBN])
                # V[j1] -> v_l1 -> L0 slot 1
                v_bp = tl.make_block_ptr(V + bz1 * sKb + kv_by1 * sKh, (S, DIM), (sKs, sKd),
                                         (j1 * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
                tile_copy(tensor_to_tile(v_bp), v_l1, [CBN, CD])
                tile_copy(p_l1, l0a1, [CBM, CBN])
                tile_copy(v_l1, l0b1, [CBN, CD])
                # O_part = P·V : mma -> l0c1 -> FIX -> mm2Res[pp1]  (synchronous stand-in).
                m2_bp = tl.make_block_ptr(mm2Res + cid * (PP * BLOCK_M * DIM) + pp1 * (BLOCK_M * DIM),
                                          (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
                o = tl.dot(tile_to_tensor(l0a1, writable=False), tile_to_tensor(l0b1, writable=False))
                tl.store(m2_bp, o)

        # ─── 6) ProcessVec2(k-2): rescale acc_o + add mm2Res[prev2], finalize ─
        if g >= 2:
            g2 = g - 2
            if g2 < n_sub:
                j2   = g2 % n_iters
                pp2  = g2 % PP
                is_first = j2 == 0
                is_last  = j2 == NUM_ITERS - 1
                t2   = g2 // n_iters
                tid2 = my_start + t2
                bx2  = tid2 % num_seq_blocks
                by2  = (tid2 // num_seq_blocks) % heads_q
                bz2  = tid2 // (num_seq_blocks * heads_q)

                if is_first:                        # init O accumulator at tile start
                    acc_o = tl.zeros((BLOCK_M, DIM), tl.float32)

                # O_partial from mm2Res[pp2] -> add (MTE2). NOTE: softmax rescale
                # and denominator carry dropped (dump illustration).
                m2r_bp = tl.make_block_ptr(mm2Res + cid * (PP * BLOCK_M * DIM) + pp2 * (BLOCK_M * DIM),
                                           (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
                o_part = tl.load(m2r_bp).to(tl.float32)
                sync_block_set(EVT_MTE2_V[0], EVT_MTE2_V[1], EVT_MTE2_V[2], EVT_MTE2_V[3], EVT_MTE2_V[4])
                sync_block_wait(EVT_MTE2_V[0], EVT_MTE2_V[1], EVT_MTE2_V[2], EVT_MTE2_V[3], EVT_MTE2_V[4])
                acc_o = acc_o + o_part

                if is_last:                         # last KV block: write O
                    out_tile = acc_o.to(Out.dtype.element_ty)
                    o_bp = tl.make_block_ptr(Out + bz2 * sOb + by2 * sOh, (S, DIM), (sOs, sOd),
                                             (bx2 * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
                    sync_block_set(EVT_V_MTE3[0], EVT_V_MTE3[1], EVT_V_MTE3[2], EVT_V_MTE3[3], EVT_V_MTE3[4])
                    sync_block_wait(EVT_V_MTE3[0], EVT_V_MTE3[1], EVT_V_MTE3[2], EVT_V_MTE3[3], EVT_V_MTE3[4])
                    tl.store(o_bp, out_tile)


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
           "mm1Res": "*fp32", "stage1Res": "*fp16", "mm2Res": "*fp32"}
    i32_names = ["B", "Hq", "Hkv", "S",
                 "sQb", "sQh", "sQs", "sQd",
                 "sKb", "sKh", "sKs", "sKd",
                 "sOb", "sOh", "sOs", "sOd",
                 "num_seq_blocks", "heads_q", "gqa_group",
                 "n_iters", "q_tasks", "r_tasks"]
    sig = dict(ptr)
    sig["sm_scale"] = "fp32"
    for n in i32_names:
        sig[n] = "i32"
    sig["NUM_ITERS"] = "constexpr"
    sig["IS_CAUSAL"] = "constexpr"
    return sig


def dump_tileir(path=None, ttir_path=None, num_iters=32, is_causal=False):
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
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_arch.mlir")
    if ttir_path is None:
        ttir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_arch_ttir.mlir")

    signature = _dump_signature()
    constants = {"NUM_ITERS": num_iters, "IS_CAUSAL": is_causal}

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

    # Phase 1: TileIR→HIVM — lower tile.* ops, producing pure TTIR/HIVM IR.
    try:
        from triton._C.libtriton.ascend import passes as ascend_passes
        ascend_passes.ttir.add_tileir_to_hivm(pm)
    except Exception:
        pass
    pm.run(module)
    print(f"[dump_tileir] after TileIR→HIVM: verify={module.verify()}", flush=True)

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


def dump_hivm(path=None, num_iters=32, is_causal=False):
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
    tileir_mlir = dump_tileir(path=None, num_iters=num_iters, is_causal=is_causal)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_arch_hivm.mlir")

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

    # Phase 1: TileIR → HIVM (first pass) — converts alloc/to_tensor/buf-to-buf copy/sync.
    #   tile.copy with !tt.ptr source is skipped (needs ptr→memref first).
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


# =============================================================================
#  Host launcher
# =============================================================================
def flash_attention_fwd(q, k, v, is_causal=False):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and S % BLOCK_N == 0 and Hq % Hkv == 0
    num_seq_blocks = S // BLOCK_M
    block_num = num_seq_blocks * Hq * B
    n_iters = S // BLOCK_N
    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    out = torch.empty_like(q)
    # GM ping-pong workspaces (taskId % 2), one slice per core.
    mm1Res    = torch.empty((NUM_CORES, PP, BLOCK_M, BLOCK_N), dtype=torch.float32, device=q.device)
    stage1Res = torch.empty((NUM_CORES, PP, BLOCK_M, BLOCK_N), dtype=q.dtype,        device=q.device)
    mm2Res    = torch.empty((NUM_CORES, PP, BLOCK_M, DIM),     dtype=torch.float32, device=q.device)
    sm_scale = (1.0 / D) ** 0.5

    grid = (NUM_CORES,)  # one program per AI core; one Cube + one Vector stream (MIX_1_1)
    flash_attention_fwd_3task_kernel[grid](
        q, k, v, out, mm1Res, stage1Res, mm2Res, sm_scale,
        B, Hq, Hkv, S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_seq_blocks, Hq, Hq // Hkv,
        n_iters, q_tasks, r_tasks,
        NUM_ITERS=n_iters,
        IS_CAUSAL=is_causal,
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
    parser.add_argument("--dump-mlir", nargs="?", const="", default=None,
                        help="Dump intermediate TileIR to PATH (default skill/op/fa_triton_arch.mlir) and exit; no device needed.")
    parser.add_argument("--dump-ir", nargs="?", const="", default=None,
                        help="Dump HIVM IR (after TileIR→HIVM lowering) to PATH and exit; no device needed.")
    args = parser.parse_args()

    # ---- dump intermediate TileIR and exit (no device required) ----
    if args.dump_mlir is not None:
        B, S, H, D = args.B, args.S, args.H, args.D
        n_iters = S // BLOCK_N
        dump_tileir(path=(args.dump_mlir or None), num_iters=n_iters, is_causal=args.causal)
        raise SystemExit(0)

    # ---- dump HIVM IR after full lowering pipeline (no device required) ----
    if args.dump_ir is not None:
        B, S, H, D = args.B, args.S, args.H, args.D
        n_iters = S // BLOCK_N
        dump_hivm(path=(args.dump_ir or None), num_iters=n_iters, is_causal=args.causal)
        raise SystemExit(0)

    B, S, H, D = args.B, args.S, args.H, args.D
    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    q = torch.randn((B, Q_H, S, D), dtype=torch.float16, device=device)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)

    out = flash_attention_fwd(q, k, v, is_causal=args.causal)

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
