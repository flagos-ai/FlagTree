import argparse
import os

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

pipe = tle.dsa.ascend.PIPE

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_CORES = 24
BLOCK_M = 32
BLOCK_N = 32
DIM = 32

# =============================================================================
#  Step sub-functions (each called from the main kernel; decorated @triton.jit
#  so the compiler sees them as inlinable device functions).
# =============================================================================


@triton.jit
def _mm1_qkt(
    # inputs
    K,
    q_block,  # pre-loaded Q tensor (BLOCK_M, DIM), passed from outer loop
    workspace_s,
    # task geometry
    cid,
    kv_idx,
    batch_idx,
    kv_head_idx,
    # strides
    sKb,
    sKh,
    sKs,
    sKd,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """MM1: compute S = Q * K^T for one KV block and store into workspace_s."""
    k_bp = tl.make_block_ptr(K + batch_idx * sKb + kv_head_idx * sKh, (S, DIM), (sKs, sKd), (kv_idx * BLOCK_N, 0),
                             (BLOCK_N, DIM), (1, 0))
    k_block = tl.load(k_bp)

    # attn_score = Q * K^T  — K is (BLOCK_N, DIM), transpose to (DIM, BLOCK_N)
    attn_score = tl.dot(q_block, tl.trans(k_block), tl.zeros((BLOCK_M, BLOCK_N), tl.float32))

    score_store_bp = tl.make_block_ptr(workspace_s + cid * BLOCK_M * BLOCK_N, (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                                       (BLOCK_M, BLOCK_N), (1, 0))
    tl.sync_block_wait("vector", "cube", 2, pipe.PIPE_MTE2, pipe.PIPE_S)
    tl.store(score_store_bp, attn_score.to(workspace_s.dtype.element_ty))
    tl.sync_block_set("cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_S)


@triton.jit
def _mm2_pv(
    # inputs
    V,
    workspace_p,
    workspace_pv,
    # task geometry
    cid,
    kv_idx,
    batch_idx,
    kv_head_idx,
    # strides
    sKb,
    sKh,
    sKs,
    sKd,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """MM2: compute O_part = P * V for one KV block and store into workspace_pv."""
    v_bp = tl.make_block_ptr(V + batch_idx * sKb + kv_head_idx * sKh, (S, DIM), (sKs, sKd), (kv_idx * BLOCK_N, 0),
                             (BLOCK_N, DIM), (1, 0))
    tl.sync_block_wait("vector", "cube", 1, pipe.PIPE_MTE3, pipe.PIPE_S)
    v_block = tl.load(v_bp)
    tl.sync_block_set("vector", "cube", 3, pipe.PIPE_MTE2, pipe.PIPE_S)

    prob_load_bp = tl.make_block_ptr(workspace_p + cid * BLOCK_M * BLOCK_N, (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                                     (BLOCK_M, BLOCK_N), (1, 0))
    p_block = tl.load(prob_load_bp)

    # pv_part = P * V  — (BLOCK_M, BLOCK_N) @ (BLOCK_N, DIM) = (BLOCK_M, DIM)
    pv_part = tl.dot(p_block, v_block, tl.zeros((BLOCK_M, DIM), tl.float32))

    pv_store_bp = tl.make_block_ptr(workspace_pv + cid * BLOCK_M * DIM, (BLOCK_M, DIM), (DIM, 1), (0, 0),
                                    (BLOCK_M, DIM), (1, 0))
    tl.sync_block_wait("vector", "cube", 4, pipe.PIPE_MTE2, pipe.PIPE_S)
    tl.store(pv_store_bp, pv_part.to(workspace_pv.dtype.element_ty))
    tl.sync_block_set("cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_S)


@triton.jit
def _vec1_softmax(
    workspace_s,
    workspace_p,
    workspace_rescale,
    workspace_expsum,
    cid,
    kv_idx,
    running_neg_max,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    global_head_idx,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Vec1: softmax over one score block -> workspace_p + workspace_rescale/expsum.

    Takes running_neg_max as an in-register accumulator (passed from main loop).
    Stores rescale and block_expsum to GM (workspace_rescale / workspace_expsum)
    so Vec2 can read them without relying on register transfer across step boundaries.
    Returns new_neg_max for the next iteration.
    """
    # load score written by MM1
    score_load_bp = tl.make_block_ptr(workspace_s + cid * BLOCK_M * BLOCK_N, (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                                      (BLOCK_M, BLOCK_N), (1, 0))
    tl.sync_block_wait("cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_S)
    attn_score_block = tl.load(score_load_bp).to(tl.float32)
    tl.sync_block_set("vector", "cube", 2, pipe.PIPE_MTE2, pipe.PIPE_S)

    if IS_CAUSAL:
        q_row_idx = global_head_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        kv_col_idx = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        causal_mask = q_row_idx[:, None] >= kv_col_idx[None, :]
        attn_score_block = tl.where(causal_mask, attn_score_block, float("-inf"))

    block_row_max = tl.max(attn_score_block, axis=-1, keep_dims=True)
    new_neg_max = tl.minimum(-block_row_max * sm_scale, running_neg_max)
    rescale = tl.exp(new_neg_max - running_neg_max)  # <= 1
    softmax_p = tl.exp(sm_scale * attn_score_block + new_neg_max)
    block_expsum = tl.sum(softmax_p, axis=-1, keep_dims=True)

    # softmax_p -> workspace_p (read by MM2)
    prob_bp = tl.make_block_ptr(workspace_p + cid * BLOCK_M * BLOCK_N, (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                                (BLOCK_M, BLOCK_N), (1, 0))
    tl.sync_block_wait("vector", "cube", 3, pipe.PIPE_MTE2, pipe.PIPE_S)
    tl.store(prob_bp, softmax_p.to(workspace_p.dtype.element_ty))
    tl.sync_block_set("vector", "cube", 1, pipe.PIPE_MTE3, pipe.PIPE_S)

    # rescale and block_expsum -> GM (read by Vec2 instead of register transfer)
    re_bp = tl.make_block_ptr(workspace_rescale + cid * BLOCK_M, (BLOCK_M, 1), (1, 1), (0, 0), (BLOCK_M, 1), (1, 0))
    tl.store(re_bp, rescale)
    es_bp = tl.make_block_ptr(workspace_expsum + cid * BLOCK_M, (BLOCK_M, 1), (1, 1), (0, 0), (BLOCK_M, 1), (1, 0))
    tl.store(es_bp, block_expsum)

    return new_neg_max


@triton.jit
def _vec2_accumulate(
    workspace_pv,
    workspace_rescale,
    workspace_expsum,
    cid,
    acc_o,
    softmax_denom,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
):
    """Vec2: rescale acc_o and accumulate one KV block's P*V.

    Reads rescale and block_expsum from GM (written by Vec1) rather than
    accepting them as register arguments, matching the cross-scope GM handoff
    pattern used in fa_cv.py.
    Takes and returns (acc_o, softmax_denom) as in-register accumulators.
    Finalization (divide by denom) is done in the main loop after the last KV block.
    """
    # load pv_part written by MM2
    pv_bp = tl.make_block_ptr(workspace_pv + cid * BLOCK_M * DIM, (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM),
                              (1, 0))
    tl.sync_block_wait("cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_S)
    pv_acc = tl.load(pv_bp).to(tl.float32)
    tl.sync_block_set("vector", "cube", 4, pipe.PIPE_MTE2, pipe.PIPE_S)

    # load rescale and block_expsum from GM (written by Vec1)
    re_bp = tl.make_block_ptr(workspace_rescale + cid * BLOCK_M, (BLOCK_M, 1), (1, 1), (0, 0), (BLOCK_M, 1), (1, 0))
    rescale = tl.load(re_bp).to(tl.float32)
    es_bp = tl.make_block_ptr(workspace_expsum + cid * BLOCK_M, (BLOCK_M, 1), (1, 1), (0, 0), (BLOCK_M, 1), (1, 0))
    block_expsum = tl.load(es_bp).to(tl.float32)

    # rescale and accumulate
    rescale_bc = tl.broadcast_to(rescale, (BLOCK_M, DIM))
    acc_o = acc_o * rescale_bc + pv_acc
    softmax_denom = softmax_denom * rescale + block_expsum

    return acc_o, softmax_denom


# =============================================================================
#  Serial single-slot FA kernel — no sync, no tle.scope (TileIR / tle.dsa form)
#
#  Outer loop: one output tile per iteration.
#  Inner loop: MM1 → Vec1 → MM2 → Vec2 for each KV block.
#  Softmax accumulators (acc_o, softmax_denom, running_neg_max) are local
#  registers declared before the inner loop; finalization after the inner loop.
#
#  grid = (NUM_CORES,). Each program drives one Cube + one Vector engine.
# =============================================================================
@triton.jit
def flash_attention_fwd_3task_kernel(
    Q,
    K,
    V,
    Out,
    workspace_s,
    workspace_p,
    workspace_pv,  # transient per-KV-block workspaces
    workspace_rescale,
    workspace_expsum,  # softmax state: Vec1 -> Vec2 via GM
    sm_scale,
    B,
    Hq,
    Hkv,
    S,
    sQb,
    sQh,
    sQs,
    sQd,
    sKb,
    sKh,
    sKs,
    sKd,
    sOb,
    sOh,
    sOs,
    sOd,
    block_num_per_core,
    rem_block_num,
    NUM_KV_BLOCKS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    cid = tl.program_id(0)

    # ---- derive loop-geometry scalars ----
    num_seq_blocks = S // BLOCK_M
    gqa_group = Hq // Hkv

    # ---- static task distribution ----
    block_start = cid * block_num_per_core + tl.where(cid < rem_block_num, cid, rem_block_num)
    block_num = block_num_per_core + tl.where(cid < rem_block_num, 1, 0)

    # =========================================================================
    #  Outer loop: one output tile per iteration.
    # =========================================================================
    tl.sync_block_set("vector", "cube", 2, pipe.PIPE_MTE2, pipe.PIPE_S)
    tl.sync_block_set("vector", "cube", 3, pipe.PIPE_MTE2, pipe.PIPE_S)
    tl.sync_block_set("vector", "cube", 4, pipe.PIPE_MTE2, pipe.PIPE_S)
    for tile_idx in range(block_num):
        output_block_id = block_start + tile_idx
        global_head_idx = output_block_id % num_seq_blocks
        head_idx = (output_block_id // num_seq_blocks) % Hq
        batch_idx = output_block_id // (num_seq_blocks * Hq)
        kv_head_idx = head_idx // gqa_group

        # ---- load Q for this tile once; reused across all KV blocks ----
        q_bp = tl.make_block_ptr(Q + batch_idx * sQb + head_idx * sQh, (S, DIM), (sQs, sQd),
                                 (global_head_idx * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
        q_block = tl.load(q_bp)

        # ---- per-tile softmax accumulators (local registers) ----
        acc_o = tl.zeros((BLOCK_M, DIM), tl.float32)
        softmax_denom = tl.zeros((BLOCK_M, 1), tl.float32)
        running_neg_max = tl.full((BLOCK_M, 1), 2**30, tl.float32)

        # =========================================================================
        #  Inner loop: MM1 → Vec1 → MM2 → Vec2 per KV block (serial, no sync).
        # =========================================================================
        for kv_idx in range(NUM_KV_BLOCKS):

            # ===== Step 1: MM1 — S = Q * K^T =====
            _mm1_qkt(
                K,
                q_block,
                workspace_s,
                cid,
                kv_idx,
                batch_idx,
                kv_head_idx,
                sKb,
                sKh,
                sKs,
                sKd,
                S,
                BLOCK_M,
                BLOCK_N,
                DIM,
            )

            # ===== Step 2: Vec1 — softmax(S) → P; stores rescale/expsum to GM =====
            running_neg_max = _vec1_softmax(
                workspace_s,
                workspace_p,
                workspace_rescale,
                workspace_expsum,
                cid,
                kv_idx,
                running_neg_max,
                sm_scale,
                IS_CAUSAL,
                global_head_idx,
                BLOCK_M,
                BLOCK_N,
            )

            # ===== Step 3: MM2 — O_part = P * V =====
            _mm2_pv(
                V,
                workspace_p,
                workspace_pv,
                cid,
                kv_idx,
                batch_idx,
                kv_head_idx,
                sKb,
                sKh,
                sKs,
                sKd,
                S,
                BLOCK_M,
                BLOCK_N,
                DIM,
            )

            # ===== Step 4: Vec2 — loads rescale/expsum from GM, accumulates =====
            acc_o, softmax_denom = _vec2_accumulate(
                workspace_pv,
                workspace_rescale,
                workspace_expsum,
                cid,
                acc_o,
                softmax_denom,
                BLOCK_M,
                DIM,
            )

        # ---- finalize: write output for this tile ----
        denom_bc = tl.broadcast_to(softmax_denom, (BLOCK_M, DIM))
        out_block = (acc_o / denom_bc).to(Out.dtype.element_ty)
        o_bp = tl.make_block_ptr(Out + batch_idx * sOb + head_idx * sOh, (S, DIM), (sOs, sOd),
                                 ((global_head_idx * BLOCK_M).to(tl.int32), 0), (BLOCK_M, DIM), (1, 0))
        tl.store(o_bp, out_block)

    tl.sync_block_wait("vector", "cube", 2, pipe.PIPE_MTE2, pipe.PIPE_S)
    tl.sync_block_wait("vector", "cube", 3, pipe.PIPE_MTE2, pipe.PIPE_S)
    tl.sync_block_wait("vector", "cube", 4, pipe.PIPE_MTE2, pipe.PIPE_S)

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
    ptr = {
        "Q": "*fp16", "K": "*fp16", "V": "*fp16", "Out": "*fp16", "workspace_s": "*fp16", "workspace_p": "*fp16",
        "workspace_pv": "*fp16", "workspace_rescale": "*fp32", "workspace_expsum": "*fp32"
    }
    i32_names = [
        "B", "Hq", "Hkv", "S", "sQb", "sQh", "sQs", "sQd", "sKb", "sKh", "sKs", "sKd", "sOb", "sOh", "sOs", "sOd",
        "block_num_per_core", "rem_block_num"
    ]
    sig = dict(ptr)
    sig["sm_scale"] = "fp32"
    for n in i32_names:
        sig[n] = "i32"
    sig["NUM_KV_BLOCKS"] = "constexpr"
    sig["IS_CAUSAL"] = "constexpr"
    sig["BLOCK_M"] = "constexpr"
    sig["BLOCK_N"] = "constexpr"
    sig["DIM"] = "constexpr"
    return sig


def dump_tileir(path=None, ttir_path=None, num_kv_blocks=32, is_causal=False):
    """Compile the kernel to TTIR (containing tile.* ops) and write it to `path`.

    Also runs the TileIR→HIVM pass to lower tile.* ops and dumps the resulting
    pure TTIR to `ttir_path`. Requires no NPU/GPU — pure front-end compilation.

    Returns the TileIR MLIR string. The TTIR dump is written as a side effect.
    """
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    try:
        from triton._C.libtriton import tle as tle_ir
        _has_tle_ir = True
    except Exception:
        tle_ir = None
        _has_tle_ir = False

    # The kernel reads module-level shape constants (BLOCK_M, DIM, ...) as plain
    # globals; allow that during this front-end-only dump.
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_4func.mlir")
    if ttir_path is None:
        ttir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_4func_ttir.mlir")

    signature = _dump_signature()
    constants = {
        "NUM_KV_BLOCKS": num_kv_blocks,
        "IS_CAUSAL": is_causal,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "DIM": DIM,
    }

    src = ASTSource(flash_attention_fwd_3task_kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    if _has_tle_ir:
        tle_ir.load_dialects(context)
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
        print("[dump_tileir] TileIR output is still valid; skipping HIVM lowering.", flush=True)
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
        print("[dump_tileir] WARNING: module.verify() failed after TTIR optimization — IR may be illegal")
    else:
        print(f"[dump_tileir] TTIR optimization complete: verify={ttir_ok}")

    ttir_mlir = str(module)
    with open(ttir_path, "w") as f:
        f.write(ttir_mlir)
    print(f"[dump_tileir] wrote optimized TTIR to {ttir_path}")

    return mlir


def dump_hivm(path=None, is_causal=False):
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
    tileir_mlir = dump_tileir(path=None, is_causal=is_causal)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_4func_hivm.mlir")

    # Step 2: parse the TileIR module and run the pass pipeline
    context = ir.context()
    ir.load_dialects(context)
    try:
        from triton._C.libtriton import tle as tle_ir
        tle_ir.load_dialects(context)
    except Exception:
        pass
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


def dump_linalg(path=None, is_causal=False):
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
      ⑤c fold_staging_copy               — merge redundant GBM→staging→cbuf pairs
      ⑤b fold memref→ptr→memref chains   — eliminate casts created by ⑤
      ⑥ final canonicalize + CSE + DCE   — erase dead ops

    The key innovation is phase ⑤b: the linalg-incubator pass creates
    unrealized_conversion_cast chains (memref → !tt.ptr → memref) as
    type-conversion materializations during its partial conversion of
    !tt.ptr<tensor<>> values.  These chains are the root cause of the
    "unresolved materialization" error.  By folding them immediately after
    the linalg pass (even when it fails) and then canonicalizing, we
    recover a valid linalg-dialect module.

    Returns the final Linalg MLIR string.  No NPU/GPU required.
    """
    from triton._C.libtriton import ir, passes, ascend

    # Step 1: compile to TTIR (TileIR)
    tileir_mlir = dump_tileir(path=None, is_causal=is_causal)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_4func_linalg.mlir")

    # Step 2: parse the TileIR module into a fresh context
    context = ir.context()
    ir.load_dialects(context)
    try:
        from triton._C.libtriton import tle as tle_ir
        tle_ir.load_dialects(context)
    except Exception:
        pass
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
    pm = ir.pass_manager(context)
    pm.enable_debug()
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
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_erase_linalg_casts(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module)
    print(f"[dump_linalg] ①b erase_linalg_casts: verify={module.verify()}", flush=True)

    # ── ② Structured (r1) + discrete mask ────────────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, False, False)
    pm.run(module)
    print(f"[dump_linalg] ② structure(r1)+discrete_mask: verify={module.verify()}", flush=True)

    # ── ③ Unstructured + HIVM + HFusion + LLVM ──────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_triton_to_unstructure_incubated(pm, False, False)
    ascend.passes.ttir.add_triton_to_hivm(pm)
    ascend.passes.ttir.add_triton_to_hfusion(pm)
    ascend.passes.ttir.add_triton_to_llvm(pm)
    pm.run(module)
    print(f"[dump_linalg] ③ unstructure+hivm+hfusion+llvm: verify={module.verify()}", flush=True)

    # ── ④ Bubble-up + structured (r2) ────────────────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_bubble_up_operation(pm)
    ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ④ bubble_up+structure(r2): verify={module.verify()}", flush=True)

    # ── ④b Inline + canonicalize ← REQUIRED to avoid C++ assertion ─────
    # Without this, the linalg incubator crashes with cast<RankedTensorType>
    # in MaskAnalysis when it encounters non-tensor types.
    pm = ir.pass_manager(context)
    pm.enable_debug()
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
    try:
        pm = ir.pass_manager(context)
        pm.enable_debug()
        ascend.passes.ttir.add_triton_to_linalg_incubated(pm, False, True, False, False, False)
        pm.run(module)
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: verify={module.verify()}", flush=True)
    except RuntimeError:
        print(
            "[dump_linalg] ⑤ triton_to_linalg_incubated: partial conversion "
            "(this is expected — the pass creates cast chains that need "
            "post-processing)", flush=True)

    # ── ⑤c Fold staging memref.alloc + memref.copy pairs ─────────────────
    #     TritonToLinalgIncubated creates staging allocs (default address
    #     space) for tt.load -> memref.copy chains.  When the downstream
    #     copy target has an explicit memory space (e.g. cbuf), the staging
    #     is redundant.  This pass merges the two copies into one direct
    #     GBM -> on-chip transfer, eliminating an alloc + copy + annotation.
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_fold_staging_copy(pm)
    pm.run(module)
    print(f"[dump_linalg] ⑤c fold_staging_copy: verify={module.verify()}", flush=True)

    # ── ⑤b Run erase-linalg-casts one more time to reconcile any
    #     unrealized_conversion_cast ops that the partial linalg conversion
    #     may have introduced around the on-chip allocations / function
    #     boundary.
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_erase_linalg_casts(pm)
    pm.run(module)
    print(f"[dump_linalg] ⑤b erase_linalg_casts (post): verify={module.verify()}", flush=True)

    # ── ⑥ Final canonicalize → erase dead casts ─────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
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


# def gen_bin():


# =============================================================================
#  Host launcher
# =============================================================================
def flash_attention_fwd(q, k, v, is_causal=False):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and S % BLOCK_N == 0 and Hq % Hkv == 0
    num_seq_blocks = S // BLOCK_M
    block_num = num_seq_blocks * Hq * B
    num_kv_blocks = S // BLOCK_N  # KV blocks per output tile (CB=1: one per task)

    block_num_per_core = block_num // NUM_CORES
    rem_block_num = block_num % NUM_CORES

    out = torch.empty_like(q)
    # Single-slot transient workspaces (accumulators are local registers in the kernel)
    workspace_s = torch.empty((NUM_CORES, BLOCK_M, BLOCK_N), dtype=torch.float16, device=q.device)
    workspace_p = torch.empty((NUM_CORES, BLOCK_M, BLOCK_N), dtype=q.dtype, device=q.device)
    workspace_pv = torch.empty((NUM_CORES, BLOCK_M, DIM), dtype=torch.float16, device=q.device)
    # GM softmax state: Vec1 stores rescale/expsum here; Vec2 loads them back
    workspace_rescale = torch.empty((NUM_CORES, BLOCK_M), dtype=torch.float32, device=q.device)
    workspace_expsum = torch.empty((NUM_CORES, BLOCK_M), dtype=torch.float32, device=q.device)
    sm_scale = (1.0 / D)**0.5

    grid = (NUM_CORES, )
    flash_attention_fwd_3task_kernel[grid](
        q,
        k,
        v,
        out,
        workspace_s,
        workspace_p,
        workspace_pv,
        workspace_rescale,
        workspace_expsum,
        sm_scale,
        B,
        Hq,
        Hkv,
        S,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_num_per_core,
        rem_block_num,
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
    parser.add_argument("--D", type=int, default=DIM)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--dump-mlir", nargs="?", const="", default=None,
                        help="Dump intermediate TileIR to PATH and exit; no device needed.")
    parser.add_argument("--dump-ir", nargs="?", const="", default=None,
                        help="Dump HIVM IR (after TileIR→HIVM lowering) to PATH and exit; no device needed.")
    parser.add_argument(
        "--dump-linalg", nargs="?", const="", default=None,
        help="Dump Linalg IR (full lowering through linalg, casts eliminated) to PATH and exit; no device needed.")
    args = parser.parse_args()

    B, S, H, D = args.B, args.S, args.H, args.D
    # ---- dump intermediate TileIR and exit (no device required) ----
    if args.dump_mlir is not None:
        dump_tileir(path=(args.dump_mlir or None), is_causal=args.causal)
        raise SystemExit(0)

    # ---- dump HIVM IR after full lowering pipeline (no device required) ----
    if args.dump_ir is not None:
        dump_hivm(path=(args.dump_ir or None), is_causal=args.causal)
        raise SystemExit(0)

    # ---- dump Linalg IR after full TileIR→Linalg lowering (no device required) ----
    if args.dump_linalg is not None:
        dump_linalg(path=(args.dump_linalg or None), is_causal=args.causal)
        raise SystemExit(0)

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
            return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                                    is_causal=args.causal).to(torch.float16)

        torch.testing.assert_close(ref(q, k, v), out, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
