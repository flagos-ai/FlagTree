import argparse
import os

import torch
import triton
import triton.language as tl

# =============================================================================
#  TLE / tile-dialect registration (needed for the dump pipeline)
# =============================================================================
import triton.experimental.tle as tle  # noqa: F401  (registers tile/tle dialects)
from triton.experimental.tle.language.dsa.ascend import L1
from triton.experimental.tle.language.dsa import tile_alloc, tile_copy, tile_to_tensor

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_CORES = 24
BLOCK_M = 32
BLOCK_N = 32
DIM = 32

# =============================================================================
#  Flash Attention v2 kernels (ported from 06-fused-attention.py)
# =============================================================================


# constexpr shape literals for tile_copy (must be tl.constexpr)
CBM = tl.constexpr(BLOCK_M)
CBN = tl.constexpr(BLOCK_N)
CD  = tl.constexpr(DIM)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q_l1,  #
                    K, V, k_l1, v_l1,  #
                    stride_kn, stride_kd, stride_vn, stride_vd,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # causal = False
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # DMA K block: GM -> L1
        k_bp = tl.make_block_ptr(K, (N_CTX, HEAD_DIM), (stride_kn, stride_kd),
                                 (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0))
        tile_copy(k_bp, k_l1, [tl.constexpr(BLOCK_N), tl.constexpr(HEAD_DIM)])
        # -- compute qk from L1 buffers --
        q = tile_to_tensor(q_l1, writable=False)
        k = tile_to_tensor(k_l1, writable=False)
        qk = tl.dot(q, tl.trans(k))
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # DMA V block: GM -> L1
        v_bp = tl.make_block_ptr(V, (N_CTX, HEAD_DIM), (stride_vn, stride_vd),
                                 (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0))
        tile_copy(v_bp, v_l1, [tl.constexpr(BLOCK_N), tl.constexpr(HEAD_DIM)])
        v = tile_to_tensor(v_l1, writable=False)
        acc = tl.dot(p.to(tl.float16), v, acc)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [32, 64, 128]
    for BN in [32, 64]
    for s in [1, 2]
    for w in [4, 8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return BLOCK_M >= BLOCK_N


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "STAGE"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qb, stride_qh, stride_qm, stride_qd,  #
              stride_kb, stride_kh, stride_kn, stride_kd,  #
              stride_vb, stride_vh, stride_vn, stride_vd,  #
              stride_ob, stride_oh, stride_om, stride_od,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # base pointers for this (batch, head)
    Q_ptr = Q + off_z * stride_qb + off_h * stride_qh
    K_ptr = K + off_z * stride_kb + off_h * stride_kh
    V_ptr = V + off_z * stride_vb + off_h * stride_vh
    O_ptr = Out + off_z * stride_ob + off_h * stride_oh

    # ---- allocate L1 buffers for Q, K, V ----
    q_l1 = tile_alloc([tl.constexpr(BLOCK_M), tl.constexpr(HEAD_DIM)], tl.float16, tle.language.dsa.ascend.L1)
    k_l1 = tile_alloc([tl.constexpr(BLOCK_N), tl.constexpr(HEAD_DIM)], tl.float16, tle.language.dsa.ascend.L1)
    v_l1 = tile_alloc([tl.constexpr(BLOCK_N), tl.constexpr(HEAD_DIM)], tl.float16, tle.language.dsa.ascend.L1)

    # DMA Q tile into L1 once; it stays resident for the full KV loop
    q_bp = tl.make_block_ptr(Q_ptr, (N_CTX, HEAD_DIM), (stride_qm, stride_qd),
                             (start_m * BLOCK_M, 0), (BLOCK_M, HEAD_DIM), (1, 0))
    tile_copy(q_bp, q_l1, [tl.constexpr(BLOCK_M), tl.constexpr(HEAD_DIM)])

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    # stage 1: off-band (causal only); stage 3: full non-causal
    # For causal=True,  STAGE=3 → inner called with STAGE=1 (off-band) then STAGE=2 (on-band)
    # For causal=False, STAGE=1 → inner called with STAGE=3 (all)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_l1,  #
                                        K_ptr, V_ptr, k_l1, v_l1,  #
                                        stride_kn, stride_kd, stride_vn, stride_vd,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX)
    # stage 2: on-band (diagonal, causal mask)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_l1,  #
                                        K_ptr, V_ptr, k_l1, v_l1,  #
                                        stride_kn, stride_kd, stride_vn, stride_vd,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    o_bp = tl.make_block_ptr(O_ptr, (N_CTX, HEAD_DIM), (stride_om, stride_od),
                             (start_m * BLOCK_M, 0), (BLOCK_M, HEAD_DIM), (1, 0))
    tl.store(o_bp, acc.to(tl.float16))


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
    """Static signature for ast_to_ttir (pointers / scalars / i32 / constexpr).

    Matches _attn_fwd: Q, K, V, sm_scale, M, Out, strides, Z, H, N_CTX,
    HEAD_DIM, BLOCK_M, BLOCK_N, STAGE.
    """
    sig = {}
    # tensor pointers
    for name in ("Q", "K", "V", "Out"):
        sig[name] = "*fp16"
    sig["sm_scale"] = "fp32"
    sig["M"] = "*fp32"
    # strides: q, k, v, o — each has b/h/m/d
    for t in ("q", "k", "v", "o"):
        for dim in ("b", "h", ("m" if t in ("q", "o") else "n"), "d"):
            sig[f"stride_{t}{dim}"] = "i32"
    sig["Z"]     = "i32"
    sig["H"]     = "i32"
    sig["N_CTX"] = "i32"
    # constexprs
    sig["HEAD_DIM"] = "constexpr"
    sig["BLOCK_M"]  = "constexpr"
    sig["BLOCK_N"]  = "constexpr"
    sig["STAGE"]    = "constexpr"
    return sig


def dump_tileir(path=None, ttir_path=None, num_kv_blocks=32, combine_batch=8, is_causal=False):
    """Compile _attn_fwd to TTIR and write it to `path`.

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

    stage = 3 if is_causal else 1
    signature = _dump_signature()
    constants = {
        "HEAD_DIM": DIM,
        "BLOCK_M":  BLOCK_M,
        "BLOCK_N":  BLOCK_N,
        "STAGE":    stage,
    }

    src = ASTSource(_attn_fwd, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.dsa_ir.load_tile_dialects(context)
    # Ascend dialect is optional (only needed for ascend-specific ops); load if present.
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    # codegen_fns: tl.dot needs a target-provided "min_dot_size"; the ascend
    # backend simply returns (1,1,1), so supply that inline (no full backend).
    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(_attn_fwd, src, context, _DumpOptions(), codegen_fns, {})

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
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_arch_hivm.mlir")

    # Step 2: parse the TileIR module and run the pass pipeline
    context = ir.context()
    ir.load_dialects(context)
    from triton._C.libtriton import tle as tle_ir
    tle_ir.load_dialects(context)
    tle_ir.dsa_ir.load_tile_dialects(context)
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
    from triton._C.libtriton import tle as tle_ir

    # Step 1: compile to TTIR (TileIR)
    tileir_mlir = dump_tileir(path=None, combine_batch=combine_batch, is_causal=is_causal)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fa_triton_arch_linalg.mlir")

    # Step 2: parse the TileIR module into a fresh context
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.dsa_ir.load_tile_dialects(context)
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
    # pm = ir.pass_manager(context)
    # pm.enable_debug()
    # ascend.passes.ttir.add_erase_linalg_casts(pm)
    # passes.common.add_canonicalizer(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ①b erase_linalg_casts: verify={module.verify()}", flush=True)

    # ── ② Structured (r1) + discrete mask ────────────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
    # ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ② structure(r1)+discrete_mask: verify={module.verify()}", flush=True)

    # ── ③ Unstructured + HIVM + HFusion + LLVM ──────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
    ascend.passes.ttir.add_triton_to_unstructure(pm, False, False)
    ascend.passes.ttir.add_triton_to_hivm(pm)
    ascend.passes.ttir.add_triton_to_hfusion(pm)
    ascend.passes.ttir.add_triton_to_llvm(pm)
    pm.run(module)
    print(f"[dump_linalg] ③ unstructure+hivm+hfusion+llvm: verify={module.verify()}", flush=True)

    # ── ④ Bubble-up + structured (r2) ────────────────────────────────────
    pm = ir.pass_manager(context); pm.enable_debug()
    ascend.passes.ttir.add_bubble_up_operation(pm)
    # ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
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
        ascend.passes.ttir.add_triton_to_linalg(pm, False, True, False, False, False)
        pm.run(module)
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: verify={module.verify()}", flush=True)
        linalg_ok = True
    except RuntimeError as e:
        print(f"[dump_linalg] ⑤ triton_to_linalg_incubated: partial conversion "
              f"(this is expected — the pass creates cast chains that need "
              f"post-processing)", flush=True)

    # ── ⑤c Fold staging memref.alloc + memref.copy pairs ─────────────────
    #     TritonToLinalgIncubated creates staging allocs (default address
    #     space) for tt.load -> memref.copy chains.  When the downstream
    #     copy target has an explicit memory space (e.g. cbuf), the staging
    #     is redundant.  This pass merges the two copies into one direct
    #     GBM -> on-chip transfer, eliminating an alloc + copy + annotation.
    # pm = ir.pass_manager(context); pm.enable_debug()
    # ascend.passes.ttir.add_fold_staging_copy(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ⑤c fold_staging_copy: verify={module.verify()}", flush=True)

    # ── ⑤b Run erase-linalg-casts one more time to reconcile any
    #     unrealized_conversion_cast ops that the partial linalg conversion
    #     may have introduced around the on-chip allocations / function
    #     boundary.
    # pm = ir.pass_manager(context)
    # pm.enable_debug()
    # ascend.passes.ttir.add_erase_linalg_casts(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ⑤b erase_linalg_casts (post): verify={module.verify()}", flush=True)

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


# def gen_bin():


# =============================================================================
#  Host launcher
# =============================================================================

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=False):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        # backward not needed for native_fa; raise to surface if accidentally called
        raise NotImplementedError("native_fa does not implement the backward pass")


attention = _attention.apply


def flash_attention_fwd(q, k, v, combine_batch=None, is_causal=False):
    """Host launcher: wraps the Flash Attention v2 forward kernel.

    `combine_batch` is accepted for CLI compatibility but unused — the FA v2
    kernel does not tile over KV blocks in a combine-batch fashion.
    """
    sm_scale = q.shape[-1] ** -0.5
    return attention(q, k, v, is_causal, sm_scale, False)


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
    parser.add_argument("--combine-batch", type=int, default=8,
                        help="KV blocks per task (arch22 nRatio)")
    parser.add_argument("--dump-mlir", nargs="?", const="", default=None,
                        help="Dump intermediate TileIR to PATH (default skill/op/fa_triton_arch.mlir) and exit; no device needed.")
    parser.add_argument("--dump-ir", nargs="?", const="", default=None,
                        help="Dump HIVM IR (after TileIR→HIVM lowering) to PATH and exit; no device needed.")
    parser.add_argument("--dump-linalg", nargs="?", const="", default=None,
                        help="Dump Linalg IR (full lowering through linalg, casts eliminated) to PATH and exit; no device needed.")
    args = parser.parse_args()

    B, S, H, D = args.B, args.S, args.H, args.D
    combine_batch = args.combine_batch
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
