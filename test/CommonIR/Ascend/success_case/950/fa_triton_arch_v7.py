"""FlashAttention forward — Ascend 910B, Triton + TLE tile-DSA, architecture v7.

v7 = fa_triton_arch_v6_casual.py (sparse causal schedule)
   + fa_v6_p0_fcasual.py       (the P0 softmax-max fix + IR dump tooling)

Neither parent is a superset of the other, and the two wins are independent:
v6 removes causal work that never had to be issued, P0 removes per-chain
`pipe_barrier<PIPE_V>` from the work that remains. v7 carries both.

---------------------------------------------------------------------------
Inherited from v6 — schedule-level work removal (ref = log/knowledge/
cann_fa_impl_reference.md, AscendC arch22 FlashAttentionScore)
---------------------------------------------------------------------------

1. **Causal blocks past the diagonal are never issued** (ref §4.1
   `GetS2LoopRange`, where sparse narrowing sits *outside* the pipeline).

   Under causal, Q row-block `i` only needs KV blocks `0 .. ((i+1)*BLOCK_M-1)
   // BLOCK_N`. Running MM1/Vec1/MM2/Vec2 over the full KV range and masking the
   result to zero pays full Cube cost for blocks that contribute nothing. v7
   gives every tile its own task count, so those tasks never enter the pipeline.
   At S=2048 that is just under half the total work.

   The cost is that task counts vary per tile, so the flat `task -> (tile, kv
   group)` decode stops being arithmetic. The host precomputes the whole mapping
   and packs each entry into one i32; the kernel decodes with a single scalar
   load plus a divmod (ref §5.4 does exactly this — `sparseStartIdx[]` is
   precomputed and "kernel 侧只管读 ... 不感知这套逻辑").

2. **Causal load balancing across cores** (ref §5.4 `PartitionSparseData` /
   `SetSparseStartIdx`). Once (1) lands, tile costs are wildly unequal — tile 0
   does 1 KV block, tile 15 does 16 — and a contiguous arithmetic split leaves
   every core waiting on whichever one drew the heavy tail. arch22
   binary-searches the max per-core load and greedily packs tiles to fit; v7
   uses the greedy half (longest-processing-time-first into the least-loaded
   core), which is within a few percent of optimal for this cost distribution
   and needs no search. The non-causal path keeps the arithmetic split, since
   there every tile costs the same and a permutation only hurts locality.

3. **Per-tile `last_kv_idx`** instead of `num_kv_blocks - 1`: under the sparse
   schedule each tile finalises its output row at its own diagonal block, so
   Vec2 is told which block closes the tile (closed form, no lookup table).

4. **One task decode per tick instead of four** (ref §4.1 `ComputeAxisIdx`).
   Task `g` is decoded once in `_task_coords`; the result is carried in scalar
   registers for next tick's `g-1` stages. Worth ~nothing on the dense path but
   it is what keeps the sparse path to a single GM load per tick.

5. **CB > MAX_CB raises instead of silently corrupting.** The Vec1->Vec2
   hand-off lives in registers named `rescale_<cb><sub>`, which only exist for
   `cb in {0,1}`. `--combine-batch 8` (arch22's own `nRatio` default) would read
   a zeroed rescale for blocks 2..7 and get ~45% of its output wrong with no
   diagnostic. The ceiling stays and becomes an assertion, because measurement
   says it costs nothing — see "measured and rejected" below.

---------------------------------------------------------------------------
Inherited from P0 — instruction-level fix inside the surviving work
---------------------------------------------------------------------------

6. **`tl.minimum` replaced by an explicit `tl.where` select** in the online-max
   update. `tl.minimum` lowers to a NaN-correct expansion costing 7
   `pipe_barrier<PIPE_V>` per chain against 2 ops of real work:

       vmuls(-1.0) | vadds(+0.0) | vcmp_eq(a,a) vnot | vcmp_eq(b,b) vnot
                   | vmin | vsel | vsel

   The two `vcmp_eq`/`vnot` pairs are isnan() tests on each operand and the two
   `vsel` propagate a NaN if either input is one. Neither operand can be NaN
   here: `block_row_max` comes out of a max-reduce over scores MM1 produced from
   finite inputs, and `running_max` is either the 10.0 seed or a previous result
   of this same expression. The guard is dead weight, so v7 asks for the select
   directly. This runs `SUB_VEC_NUM * CB` times per task, i.e. on every surviving
   block after (1) has pruned the rest.

7. **IR dump entry points** (`--dump-mlir` / `--dump-ir` / `--dump-linalg`),
   carried over so v7 can be inspected without a device. See also
   `BISHENGIR_DUMP_IR_AFTER_ALL=1` for the backend-side dump.

---------------------------------------------------------------------------
New in v7 — autotuning
---------------------------------------------------------------------------

8. **`--autotune` searches (BLOCK_M, BLOCK_N, CB, SUB_VEC_NUM, RING,
   num_stages) by measurement** and caches the winner per (device, shape,
   causal) key, optionally to disk.

   The search is host-side rather than `@triton.autotune`, because these are not
   merely constexprs: BLOCK_M/BLOCK_N/CB/RING size `workspace_s` and
   `workspace_pv`, and BLOCK_M/BLOCK_N/CB decide the *contents* of `task_map`,
   since the sparse causal schedule is built on the host out of the tile shape.
   `@triton.autotune` varies constexprs against a fixed argument list and cannot
   express either. Re-entering the launcher per candidate has the side benefit
   that every candidate is timed through exactly the path production uses.

   Default strategy is one coordinate-descent sweep off the baseline —
   `sum(len(axis))` compiles rather than `prod(len(axis))`, 8 against 72 at the
   default space — because a bishengir compile costs tens of seconds. `--tune-
   strategy grid` does the full cross product when that budget exists.

   Two hard ceilings bound the space rather than being searched: `CB <= 2` and
   `SUB_VEC_NUM <= 2`, both because the Vec1->Vec2 hand-off is unrolled into
   `rescale_<cb><sub>` registers (item 5), and `RING <= 5` because three event-id
   banks of RING slots must fit in event_id [0,15]. Configs that cannot fit UB or
   the workspace budget are pruned by host arithmetic before a compile is spent
   on them; anything that survives that and still fails to compile is recorded
   with its full error under `$FA_V7_LOG_DIR` and skipped. A candidate that
   compiles but disagrees with the fp32 reference is rejected before it is timed,
   so it cannot win on speed alone.

   Candidates are timed the way `--bench` times: `--tune-iters` launches per
   batch, one `synchronize()` per batch, best of three batches. Synchronising per
   launch instead charges each candidate a host round-trip (~0.23 ms at B=4
   S=2048) and buries the RING axis, whose true spread is ~0.03 ms.

   What the search found at B=4 H=16 S=2048: BLOCK_M=128 / BLOCK_N=256 is a real
   optimum — M64 costs +33% and N128 +36%, because shrinking either axis
   multiplies the tick count and every extra tick pays another cross-core
   semaphore round plus a GM round-trip through `workspace_s`. It is the largest
   tile that fits UB. RING is nearly flat by comparison (R2 and R4 within 0.4%),
   with the wrinkle that R3 is bimodal across runs — sometimes 1.0016 ms, more
   often 1.034 — so the default moved to R2, which is never worse and costs one
   less ring slot. Since that spread exceeds the gap between R2 and R4, treat a
   sub-1% win on this axis as unresolved rather than real.

Kept from both parents because it was already right: BLOCK_M=128 / BLOCK_N=256,
the skew-1 pipeline `Cube[MM1(g), MM2(g-1)] / Vector[Vec1(g), Vec2(g-1)]`,
the sequential `SUB_VEC_NUM=2` sub-tile split that fits 128x256 into
192 KB UB, the register hand-off itself, fp16-native reductions
(`_fp16_max_combine` — `tl.max` promotes to fp32 and the Ascend backend hoists
that `extf` above `tt.reduce`, so every 64x256 block would pay a widen+narrow
pair), host-side Q pre-scaling, `workspace_p` allocated separately from `workspace_s`, the
workspace cache, and `USE_CUSTOM_COMPILE_OPT=1` to drop the redundant
auto-injected block sync.

---------------------------------------------------------------------------
Measured and rejected — recorded so they are not retried blindly
---------------------------------------------------------------------------

All at B=4 H=16 S=2048 D=64 fp16, 20 timed iterations, each variant in its own
process (these files share kernel names and module-level constexprs, so two of
them in one interpreter will run one file's kernel against another's semaphore
layout and hang the Vector core — which is why v7 renames both).

- **Per-block causal mask classification** (ref §7.2 `GetAttenMaskComputeMode` /
  `NO_NEED_COMPUTE_MODE`): skip the mask on blocks fully below the diagonal and
  only pay for the straddling one. This looks like the obvious companion to the
  sparse schedule and is a **1.5x regression** — 1.83 ms against 1.18 ms, enough
  to cancel the entire sparse-scheduling win. The block index is a scheduler
  value, so the predicate is not compile-time; guarding the mask, the reduction
  and the all-masked fix-up turns uniform vector work into divergent control
  flow, and that costs more than the compares saved. AscendC evaluates the same
  test on its scalar unit and switches whole templated paths, which is why the
  idea transfers to arch22 but not to Triton. v7 masks unconditionally under
  `IS_CAUSAL`.
- **Lifting the CB ceiling via a GM hand-off** — route rescale/expsum through
  `workspace_rescale`/`workspace_expsum` indexed by ring slot, which is what
  arch22 does (`softmaxExpBuf[taskId % 2]`, ref §6.2) and what v4/v5 did.
  Correct at every CB, and **2x slower**: 2.15 ms at CB=2 against 1.05 ms. The
  factors are 2 KB per task against 512 KB of score traffic, so this is not
  bandwidth — it is ~12 extra small GM transfers per task whose fixed latency
  the Cube side cannot cover.
- **CB > 2 at all.** With the GM hand-off making CB=4 and CB=8 runnable, the
  sweep is 2.15 / 2.09 / 2.23 ms for CB = 2 / 4 / 8 — flat within noise. Larger
  `nRatio` does not pay off here, so the CB<=2 ceiling costs nothing and the
  register hand-off is strictly the better trade.
- **Deeper skew** — `Cube[MM1(g), MM2(g-2)] / Vector[Vec1(g-1), Vec2(g-3)]`, so
  every cross-core dependency is a tick old rather than same-tick. This is what
  arch22 gets for free by issuing BMM1 asynchronously (`IterateAll<false>`, ref
  §4.2); `tl.dot` has no issue/await split to mirror. Measured 2.15 ms, i.e.
  identical to skew-1 — but measured *on top of* the GM hand-off that dominated
  both, so this number only says "not a win there". It also needs RING live
  copies of the hand-off state instead of one, since Vec1 laps Vec2 by two
  tasks. If revisited, do it over the register hand-off so the effect is visible.
- **A `(CB, SUB_M)` register tile** addressed by `tl.where(rows == cb_idx, ...)`
  to make the hand-off CB-general while staying on-chip: `bishengir-compile`
  fails with "Failed to run BiShengHIR pipeline" in `[ConvertLinalgIRToBinary]`.
  Moot given the CB sweep above.
"""

import argparse
import json
import os
import sys
import traceback
from typing import NamedTuple

# ---- Force use of locally built FlagTree instead of any installed version ----
FLAGTREE_BUILD_DIR = "/dev/shm/yuansheng/ref_flagtree/FlagTree/build/lib.linux-aarch64-cpython-312"
if os.path.exists(FLAGTREE_BUILD_DIR):
    # Remove any old triton/flagtree paths from sys.path
    sys.path = [p for p in sys.path if 'triton' not in p.lower() and 'flagtree' not in p.lower()]
    # Insert the local build at the front
    sys.path.insert(0, FLAGTREE_BUILD_DIR)
    print(f"[INFO] Using FlagTree from: {FLAGTREE_BUILD_DIR}")

# Keep the custom compile options and use the block synchronization below.
os.environ["USE_CUSTOM_COMPILE_OPT"] = "1"

import torch
import triton

# Patch the compiler's cached variable (read at module-load, so the env var alone
# is not enough).
try:
    import triton.backends.ascend.compiler as _ascend_compiler
    _ascend_compiler._USE_CUSTOM_COMPILE_OPT = "1"
    _original_compile_910_95 = _ascend_compiler.linalg_to_bin_enable_npu_compile_910_95
    def _compile_910_95_with_manual_sync(linalg, metadata, opt):
        metadata["disable_auto_inject_block_sync"] = True
        return _original_compile_910_95(linalg, metadata, opt)
    _ascend_compiler.linalg_to_bin_enable_npu_compile_910_95 = _compile_910_95_with_manual_sync
except (ImportError, AttributeError):
    pass

# ---- LLVM version mismatch handled in compiler.py ----
# libtriton.so (LLVM22) produces MLIR bytecode v6 that bishengir-opt (LLVM19) cannot
# parse. The compiler pipeline is patched to apply syntax fixup + bishengir-opt on
# text directly, bypassing the bytecode round-trip while keeping canonicalization.
import triton.language as tl

import triton.experimental.tle as tle  # registers tile/tle dialects; tle.scope
from triton.experimental.tle.language.dsa.ascend import PIPE, sync_block_set, sync_block_wait

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_CORES = 20
DIM = 64

# ---- Hardware ceilings used to prune the tuning space -----------------------
UB_BYTES = 192 * 1024  # per-Vector-core unified buffer
L0C_BYTES = 128 * 1024  # fp32 matmul accumulator budget we hold ourselves to
MAX_EVENT_ID = 16  # sync_block event_id lives in [0, 16)

# The Vec1->Vec2 hand-off is unrolled into named registers, one pair per
# (cb_idx, sub_idx), so both of those axes are capped. See docstring item 5:
# measurement says raising the CB ceiling is not worth the GM round-trip it
# would cost, and SUB_VEC_NUM is bounded by the same unrolling plus the one
# accumulator/running-max pair that exists per sub-tile.
MAX_CB = 2
MAX_SUB_VEC_NUM = 2

# ---- Default tile shape (the measured baseline; see the docstring) ----------
# These remain module-level so the dump entry points and any importer keep
# working unchanged. The autotuner does not mutate them — it passes a per-launch
# FAConfig instead, and every one of these values reaches the kernel as an
# explicit constexpr argument.
BLOCK_M = 128
BLOCK_N = 256
# Task-ring depth. Measured at B=4 H=16 S=2048 non-causal, --bench 100 x3 runs:
# R2 0.9981/0.9985/1.0016 ms, R4 1.0015/1.0016, R3 1.0338/1.0340/1.0016. R2 is
# never worse than R3 and holds one fewer slot of `workspace_s`/`workspace_pv`,
# so it is the default; R3's spread is bimodal, which is what made the earlier
# 3-deep pipeline look like a wash.
RING = 2
# The Vector scope walks a BLOCK_M x BLOCK_N softmax tile in SUB_VEC_NUM
# sequential sub-tiles of SUB_M x BLOCK_N. This halves peak UB pressure, which
# is what lets PlanMemory fit BLOCK_M=128 / BLOCK_N=256 inside 192 KB. The Cube
# side is untouched: MM1/MM2 still produce and consume full BLOCK_M tiles.
SUB_VEC_NUM = 2
SUB_M = BLOCK_M // SUB_VEC_NUM
NUM_STAGES = 4

# ---- Cross-core semaphores -------------------------------------------------
# Keep S, P and PV in separate event-id banks on the tested A5 backend.
# Reusing an id for different pipe pairs caused incorrect Vec1/P results.
# Within each direction, each slot has one event for each buffer family:
#   S:  slot             S_READY cube->vector / S_FREE vector->cube
#   P:  RING + slot      P_FREE cube->vector / P_READY vector->cube
#   PV: 2 * RING + slot  PV_READY cube->vector / PV_FREE vector->cube
# The prologue primes FREE events; the epilogue consumes the remaining tokens.
MAX_RING = MAX_EVENT_ID // 3


# =============================================================================
#  Tuning: the config object and the search space
# =============================================================================
class FAConfig(NamedTuple):
    """One launch configuration.

    Every field reaches the kernel as a constexpr (or, for `num_stages`, as a
    launch attribute), and the first four additionally decide host-side state:
    `block_m`/`block_n`/`combine_batch`/`ring` size the GM ring workspaces, and
    `block_m`/`block_n`/`combine_batch` determine the sparse causal `task_map`
    contents. That coupling is why the search lives here and not in
    `@triton.autotune`, which varies constexprs against a fixed argument list.
    """
    block_m: int = BLOCK_M
    block_n: int = BLOCK_N
    combine_batch: int = 2
    sub_vec_num: int = SUB_VEC_NUM
    ring: int = RING
    num_stages: int = NUM_STAGES

    @property
    def sub_m(self):
        return self.block_m // self.sub_vec_num

    def label(self):
        return (f"M{self.block_m}/N{self.block_n}/CB{self.combine_batch}/"
                f"SV{self.sub_vec_num}/R{self.ring}/ns{self.num_stages}")


DEFAULT_CONFIG = FAConfig()

# Coordinate-descent axes, each listed best-known-first so the baseline is
# candidate 0 on every axis. Kept deliberately small: a bishengir compile is
# tens of seconds, so the default budget is sum(len(axis)) ~= 8 compiles rather
# than the 72 of the full cross product.
TUNE_SPACE = {
    "block_m": [128, 64, 256],
    "block_n": [256, 128],
    "combine_batch": [2, 1],
    "sub_vec_num": [2, 1],
    "ring": [2, 3, 4],
    "num_stages": [4, 2],
}


def _ub_bytes(cfg):
    """Peak Vector-core UB for one sub-tile pass, in bytes (fp16 unless noted).

    Live at the widest point inside `_vec1_softmax`: the score sub-tile, the
    exponentiated copy destined for `workspace_p`, and the fp32 temporary the
    causal mask/compare chain materialises alongside them. Vec2's accumulator
    (sub_m x DIM) plus its PV load are charged too since the two stages coexist
    on the same core within a tick.
    """
    sub = cfg.sub_m * cfg.block_n
    vec1 = 2 * sub * 2 + sub * 4  # score + P (fp16) + one fp32 temp
    vec2 = 2 * (cfg.sub_m * DIM) * 2  # acc_o + pv load
    return vec1 + vec2


def _l0c_bytes(cfg):
    """Cube-side fp32 accumulator for one `tl.dot`: MM1 dominates MM2 here."""
    return cfg.block_m * cfg.block_n * 4


def _workspace_bytes(cfg):
    """GM ring footprint across all cores, in bytes.

    `workspace_p` aliases `workspace_s`, so it is not charged separately.
    """
    per_core = cfg.ring * cfg.combine_batch * cfg.block_m * (cfg.block_n + DIM) * 2
    return NUM_CORES * per_core


def config_is_valid(cfg, S):
    """Reject a config on host arithmetic, before a compile is spent on it.

    Returns (ok, reason). The reasons are the structural invariants the kernel
    relies on plus the two register-unrolling ceilings from docstring item 5.
    """
    if cfg.block_m <= 0 or cfg.block_n <= 0:
        return False, "non-positive block size"
    if S % cfg.block_m:
        return False, f"S={S} not divisible by BLOCK_M={cfg.block_m}"
    if S % cfg.block_n:
        return False, f"S={S} not divisible by BLOCK_N={cfg.block_n}"
    if cfg.sub_vec_num < 1 or cfg.sub_vec_num > MAX_SUB_VEC_NUM:
        return False, f"SUB_VEC_NUM={cfg.sub_vec_num} outside [1,{MAX_SUB_VEC_NUM}] (register hand-off)"
    if cfg.block_m % cfg.sub_vec_num:
        return False, f"BLOCK_M={cfg.block_m} not divisible by SUB_VEC_NUM={cfg.sub_vec_num}"
    if cfg.combine_batch < 1 or cfg.combine_batch > MAX_CB:
        return False, f"CB={cfg.combine_batch} outside [1,{MAX_CB}] (register hand-off)"
    nkv = S // cfg.block_n
    if nkv % cfg.combine_batch:
        return False, f"num_kv_blocks={nkv} not divisible by CB={cfg.combine_batch}"
    if cfg.ring < 2:
        return False, f"RING={cfg.ring} < 2 leaves no slot to overlap"
    if cfg.ring > MAX_RING:
        return False, f"three banks * RING={cfg.ring} exceed event_id [0,{MAX_EVENT_ID})"
    if cfg.num_stages < 1:
        return False, "num_stages < 1"
    if _ub_bytes(cfg) > UB_BYTES:
        return False, f"UB {_ub_bytes(cfg)} > {UB_BYTES}"
    if _l0c_bytes(cfg) > L0C_BYTES:
        return False, f"L0C {_l0c_bytes(cfg)} > {L0C_BYTES}"
    return True, "ok"


# =============================================================================
#  Task decode  (== arch22 ComputeAxisIdx, ref §4.1)
# =============================================================================
@triton.jit
def _task_coords(t, tile_base, task_map, num_seq_blocks, heads_q, gqa_group, conbined_block_num,
                 SPARSE: tl.constexpr):
    """Decode this core's flat task id `t` into tile coordinates + KV group.

    Two schedules share one decode:

    * dense (`SPARSE=False`): every tile owns `conbined_block_num` tasks, so the
      split is pure arithmetic and `tile_base` is this core's first tile.
    * sparse (`SPARSE=True`, causal): tiles own different task counts, so the
      host precomputes the whole mapping and packs each entry into one i32,
      `output_block_id * conbined_block_num + idx_in_conbine`. The decode is a
      single scalar GM load plus a divmod (ref §5.4: the kernel "只管读" a
      precomputed index and does not reconstruct the schedule).

      An earlier version instead stored a per-tile task prefix sum and scanned
      it for the owning tile. That is O(tiles-per-core) scalar loads — about 51
      at this shape — repeated for all four stages every tick, and it cost more
      than the skipped causal work was worth. Hence the flat packed table.

    Returns (idx_in_conbine, global_head_idx, head_idx, batch_idx, kv_head_idx).
    """
    if SPARSE:
        packed = tl.load(task_map + t)
        output_block_id = packed // conbined_block_num
        idx_in_conbine = packed % conbined_block_num
    else:
        idx_in_conbine = t % conbined_block_num
        output_block_id = tile_base + t // conbined_block_num

    global_head_idx = output_block_id % num_seq_blocks  # Q row-block index
    head_idx = (output_block_id // num_seq_blocks) % heads_q
    batch_idx = output_block_id // (num_seq_blocks * heads_q)
    kv_head_idx = head_idx // gqa_group  # GQA: several Q heads share one KV head
    return idx_in_conbine, global_head_idx, head_idx, batch_idx, kv_head_idx


# =============================================================================
#  Cube stage 1 — MM1
# =============================================================================
@triton.jit
def _mm1_qkt(
    Q,
    K,
    workspace_s,
    cid,
    idx_in_conbine,
    global_head_idx,
    batch_idx,
    head_idx,
    kv_head_idx,
    ring_slot,
    sQb,
    sQh,
    sQs,
    sQd,
    sKb,
    sKh,
    sKs,
    sKd,
    S,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """MM1: S = Q @ K^T over CB KV blocks -> workspace_s[cid, ring_slot].

    Q is re-fetched per task: Cube register state does not survive a scope
    switch, because the Cube and Vector scopes compile to interleaved
    instruction streams that share L1.

    Each KV block gets a fresh zero accumulator, so the `tl.dot` result is that
    block's score directly, and it lands in GM before any Vector arithmetic
    touches it — on A2, L2/GM is the only AIC->AIV data path (ref §1.1).
    """
    # workspace_s[ring_slot] free? (Vec1 released it RING ticks ago)
    sync_block_wait("vector", "cube", ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    q_row_offs = global_head_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_col_offs = tl.arange(0, DIM)
    q_ptr = (Q + batch_idx * sQb + head_idx * sQh + q_row_offs[:, None] * sQs + q_col_offs[None, :] * sQd)
    q_l1 = tl.load(q_ptr)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx

        k_bp = tl.make_block_ptr(K + batch_idx * sKb + kv_head_idx * sKh, (S, DIM), (sKs, sKd), (kv_idx * BLOCK_N, 0),
                                 (BLOCK_N, DIM), (1, 0))
        k_block = tl.load(k_bp)

        # Both operands stay plain register tensors: routing either through
        # tile_to_tensor makes the lowered func.call see a 4D cbuf memref, which
        # the matmul intrinsic rejects (it wants 2D).
        attn_score_l0c = tl.dot(q_l1.to(tl.float16), tl.trans(k_block).to(tl.float16))

        score_store_bp = tl.make_block_ptr(
            workspace_s +
            (cid * RING * CB * BLOCK_M * BLOCK_N + ring_slot * CB * BLOCK_M * BLOCK_N + cb_idx * BLOCK_M * BLOCK_N),
            (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
        tl.store(score_store_bp, attn_score_l0c.to(workspace_s.dtype.element_ty))

    # every CB score block written -> Vec1 may start on this slot
    sync_block_set("cube", "vector", ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


# =============================================================================
#  Cube stage 3 — MM2
# =============================================================================
@triton.jit
def _mm2_pv(
    V,
    workspace_p,
    workspace_pv,
    cid,
    idx_in_conbine,
    batch_idx,
    kv_head_idx,
    ring_slot,
    sKb,
    sKh,
    sKs,
    sKd,
    S,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    """MM2: O_part = P @ V over CB blocks -> workspace_pv[cid, ring_slot]."""
    # P from Vec1 ready?
    sync_block_wait("vector", "cube", RING + ring_slot, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)
    # workspace_pv[ring_slot] free? (Vec2 released it RING ticks ago)
    sync_block_wait("vector", "cube", 2 * RING + ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx

        v_bp = tl.make_block_ptr(V + batch_idx * sKb + kv_head_idx * sKh, (S, DIM), (sKs, sKd),
                                 (kv_idx * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
        v_block = tl.load(v_bp)

        prob_load_bp = tl.make_block_ptr(
            workspace_p +
            (cid * RING * CB * BLOCK_M * BLOCK_N + ring_slot * CB * BLOCK_M * BLOCK_N + cb_idx * BLOCK_M * BLOCK_N),
            (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
        p_block = tl.load(prob_load_bp)

        pv_part_l0c = tl.dot(p_block.to(tl.float16), v_block.to(tl.float16))

        pv_store_bp = tl.make_block_ptr(
            workspace_pv + cid * RING * CB * BLOCK_M * DIM + ring_slot * CB * BLOCK_M * DIM + cb_idx * BLOCK_M * DIM,
            (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
        tl.store(pv_store_bp, pv_part_l0c.to(workspace_pv.dtype.element_ty))

    # P consumed -> Vec1 may reuse the slot;  PV ready -> Vec2 may start
    sync_block_set("cube", "vector", RING + ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)
    sync_block_set("cube", "vector", 2 * RING + ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)


@triton.jit
def _fp16_max_combine(a, b):
    """Element-wise max that stays in the operand dtype (fp16).

    `tl.max` promotes fp16 to fp32 before reducing, and the Ascend backend
    hoists that `extf` above the reduction, so every 64x256 score block would
    pay a full widen+narrow pair. An explicit `tl.maximum` combiner keeps the
    whole reduction in fp16.
    """
    return tl.maximum(a, b)


# =============================================================================
#  Vector stage 2 — Vec1 (online softmax)
# =============================================================================
@triton.jit
def _vec1_softmax(
    workspace_s,
    workspace_p,
    cid,
    idx_in_conbine,
    global_head_idx,
    ring_slot,
    running_max_0,
    running_max_1,
    IS_CAUSAL: tl.constexpr,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUB_M: tl.constexpr,
    SUB_VEC_NUM: tl.constexpr,
):
    """Vec1: online softmax over CB score blocks -> workspace_p + register factors.

    Reads workspace_s[ring_slot], applies the causal mask, stabilises and
    exponentiates, stores P for MM2, and returns the per-(kv block, sub-tile)
    rescale/expsum pair in registers.

    Returns (running_max_0, running_max_1,
             rescale_00, expsum_00, rescale_01, expsum_01,
             rescale_10, expsum_10, rescale_11, expsum_11).
    """
    # all CB score blocks ready?
    sync_block_wait("cube", "vector", ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
    # workspace_p[ring_slot] free? (MM2 released it)
    sync_block_wait("cube", "vector", RING + ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

    # Reset the running max at each output tile's first KV group. `running_max`
    # holds the *negated* max, and 10.0 is a sentinel that the select below
    # displaces as soon as a real block arrives.
    if idx_in_conbine == 0:
        running_max_0 = tl.full((SUB_M, ), 10.0, tl.float16)
        running_max_1 = tl.full((SUB_M, ), 10.0, tl.float16)

    rescale_00 = tl.zeros((SUB_M, ), tl.float16)
    expsum_00 = tl.zeros((SUB_M, ), tl.float16)
    rescale_01 = tl.zeros((SUB_M, ), tl.float16)
    expsum_01 = tl.zeros((SUB_M, ), tl.float16)
    rescale_10 = tl.zeros((SUB_M, ), tl.float16)
    expsum_10 = tl.zeros((SUB_M, ), tl.float16)
    rescale_11 = tl.zeros((SUB_M, ), tl.float16)
    expsum_11 = tl.zeros((SUB_M, ), tl.float16)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx
        kv_col_start = kv_idx * BLOCK_N

        for sub_idx in tl.static_range(SUB_VEC_NUM):
            offset_m = sub_idx * SUB_M
            q_row_start = global_head_idx * BLOCK_M + offset_m

            score_load_bp = tl.make_block_ptr(
                workspace_s + cid * RING * CB * BLOCK_M * BLOCK_N + ring_slot * CB * BLOCK_M * BLOCK_N +
                cb_idx * BLOCK_M * BLOCK_N + offset_m * BLOCK_N, (SUB_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                (SUB_M, BLOCK_N), (1, 0))
            attn_score_block = tl.load(score_load_bp)

            if sub_idx == 0:
                running_max = running_max_0
            else:
                running_max = running_max_1

            # Masking is applied to every block of a causal run, branching only
            # on the `IS_CAUSAL` constexpr.
            #
            # ref §7.2 classifies each block (`GetAttenMaskComputeMode`) so that
            # only the diagonal-straddling one pays for a mask. That was tried
            # here and is a 1.5x *regression*: 1.83 ms against 1.18 ms. The block
            # index comes from the scheduler at runtime, so `kv_col_start >
            # q_row_start` is not a compile-time predicate — instead of
            # specialising, it wraps the mask, the reduction and the all-masked
            # fix-up in real control flow that the vector units run divergently,
            # and the lost uniformity costs far more than the saved compares.
            # AscendC resolves the same test on the scalar unit and switches
            # whole templated code paths, which is why it wins there and not here.
            if IS_CAUSAL:
                q_row_idx = q_row_start + tl.arange(0, SUB_M)
                kv_col_idx = kv_col_start + tl.arange(0, BLOCK_N)
                causal_mask = q_row_idx[:, None] >= kv_col_idx[None, :]
                attn_score_block = tl.where(causal_mask, attn_score_block, tl.full((1, ), -256.0, tl.float16))

            block_row_max = tl.reduce(attn_score_block, axis=-1, combine_fn=_fp16_max_combine)
            prev_neg_max = running_max

            # P0: `tl.minimum` lowers to a NaN-correct expansion costing 7
            # pipe_barrier<PIPE_V> per chain against 2 ops of real work:
            #   vmuls(-1.0) | vadds(+0.0) | vcmp_eq(a,a) vnot | vcmp_eq(b,b) vnot
            #   | vmin | vsel | vsel
            # The two vcmp_eq/vnot pairs are isnan() tests on each operand and
            # the two vsel propagate a NaN if either input is one. Neither
            # operand can be NaN here: block_row_max comes out of a max-reduce
            # over scores MM1 produced from finite inputs, and running_max is
            # either the 10.0 seed or a previous result of this same expression.
            # So the guard is dead weight; tl.where asks for the select directly.
            neg_block_row_max = -block_row_max
            candidate_neg_max = tl.where(neg_block_row_max < running_max, neg_block_row_max, running_max)

            if IS_CAUSAL:
                # A fully masked row has max == -256, which would drive exp() and
                # the denominator to a degenerate 0/0; hold the previous max
                # instead (ref §7.2 "全掩行" / AdjustSoftMaxRes). Still needed
                # even under the sparse schedule: a tile's last CB group can
                # overshoot its diagonal block, so the tail blocks inside that
                # group really are fully masked.
                row_all_masked = (block_row_max < -200.0)
                neg_max_new = tl.where(row_all_masked, running_max, candidate_neg_max).to(tl.float16)
            else:
                neg_max_new = candidate_neg_max.to(tl.float16)

            softmax_p = tl.exp(attn_score_block + neg_max_new[:, None])

            # Preserve the validated FP32 cross-KV rescale calculation.
            rescale = tl.exp(neg_max_new.to(tl.float32) - prev_neg_max.to(tl.float32)).to(tl.float16)
            block_expsum = tl.sum(softmax_p, axis=-1, keep_dims=False).to(tl.float16)

            # Register hand-off to Vec2 (positional: <cb_idx><sub_idx>). Capped
            # at CB <= MAX_CB by construction; the host asserts it.
            if cb_idx == 0 and sub_idx == 0:
                rescale_00 = rescale
                expsum_00 = block_expsum
            if cb_idx == 0 and sub_idx == 1:
                rescale_01 = rescale
                expsum_01 = block_expsum
            if cb_idx == 1 and sub_idx == 0:
                rescale_10 = rescale
                expsum_10 = block_expsum
            if cb_idx == 1 and sub_idx == 1:
                rescale_11 = rescale
                expsum_11 = block_expsum

            prob_store_bp = tl.make_block_ptr(
                workspace_p + cid * RING * CB * BLOCK_M * BLOCK_N + ring_slot * CB * BLOCK_M * BLOCK_N +
                cb_idx * BLOCK_M * BLOCK_N + offset_m * BLOCK_N, (SUB_M, BLOCK_N), (BLOCK_N, 1), (0, 0),
                (SUB_M, BLOCK_N), (1, 0))
            tl.store(prob_store_bp, softmax_p.to(workspace_p.dtype.element_ty))

            if sub_idx == 0:
                running_max_0 = neg_max_new
            else:
                running_max_1 = neg_max_new

    # scores consumed -> MM1 may reuse the slot;  P ready -> MM2 may start
    sync_block_set("vector", "cube", ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
    sync_block_set("vector", "cube", RING + ring_slot, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

    return (running_max_0, running_max_1, rescale_00, expsum_00, rescale_01, expsum_01, rescale_10, expsum_10,
            rescale_11, expsum_11)


# =============================================================================
#  Vector stage 4 — Vec2 (FlashAttention online refresh)
# =============================================================================
@triton.jit
def _vec2_accumulate(
    Out,
    workspace_pv,
    cid,
    idx_in_conbine,
    global_head_idx,
    batch_idx,
    head_idx,
    ring_slot,
    acc_o_0,
    softmax_denom_0,
    acc_o_1,
    softmax_denom_1,
    prev_rescale_00,
    prev_expsum_00,
    prev_rescale_01,
    prev_expsum_01,
    prev_rescale_10,
    prev_expsum_10,
    prev_rescale_11,
    prev_expsum_11,
    last_kv_idx,
    sOb,
    sOh,
    sOs,
    sOd,
    S,
    CB: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    SUB_M: tl.constexpr,
    SUB_VEC_NUM: tl.constexpr,
):
    """Vec2: acc_o = acc_o * rescale + P@V, then finalise on this tile's last block.

    `last_kv_idx` is the tile's own final KV-block index, passed in rather than
    assumed to be `num_kv_blocks - 1`: under causal each tile stops at its own
    diagonal block (ref §4.1 `GetS2LoopRange`).

    Returns (acc_o_0, softmax_denom_0, acc_o_1, softmax_denom_1).
    """
    # all CB P@V blocks ready?
    sync_block_wait("cube", "vector", 2 * RING + ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

    for cb_idx in tl.static_range(CB):
        kv_idx = idx_in_conbine * CB + cb_idx

        for sub_idx in tl.static_range(SUB_VEC_NUM):
            offset_m = sub_idx * SUB_M

            pv_load_bp = tl.make_block_ptr(
                workspace_pv + cid * RING * CB * BLOCK_M * DIM + ring_slot * CB * BLOCK_M * DIM +
                cb_idx * BLOCK_M * DIM + offset_m * DIM, (SUB_M, DIM), (DIM, 1), (0, 0), (SUB_M, DIM), (1, 0))
            pv_acc = tl.load(pv_load_bp)

            if cb_idx == 0 and sub_idx == 0:
                rescale = prev_rescale_00
                block_expsum = prev_expsum_00
            if cb_idx == 0 and sub_idx == 1:
                rescale = prev_rescale_01
                block_expsum = prev_expsum_01
            if cb_idx == 1 and sub_idx == 0:
                rescale = prev_rescale_10
                block_expsum = prev_expsum_10
            if cb_idx == 1 and sub_idx == 1:
                rescale = prev_rescale_11
                block_expsum = prev_expsum_11

            if sub_idx == 0:
                acc_o = acc_o_0
                softmax_denom = softmax_denom_0
            else:
                acc_o = acc_o_1
                softmax_denom = softmax_denom_1

            if kv_idx == 0:
                # ref §4.4: the first KV block needs no rescale, just seed.
                acc_o = pv_acc
                softmax_denom = block_expsum
            else:
                acc_o = acc_o * rescale[:, None] + pv_acc
                softmax_denom = softmax_denom * rescale + block_expsum

            if kv_idx == last_kv_idx:
                # ref §4.4 / §7.4: divide by the denominator and store straight
                # into the final layout — no staging tile, no transpose.
                output_block = (acc_o / softmax_denom[:, None]).to(Out.dtype.element_ty)
                o_bp = tl.make_block_ptr(Out + batch_idx * sOb + head_idx * sOh, (S, DIM), (sOs, sOd),
                                         (global_head_idx * BLOCK_M + offset_m, 0), (SUB_M, DIM), (1, 0))
                tl.store(o_bp, output_block)

            if sub_idx == 0:
                acc_o_0 = acc_o
                softmax_denom_0 = softmax_denom
            else:
                acc_o_1 = acc_o
                softmax_denom_1 = softmax_denom

    # PV consumed -> MM2 may reuse the slot
    sync_block_set("vector", "cube", 2 * RING + ring_slot, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    return acc_o_0, softmax_denom_0, acc_o_1, softmax_denom_1


# =============================================================================
#  The skew-1 4-stage scheduler kernel
#
#  grid = (NUM_CORES,). Each program drives one Cube + one Vector engine
#  (MIX_1_1). Stage placement per tick g:
#
#      Cube  : MM1(g)    MM2(g-1)
#      Vector: Vec1(g)   Vec2(g-1)
#
#  Semaphore ledger:
#
#      channel    producer     consumer
#      S_READY    MM1(g)       Vec1(g)
#      S_FREE     Vec1(g)      MM1(g+RING)
#      P_READY    Vec1(g)      MM2(g)
#      P_FREE     MM2(g)       Vec1(g+RING)
#      PV_READY   MM2(g)       Vec2(g)
#      PV_FREE    Vec2(g)      MM2(g+RING)
#
#  The softmax factors need no channel: producer and consumer are both on the
#  Vector core, so the snapshot into `prev_*` plus program order is the whole
#  protocol. Vec1(g) runs before Vec2(g-1) in the same tick and would otherwise
#  clobber the registers Vec2 still needs.
# =============================================================================
@triton.jit
def flash_attention_fwd_kernel_v7(
    Q,
    K,
    V,
    Out,
    workspace_s,
    workspace_p,
    workspace_pv,  # GM ring workspaces
    task_map,  # i32[NUM_CORES, max_tasks]: packed tile*cbn + group (sparse only)
    core_ntasks,  # i32[NUM_CORES]: tasks owned (sparse) / unused (dense)
    max_tasks,  # row stride of task_map (sparse only)
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
    num_seq_blocks,
    heads_q,
    gqa_group,
    num_kv_blocks,  # KV blocks per output tile (= S // BLOCK_N)
    conbined_block_num,  # tasks per output tile (= num_kv_blocks // CB), dense only
    block_num_per_core,
    rem_block_num,
    CB: tl.constexpr,  # KV blocks per task (arch22 nRatio)
    NUM_KV_BLOCKS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SPARSE: tl.constexpr,  # use the host's balanced sparse schedule
    RING: tl.constexpr,  # task-ring depth; tuned
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
    SUB_M: tl.constexpr,
    SUB_VEC_NUM: tl.constexpr,
):
    cid = tl.program_id(0)

    # ---- task distribution (== AICPU GetFASectionInfo metadata, ref §5.4) ----
    if SPARSE:
        # Host-computed balanced schedule: this core's packed task row.
        my_task_map = task_map + cid * max_tasks
        num_global_tasks = tl.load(core_ntasks + cid)
        tile_base = 0  # unused on the sparse path
    else:
        my_task_map = task_map
        block_start = cid * block_num_per_core + tl.where(cid < rem_block_num, cid, rem_block_num)
        block_num = block_num_per_core + tl.where(cid < rem_block_num, 1, 0)
        num_global_tasks = block_num * conbined_block_num
        tile_base = block_start

    # ---- Vector-side persistent state (UB registers) ----
    # One set per sub-tile: the sub-tiles run sequentially but each carries its
    # own FlashAttention accumulator across the tile's KV groups (ref §6.2, the
    # output-tile buffer family).
    acc_o_0 = tl.zeros((SUB_M, DIM), tl.float16)
    acc_o_1 = tl.zeros((SUB_M, DIM), tl.float16)
    softmax_denom_0 = tl.zeros((SUB_M, ), tl.float16)
    softmax_denom_1 = tl.zeros((SUB_M, ), tl.float16)
    running_max_0 = tl.full((SUB_M, ), 10.0, tl.float16)
    running_max_1 = tl.full((SUB_M, ), 10.0, tl.float16)

    # Vec1(g) -> Vec2(g) softmax factors, one pair per (cb_idx, sub_idx).
    rescale_00 = tl.zeros((SUB_M, ), tl.float16)
    expsum_00 = tl.zeros((SUB_M, ), tl.float16)
    rescale_01 = tl.zeros((SUB_M, ), tl.float16)
    expsum_01 = tl.zeros((SUB_M, ), tl.float16)
    rescale_10 = tl.zeros((SUB_M, ), tl.float16)
    expsum_10 = tl.zeros((SUB_M, ), tl.float16)
    rescale_11 = tl.zeros((SUB_M, ), tl.float16)
    expsum_11 = tl.zeros((SUB_M, ), tl.float16)

    # =========================================================================
    #  PROLOGUE: prime the FREE channels so the first RING ticks can proceed
    # =========================================================================
    with tle.scope(core_mode="cube"):
        for s in tl.static_range(RING):
            sync_block_set("cube", "vector", RING + s, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

    with tle.scope(core_mode="vector"):
        for s in tl.static_range(RING):
            sync_block_set("vector", "cube", s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
            sync_block_set("vector", "cube", 2 * RING + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)

    # =========================================================================
    #  Main loop. One trailing tick drains stage g-1 (ref §4.1 does the same
    #  with `multiCoreInnerLimit += 2` instead of a separate epilogue).
    # =========================================================================
    # Task g-1's coordinates, carried in scalar registers rather than re-decoded.
    # The two g-1 stages (MM2, Vec2) used to each run their own decode, which on
    # the sparse path means a GM load apiece; now the tick's single decode feeds
    # stage g and last tick's values feed stage g-1.
    p_idx_in_conbine = 0
    p_global_head_idx = 0
    p_head_idx = 0
    p_batch_idx = 0
    p_kv_head_idx = 0

    for g in range(num_global_tasks + 1):

        # One decode per tick, shared by both scopes (ref §7.5: keep the loop
        # body free of redundant scalar work).
        if g < num_global_tasks:
            (idx_in_conbine, global_head_idx, head_idx, batch_idx,
             kv_head_idx) = _task_coords(g, tile_base, my_task_map, num_seq_blocks, heads_q, gqa_group,
                                         conbined_block_num, SPARSE)
        else:
            idx_in_conbine = 0
            global_head_idx = 0
            head_idx = 0
            batch_idx = 0
            kv_head_idx = 0

        # -----------------------------------------------------------------
        #  Cube scope: MM1(g), MM2(g-1)
        # -----------------------------------------------------------------
        with tle.scope(core_mode="cube"):
            if g < num_global_tasks:
                _mm1_qkt(
                    Q,
                    K,
                    workspace_s,
                    cid,
                    idx_in_conbine,
                    global_head_idx,
                    batch_idx,
                    head_idx,
                    kv_head_idx,
                    g % RING,
                    sQb,
                    sQh,
                    sQs,
                    sQd,
                    sKb,
                    sKh,
                    sKs,
                    sKd,
                    S,
                    CB,
                    RING,
                    BLOCK_M,
                    BLOCK_N,
                    DIM,
                )

            if g >= 1:
                idx_in_conbine_1 = p_idx_in_conbine
                batch_idx_1 = p_batch_idx
                kv_head_idx_1 = p_kv_head_idx
                _mm2_pv(
                    V,
                    workspace_p,
                    workspace_pv,
                    cid,
                    idx_in_conbine_1,
                    batch_idx_1,
                    kv_head_idx_1,
                    (g - 1) % RING,
                    sKb,
                    sKh,
                    sKs,
                    sKd,
                    S,
                    CB,
                    RING,
                    BLOCK_M,
                    BLOCK_N,
                    DIM,
                )

        # -----------------------------------------------------------------
        #  Vector scope: Vec1(g), Vec2(g-1)
        # -----------------------------------------------------------------
        with tle.scope(core_mode="vector"):
            # Vec2 below consumes tick g-1's factors, but Vec1(g) runs first and
            # overwrites the registers, so snapshot them here.
            prev_rescale_00 = rescale_00
            prev_expsum_00 = expsum_00
            prev_rescale_01 = rescale_01
            prev_expsum_01 = expsum_01
            prev_rescale_10 = rescale_10
            prev_expsum_10 = expsum_10
            prev_rescale_11 = rescale_11
            prev_expsum_11 = expsum_11

            if g < num_global_tasks:
                (running_max_0, running_max_1, rescale_00, expsum_00, rescale_01, expsum_01, rescale_10, expsum_10,
                 rescale_11, expsum_11) = _vec1_softmax(
                     workspace_s,
                     workspace_p,
                     cid,
                     idx_in_conbine,
                     global_head_idx,
                     g % RING,
                     running_max_0,
                     running_max_1,
                     IS_CAUSAL,
                     CB,
                     RING,
                     BLOCK_M,
                     BLOCK_N,
                     SUB_M,
                     SUB_VEC_NUM,
                 )

            if g >= 1:
                idx_in_conbine_1 = p_idx_in_conbine
                global_head_idx_1 = p_global_head_idx
                head_idx_1 = p_head_idx
                batch_idx_1 = p_batch_idx
                # Which KV block finalises this tile's output row?
                #
                # Sparse (causal): the tile stops at its diagonal block, rounded
                # up to a whole CB group — the same closed form the host uses to
                # size the schedule, so no lookup table is needed.
                # Dense (incl. dense causal): every tile walks the full range.
                if SPARSE:
                    diag_kv = ((global_head_idx_1 + 1) * BLOCK_M - 1) // BLOCK_N
                    n_groups = (diag_kv + CB) // CB  # ceil((diag_kv + 1) / CB)
                    last_kv_idx = n_groups * CB - 1
                else:
                    last_kv_idx = NUM_KV_BLOCKS - 1

                acc_o_0, softmax_denom_0, acc_o_1, softmax_denom_1 = _vec2_accumulate(
                    Out,
                    workspace_pv,
                    cid,
                    idx_in_conbine_1,
                    global_head_idx_1,
                    batch_idx_1,
                    head_idx_1,
                    (g - 1) % RING,
                    acc_o_0,
                    softmax_denom_0,
                    acc_o_1,
                    softmax_denom_1,
                    prev_rescale_00,
                    prev_expsum_00,
                    prev_rescale_01,
                    prev_expsum_01,
                    prev_rescale_10,
                    prev_expsum_10,
                    prev_rescale_11,
                    prev_expsum_11,
                    last_kv_idx,
                    sOb,
                    sOh,
                    sOs,
                    sOd,
                    S,
                    CB,
                    RING,
                    BLOCK_M,
                    DIM,
                    SUB_M,
                    SUB_VEC_NUM,
                )

        # Hand this tick's coordinates to next tick's g-1 stages.
        p_idx_in_conbine = idx_in_conbine
        p_global_head_idx = global_head_idx
        p_head_idx = head_idx
        p_batch_idx = batch_idx
        p_kv_head_idx = kv_head_idx

    # =========================================================================
    #  EPILOGUE: drain the primed FREE tokens no stage consumed
    # =========================================================================
    with tle.scope(core_mode="cube"):
        for s in tl.static_range(RING):
            sync_block_wait("cube", "vector", RING + s, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

    with tle.scope(core_mode="vector"):
        for s in tl.static_range(RING):
            sync_block_wait("vector", "cube", s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
            sync_block_wait("vector", "cube", 2 * RING + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)


# =============================================================================
#  Host: causal schedule construction  (== arch22 InitSparseValidArray /
#  PartitionSparseData / SetSparseStartIdx, ref §5.4)
# =============================================================================
def _build_causal_schedule(num_seq_blocks, heads_q, batch, CB, num_kv_blocks, device, block_m=BLOCK_M,
                          block_n=BLOCK_N):
    """Build the balanced sparse schedule for a causal run.

    Two jobs, both of which arch22 does on the host:

    * **narrow** (ref §4.1 `GetS2LoopRange`): Q row-block `i` spans rows
      `[i*BLOCK_M, (i+1)*BLOCK_M)`, so its last visible KV block is
      `((i+1)*BLOCK_M - 1) // BLOCK_N`. Tasks past that never get issued.
      Rounded up to a whole CB group, a tile owns `ceil((last_kv + 1) / CB)`
      tasks.
    * **balance** (ref §5.4): tile costs now range from 1 task to
      `num_kv_blocks/CB`, so hand tiles out longest-first to whichever core is
      currently least loaded. That is the greedy half of arch22's
      binary-search-plus-greedy; for this cost distribution it lands within a
      few percent of the optimum without the search.

    Returns (task_map, core_ntasks, max_tasks): an i32[NUM_CORES, max_tasks]
    device tensor of packed `tile * conbined_block_num + group` entries, the
    per-core task count, and the row stride.
    """
    num_tiles = num_seq_blocks * heads_q * batch

    # -- narrow: per-tile last KV block and task count -------------------------
    row_block = torch.arange(num_seq_blocks, dtype=torch.int64)
    last_kv_per_row = ((row_block + 1) * block_m - 1) // block_n
    last_kv_per_row = torch.clamp(last_kv_per_row, max=num_kv_blocks - 1)
    # A task covers CB KV blocks, so round the count up to whole groups. The
    # final group may reach past `last_kv`; those blocks are fully masked and
    # Vec1's all-masked-row path keeps them from polluting the accumulator.
    tasks_per_row = (last_kv_per_row + 1 + CB - 1) // CB
    # The tile's effective last KV block is the end of its last whole group,
    # `tasks_per_row * CB - 1`. Vec2 recomputes that closed form itself rather
    # than reading it from a table, so it is not materialised here.
    tile_tasks = tasks_per_row.repeat(batch * heads_q)

    # -- balance: longest-processing-time-first greedy packing ------------------
    order = torch.argsort(tile_tasks, descending=True, stable=True)
    per_core_tiles = [[] for _ in range(NUM_CORES)]
    per_core_load = [0] * NUM_CORES
    for tile in order.tolist():
        c = min(range(NUM_CORES), key=lambda i: per_core_load[i])
        per_core_tiles[c].append(tile)
        per_core_load[c] += int(tile_tasks[tile])

    # -- emit: one packed i32 per task, so the kernel decodes with one load -----
    # Each entry is `tile * conbined_block_num + group`. Cores walk their tiles
    # in ascending tile order, because consecutive tiles share a (batch, head)
    # and that keeps the Q/K/V reads local.
    conbined_block_num = num_kv_blocks // CB
    max_tasks = int(max(per_core_load))
    task_map = torch.zeros((NUM_CORES, max_tasks), dtype=torch.int32)
    core_ntasks = torch.zeros((NUM_CORES, ), dtype=torch.int32)
    for c in range(NUM_CORES):
        j = 0
        for tile in sorted(per_core_tiles[c]):
            for grp in range(int(tile_tasks[tile])):
                task_map[c, j] = tile * conbined_block_num + grp
                j += 1
        core_ntasks[c] = j

    return task_map.to(device), core_ntasks.to(device), max_tasks


# =============================================================================
#  Intermediate-IR dumps (no device required)
#
#  Compiles the kernel straight to TTIR with the tile/tle/ascend dialects
#  registered, then writes str(module). The tile.* ops emitted by the tile-DSA
#  builtins appear in this dump — that is the intermediate CommonIR. Mirrors
#  python/test/tle/test_bind_buffer.py::compile_kernel.
#
#  For the backend-side dump, set BISHENGIR_DUMP_IR_AFTER_ALL=1 instead.
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
    sig = {
        "Q": "*fp16",
        "K": "*fp16",
        "V": "*fp16",
        "Out": "*fp16",
        "workspace_s": "*fp16",
        "workspace_p": "*fp16",
        "workspace_pv": "*fp16",
        "task_map": "*i32",
        "core_ntasks": "*i32",
    }
    i32_names = [
        "max_tasks", "B", "Hq", "Hkv", "S", "sQb", "sQh", "sQs", "sQd", "sKb", "sKh", "sKs", "sKd", "sOb", "sOh",
        "sOs", "sOd", "num_seq_blocks", "heads_q", "gqa_group", "num_kv_blocks", "conbined_block_num",
        "block_num_per_core", "rem_block_num"
    ]
    for n in i32_names:
        sig[n] = "i32"
    for n in ["IS_CAUSAL", "SPARSE", "RING", "BLOCK_M", "BLOCK_N", "DIM", "SUB_M", "SUB_VEC_NUM"]:
        sig[n] = "constexpr"
    return sig


def _dump_constants(num_kv_blocks, combine_batch, is_causal, sparse, config=None):
    """Constexpr values for a dump. `config` defaults to the tuned baseline."""
    cfg = config or DEFAULT_CONFIG
    if combine_batch is not None:
        cfg = cfg._replace(combine_batch=combine_batch)
    return {
        "CB": cfg.combine_batch,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "IS_CAUSAL": is_causal,
        "SPARSE": sparse,
        "RING": cfg.ring,
        "BLOCK_M": cfg.block_m,
        "BLOCK_N": cfg.block_n,
        "DIM": DIM,
        "SUB_M": cfg.sub_m,
        "SUB_VEC_NUM": cfg.sub_vec_num,
    }


def _load_all_dialects(context):
    from triton._C.libtriton import ir
    from triton._C.libtriton import tle as tle_ir
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.dsa_ir.load_tile_dialects(context)
    # Ascend dialect is optional (only needed for ascend-specific ops).
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass


def dump_commonir(path=None, ttir_path=None, num_kv_blocks=32, combine_batch=2, is_causal=False, sparse=False,
                  config=None):
    """Compile the kernel to TTIR (containing tile.* ops) and write it to `path`.

    Also runs the CommonIR→HIVM pass to lower tile.* ops and dumps the resulting
    pure TTIR to `ttir_path`. Requires no NPU/GPU — pure front-end compilation.

    Returns the CommonIR MLIR string. The TTIR dump is written as a side effect.
    """
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir

    # The kernel reads module-level shape constants (BLOCK_M, DIM, ...) as plain
    # globals; allow that during this front-end-only dump.
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    here = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        path = os.path.join(here, "fa_v7_casual.mlir")
    if ttir_path is None:
        ttir_path = os.path.join(here, "fa_v7_casual_ttir.mlir")

    src = ASTSource(flash_attention_fwd_kernel_v7, _dump_signature(),
                    _dump_constants(num_kv_blocks, combine_batch, is_causal, sparse, config))
    context = ir.context()
    _load_all_dialects(context)

    # codegen_fns: tl.dot needs a target-provided "min_dot_size"; the ascend
    # backend simply returns (1,1,1), so supply that inline (no full backend).
    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(flash_attention_fwd_kernel_v7, src, context, _DumpOptions(), codegen_fns, {})

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_commonir: module.verify() failed — IR is not legal")

    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_commonir] module.verify() = {ok}; wrote legal CommonIR to {path}")

    # ---- dump TTIR (CommonIR→HIVM + TTIR optimization passes) ----
    from triton._C.libtriton import passes as ir_passes
    pm = ir.pass_manager(context)
    pm.enable_debug()
    # Inline first, so tile.buf types never cross a call boundary.
    ir_passes.common.add_inliner(pm)
    try:
        from triton._C.libtriton.ascend import passes as ascend_passes
        ascend_passes.ttir.add_commonir_to_hivm(pm)
    except Exception:
        pass
    try:
        pm.run(module)
        print(f"[dump_commonir] after CommonIR→HIVM: verify={module.verify()}", flush=True)
    except RuntimeError as e:
        print(f"[dump_commonir] CommonIR→HIVM pass failed (non-fatal): {e}", flush=True)
        print("[dump_commonir] CommonIR output is still valid; skipping HIVM lowering.", flush=True)
        return mlir

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

    ttir_ok = module.verify()
    if not ttir_ok:
        print("[dump_commonir] WARNING: module.verify() failed after TTIR optimization — IR may be illegal")
    else:
        print(f"[dump_commonir] TTIR optimization complete: verify={ttir_ok}")

    with open(ttir_path, "w") as f:
        f.write(str(module))
    print(f"[dump_commonir] wrote optimized TTIR to {ttir_path}")

    return mlir


def _parse_back(commonir_mlir):
    """Re-parse a CommonIR string into a fresh (context, module) pair."""
    import tempfile
    from triton._C.libtriton import ir
    context = ir.context()
    _load_all_dialects(context)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(commonir_mlir)
        tmp_path = f.name
    module = ir.parse_mlir_module(tmp_path, context)
    os.unlink(tmp_path)
    return context, module


def dump_hivm(path=None, num_kv_blocks=32, combine_batch=2, is_causal=False, sparse=False, config=None):
    """Compile to CommonIR, then lower through the CommonIR→HIVM pipeline.

    Phases:
      ① add_commonir_to_hivm  — tile.alloc → memref.alloc, tile.copy → hivm.copy
                                (on-chip) or memref.copy (GM↔local), tile.load/
                                store → hivm.load/store, sync ops → hivm sync
      ② add_triton_to_hivm    — Triton CustomOp → HIVM SyncOp
      ③ inline + canonicalize — clean up

    Returns the MLIR string. No NPU/GPU required.
    """
    from triton._C.libtriton import ir, passes, ascend

    commonir_mlir = dump_commonir(num_kv_blocks=num_kv_blocks, combine_batch=combine_batch, is_causal=is_causal,
                                  sparse=sparse, config=config)
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_v7_casual_hivm.mlir")

    context, module = _parse_back(commonir_mlir)

    pm1 = ir.pass_manager(context)
    pm1.enable_debug()
    ascend.passes.ttir.add_commonir_to_hivm(pm1)
    pm1.run(module)
    print(f"[dump_hivm] ① commonir_to_hivm: verify={module.verify()}", flush=True)

    pm2 = ir.pass_manager(context)
    pm2.enable_debug()
    ascend.passes.ttir.add_triton_to_hivm(pm2)
    pm2.run(module)
    print(f"[dump_hivm] ② triton_to_hivm: verify={module.verify()}", flush=True)

    pm3 = ir.pass_manager(context)
    pm3.enable_debug()
    passes.common.add_inliner(pm3)
    passes.common.add_canonicalizer(pm3)
    pm3.run(module)
    print(f"[dump_hivm] ③ canonicalize: verify={module.verify()}", flush=True)

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_hivm: module.verify() failed after pipeline — IR is not legal")

    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_hivm] module.verify() = {ok}; wrote HIVM IR to {path}")
    return mlir


def dump_linalg(path=None, num_kv_blocks=32, combine_batch=2, is_causal=False, sparse=False, config=None):
    """Compile the kernel through the full CommonIR→Linalg lowering pipeline.

    Phases:
      ① add_commonir_to_hivm                 — tile.* → memref/hivm
      ② add_discrete_mask_access_conversion  — non-contiguous mask handling
      ③ add_triton_to_unstructure            — scalarize unstructured accesses
         add_triton_to_hivm                  — CustomOp → HIVM SyncOp
         add_triton_to_hfusion               — Triton → HFusion
         add_triton_to_llvm                  — Triton → LLVM
      ④ add_bubble_up_operation              — push extracts upward
      ④b inline + canonicalize               — REQUIRED: without it the linalg
                                               incubator hits a
                                               cast<RankedTensorType> assertion
                                               in MaskAnalysis on non-tensor types
      ⑤ add_triton_to_linalg                 — Triton→Linalg; may report a
                                               partial conversion, which is
                                               expected and non-fatal here

    Returns the final Linalg MLIR string. No NPU/GPU required.
    """
    from triton._C.libtriton import ir, passes, ascend

    commonir_mlir = dump_commonir(num_kv_blocks=num_kv_blocks, combine_batch=combine_batch, is_causal=is_causal,
                                  sparse=sparse, config=config)
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_v7_casual_linalg.mlir")

    context, module = _parse_back(commonir_mlir)

    pm = ir.pass_manager(context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    ascend.passes.ttir.add_commonir_to_hivm(pm)
    pm.run(module)
    print(f"[dump_linalg] ① commonir_to_hivm: verify={module.verify()}", flush=True)

    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ② discrete_mask: verify={module.verify()}", flush=True)

    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_triton_to_unstructure(pm, False, False)
    ascend.passes.ttir.add_triton_to_hivm(pm)
    ascend.passes.ttir.add_triton_to_hfusion(pm)
    ascend.passes.ttir.add_triton_to_llvm(pm)
    pm.run(module)
    print(f"[dump_linalg] ③ unstructure+hivm+hfusion+llvm: verify={module.verify()}", flush=True)

    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_bubble_up_operation(pm)
    pm.run(module)
    print(f"[dump_linalg] ④ bubble_up: verify={module.verify()}", flush=True)

    pm = ir.pass_manager(context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module)
    print(f"[dump_linalg] ④b inline+canonicalize: verify={module.verify()}", flush=True)

    try:
        pm = ir.pass_manager(context)
        pm.enable_debug()
        ascend.passes.ttir.add_triton_to_linalg(pm, False, True, False, False, False)
        pm.run(module)
        print(f"[dump_linalg] ⑤ triton_to_linalg: verify={module.verify()}", flush=True)
    except RuntimeError:
        print(
            "[dump_linalg] ⑤ triton_to_linalg: partial conversion (expected — the "
            "pass creates memref→!tt.ptr→memref cast chains)", flush=True)

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
# The GM scratch is tens of MB, so don't re-allocate it per call.
_workspace_cache: dict = {}
_schedule_cache: dict = {}

# Tuned config per (shape, causal, schedule) — see autotune_config.
_autotune_cache: dict = {}


# =============================================================================
#  Autotuner
# =============================================================================
# How many times each candidate's timing batch is repeated; the minimum wins.
_TUNE_REPEATS = 3


def _log_dir():
    """Where compile failures go. remind.md: save the full error, then ask."""
    return os.environ.get("FA_V7_LOG_DIR", "/data/jianheng/works/FlagTree/log")


def _record_failure(cfg, shape_key, exc):
    """Persist a candidate's full traceback and return the path (or None)."""
    try:
        d = _log_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"fa_v7_autotune_fail_{cfg.label().replace('/', '_')}.log")
        with open(path, "w") as f:
            f.write(f"config: {cfg}\nlabel: {cfg.label()}\nshape: {shape_key}\n"
                    f"UB={_ub_bytes(cfg)}B L0C={_l0c_bytes(cfg)}B "
                    f"workspace={_workspace_bytes(cfg) / 2**20:.1f}MiB\n\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return path
    except OSError:
        return None


def _time_config(q, k, v, cfg, is_causal, balance_causal, warmup, iters):
    """Per-launch wall time for one config, in ms. Raises on failure.

    Measured exactly the way `--bench` measures: `iters` launches enqueued
    back-to-back with a single `synchronize()` at each end, then divided. The
    earlier version synchronised after *every* launch, which charged each
    candidate one full host round-trip (~0.23 ms at B=4 S=2048 — it reported
    1.2594 ms where `--bench 50` reports 1.0270 ms for the same config). That
    offset is not constant enough to cancel out: it buried the RING axis, whose
    real spread is ~0.03 ms, in launch-latency jitter. Repeated `_REPEATS` times
    with the minimum kept, since the floor is the least noisy statistic here.

    Timed through `flash_attention_fwd` so a candidate takes production's exact
    path, pre-scale hoisted out of the loop as `--bench` does it.
    """
    import time
    sm_scale = (1.0 / q.shape[-1])**0.5
    q_pre = q * sm_scale
    n = max(1, iters)

    def run():
        return flash_attention_fwd(q_pre, k, v, is_causal=is_causal, _q_prescaled=True,
                                   balance_causal=balance_causal, config=cfg)

    for _ in range(max(1, warmup)):
        run()
    torch.npu.synchronize()

    best = float("inf")
    for _ in range(_TUNE_REPEATS):
        t0 = time.perf_counter()
        for _ in range(n):
            run()
        torch.npu.synchronize()
        best = min(best, (time.perf_counter() - t0) / n * 1e3)
    return best


def _config_is_correct(q, k, v, cfg, is_causal, balance_causal, ref, atol=2e-2, rtol=2e-2):
    """True if `cfg` matches `ref` at this shape.

    A candidate that compiles and runs fast but computes the wrong thing would
    otherwise win the search outright, which is the worst failure mode available
    to a tuner. `config_is_valid` catches the ceilings it can see from host
    arithmetic; this catches everything else, once, before the config is timed.

    `ref` is computed once by the caller — the fp32 reference costs more than a
    measurement at the larger shapes, so it must not be per-candidate.
    """
    out = flash_attention_fwd(q, k, v, is_causal=is_causal, balance_causal=balance_causal, config=cfg)
    return torch.allclose(ref, out, rtol=rtol, atol=atol)


class _TuneCursor:
    """Mutable cell holding the best config found so far.

    Coordinate descent needs to read the running winner between candidates. The
    caller assigns `cursor.best`; the generator reads it on its next step. This
    keeps the drive loop an ordinary `for`, so the `continue` paths for pruned,
    wrong, and failed candidates cannot desynchronise a send/yield handshake.
    """

    def __init__(self, base):
        self.best = base


def _tune_candidates(base, space, strategy, cursor=None):
    """Enumerate candidates, baseline first, without duplicates.

    * `coord` — one coordinate-descent sweep: vary a single axis at a time off
      the baseline, so the cost is sum(len(axis)) compiles. Each axis is listed
      best-known-first, and the winner of each axis is carried into the next via
      `cursor.best`, so one pass already captures the interactions that matter
      most (a wider BLOCK_N wants a different SUB_VEC_NUM, for instance) at a
      fraction of grid cost. Default, because a bishengir compile is tens of
      seconds.
    * `grid` — the full cross product, for when that budget exists.

    `cursor` may be None, in which case `coord` descends from `base` throughout.
    """
    if strategy == "grid":
        import itertools
        keys = list(space)
        seen = set()
        for combo in itertools.product(*(space[kk] for kk in keys)):
            cfg = base._replace(**dict(zip(keys, combo)))
            if cfg not in seen:
                seen.add(cfg)
                yield cfg
        return

    seen = {base}
    yield base
    for axis, values in space.items():
        for val in values:
            cur = cursor.best if cursor is not None else base
            cfg = cur._replace(**{axis: val})
            if cfg in seen:
                continue
            seen.add(cfg)
            yield cfg


def autotune_config(q, k, v, is_causal=False, balance_causal=True, combine_batch=None, space=None, strategy="coord",
                    warmup=2, iters=20, verbose=True, cache=True, cache_file=None, check=True):
    """Search for the fastest FAConfig at this shape and cache it.

    Why this is not `@triton.autotune`: BLOCK_M/BLOCK_N/CB/RING size the GM ring
    workspaces, and BLOCK_M/BLOCK_N/CB decide the *contents* of `task_map`,
    because the sparse causal schedule is built on the host from the tile shape.
    `@triton.autotune` varies constexprs against a fixed argument list and can
    express neither. Re-entering the launcher per candidate also means each one
    is timed through the exact path production uses.

    Invalid candidates are pruned by host arithmetic (`config_is_valid`) before a
    compile is spent. A candidate that survives that and still fails to compile
    gets its full traceback written under `$FA_V7_LOG_DIR` and is skipped, so one
    bad corner of the space cannot abort the sweep. A candidate that compiles but
    computes the wrong answer is rejected by `_config_is_correct` before it is
    timed, so it cannot win the search.

    Returns the winning FAConfig.
    """
    B, Hq, S, D = q.shape
    sparse = bool(is_causal and balance_causal)
    key = (str(q.device), B, Hq, k.shape[1], S, D, str(q.dtype), bool(is_causal), sparse)

    if cache_file:
        disk = _load_tuned(cache_file)
        if key in disk:
            _autotune_cache[key] = disk[key]
    if cache and key in _autotune_cache:
        return _autotune_cache[key]

    base = DEFAULT_CONFIG
    if combine_batch is not None:
        base = base._replace(combine_batch=combine_batch)
    space = dict(TUNE_SPACE if space is None else space)
    if combine_batch is not None:
        space.pop("combine_batch", None)  # explicitly pinned by the caller

    # A baseline that does not fit this shape (e.g. S=128 with BLOCK_N=256) is
    # replaced by the first valid point in the space, so tuning still has a
    # reference to descend from.
    if not config_is_valid(base, S)[0]:
        for cand in _tune_candidates(base, space, "grid"):
            if config_is_valid(cand, S)[0]:
                base = cand
                break

    # One fp32 reference for the whole search. It is head-by-head and slow, so
    # amortise it; if it cannot be built (memory, unsupported shape) fall back to
    # timing-only rather than abandoning the search.
    ref = None
    if check:
        try:
            ref = _reference(q, k, v, is_causal)
        except Exception as e:
            if verbose:
                print(f"  [note] reference unavailable ({type(e).__name__}), tuning without the "
                      f"correctness gate")

    best_cfg, best_ms = None, float("inf")
    tried = skipped = failed = wrong = 0

    # The running winner is carried to the generator through a mutable cell
    # rather than generator.send(), so this stays an ordinary for-loop and the
    # `continue` paths below cannot get out of step with the feedback protocol.
    cursor = _TuneCursor(base)
    for cfg in _tune_candidates(base, space, strategy, cursor):
        ok, why = config_is_valid(cfg, S)
        if not ok:
            skipped += 1
            if verbose:
                print(f"  [skip] {cfg.label():<38} {why}")
            continue

        try:
            if ref is not None and not _config_is_correct(q, k, v, cfg, is_causal, balance_causal, ref):
                wrong += 1
                if verbose:
                    print(f"  [WRONG] {cfg.label():<37} compiles but mismatches the reference")
                continue
            ms = _time_config(q, k, v, cfg, is_causal, balance_causal, warmup, iters)
        except Exception as e:  # compile or launch failure: record and move on
            failed += 1
            path = _record_failure(cfg, key, e)
            if verbose:
                first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                print(f"  [FAIL] {cfg.label():<38} {first[:90]}")
                if path:
                    print(f"         full error -> {path}")
            continue

        tried += 1
        mark = ""
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
            cursor.best = cfg
            mark = "  <- best"
        if verbose:
            print(f"  [ok]   {cfg.label():<38} {ms:8.4f} ms{mark}")

    if best_cfg is None:
        raise RuntimeError(f"autotune: no config ran at B={B} H={Hq} S={S} D={D} "
                           f"({skipped} pruned, {failed} failed to compile, {wrong} wrong; "
                           f"see {_log_dir()})")

    if verbose:
        print(f"autotune: {best_cfg.label()} at {best_ms:.4f} ms "
              f"({tried} timed, {skipped} pruned, {failed} failed, {wrong} wrong)")
    if cache:
        _autotune_cache[key] = best_cfg
    if cache_file:
        _store_tuned(cache_file, key, best_cfg, best_ms)
    return best_cfg


def _load_tuned(path):
    """Read a tuning cache written by `_store_tuned`. Missing/corrupt -> {}."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for entry in raw.get("entries", []):
        try:
            out[tuple(entry["key"])] = FAConfig(**entry["config"])
        except (KeyError, TypeError):
            continue
    return out


def _store_tuned(path, key, cfg, ms):
    """Merge one result into the on-disk tuning cache (best-effort)."""
    existing = _load_tuned(path)
    existing[tuple(key)] = cfg
    entries = [{"key": list(kk), "config": vv._asdict()} for kk, vv in existing.items()]
    # Keep the timing of the entry just measured, purely as a human-readable note.
    for e in entries:
        if tuple(e["key"]) == tuple(key):
            e["ms"] = round(ms, 4)
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": 1, "entries": entries}, f, indent=1)
    except OSError as e:
        print(f"[autotune] could not write cache {path}: {e}")


def flash_attention_fwd(q, k, v, combine_batch=None, is_causal=False, _q_prescaled=False, balance_causal=True,
                        config=None, autotune=False):
    """Launch the v7 kernel.

    `config` pins an explicit FAConfig; `autotune=True` searches for one and
    caches it per (device, shape, causal, schedule) key. With neither, the
    measured baseline DEFAULT_CONFIG is used. `combine_batch` stays supported as
    a direct override of that one axis, so existing callers and the CLI flag
    behave exactly as before — except against a tuned config, where the tuner has
    already searched that axis under the pin and re-applying it here would clamp
    the result to a stale `block_n`.
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and Hq % Hkv == 0

    tuned = False
    if config is None and autotune:
        config = autotune_config(q, k, v, is_causal=is_causal, balance_causal=balance_causal,
                                 combine_batch=combine_batch)
        tuned = True
    if config is None:
        config = DEFAULT_CONFIG
    if combine_batch is not None and not tuned:
        config = config._replace(combine_batch=min(combine_batch, S // config.block_n))

    block_m, block_n = config.block_m, config.block_n
    sub_vec_num, sub_m = config.sub_vec_num, config.sub_m
    ring, num_stages = config.ring, config.num_stages

    ok, why = config_is_valid(config, S)
    # The two register-unrolling ceilings used to be bare asserts here; they are
    # now part of config_is_valid, so one check covers a hand-written config, an
    # autotuned one, and a --combine-batch override alike.
    assert ok, (f"invalid config {config.label()} at S={S}: {why}. CB/SUB_VEC_NUM above their "
                f"ceilings would read uninitialised rescale registers and silently corrupt the "
                f"output; see the module docstring item 5.")

    num_seq_blocks = S // block_m
    block_num = num_seq_blocks * Hq * B
    num_kv_blocks = S // block_n
    CB = config.combine_batch
    conbined_block_num = num_kv_blocks // CB

    block_num_per_core = block_num // NUM_CORES
    rem_block_num = block_num % NUM_CORES

    # ---- schedule (ref §5.4) ----
    sparse = bool(is_causal and balance_causal)
    if sparse:
        sched_key = (q.device, CB, num_seq_blocks, Hq, B, block_m, block_n)
        if sched_key not in _schedule_cache:
            _schedule_cache[sched_key] = _build_causal_schedule(num_seq_blocks, Hq, B, CB, num_kv_blocks, q.device,
                                                                block_m=block_m, block_n=block_n)
        task_map, core_ntasks, max_tasks = _schedule_cache[sched_key]
    else:
        # Dense path: the kernel derives everything arithmetically, but the
        # parameters must still be real tensors.
        task_map = core_ntasks = torch.zeros((1, ), dtype=torch.int32, device=q.device)
        max_tasks = 1

    out = torch.empty_like(q)

    # Keyed on every dimension of the ring, since the tuner varies all of them.
    cache_key = (q.device, CB, ring, block_m, block_n)
    if cache_key in _workspace_cache:
        workspace_s, workspace_p, workspace_pv = _workspace_cache[cache_key]
    else:
        workspace_s = torch.empty((NUM_CORES, ring, CB, block_m, block_n), dtype=torch.float16, device=q.device)
        workspace_p = torch.empty_like(workspace_s)
        workspace_pv = torch.empty((NUM_CORES, ring, CB, block_m, DIM), dtype=torch.float16, device=q.device)
        _workspace_cache[cache_key] = (workspace_s, workspace_p, workspace_pv)

    sm_scale = (1.0 / D)**0.5
    q_scaled = q if _q_prescaled else q * sm_scale

    grid = (NUM_CORES, )  # one program per AI core; one Cube + one Vector stream
    flash_attention_fwd_kernel_v7[grid](
        q_scaled,
        k,
        v,
        out,
        workspace_s,
        workspace_p,
        workspace_pv,
        task_map,
        core_ntasks,
        max_tasks,
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
        num_seq_blocks,
        Hq,
        Hq // Hkv,
        num_kv_blocks,
        conbined_block_num,
        block_num_per_core,
        rem_block_num,
        CB=CB,
        NUM_KV_BLOCKS=num_kv_blocks,
        IS_CAUSAL=is_causal,
        SPARSE=sparse,
        RING=ring,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        DIM=DIM,
        SUB_M=sub_m,
        SUB_VEC_NUM=sub_vec_num,
        num_stages=num_stages,
    )
    return out


# =============================================================================
#  CLI
# =============================================================================
def _reference(q, k, v, is_causal):
    """fp32 SDPA reference, head by head — a one-shot call at these shapes is ~1 GB."""
    if k.shape[1] != q.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    ref = torch.empty_like(q)
    B, H = q.shape[0], q.shape[1]
    for b in range(B):
        for h in range(H):
            ref[b, h] = torch.nn.functional.scaled_dot_product_attention(q[b, h].float(), k[b, h].float(),
                                                                         v[b, h].float(),
                                                                         is_causal=is_causal).to(q.dtype)
    return ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--S", type=int, default=2048)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--D", type=int, default=DIM)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--combine-batch", type=int, default=None,
                        help=f"KV blocks per task (arch22 nRatio), <= {MAX_CB}")
    parser.add_argument("--no-balance", action="store_true",
                        help="causal: contiguous dense split instead of the balanced sparse one")
    parser.add_argument("--bench", type=int, default=0, help="timed iterations (0 = correctness only)")
    parser.add_argument("--autotune", action="store_true",
                        help="search (BLOCK_M, BLOCK_N, CB, SUB_VEC_NUM, RING, num_stages) before running")
    parser.add_argument("--tune-strategy", choices=["coord", "grid"], default="coord",
                        help="coord: one coordinate-descent sweep (~8 compiles); grid: full cross product")
    parser.add_argument("--tune-iters", type=int, default=20,
                        help="launches per timing batch, per candidate (batched, one sync per batch)")
    parser.add_argument("--tune-warmup", type=int, default=5, help="warmup iterations per candidate")
    parser.add_argument("--tune-cache", type=str, default=None,
                        help="JSON file to read/write tuned configs, keyed by shape")
    parser.add_argument("--config", type=str, default=None,
                        help="pin a config instead of tuning: BLOCK_M,BLOCK_N,CB,SUB_VEC_NUM,RING,num_stages")
    parser.add_argument("--dump-mlir", nargs="?", const="", default=None,
                        help="Dump intermediate CommonIR to PATH and exit; no device needed.")
    parser.add_argument("--dump-ir", nargs="?", const="", default=None,
                        help="Dump HIVM IR (after CommonIR→HIVM lowering) to PATH and exit; no device needed.")
    parser.add_argument("--dump-linalg", nargs="?", const="", default=None,
                        help="Dump Linalg IR (full lowering through linalg) to PATH and exit; no device needed.")
    args = parser.parse_args()

    B, S, H, D = args.B, args.S, args.H, args.D
    balance = not args.no_balance

    pinned = None
    if args.config:
        fields = [int(x) for x in args.config.replace(" ", "").split(",")]
        assert len(fields) == 6, ("--config wants 6 comma-separated ints: "
                                  "BLOCK_M,BLOCK_N,CB,SUB_VEC_NUM,RING,num_stages")
        pinned = FAConfig(*fields)
        ok, why = config_is_valid(pinned, S)
        assert ok, f"--config {pinned.label()} invalid at S={S}: {why}"

    _dump_cfg = pinned or DEFAULT_CONFIG
    _cb = args.combine_batch if args.combine_batch is not None else _dump_cfg.combine_batch
    _nkv = S // _dump_cfg.block_n
    _sparse = bool(args.causal and balance)

    # ---- device-free IR dumps ----
    if args.dump_mlir is not None:
        dump_commonir(path=(args.dump_mlir or None), num_kv_blocks=_nkv, combine_batch=_cb, is_causal=args.causal,
                      sparse=_sparse, config=_dump_cfg)
        raise SystemExit(0)
    if args.dump_ir is not None:
        dump_hivm(path=(args.dump_ir or None), num_kv_blocks=_nkv, combine_batch=_cb, is_causal=args.causal,
                  sparse=_sparse, config=_dump_cfg)
        raise SystemExit(0)
    if args.dump_linalg is not None:
        dump_linalg(path=(args.dump_linalg or None), num_kv_blocks=_nkv, combine_batch=_cb, is_causal=args.causal,
                    sparse=_sparse, config=_dump_cfg)
        raise SystemExit(0)

    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    q = torch.randn((B, Q_H, S, D), dtype=torch.float16, device=device)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)

    cfg = pinned
    # Once a config is tuned or pinned its CB is already final; passing
    # --combine-batch on to the launcher as well would re-clamp it.
    cb_override = args.combine_batch
    if cfg is None and args.autotune:
        print(f"autotune: B={B} H={Q_H} S={S} D={D} causal={args.causal} "
              f"strategy={args.tune_strategy}")
        cfg = autotune_config(q, k, v, is_causal=args.causal, balance_causal=balance,
                              combine_batch=args.combine_batch, strategy=args.tune_strategy, warmup=args.tune_warmup,
                              iters=args.tune_iters, cache_file=args.tune_cache, check=not args.no_check)
        cb_override = None
    elif cfg is not None:
        cb_override = None
    if cfg is None:
        cfg = DEFAULT_CONFIG
    print(f"config: {cfg.label()}")

    out = flash_attention_fwd(q, k, v, cb_override, is_causal=args.causal, balance_causal=balance, config=cfg)

    if not args.no_check:
        ref = _reference(q, k, v, args.causal)
        nan_count = torch.isnan(out).sum().item()
        total = out.numel()
        print(f"[diag] out NaN: {nan_count}/{total}  out sample: {out[0,0,0,:4]}")
        print(f"[diag] ref sample: {ref[0,0,0,:4]}")
        if nan_count > 0:
            # Find first NaN location
            nan_mask = torch.isnan(out)
            idx = torch.nonzero(nan_mask.view(-1))[0].item()
            print(f"[diag] first NaN at flat index {idx}")
        # Detailed debug: per-batch/head NaN pattern
        out_cpu = out.cpu().float()
        ref_cpu = ref.cpu().float()
        nan_mask_cpu = torch.isnan(out_cpu)
        for b_i in range(min(B, 2)):
            for h_i in range(min(Q_H, 3)):
                sl = out_cpu[b_i, h_i]
                n = torch.isnan(sl).sum().item()
                first_nan_s = -1
                for si in range(S):
                    if torch.isnan(sl[si, 0]).item():
                        first_nan_s = si
                        break
                print(f"  [b={b_i},h={h_i}] NaN={n}/{sl.numel()} first_nan_seq={first_nan_s}")
        # Non-NaN accuracy
        non_nan = ~nan_mask_cpu
        if non_nan.any():
            diff = (out_cpu[non_nan] - ref_cpu[non_nan]).abs()
            print(f"  Non-NaN: count={non_nan.sum().item()} max_diff={diff.max():.4f} mean_diff={diff.mean():.4f}")
            # Sample values
            print(f"  out[0,0,0,:8]={out_cpu[0,0,0,:8].tolist()}")
            print(f"  ref[0,0,0,:8]={ref_cpu[0,0,0,:8].tolist()}")
            print(f"  out[0,0,3,:8]={out_cpu[0,0,3,:8].tolist()}")
            print(f"  ref[0,0,3,:8]={ref_cpu[0,0,3,:8].tolist()}")
            print(f"  out[0,0,4,:8]={out_cpu[0,0,4,:8].tolist()}")
            print(f"  ref[0,0,4,:8]={ref_cpu[0,0,4,:8].tolist()}")
        torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")

    if args.bench:
        import time
        sm_scale = (1.0 / D)**0.5
        q_pre = q * sm_scale  # hoist the pre-scale out of the timing loop
        for _ in range(3):
            flash_attention_fwd(q_pre, k, v, cb_override, is_causal=args.causal, _q_prescaled=True,
                                balance_causal=balance, config=cfg)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.bench):
            flash_attention_fwd(q_pre, k, v, cb_override, is_causal=args.causal, _q_prescaled=True,
                                balance_causal=balance, config=cfg)
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) / args.bench * 1e3
        flops = 4.0 * B * Q_H * S * S * D
        if args.causal:
            flops *= 0.5
        print(f"v7  B={B} H={Q_H} S={S} D={D} causal={args.causal} balance={balance} "
              f"{cfg.label()}: {ms:.4f} ms   {flops / (ms * 1e-3) / 1e12:.3f} TFLOP/s")
