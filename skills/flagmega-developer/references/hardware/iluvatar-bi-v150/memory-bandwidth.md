# BI-V150 memory and execution notes

graph expert and kernel expert use these **historical** measurements and
investigations (vendor confirmation 2026-10-08, last investigation 2026-10-09).
Verify current source/device behavior before applying recorded conclusions.

## 1. Workflow

1. Recorded one CUDA device = one die: 16 MPs, 1.5GHz, 32GB, 16MB L2,
   cc7.1, warpSize64; dual-die board reported by ixsmi. Per-die peak819.2GB/s.
2. Separate one-CTA/MP grid-synchronized admission from ordinary high-occupancy
   launches; inspect actual native width/resources before claiming bandwidth limits.
3. Compare coalesced narrow/wide loads or double-buffered SME under matched
   geometry/workload; preserve publication/ABI and verify separately from timing.
4. For hangs/wrong values, pin source/native/environment, check fresh XID, and use
   bounded independent-process repros. Distinguish confirmed repairs from refuted
   explanations; do not remove required synchronization as a symptom workaround.

## 2. Checklist

<a id="bandwidth"></a>

| Recorded path | GB/s | Peak fraction |
| --- | --- | --- |
| Stores, any width | 749 | 91% |
| SME double-buffered / drain each iteration | 730 / 690 | 89% / 84% |
| 32-bit / 64-bit load | 676 / 589 | 82% / 72% |
| 128-bit / 16-bit load | 373 / 574 | 46% / 70% |

Wide loads capped near1 load/MP-cycle versus scalar6.9; flat over32K–2M resident
threads, dependent latency122ns and scaling1–16 MPs supported the recorded issue
bottleneck. One-CTA/MP favors128-bit; high occupancy can reverse the ordering:

| CTAs, 1024 threads, 512MiB | 32-bit | 64-bit | 128-bit GB/s |
| --- | --- | --- | --- |
| 16 | 169 | 324 | 353 |
| 32 | 320 | 541 | 366 |
| 64 | 596 | 588 | 382 |
| 128 | 658 | 569 | 366 |
| 256 | 660 | 574 | 367 |
| 1024 | 664 | 591 | 383 |
| 2048 | 676 | 604 | 389 |

1. Backend getVectorSize's128-bit cap cited Trial74's16-CTA result and has no
   occupancy input. Recorded cooperative_launch_admission=False restricted grid
   barriers to available_sm_count. Ordinary launches above~4 CTAs/MP need width
   comparison, not a global default rollback. Diagnostic disable_load_vectorize=True
   maps to lower-to-LLVM disable-load-vectorize, leaving TTIR/TTGIR/stores/atomics/SME;
   a64-CTA/MP reduction measured342→455GB/s.
2. Coalesced grid-stride matters: per-thread contiguous ranges measured40–64GB/s;
   extra address streams did not help. Below16MB tests L2, saturation~128MiB;
   float4 copy453/triad405/D2D589 inherited read limits, stores stayed width-insensitive.
3. SME next-stage issue before sl_waitcnt gave730GB/s; four tiles/warp gave698.
   ABI: native ext_vector_type(4), abase={gptr_lo,gptr_hi,-1,positive stride multiple64},
   16-row×64-byte tile/warp, __ivcorex_sme_load_16x1b64 and __ivcorex_sl_waitcnt(0).
   CUDA int4 struct is incompatible; recorded ivcore11 lacked plain cp_async builtins.
4. Inspect actual sme_load/dword/dwordx2/dwordx4 instructions and spills. Timed
   atomic verification/single accumulators can measure serialization/latency;
   torch.add(a,a,out=c) is not distinct-buffer triad. Preserve exact covered bytes.
   Position-varying stores743.7/742.6 and incompressible bits735 ruled out elision/
   compression in the recorded749 result; redirected stdout alone cannot diagnose a hang.

<a id="known-issues"></a>

| Historical issue | Evidence / causal limit |
| --- | --- |
| Lazy loading | Vendor-confirmed2026-09-20: real token load/create_state/prepare/run_into hung3/3 with fresh XID24; eager passed3/3 ~10.2–10.3s after reset. Set CUDA_MODULE_LOADING=0 before first device touch on this target; later loader.py initialization cannot repair earlier CUDA initialization. XID22 was related. |
| Stale block_table | execute_decode refreshes width columns; captured replay bypasses initial validation. Historical clamp existed, but three corruption/replay injections completed: unproved cause of the hang, not a causal final fix. |
| Acquire visibility | Trials72–73 lost finite Qwen logits without ml_lsa_wbinv. Preserve release/acquire; “barrier tax”4.3–4.7ms was actually N8/K16 down projection4.46ms/layer. Larger tiles kept barriers and reduced full model~134→16ms. |
| Producer publication | Trials103–106 reproduced wrong payload in resident16-CTA exchange. Elected release after CTA barrier did not publish other warps' writes; per-thread membar.gl before rendezvous/elected arrival fixed the coherence defect, retaining acquire. Grid/grid-axis regression and500 bitwise replays passed. |
| Primitive scope | barrier.alu is CTA convergence without full memory fence; vote.all is warp scope. Asymmetric/skew probes completed, so it is not an arrival counter. Barrier0-only conversion cannot rescope LLVM intrinsics; base8288/8296 attribution refuted an earlier8-producer/0-consumer claim. Scope alone did not explain the hang. |
| Isolated WS tt.call | promoteOperands read enclosing shared/scratch/profile bases before partition inlining. __ILUVATAR_TLE__ captureIntoPartition added explicit captures and reread after append; default/partition compile passed~2.65s. The SME hang had tt.call0, so this fix was separate. |
| Multi-step SME pipe | Open at the recorded investigation; preserve the bounded reproducer/latest refutations below and verify current status. |

<a id="sme-pipe-reproducer"></a>

1. third_party/iluvatar/python/test/unit/tle/test_sme_pipe_device_execution.py:
   capacity2,4 producer/16 default warps,4 steps; only SME layout/encoding/input_stride
   changes. no-SME match=True, SME RC124; FLAGTREE_RUN_HANGING_SME_PIPE=1 opt-in.
   Prior compile-only loops/WS tests did not prove this device path.
2. Same-group4-step SME passed2.48s; cross-group wait/release without tile reads
   still hung. Correctly unrolled C++771 SASS lines/16 SME/8 req/8 wait passed,
   versus Triton908/16/8/8; the initial314-line4-SME replica was not equivalent.
   Restoring four missing same-block pre-arrival wbinv fences9→5→9 repaired
   coherence but retained the hang. Histogram/count agreement is not CFG equivalence.
3. Stock single-step was probabilistic (2 pass/4 hang, later4/3); removing publication
   rendezvous gave8/0 but multi-step SME still0/4 and0/6 versus no-SME2/0.
   All-lane arrival/drop-close/scratch/fixed-stage probes also hung; native constant-false
   guards once accidentally disabled all rendezvous, invalidating that experiment.
4. Latest mask hypothesis was REFUTED: save206 overwritten213/restore217 before
   req218 was real; all-lane polling removed double saves23→18 with req4 and paired
   restore, still0 pass/7 hang. Do not promote mask narrowing/lane0 polling to root
   cause. Vendor-emitted wait/req (IR waitcnt0), blocl/mov_v2s differences remained leads.

## 3. Index

1. [Graph expert](../../agents/graph-expert.md), [kernel expert](../../agents/kernel-expert.md).
2. [Causal validation](../../agents/reviewer.md), [profiling](../../kernel-trace-method.md).
3. Current backend getVectorSize/emitIluvatarSmeTileLoads and runtime admission;
   publication test: third_party/iluvatar/python/test/unit/tle/test_grid_barrier_publication.py.
4. Historical call sites: serving execute_decode/_paged_attention_state.py,
   paged_attention_partial decode.py.jinja/kernel_call_renderers.py,
   CallOpConversion::promoteOperands/captureIntoPartition; inspect actual versions.

## 4. Lessons

1. Faulting/missing-arrival CTAs can masquerade as protocol deadlocks; inspect
   actual XID, initialized metadata and publication before blaming the waiting counter.
2. Per-store instrumentation can mask races; compile-only/instrumented passes do
   not prove ordinary multi-step correctness. Confirmed fixes may leave original hangs open.
3. COREX synchronize can hold GIL, preventing reader/watchdog snapshots during wedges;
   use bounded separate processes. Historical resets do not authorize shared-device resets.
4. Single-run exclusions were unreliable. Preserve statistical effects, true repairs
   and latest refutations; reuse measured occupancy-specific conclusions, not stale universal defaults.
