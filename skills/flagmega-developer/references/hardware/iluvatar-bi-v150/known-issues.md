# Iluvatar BI-V150 (COREX) known issues

Hardware/driver-level issues specific to this target, distinct from ordinary
compiler or kernel bugs. Check this before spending agent time re-diagnosing
a hang, fault, or crash as a FlagMega logic defect.

## CUDA lazy module loading causes a real device fault, not a livelock

**Symptom.** A real decode call, or a megakernel using in-kernel grid
barriers (`tle.distributed_barrier`), hangs — `cudaDeviceSynchronize`
blocks forever, GPU utilization reads high (busy, not idle), no immediate
error. Historically characterized as a "probabilistic livelock" because it
does not reproduce on every attempt (see `.local/perf-iteration/
ITERATION.md`, Trials 18-26 and 34-49 in the Qwen3-1.7B tutorial for the
full investigation).

**Root cause.** CUDA's default lazy module loading does not reliably
resolve this backend's device code — vendor-confirmed 2026-09-20. A
faulting launch never reaches its subsequent barrier call, so every other
CTA in that barrier's group spins forever waiting for an arrival that never
comes. This is why exhaustive verification of the barrier protocol itself
(isolated CUDA C++ reproduction, LLVM IR comparison, SASS instruction-level
diff — all clean, no divergence) could never find anything wrong: the
barrier was never the bug, only the mechanism that turns an unrelated
module-loading fault into an apparent permanent hang.

**Reproduction.** With `CUDA_MODULE_LOADING` unset (default lazy), a
minimal `load()` -> `create_state()` -> `prepare()` -> `run_into()` call
with a real token, normally ~10s, hung on 3/3 tested GPUs (60-90s timeout),
each producing a fresh `dmesg -T | grep XID` entry (`XID: 24 mmu page
fault!`) with a timestamp exactly matching the test run. Resetting the same
GPUs (`ixsmi -i <ids> -r`) and re-running with `CUDA_MODULE_LOADING=0`
restored: 3/3 passed cleanly, ~10.2-10.3s, immediately after.

**Workaround (required, not optional).** Force eager module loading before
CUDA initializes in the process:

```sh
export CUDA_MODULE_LOADING=0
```

This must be set before the process's first CUDA driver call — the driver
reads it once. `python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/
.local/env.sh` sets it for shell-driven workflows. The runtime itself also
sets it defensively in `python/triton/flagmega/runtime/loader.py`'s `load()`,
scoped to `target == "iluvatar-bi-v150"`, right before the function's first
device touch — this protects any caller that only reaches COREX through
this loader, but cannot help if the calling process already initialized a
CUDA device beforehand (e.g. some other library touched `torch.cuda` first).
If you are embedding this runtime in a larger process, set the environment
variable yourself, as early as possible, rather than relying on the loader.

**Do not** re-attribute a hang matching this symptom to the barrier
protocol, the megakernel design, or `distributed_barrier`'s implementation
without first checking `dmesg -T | grep XID` for a fresh `XID: 24` (or
`XID: 22`, a related illegal-instruction variant tied to the same
lazy-loading/`noinline` interaction) at the failure's timestamp, and
confirming the failure reproduces with `CUDA_MODULE_LOADING` explicitly
unset before ruling this out.

## Stale `block_table` entries can cause an out-of-bounds `kv_cache` read

**Symptom.** Same class as above (a faulting CTA never reaches its barrier,
the rest of that group hangs) but through a different mechanism, specific
to the vLLM-integrated serving path (`vllm_adapter.py`), not the standalone
`ArtifactBackend`/chat_cli path (whose `block_table` is a constant identity
mapping and cannot go stale).

**Root cause.** `vllm_adapter.py`'s `execute_decode` only refreshes
`state.block_table`'s first `width` columns each call
(`self.state.block_table[:, :width].copy_(metadata.block_table)`); any
column beyond `width` retains whatever a previous, unrelated call wrote.
`model.prepare()` validates `block_table` bounds once, at initial CUDA
graph capture (`_validate_page_table`,
`python/triton/flagmega/ir/ops/nn/_paged_attention_state.py`), but every
decode step after the first calls `self.graph.replay()` directly, which by
design bypasses all Python-level validation. If a corrupted or stale entry
outside `[0, num_blocks)` is ever read through this path, the corresponding
`kv_cache` address computation can go genuinely out of bounds.

**Fix (already applied).** `paged_attention_partial`'s kernel
(`python/triton/flagmega/codegen/triton/kernels/paged_attention_partial/
decode.py.jinja`) clamps the loaded physical block index into
`[0, num_blocks)` before using it in any address computation — the only
correct place to guard this for a CUDA-graph-based serving path, since
graph replay cannot run Python-level checks. `num_blocks` is threaded in as
a new `tl.constexpr` from `kernel_call_renderers.py`'s existing
`cache_shape[0]`.

**Status.** Architecturally sound (a narrowing clamp cannot introduce a new
hazard) but not confirmed as the cause of any specific observed hang: three
direct fault-injection attempts against this exact mechanism (constructing
a valid CUDA graph, then corrupting `block_table` post-capture, then
replaying) all completed without a fault. Keep this fix in place as
defense-in-depth; do not treat it as a substitute for checking the
lazy-loading issue above first.

## 3. Cross-CTA visibility requires acquire-side invalidation

Removing the acquire-side `ml_lsa_wbinv` from the grid barrier produced
nonfinite logits in the Qwen3-1.7B megakernel (perf-iteration Trials 72–73).
Keep the release arrival / acquire poll memory-ordering contract. Vendor
clang and Triton's LLVM backend do not emit identical instructions for the
same source-level atomic ordering.

**Corrected performance attribution (2026-09-24).** The earlier claim of
4.3–4.7 ms per barrier was wrong. Trial 74's C++ microbenchmarks and careful
consumer-call removal showed cheap barriers; Trial 77 finally measured the
previously omitted MLP down projection at 4.46 ms per layer. Its tiny N=8,
K=16 tiles serialized thousands of reductions. Increasing those tiles while
keeping all barriers cut full-model time from about 134 ms to 16 ms. Do not
use the historical “barrier tax” as evidence for removing coherence fences
or redesigning the phase schedule. Measure the actual BF16 call ABI and all
operators, including the fused down-projection/residual/norm-statistics op.


## 4. Grid barrier must publish every producer warp before arrival

Confirmed 2026-09-25 by Qwen3 tutorial Trial 103–106. The old TLE grid lowering
used a CTA barrier followed by one thread's release atomic. That atomic did
not publish the other warps' pending global writes. A resident 16-CTA integer
exchange reproduced incorrect data without any model or runtime integration;
replaying the identical full-model state eventually produced NaNs as well.

`GridBarrierLowering` now emits device `membar.gl` on every thread before the
CTA rendezvous and elected arrival. Keep this fence and the acquire poll.
The all-thread fence is essential for payload visibility, distinct from the
counter protocol. Both grid and grid-axis groups are covered by
`third_party/iluvatar/python/test/unit/tle/test_grid_barrier_publication.py`.

Instrumenting every output store can hide the race by changing timing. Do not
infer correctness from one successful decode, one standalone run, or instrumented
source alone. Trial 106 passed 500 restores/replays of the original failing
model snapshot bitwise and the complete independent-token serving matrix.


## 5. `barrier.alu` is CTA-scope, not warp-scope; `vote.all` is the warp primitive

Measured on hardware 2026-10-09 (ivcore11, GPU 7). Two source comments in the
tree disagreed about `llvm.bi.sl.barrier.alu` / `__syncthreads_alu`:
`third_party/iluvatar/lib/Analysis/Membar.cpp` called it "light CTA
thread/warp sync only" and `.../TritonILUVATARGPUToLLVM/BarrierOpToLLVM.cpp`
called it a "full-CTA rendezvous". The second one is correct.

`.local/bugreport/cpp_reference/barrier_alu_scope.cu`: warp 0 spins on a
shared flag that warp 1 sets only *after* its own `__syncthreads_alu()`.

```
scope_probe    err=no error   out[0]=-1   (-1 => CTA-wide, 1 => warp-local)
asym_probe     err=no error   out[0]=1
```

`out[0]=-1` means warp 1 could **not** pass the instruction alone; it only
got through after warp 0's bounded spin expired and warp 0 exited. So
`barrier.alu` is lighter only in MEMORY semantics (no memory-system fence
versus `sl_barrier`); a single warp cannot clear it by itself.

**But it releases on CONVERGENCE, not on an arrival count.** `asym_probe`
(warp 0 executes it twice, warp 1 once) completes, and
`alu_loop.cu` shows two warp groups each running their own loop with a
`barrier.alu` inside clear it fine even when their iteration counts differ
(`skew=1`, `skew=2` both pass). Do NOT read "CTA scope" as "mismatched
iteration counts deadlock" — they do not. What does block is a warp group
reaching the instruction while the others are parked somewhere that they
cannot leave.

`.local/bugreport/cpp_reference/vote_all_scope.cu`: `__ivcorex_vote_all`
(`llvm.bi.vote.all`) is the warp-scope convergent primitive, and it does
order the lanes' prior shared-memory writes against lane 0's read.

```
vote_scope     err=no error   out=1      (1 => WARP-scope)
vote_converges err=no error   out=2080   (= 64*65/2, lane 0 saw all 64 lanes)
```

**How to apply.** A rendezvous that only needs the lanes of one warp to
converge — for example aggregating one completion per warp — must use
`vote.all`, not `barrier.alu`. Emitting `barrier.alu` at a program point that
only one `ttg.warp_specialize` group reaches would block that group on warps
which are concurrently executing a different region.

**This entry is a primitive-selection rule, NOT a hang root cause.** Do not
cite it as one without a hang-before/pass-after pair on hardware. Evidence
accumulated 2026-10-09 against using it that way:

- A device A/B over `barrier.alu` versus `vote.all` on a looping SME pipe
  hung identically (both RC=124), so the primitive choice did not change the
  outcome.
- The `barrier.alu` occurrences in that kernel turned out to be SYMMETRIC.
  Instrumenting `annotateSmeBarrierMarkers` printed
  `defaultMarkers=1 partitionMarkers=1`, and attributing each occurrence by
  the shared-memory base it touches (8288/8296 = full barrier) showed four
  after the producer's commit arrive and four after the consumer's full-barrier
  wait convergence — a matched set. An earlier "8 producer / 0 consumer"
  reading came from guessing region ownership off control-flow labels and was
  wrong; use the barrier base address, not block labels, to attribute these.

Also note `ConvertWarpSpecializeToLLVM.cpp` rescopes rendezvous by walking
for `NVVM::Barrier0Op` only; it has zero references to `CallIntrinsicOp`.
Any pass that emits `barrier.alu`, `sch_barrier` or `pipebar` as an LLVM
intrinsic into a WS region therefore keeps CTA scope through lowering.


## 6. A looping SME async-copy pipe hangs on device (open)

Confirmed 2026-10-09 by a single-variable bisect on ivcore11. Root cause is
NOT yet identified; this entry records the reproducer and the ruled-out
hypotheses so the next attempt does not redo them.

`third_party/iluvatar/python/test/unit/tle/test_sme_pipe_device_execution.py`
holds both halves. Lifecycle shape, capacity (2), warp split (4 partition +
16 default), consumer, and step count (4) are identical between them; the
only difference is the SME triple (`iluvatar_sme_shared_layout`,
`IluvatarSmeBlockEncoding` via `set_layout`, `input_stride`):

```
test_nosme_pipe_runs   -> COMPLETED match=True
test_sme_pipe_runs     -> hangs (RC=124), opt-in via
                          FLAGTREE_RUN_HANGING_SME_PIPE=1
```

**Why this was never caught.** Every multi-step SME + pipe +
`warp_specialize` test in `test_ws_explicit_copy.py` is compile-only
(`test_ws_explicit_copy_2d_sme_codegen`, `..._2w_codegen` call
`triton.compile` and never launch). The tests that do launch either use a
single `wait(0)` or, like `test_ws_explicit_copy_2d_tile_is_deterministic`,
use a plain `tle.gpu.alloc(scope=smem)` with no SME at all. Device execution
of a looping SME pipe had no coverage.

**Ruled out** (each verified, not assumed):

- Pipe lifecycle shape: removing `close`, removing `async_wait_group`, and
  making the wait adjacent to the commit all still hang.
- `wait_drained` and `tl.broadcast_to` index forms: adding both still hangs.
- Rendezvous primitive: `barrier.alu` versus `vote.all` device A/B hung
  identically.
- `barrier.alu` asymmetry: the markers are symmetric (see issue 5).
- Participant counts: `%pipe`(8272/8280) `init=256` matches its waits'
  expected 256, and `%pipe_1`(8288/8296) `init=1024` matches 1024. The
  `+= 64` arrivals are per-warp increments under a lane-0 guard, so a 4-warp
  producer contributes 256 and a 16-warp consumer 1024 — both conserved.
- `lookupNumWarps` through the WS boundary: instrumentation showed it already
  returns the correct per-region widths (writer ops 4, reader ops 16).

**Fixed along the way but NOT the cause** (the hang survives it):
`deferSmePublicationToLocalLoad` tagged a `LocalLoadOp` with
`sme_deferred_publication` and made the consumer-side wait skip emitting its
publication rendezvous, but nothing in the tree ever read that attribute
back, so the deferred publication was silently dropped. The deferral was
removed and the publication is now emitted directly.

**Debugging constraints on this target.** Device-side tracing does not work
during a hang: `synchronize()` never returns, COREX will not co-schedule a
reader kernel with a running one, and a host watchdog thread starves because
COREX's synchronize holds the GIL. Pinned-host trace buffers also produced no
snapshots. Prefer structural bisection in separate processes with their own
timeouts. Each hang wedges the card until the host is reset, so probe with
compile-only runs where possible and rotate cards.


## 7. A `tt.call` inside a warp-specialize partition failed to lower (fixed)

Fixed 2026-10-09. `CallOpConversion::promoteOperands`
(`third_party/iluvatar/lib/Conversion/TritonGPUToLLVM/ControlFlowOpToLLVM.cpp`)
appends three implicit trailing operands to every `tt.call` -- the shared,
global-scratch and profile base pointers. All three read the enclosing
KERNEL's function arguments (`getStackPointer`,
`tle::getGlobalScratchBase`, `getProfileScratchPtr`).

A `ttg.warp_specialize` partition region is `IsolatedFromAbove`, and this
conversion runs BEFORE `add_warp_specialize_to_llvm` inlines those regions
(`backend/compiler.py`: `add_to_llvmir` then `add_warp_specialize_to_llvm`).
Referencing a kernel argument from inside a partition therefore produced:

```
error: 'llvm.call' op using value defined outside the region
note: required by region isolation constraints
RuntimeError: PassManager::run failed
```

A `noinline` SME body is the natural trigger, because its lowering needs a
real call. The default region is NOT isolated, so the identical body compiles
there -- which is why `test_ws_sme_direct_producer_preserves_payload` passes
while the same body in a partition failed.

**Fix.** Under `__ILUVATAR_TLE__`, `captureIntoPartition` threads each base
in as a `ttg.warp_specialize` explicit capture so the partition reads it from
its own region argument. It re-reads `getExplicitCaptures()` after appending
instead of deriving the index from the pre-insert operand count; the latter
desynchronizes when several captures are appended in sequence (the Trial 82
defect).

Fail-before/pass-after is in
`third_party/iluvatar/python/test/unit/tle/test_sme_call_in_ws_partition.py`
(single variable: which warp group holds the body; body, storage and
num_warps identical). Before: default passed, partition failed. After: both
pass, 2.65 s. Compile-time only, so it cannot wedge a card.

**This is NOT the cause of issue 6.** The looping SME pipe fully inlines its
producer and consumer -- its TTGIR `tt.call` count is 0 -- so it never
reaches this code path, and it still hangs after this fix.

**Further ruled out for issue 6 (2026-10-09, continued).** A full instruction
histogram diff between the passing no-SME kernel and the hanging SME kernel
(identical lifecycle, capacity, warp split, consumer, steps) leaves only:

```
            nosme   sme
load.kop.v1bf16  33     0     (ordinary global load, replaced by SME)
sme.load.*        0    17
barrier.alu       0     9
vote.all          0     5
membar.cta       10     6
```

- The `barrier.alu` difference is not the cause: disabling the pipe's
  `markSmePayload` calls drops it to 0, making the synchronization structure
  byte-identical to the passing kernel (membar.cta 10, barrier.cta.sync 11,
  barrier.alu 0), and it STILL hangs.
- The four missing `membar.cta` are deliberate: the `sme_async_payload`
  arrive path in `BarrierOpToLLVM.cpp` omits the CTA fence on the grounds
  that the SME publication marker already supplies that ordering. Since
  removing the marker entirely also still hangs, fence count is not the
  discriminator either.
- SME + warp-specialize with the write and read in the SAME warp group, over
  a 4-step loop, PASSES (2.48 s). Only the cross-group pipe form hangs.
- Consumer-side reads are not required for the hang: a consumer that only
  does `wait`/`release`/`drain` and never touches the SME tile still hangs,
  so the producer side is where progress stops.
- The hang is not a visibility problem. The consumer spins on the barrier
  counter that the producer updates with `atomicrmw`; stale SME payload would
  corrupt data, not block the counter. (Credit to the user for catching this
  reasoning error.)
- Issue 7's `tt.call`-in-partition compile defect is unrelated: this kernel
  inlines producer and consumer, so its TTGIR `tt.call` count is 0.

**Issue 6 continued — two findings from a SASS-level C++ replica (2026-10-09).**

*Replicate the structure properly before comparing.* An instruction-type
histogram is not enough. A first C++ replica looked structurally equivalent
but its loop was not unrolled (314 SASS lines, 4 `sme_load`, 2
`barrier_req/wait`) while Triton fully unrolls (908 lines, 16 `sme_load`,
8 `barrier_req/wait`). With `#pragma unroll 4` the replica matches
(771 lines, 16 `sme_load`, 8/8 `barrier_req/wait`) -- and still PASSES, so
the hang is not in that structure.

*Fixed: SME commit lost its `wbinv`.* The replica's passing invariant is one
`ml_lsa_wbinv` immediately before each release atomic, same block. Measured
in Triton:

```
nosme (passes): wbinv=9, every arrive preceded at distance 1
                add@167<-wbinv@166, add@209<-208, add@250<-249, ...
sme   (hangs):  wbinv=5, the first FOUR arrives had NO preceding wbinv
```

`BarrierOpToLLVM.cpp`'s `sme_async_payload` arrive path omitted the CTA
fence. Restoring it is not enough by itself: emitted in the predecessor
block, a conditional branch sits between it and the atomic and the vendor
backend produces no `wbinv` on that path. It must go in the SAME block as
the release atomic. After moving it there, `wbinv=9` with every arrive
preceded at distance 1, matching the passing build. Since BI-V150 has no
inter-SM L1 coherence (issue 3), this is a real correctness fix -- but the
pipe STILL hangs, so it is not the hang's cause.

*Remaining lead: `sl_blocl_b64` + `ml_mov_v2s_b32`.* After aligning `wbinv`,
the standout difference is:

```
nosme (passes): blocl=10  mov_v2s=14  no blocl followed by mov_v2s
sme   (hangs):  blocl=14  mov_v2s=52  blocl@153/188/236 each followed by 4-6
```

`sl_blocl_b64` is the hardware-loop instruction that writes `x0`;
`ml_mov_v2s_b32 ..., x0` extracts from it. This is the same signature the
2026-09-20 grid-barrier investigation recorded (see memory
`iluvatar-grid-barrier-deadlock`): hanging kernels carried ~73 scattered
`ml_mov_v2s_b32` extractions of `x0`, passing variants ~3. `blocl@153` sits
right at the SME region (`sme_load` at 160-163). Treat this as the next
lead, and note it points at vendor codegen for this CFG shape rather than at
FlagTree's own lowering.

**Issue 6 continued — the hang is PROBABILISTIC; use statistics, not single
runs (2026-10-09).**

The decisive methodological finding: a single `pytest` run is not a verdict.
Measured on the minimal single-step SME pipe with the stock compiler:

```
6 parallel runs (one per card): 2 passed / 4 hung   -> ~67% hang rate
```

This matches the rate recorded in memory `iluvatar-grid-barrier-deadlock`
("roughly 3 of 4 identical attempts hang"). Several earlier "refutations" in
this investigation used one run each and are therefore unreliable.

Run probes in PARALLEL, one per GPU — 10 runs drop from ~10 min to ~1 min,
which is what makes a statistical verdict affordable:

```
for g in 2 3 4 5 6 7; do (CUDA_VISIBLE_DEVICES=$g timeout 70 pytest ... ; \
  echo "rc=$?" >> /tmp/log) & done; wait
```

**Statistically supported result.** Suppressing the SME publication
rendezvous (`barrier.alu`, emitted by `rewriteSmePublicationBarriers`)
removes the probabilistic hang of the SINGLE-STEP case:

```
single step, rendezvous on : 2 passed / 4 hung   (~67%)
single step, rendezvous off: 8 passed / 0 hung   (0%)
```

At a 67% per-run hang rate, 8 consecutive passes has probability ~0.03%, so
this is a real effect rather than luck.

**A second, DETERMINISTIC factor exists for multi-step pipes.** With the
rendezvous already suppressed:

```
2 steps, SME, no rendezvous      : 0 passed / 4 hung  (100%)
4 steps, SME, no rendezvous      : 0 passed / 6 hung  (100%)
2 steps, NO SME, no rendezvous   : 2 passed / 0 hung
```

So any SME async copy makes a >=2-step pipe hang deterministically, while the
same pipe without SME completes. SASS diff between those two (K=2 passing vs
K=0 hanging) adds only: `sme_load` +4, `savetmsk` +4, `blocl` +1 and
`sl_wait lmcnt(0) g2scnt(0)` +1 — **no new barrier at all**.

**Also ruled out this round** (each with parallel repetitions, not one run):

- Lane-elected arrive: forcing the SME arrive onto the all-lane scalar path
  still hangs 0/4.
- `close` tag's single-lane region: dropping `close` entirely still hangs 0/2.
- SME destination: copying into a standalone scratch allocation instead of the
  pipe slot still hangs 0/2.
- Stage index form: fixed `acquire(0),acquire(0)` and distinct
  `acquire(0),acquire(1)` both hang 0/3.
- `sl_wait g2scnt(0)` is NOT emitted by FlagTree (IR `waitcnt` count is 0 for
  the hanging kernel); like `sl_barrier_req`, the vendor backend inserts it.

**Data-integrity warning.** A probe that gated the rendezvous emission on a
condition which was always false silently disabled it for ALL runs
(`barrier_req` was 0 where the stock build has 2). Always confirm the probe
changed the SASS the way you intended before reading any run result.

**Issue 6 — the deterministic multi-step factor, located (2026-10-09).**

Sample size matters: 2 repetitions cannot separate a 67% hang rate from
100%. With 7 parallel runs each:

```
1 step  (SME, stock compiler): 4 passed / 3 hung   -> ~43%, probabilistic
2 steps (SME on step 1 only) : 0 passed / 7 hung   -> 100%, deterministic
```

The deterministic case's SASS shows a thread-mask save being overwritten
inside the acquire spin loop. Second-step commit (`sl_barrier_req`@218, no
SME on this step):

```
206: sl_and_savetmsk_b64 s[22:23], s[4:5], tmsk   <- save mask
207: sl_cbr_tmskaz ...                             <- exit if mask empty
208: ml_slb_load_b32x1 ...                         <- acquire polls empty barrier
212: sl_jump 4294967184                            <- spin backedge
213: sl_and_savetmsk_b64 s[22:23], s[4:5], tmsk   <- OVERWRITES s[22:23]
216: ml_slb_add_u32 ...                            <- arrive
217: sl_or_b64 {tmsk, wcr}, tmsk, s[22:23]        <- restores the 213 value
218: sl_barrier_req
```

The mask saved at 206 (before narrowing for the spin) is lost: 217 restores
what 213 saved, and 213 ran with the spin's already-narrowed mask. So the
CTA-wide `barrier_req` at 218 executes with an incomplete mask.

Contrast the first step (`req`@172, with SME): save at 167, restore at 171,
no intervening overwrite — correctly paired.

Why two steps are needed: step 2's `acquire` genuinely spins (it must wait
for the consumer's release), and the `savetmsk` inside that spin clobbers the
one taken outside it. Step 1's `acquire` finds the slot already empty, never
enters the spin, and so never clobbers.

Note this is NOT the earlier `s[6:7]` hypothesis that was withdrawn: there,
the `sl_cbr_tmskaz` was a backward branch and the save was consumed on
another path. Here 207's branch and 213's overwrite lie on the same
straight-line path, and 217 demonstrably restores the overwritten value.

**Issue 6 — the mask-overwrite hypothesis is REFUTED (2026-10-09).**

The `savetmsk` overwrite documented above is real but is NOT the cause.
Forcing all lanes to poll (replacing `isLeader` with constant true in
`lowerWaitBarrier`, which is what narrows the mask) provably removed the
double save:

```
baseline : savetmsk=23  req=4  (req@218 had savetmsk@206 AND @213)
all-lane : savetmsk=18  req=4  (req@194 has a single savetmsk@189,
                                correctly paired with or_b64@193)
```

The probe demonstrably changed the SASS, and the hang was unaffected:

```
2 steps, all-lane poll: 0 passed / 7 hung   (same as the 0/7 baseline)
```

So neither the mask narrowing nor its clobbering explains the deterministic
multi-step hang. The lane-0 poll optimization is not at fault either.
