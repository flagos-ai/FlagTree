# Native MLA numerical and representation contract

Current acceptance (2026-10-17): complete 45-layer TP4, E288/top8/H4096,
cache4096 and initial token11 pass the official vLLM 0.31.0 greedy8 twice
with independent reset. Both requests return
`[3847,284,1373,8282,4218,3755,947,906]`. All four rank kernels have zero
spill. This is official-prefix acceptance; strict CPU defaults are unchanged.
Earlier failing trials below explain the required contracts and regressions.

The explicit `vllm-0.31-tp4` profile targets the pinned official
`FLASHINFER_MLA_SPARSE_SM90` BF16-cache path. Its decode computes
`Q -> BF16(Q Wk) -> latent attention -> BF16(context Wv)`.
Expanding K/V before attention changes the BF16 rounding boundaries.
The official first two forwards are captured read-only in native trial 215;
trial 266 reproduces the actual layer-3 context exactly on all four ranks.

`nn.mla_projection` expresses one batched per-head projection without
expanding a graph into one MatMul per head. The interleaved checkpoint weight
has shape `[heads * (qk_dim + value_dim), latent_dim]`.

| Operand/result | Contract |
| --- | --- |
| `value` | BF16 `[tokens, heads, dim]`; vectors contain consecutive components of the final axis. `absorb_query` consumes `qk_dim`; `expand_value` consumes `latent_dim`. |
| `weight` | Dense scalar BF16 or FP8 E4M3; select each head's K/V rows in the consumer loads. No runtime Slice or weight Pack is needed. |
| `scale` | None for BF16; FP32 block scales for FP8. Dequantize to BF16 before the projection dot. |
| `direction` | `absorb_query` computes Q times the K rows; `expand_value` computes latent context times the transposed V rows. |
| Result | BF16 `[tokens, heads, output_dim]`, optionally packed on the final axis by explicit `output_lanes`; the projection rounds once before packing. |
| Effects/constants | Pure, no aliases or state writes; constant evaluation is legal only when every input is constant. |
| Distribution | Initially plain or one fully broadcast mesh for all tensor operands, with no partials. Head splitting requires an explicit corresponding weight/indexing contract. |
| Physical ABI | Scalar strides and offsets come from verified descriptors, including lane components. No additional scratch or synchronization inside the projection. |
| Device coverage | Initial measured implementation: NVIDIA SM90, single-token chunks, BF16 Tensor Core accumulation into FP32. Other targets/shapes are rejected explicitly. CPU evaluation is diagnostic for hardware association. |

Latent cache storage reuses the existing typed paged-state ABI with one KV
head and `head_dim=latent_dim`. Both K/V fields contain the same normalized
latent. Existing paired cache update owns the publication/sequence advance;
attention remains a separate reader. This keeps state ordering visible to
bufferization and synchronization, rather than hiding a grid barrier inside
a projection kernel. The default nncase importer/state layout stays intact.
Exclusive (E) ownership limits the buffer access domain, not the participants
of an internal grid barrier. Calls declaring `internal_grid_barriers > 0`
retain every CTA within each selected chip; chip/node exclusivity stays in the
caller predicate. Writer masks and active extents still enforce E data access.
Trials 300–302 cover the deadlock before the fix and a bounded 128-CTA
cache-update/latent-attention regression after the fix.

`DsaPagedAttention(probability_rounding="online_bf16")` processes 64 top-k
slots per tile, maintains FP32 running max/denominator, casts unnormalized
exponentials to BF16 before PV, rescales the FP32 accumulator between tiles,
and normalizes last. The default `normalized` contract and its canonical hash
stay intact. Trial 269 fails before support; trial 270 passes the new and old
device tests. Trial 272 checks actual H512 operands: three ranks exact, one
BF16 element differs on the fourth rank; no complete-model success is claimed.

Acceptance requires typed emission/reload, schema rejection cases, scalar and
packed evaluator ordering, normal candidate/lowering/runtime coverage,
compute-body Pack=0 at all existing gates, and full45/TP4 independent official
greedy8 twice with reset. Prefix05 failed at the second token; prefix06
reaches the final MLA cache advance but deadlocks due to the participation
contract above. The corrected full-model prefix07 completes the first two forwards with zero
spill: token 3847 passes, token 646 fails against official 284. Standalone
operator agreement does not establish complete-model acceptance.


The native FP8 precursor GEMMs select `scaled_accumulation="fma"` in their
physical microkernel schedule. Each raw FP8 MMA tile is promoted by
`scale = fp32(activation_scale * weight_scale)` followed by
`accumulator = fma(raw_tile, scale, accumulator)`. The default compensated
schedule remains available and unchanged. The FMA selection requires the
SM90 quantized MMA path; dequantized BF16 GEMM retains its running MMA
accumulator. No new semantic op or default CPU contract is introduced.

The pinned vLLM small-batch path is
`FlashInferFp8DeepGEMMDynamicBlockScaledKernel` with FlashInfer swapAB for
M < 32. Its activation conversion is internal; the large-batch DeepGEMM
quantizer is a different reference. Trial 329 independently reconstructs
all three dense MLPs at the first two positions exactly. Trial 330 fixes
those operands and observes exact FMA promotion on all 48 local GEMMs;
other promotion schedules retain 20 mismatches. The checked-in two-block
cancellation regression fails before the selected schedule is implemented
and passes afterwards for both linear and GLU. Trial 334 removes the two
MLP2 output differences while preserving readonly file identities.

The fused Q/KV_A cuBLAS projection uses two accumulation partitions for
both Q_A and KV_A. Q_B keeps one running accumulator: trial 319 verifies
both positions and all Q_B rank shards, and splitting Q_B introduces
errors. Trial 335 then matches the actual MLA input, Q and absorbed Q
exactly, with only one latent BF16 difference on rank 3. Final acceptance
continues to require the independent complete-model official prefix.

The selected SM90 `decode_schedule="bf16_mma"` processes 16 query heads and
64 token slots per tile. QK retains one FP32 BF16-MMA accumulator across all
K16 instructions; a SIMT tree sum has a different rounding order. Online
maxima stay in raw-score units, exponentials use `scale * log2(e)`, and
unnormalized probabilities round to BF16 before the continuous PV MMA.
The output accumulator is rescaled before each PV tile and normalized with
the native approximate reciprocal. Empty sparse tiles are the identity.
The schedule requires one query token, one KV head, contiguous BF16 scalar
dimensions, complete local domains and a head dimension divisible by 16,
at most 512. It is rejected outside NVIDIA SM90. Default SIMT and normalized
probability contracts remain unchanged. Trials 347/349 provide the checked-in
fail-before/pass-after cancellation case; complete-model acceptance remained
pending at that stage; the final successful run is recorded above.

PV consumes one K16 fragment at a time while retaining its FP32 output
accumulator. Materializing the entire 64x512 BF16 value operand and its
pointer vectors raised the standalone register count to 227 and caused
216 spill bytes across the complete model's call chain (trial 354).
K16 streaming reduces the standalone count to 139 without changing MMA
association, head grouping, output buffering or the probability contract.
Testing the callee alone is insufficient: the checked-in live-caller test
preserves the output's previous values across its generated device call.
The archived whole-tile kernel spills 160 bytes in that small reproducer
(trial 375); the streaming kernel passes (374). Run it with:

```sh
pytest -q python/test/flagmega/codegen/triton/kernels/dsa_paged_attention/test_mma_online_schedule.py -k live_caller
```

Full-model zero-spill JIT and official greedy acceptance remain separate
required gates; trial 354 was rejected before publication or inference.


Sparse index traversal is an explicit numerical contract:
`index_traversal="slots"|"compact"`, defaulting to `slots`. The default is
omitted from canonical attributes to retain existing hashes. `compact`
stably filters indices visible at the current position before forming
64-entry online tiles; duplicates retain their original order. This can
change results because each tile rounds its unnormalized probabilities to
BF16. CPU interpretation and typed emission/reload preserve the attribute.
The initial device implementation requires the selected SM90 BF16-MMA path;
unsupported schedules must fail validation.

The pinned native indexer places complete pool-4 groups at the start and
its incomplete tail at offset 2048. The previous slot traversal therefore
splits lengths 5–7 over separate online tiles. Diagnostic 472 carries the
first three dense layers' states exactly through all eight positions and
finds exact MLA input/Q/absorbed-Q at every position; only lengths 5–7 have
incorrect latent/context values. Native wrapper calibration 455 matches
all 32 actual contexts independently.

Padding invariance alone is insufficient: the initial compact candidate
passed that test but failed actual native replay (478). Minimal index-only
probe 485 proves the rank computation correct and the shared scatter wrong.
`TleOptimizeLocalPointerStores` incorrectly replaced every full-shaped
pointer store with a sequential full-buffer write without proving the
explicit index map. The TLE-owned repair requires per-axis identity
coordinates; dynamic, reversed and transposed writes keep their addressing.
Checked-in MLIR and independent TLE device regressions fail before the fix
(486/489); six known-mean attention cases also fail (487). The rebuilt compiler passes 9 independent TLE device neighbors and all
108 TLE MLIR tests. Attention/NormStats/schema/profile neighbors pass 71
tests, plus duplicate/future-index and unsupported-schedule checks. Actual
replay 495 matches all 32 native latent outputs. The original first-three-layer
carry plus MLA boundary diagnostic also matches all seven exported boundaries
at all eight positions and four ranks (500/501), with zero spill. The full 45-layer JIT then passes (482: 252 registers, 65,536 shared bytes,
128-byte call stack, zero spill; 288.308 seconds JIT). Normal publication 483
retains all 64 readonly inodes in 65.268 seconds. Its source hash is
`2ad7d68d4462d0bd3b3235779f700c7603a5c5eccd33b71dad89d8344200ea12`.
The complete compact prefix12 run later fails at token five; the following
Q/KV statistics repair closes that remaining path. The rejected initial
compact candidate was never published.

Final acceptance still requires the complete model's independent official
8-token greedy sequence twice with reset. Held-history boundary diagnostics
are only localization evidence and cannot satisfy this gate.

The complete compact prefix12 run failed at token five. Its earliest
fifth-position raw-QKV discrepancy moves from layer 4 to layer 21; this proves
local progress without establishing complete-model acceptance. Independent
MLA19 replay then exposes a fourth-position Q mismatch. Actual Q_A, KV_A and
Q_B projections all match their native operands (526: 48 cases). Q_A's RMS
statistics are the first differing calculation: a 128-element loop changes
FP32 association, yielding three different BF16 normalized values and 1,353
different expanded Q values.

The explicit native profile now chooses a full 2048-element Q/KV statistics
tile, bounded by the local capacity, plus `reduction_layout="contiguous_8"`.
The TLE layout fixes eight consecutive scalars per thread independently of
pointer alignment. KV keeps its original eight-scalar owner shards. Its
finalizer selects `partial_reduction="global_tree"`, which gathers the same
partial values and reduces globally ordered owner lanes once. The previous
per-peer reductions introduced extra FP32 association boundaries. This changes
neither the mesh nor the transferred byte count, and needs no new workspace.
Defaults remain automatic local layout and `peer_sum`; FP32 mHC statistics
retain their previous schedule. Unsupported parameters/targets are rejected.

The layout regression fails before on all three misaligned packed/scalar/tail
cases (531) and passes after (532). An independent four-chip FP32 oracle
exposes 56 different mean-normalization outputs with the former peer order
(541) and passes both RMS and mean cases after (542). Actual MLA19 replay
matches Q, absorbed Q, latent, context and final projection at all eight
positions and four ranks (534/535/548), with zero spill. Full-model JIT passes (543: 255 registers, 65,536 shared bytes, 128-byte
ABI stack, zero spill). Normal publication retains all 64 readonly files
(546: 65.422 seconds). Full independent greedy8 twice then passes (547,
prefix13), including finite values, reset, sequence positions and cache tails.
Default/new cross-chip neighbors and parameter/target guards pass all 12
tests (549); fresh public compilation emits exactly the accepted source
(545), with equal immutable readonly recipes and all 2,053 readonly buffer
mappings (551).
