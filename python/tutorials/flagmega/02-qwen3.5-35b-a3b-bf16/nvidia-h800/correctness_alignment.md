# Decode benchmark alignment

The accepted October 24 scope uses native vLLM prefill for prompt lengths
32/128/1024/2048 and measures only the following 63 decode intervals in a
64-token output. Native vLLM, Initial, Agent and the standalone chat CLI each
run three rotated rounds of five requests, with two warmups per prompt.
Snapshot conversion, state restoration and prefill are excluded from timing.
Every snapshot covers all 40 layers and must round-trip exactly. The first
three independently generated decode tokens must match native; complete
64-token native equality is reported rather than required. Compiled prefill
and request throughput are outside this publication scope.

The accepted three-round campaign is complete and the two decode figures
are published in `figures/`. The long-context Agent measures
3.731/3.736/3.828/3.917 ms/token, with 1.13–1.19× median speedup over native
and 1.42–1.50× over Initial. All 240 measured requests have unchanged thermal
event counters; their source hashes, resource checks and 15120 decode
intervals are preserved in `figures/summary.json` and `figures/raw/`. The
standalone chat CLI measures 3.647/3.648/3.747/3.835 ms/token using the same
Agent artifact. Both series match all 64 native outputs in all four scenarios.
The complete previous measurement set is preserved in `figures/before_optimization/`;
`figures/long_context_comparison.json` records latency reductions of 1.4%
and 3.2% for Agent at 1024 and 2048. The final program uses 161 registers,
143340 shared bytes and a 128-block cooperative grid, with zero spills
and zero stack bytes. All 51 targeted numerical, boundary, selection and
measurement-policy checks pass.

## Decode compiler root causes

`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp::loadDShared`
created predicated shared-memory inline-asm outputs without initialization.
On inactive threads the PTX destination remained a live-in register. In
noinline device functions ptxas preserved that value in local memory, even
though the reduction discards those threads' results. This explains the
remaining spill instructions in the 256-element router softmax and QKV
partial reduction; it was not matrix accumulator pressure.

Initializing every scalar/vector output with the existing PTXBuilder helper
before the predicated load removes that false register lifetime. Active
threads execute the same load and arithmetic; constant-true loads retain
their existing lowering. The change is entirely within `__TLE__` guards.
The exact stock Initial decode artifact changes from 168 registers, 12-byte
spill stores/loads and a 16-byte stack frame to 160 registers, zero spills and
zero stack, with the same 149868-byte shared arena, 128-block cooperative
grid and N/K geometry. `python/test/tle/unit/test_tle_predicated_shared_load.py`
fails before the change and passes afterwards for four/eight warps. Forty-one
neighboring reduction, pointer-layout, ordered projection, QKV MMA and function
ABI checks pass; NVIDIA's non-TLE path and other backends are unchanged.
The full-model command is `benchmark_requests.py --runtime vllm --label baseline
--artifact <initial> --native-prefill-reference <reference/report.json>` with
the pinned checkpoint, 64 outputs and all four prompt lengths.

`agent_optimizations/decode.py::create_target` overwrote the fast SIMT GEMV
implementation with native ordered MMA unconditionally. Unordered BF16
imports consequently computed sixteen replicated MMA rows for a single-token
projection. That policy changed the Agent's decode latency from approximately
3.65 to 5.7 ms/token. The target now offers distinct SIMT and ordered MMA
implementations with separate identities and arithmetic facts. Normal
selection keeps the fast implementation for unordered imports and retains
the complete ordered accumulator when `reduction_order="mma_k16"` requires it.
This preserves both arithmetic contracts and the TMA pipeline optimization.
The three unordered-selection regressions fail before the correction; seven
target-policy checks pass afterwards. Ordered pipeline GPU comparisons retain
exact results and zero spills. The native-prefilled chat diagnostic now runs
at 3.65/3.65/3.81/3.98 ms/token for the four prompts, and all 64 outputs match
native. These diagnostic timings are excluded from the formal figure data.

## Long-context attention optimization

The growing decode cost is in
`codegen/triton/kernels/paged_attention_partial/decode.py.jinja::`
`_flagmega_paged_attention_partial`: automatic layout selection distributes
the 256-element QK reduction across warps. Each 32-token context tile then
requires shared-memory reduction and synchronization. With eight context
owners, a 2048-token prompt followed by decode needs eight or nine iterations
per owner, repeating that overhead in each of the ten full-attention layers.

The tutorial's SM90 `decode_t64` template explicitly gives each token row a
complete 32-lane warp, with head elements distributed within that warp.
`agent_optimizations/decode.py::create_target` declares the corresponding
eight-warp consumer requirement. At most 32 tokens per context owner use
the original single-iteration 32-token path. Longer contexts use 64-token
tiles; both K and V use the declared row layout. This reduces tile iterations
without the register pressure of a 128-token tile. Native online-BF16 keeps
its declared tile and traversal. In every path,
FP32 probabilities, context ownership, KV addressing and cache capacity retain
their contracts. The template flag is set only by this tutorial's NVIDIA
SM90 specialization. Portable layouts, 64-lane backend variants and the
native ordered online-BF16 MMA branch retain their implementations.

The checked-in [benchmark_attention.py](benchmark_attention.py) reproducer
uses 16 query heads, two KV
heads, a 256-element head, eight context owners and nonconsecutive physical
pages. On its 2048-token input, the original 32-token kernel takes about
39–41 microseconds and the warp-row variant about 23 microseconds, with zero
spills. An additional pinned-Torch diagnostic at lengths 1024/2048/3072
measures approximately 25/41/58 microseconds before and 15/22/29 afterwards;
its complete report is in `figures/attention_diagnostic.json`. Larger
128-token tiles were also measured and require 155–248 registers in the
isolated kernel, compared with 124 for the selected 32/64-token dispatcher.
Their complete-program gains were smaller than the isolated-kernel gains.
The final policy targets the repeated synchronization
at its source, using 32/64-token tiles within the persistent register budget.

`tests/test_decode_target.py::test_agent_attention_reduction_fits_one_warp_per_token`
fails before the policy/layout change and passes afterward.
`tests/test_attention_decode.py` validates generated GPU output against an
independent FP32 Torch softmax at context lengths 31/255/256/257,
1023/1024/1025, 2047/2048/2049 and 3071, with randomized nonconsecutive pages
and zero spill. These checks plus native online-BF16, causal-chunk and target
neighbors pass all 32 tests. Isolated-kernel timings are diagnostic evidence;
the figure data uses complete native-prefilled requests.

## Thermal conditioning

The first formal campaign on GPU 6 is excluded: software thermal slowdown
was active at 86 C, the SM clock dropped from 1980 to 1635 MHz and repeated
identical scenarios drifted by about 20 percent. The replacement campaign
runs sequentially on GPU 0, with at least 0.5 untimed idle seconds between
requests and a maximum starting temperature of 65 C. Each request records
NVML software/hardware thermal event counters before and after execution;
a new thermal event rejects the campaign. The figure validator rejects a
counter increase as well. This acceptance regression fails before the
validator correction and passes afterwards. Idle intervals are outside
the 63 measured decode intervals, so these figures report per-request decode
performance rather than a sustained request admission rate.

## Historical compiled-prefill investigation

The notes below preserve earlier component diagnostics and their original
acceptance status. Their complete-prefill and full-request publication gates
were superseded by the native-prefill decode scope above.

## Defects exposed by real prefill

| Root cause and trigger | Correction | Checked-in reproducer |
| --- | --- | --- |
| `delta_rule_block_update/blockwise.py.jinja`: strongly negative decay prefixes make upper-triangular `exp2(log_i-log_j)` overflow. Multiplying its infinity by a zero coefficient creates NaNs that contaminate state and output. | Mask the exponent before evaluating it and mask coefficient loads to the causal triangle. Mask padded decay exponents as well. Reload the immutable prefix after output MMA to avoid extending its lifetime across both fragments. | `python/test/flagmega/codegen/triton/kernels/delta_rule_block_update/test_strong_decay.py` |
| `importer/numerics/gdn_prefill.py::emit_gdn_prefill`: FP32 output conversion leaves NormApply's default BF16 intermediate rounding intact. Native GDN normalization and SiLU multiplication stay FP32 until the product is stored. | Specify FP32 normalization input semantics as well as FP32 output. | `python/test/flagmega/importer/qwen3_5_moe/test_prefill_gate_normalization.py` |
| `importer/numerics/qwen3_5_moe.py::apply_qwen35_moe_vllm_profile`: inferred BF16 projections followed by private widening casts permit the relaxed FoldMatMulCast rewrite to remove the native projection store. | Declare the native BF16 projection result explicitly, including the prefill A/B projections. The default relaxed rewrite remains available to other contracts. | `test_native_projection_stores_survive_private_widening` in the same test file, for decode and prefill |
| `passes/tir/bufferize/planner.py::_plan_call`: a callee's readonly constant result receives a new caller workspace even though no kernel writes it. Returning the normalization constant `1` therefore loses the Gemma-style `1 + weight` scale. | Bind the call result to the already loaded immutable allocation. Do not allocate an unwritten result. | `python/test/flagmega/passes/tir/bufferize/test_readonly_call_result.py`, with fast/optimized allocation and direct/nested calls |
| `passes/functions/function_boundary_layout.py::_propagate_one_iteration` and `distributed_boundary_layout.py::_propagate_one_iteration`: nested output projections take an early return before emitting a selected outer output adapter. The outer signature references a nonexistent producer. | Resolve matching transforms to the existing packed/distributed projection; emit required differing transforms and their logical view. | Nested cases in `test_readonly_call_result.py`; neighboring function-boundary-layout suites |
| `codegen/triton/kernels/qkv_rope_with_cache/_function.py.jinja`: fused prefill uses one cache position for every K/V row and advances sequence length by one. All prompt rows overwrite the same slot. | Derive the token position from each K/V ownership map, resolve its physical page, retain the chunk base in metadata and advance by the complete query length. Apply the same rule to the partial-QKV implementation. | `python/test/flagmega/codegen/triton/kernels/qkv_rope_with_cache/test_prefill_chunk.py`, including token-specific RoPE, independent V ownership and page crossings |
| `codegen/triton/kernel_call_renderers.py::_exclusive_participation_active`: a canonical global input's exclusive writer predicate restricts its consumers. A split vocabulary projection therefore executes only on CTA zero, leaving every other shard unwritten. | A canonical global input can be read by every block on the chip. Retain owner restrictions for private inputs, exclusive outputs/workspaces and chip/node ownership. | `python/test/flagmega/codegen/triton/distribution/test_canonical_exclusive_reads.py`; neighboring grid rendezvous tests |
| `passes/functions/outline_stages.py::outline_function_stages`: sequential stateful stages export a locally constructed reference subspan through an identity-only result ABI. | A declared non-owning reference view with caller-addressable operands is constructed by the caller and passed to each stage. Stage results then alias their full parameter spans. The ABI verifier remains intact. | `python/test/flagmega/passes/functions/test_outline_reference_views.py` |
| `codegen/triton/function_programs.py`: reusable program entries disable pointer-alignment specialization, turning aligned recurrent-state loads into scalar loads and increasing the block-update kernel's register lifetime. | Retain the compiled pointer specialization. `runtime/prepared.py` reads its authoritative `tt.divisibility` attributes and rejects dynamic bindings that violate that specialization; a differently aligned address requires preparation of a matching specialization. | `python/test/flagmega/runtime/test_prepared_pointer_alignment.py`; reusable-function and program-resource-lifetime suites |
| Dense projection selection: a source BF16 projection is assigned a split SIMT reduction, which changes rounding relative to native ordered K accumulation. | Carry an explicit `mma_k16` contract through MatMul, vectorization, packing and normalization fusion. Filter candidates by that contract and retain one FP32 MMA accumulator across K tiles. | `python/test/flagmega/ir/ops/math/test_matmul_reduction_order.py`; `codegen/triton/kernels/dense_matmul/test_ordered_norm_stats.py` |
| `rules/neutral/fuse_wide_glu.py`: fusion discards the gate/up projections' ordered reduction contract. | Fuse only matching reduction orders, retain the order through GLU packing and select an implementation that satisfies it. | `test_wide_glu_retains_ordered_projection_contract` in `python/test/flagmega/rules/neutral/test_fuse_wide_glu.py`; `codegen/triton/kernels/dense_matmul_glu/test_ordered_mma.py` |

The strong-decay reproducer failed with NaNs before the fix. Its kernel suite
passes 14 tests after the fix, including the large state tile under the strict
zero-spill contract. The normalization reproducer failed on all 24 elements
before the precision correction. The readonly-result GPU reproducer failed on
all 128 elements, with an absolute error of one, before the binding correction.
The fused-cache reproducer failed for both V ownership maps before the fix.
Its two kernel suites pass 43 tests after the correction. Five canonical-input
participation cases failed before the scheduling correction; the seven new
cases and twelve neighboring rendezvous cases pass afterwards.

The complete 32-token prefill formerly selected token 8 because stale vocabulary
shards contained values as large as `9.6377e17`. After correcting participation,
all logits are finite and within `[-15.5625, 15.3125]`, and the first token is
79091, matching native vLLM. An independent BF16 matrix multiplication of the
same final hidden state also selects 79091. This is a first-token check: the
following decode sequence still differs at generated token 30, where the two
native leading candidates tie. Full-request numerical acceptance remains open.
The rebuilt decode artifact also passes preparation with zero spill bytes;
its full continuation from the real prefill retains the same token-30 mismatch.

## Other required compiler corrections

Real batched prefill also exercises several paths that the older one-token
prepared-state workload did not cover:

| Area | Trigger and causal correction | Regression tests |
| --- | --- | --- |
| Multirow AddNormStats | Materialized rows need independent statistics and one residual write per element. | `codegen/triton/kernels/add_norm_stats/test_prefill_rows.py` |
| Partial GatherReduceAddNormStats | Scratch addresses must include the global row and partial owner. Statistics ownership excludes the reduced hierarchy axes. | `codegen/triton/kernels/gather_reduce_add_norm_stats/test_batched_partial_statistics.py` |
| Embedding | Rank-two input must load the token belonging to each logical row. | `codegen/triton/kernels/embedding/test_batched_embedding.py` |
| Packed QKV projection | The fused GEMV must iterate the local row domain and use its source/result offsets. | `codegen/triton/kernels/qkv_parallel_linear/test_physical_order.py` |
| TMA pipeline layouts | Eight-warp layouts inside a sixteen-warp entry must replicate their layouts over the additional consumer warps. | `codegen/triton/descriptors/test_pipeline_warp_replication.py` |
| Ordered dense projection accumulation | Split SIMT reduction changes BF16 projection rounding. An explicit ordered MMA implementation retains an FP32 fragment across K tiles. | `codegen/triton/descriptors/test_ordered_pipeline_accumulation.py`; ordered norm-stat LHS staging cases |
| Escaping replica views | Promoting a private source to caller-visible output storage invalidates the view's earlier compact backing. Recompute that backing after promotion. | `passes/tir/bufferize/test_escaping_replica_view.py` |
| Function outlining | Pure projections can precede stateful work. Explicit effect-owning stages retain each original operation once; sequential stateful stages must preserve the original mutation order. | `passes/functions/test_outline_effect_stage.py` |
| Program resource lifetimes | Complete root and nested transfer stages own both roles and the instantiated shared arena within one program. Keep trailing drains with their owning call and retain reusable functions' endpoint arguments. | `codegen/triton/pipelines/test_program_resource_lifetimes.py` |

The test paths in this table are relative to `python/test/flagmega/`.
The complete 32-token prefill artifact has passed runtime preparation with
zero spill bytes and at most 199680 bytes of shared memory. This resource
result is separate from full-model numerical acceptance.
The nested-program reproducer failed before projection supported its complete
resource interface. Six tests pass afterwards, including direct/nested H800
execution under the zero-spill contract; nineteen neighboring reusable-function,
resource-instantiation and pipeline runtime tests also pass.
The reference-view reproducer failed the identity alias verifier before view
hoisting. Thirty-six outlining and state-view checks pass afterwards, including
H800 execution with two updates of each selected layer, unchanged neighboring
layers and recurrent state, and zero spill bytes.

The rebuilt 128-token batched prefill selects the native first token, 79091,
with finite logits and passes strict preparation with zero spill bytes and
at most 131228 bytes of shared memory. The standalone block-update reproducer
spilled 72 bytes when its recurrent-state pointer was forced unspecialized;
normal alignment specialization removes the spill. The dynamic-rebinding
alignment regression failed before the runtime check, and sixteen alignment,
stream and resource checks pass afterwards; fifteen reusable-function and
program-resource neighbors also pass. Full 64-token numerical acceptance is
still pending.

The ordered projection contract makes layer zero's convolution and all
524288 recurrent-state elements exactly equal to native after a real
32-token prefill. Its ordered normalization-fusion suite passes four H800
cases, including 32 rows and both BF16/FP32 residuals, with exact eager output
and zero spill bytes. The GLU contract reproducer failed before fusion retained
the attribute. Three logical/packed GLU H800 cases pass afterwards with exact
eager output and zero spill bytes. The full continuation from the rebuilt
prefill still differs at generated token 30; these component checks do not
complete numerical acceptance.

## Long-prompt CP composition

Native prefill composes independent 512-token chunks using transfer matrices
and local states, followed by TF32-truncated state fixup. Sequential block
updates alone do not express this rounding boundary. The profile now exposes
these operations for prompts longer than 512 tokens.

Standalone device comparisons of the transfer matrix, local state and fixed
state are exact against the pinned native implementation for 32, 128, 1024
and 2048 tokens, with zero spill bytes. The checked-in combined pipeline test
passes all four lengths, including output and persistent-state equality,
under strict zero-spill preparation.

The shared block-update kernel formerly kept a BF16 register view of its
FP32 state alive across the coefficient and output MMAs because two later
projections used it. Its 128-token reproducer spilled 176 bytes. The Hopper
implementation now stores that view in one resident WGMMA operand arena;
both state projections consume the arena directly. This ends the register
view's lifetime after its first publication and keeps the FP32 state and
native MMA arithmetic intact. At 128 tokens the kernel uses 233 registers,
81920 shared-memory bytes and zero spill bytes, instead of 255 registers,
117760 shared-memory bytes and 176 spill bytes. Looping over chunks rather
than unrolling all four also removes the 2048-token residual spill.

A component-only CUDA-graph measurement, including identical state resets,
takes about 25.55 microseconds after this correction versus 22.97 microseconds
for the former spilling implementation. This is separate from full-request
speedup. The earlier FP32/BF16 shared-copy experiments were slower and are not
the selected implementation. Full-model correctness, complete 64-token
continuation and full-request performance acceptance remain open.

The performance reference is the existing decode result: Agent is about
1.25x native vLLM and 1.53x Initial. The replacement benchmark must assess
decode speedup and request throughput with real prefill before updating the
figures; extending the request scope does not make the old prepared-state
timing an end-to-end measurement.

## Reproduce the correctness checks

Run from the repository root with an exclusive SM90 GPU, a TLE-enabled
FlagTree environment and PTXAS 13.3.73:

```sh
conda activate flagtree
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
pytest python/test/flagmega/importer/qwen3_5_moe/test_prefill_gate_normalization.py \
  python/test/flagmega/importer/qwen3_5_moe/test_qwen35_numerical_profile.py
pytest python/test/flagmega/codegen/triton/kernels/delta_rule_block_update
pytest python/test/flagmega/passes/tir/bufferize/test_readonly_call_result.py \
  python/test/flagmega/passes/functions/function_boundary_layout \
  python/test/flagmega/passes/functions/distributed_boundary_layout
pytest python/test/flagmega/codegen/triton/kernels/qkv_rope_with_cache \
  python/test/flagmega/codegen/triton/kernels/gather_reduce_qkv_rope_with_cache
pytest python/test/flagmega/codegen/triton/distribution
pytest python/test/flagmega/codegen/triton/pipelines/test_program_resource_lifetimes.py
```

The pinned native contract is vLLM
`ae10e855abf4ff5e24e2088aef16029ee1cb7de8`, BF16, one H800, TP/PP one,
Inductor compilation mode three and `custom_ops=['none']`. Long prompts also
require the native CP prefill chunk-transfer contract; component tests for a
single chunk do not establish that contract for 1024/2048 tokens.


## October 23 complete-request alignment (in progress)

The accepted scope is prompt lengths 32/128/1024/2048, 64 independently
sampled output tokens, three rotated rounds of five requests and two warmups,
including the standalone chat CLI. The user relaxed complete native token
parity on October 24: publication records native output differences, while
retaining comparison with independent IR/PyTorch eager execution. Uncaptured
compiled execution is a diagnostic, not the independent eager reference.
No replacement performance figures have been published from these diagnostics.

Explicit BF16 parameter stores now carry `rounding_contract="preserve"`
through Cast/VectorizedCast, cast folding and view propagation. This repairs
the FP32 checkpoint GDN norm weight's BF16 parameter rounding: the previous
FoldCast rule removed its BF16-to-FP32 round trip. Accurate GDN SiLU uses
`z / (1 + exp(-z))`; its `math_mode` survives vectorization and fusion, including
the main VectorizeUnary rule. The 32-token layer-0 diagnostic now matches the
native recurrent core, normalized values, gate and BF16 GDN output exactly.
These are component results; they do not establish complete-model acceptance.

A second numerical defect is the native small-M output projection's K
partition. For BF16 [M,4096] × [2048,4096], the pinned CUDA kernels for M=1/32
publish two K=2048 FP32 partials and add them before BF16 rounding. M=128,
1024 and 2048 use complete-K accumulation. CUDA profiling found the small-M
split-K GEMM/reduction pair. On the actual layer-0 inputs, complete-K
accumulation differed in 153 BF16 values; two partials matched exactly.
Both Torch 2.10/cu128 and pinned native Torch 2.11/cu130 reproduce this result.
The profile now declares the two partial MatMul operations, FP32 Add and one
preserved BF16 Cast. The generated SM90 regression is exact and has no spill;
other matrix shapes retain their previous accumulation contract.

The new K slices exposed `LowerVectorizationContracts` reconstructing a
scalar Bitcast as its annotated vector Slice in graphs below 20,000 nodes.
The packed matmul then received a vector LHS in its scalar ABI. Removing the
graph-size condition retains the real Bitcast and its physical dependencies
for every graph. The checked-in minimal case fails before the change.

Prefill/decode phase bundles share verified readonly byte ranges while keeping
persistent GDN and KV state common across phases. The first full 40-layer
32-token bundle occupies 69,540,672,768 readonly bytes on one H800. Loading a
second phase copies only new content-proved ranges. The paged ABI carries one
logical chunk base in `slot_mapping[0]`; reset must keep that scalar even for
a 32-token chunk. The old reset attempted a per-token list and failed; the
scalar chunk-base regression now passes.

Independent eager evaluation starts from imported IR and executes PyTorch
operators without generated kernels or vLLM. Ordinary `TorchEvaluator.run`
now releases consumed local memo values and does not retain a trace; explicit
trace APIs keep their original retention contract. Retaining all copied expert
weight slices previously exhausted H800 memory. After the lifetime change, the
full 32+64 request completed on one H800 and matched the pre-partition compiled
bundle in all 64 outputs. Both still differed from native, so this result remains
unpublished. Updated projection artifacts are being built and validated.

Reproduction of the new checked-in cases:

```sh
pytest python/test/flagmega/codegen/triton/kernels/test_accurate_silu.py \
  python/test/flagmega/importer/qwen3_5_moe/test_partitioned_projection.py \
  python/test/flagmega/evaluator/test_value_lifetimes.py \
  python/test/flagmega/runtime/test_shared_readonly_pool.py \
  python/test/flagmega/passes/auto_distributed/test_retained_vector_boundary_dependencies.py \
  python/tutorials/flagmega/02-qwen3.5-35b-a3b-bf16/nvidia-h800/tests/test_phase_backend.py \
  python/tutorials/flagmega/02-qwen3.5-35b-a3b-bf16/nvidia-h800/tests/test_request_results.py
```

The performance target remains approximately the historical Agent/native
1.25–1.26× and Agent/Initial 1.53–1.54× decode speedup, measured again at the
complete-request boundary before any replacement figure is generated.


The K-slice projection also exposed two owner-visibility defects. In
`passes/tir/plan_function_memory.py::_can_place_block_local`, owner-local
coordinate effects did not prove that an exclusive producer populated the
private allocation of every executing consumer. A scalar view retained E
ownership while a packed matmul published split output columns. Only the
canonical block executed; 65024 of 65536 projection values were missing or
incorrect. Private placement now requires each consumer to retain the
producer's exclusive block owners. Same-owner private edges remain optimized.
Verified resident weights and all canonical slice values were correct.

In `passes/tir/bufferize/synchronization.py::_node_accesses`, a canonical
reader's E input described the writer, and was incorrectly reused as the
reader's execution owner set. The fused case therefore received a block fence
where non-E consumers needed a grid rendezvous. Canonical read accesses now
include those executing block readers while retaining chip/node ownership.
The memory placement and synchronization regressions each fail before their
fix. All 114 neighboring checks pass, including same-owner, physical-span,
in-place, and multi-chip tests. The H800 normalized-input projection regression
passes with zero spill. The separate layer-0 projection and post-norm now
match native exactly, and the corrected full prefill produces native's first
token 79091. Complete 64-token acceptance remains open.

The independent eager ordered-matmul handler previously widened BF16 inputs
to FP32 before GEMM, changing the declared MMA reduction. Its CUDA
`mma_k16` path now uses PyTorch mixed-output GEMM with BF16 operands and an
unrounded FP32 output. The checked-in CUDA case failed in 64968 of 65536
FP32 accumulators before the change; it and 16 neighboring checks now pass.
This reference still uses PyTorch operators independently of generated kernels.

The native router's FP32 order is also part of the profile. The pinned source
`csrc/libtorch_stable/moe/topk_softmax_kernels.cu` uses accurate exp, sequential
eight-element per-thread sums, a descending warp XOR butterfly, sequential
TopK score accumulation, and reciprocal multiplication. The previous generic
softmax/tree sum/division changed 166 of 256 layer-0 routing weights by up to
5.96e-8 and changed one routed BF16 output. The profile now emits
`nn.staged_routing(arithmetic="exp_grouped_reciprocal")`, which owns both
scoring and normalization. Default routing and the existing sigmoid contract
retain their behavior. The no-bias renderer also accepts its actual optional
buffer ABI. Initial, Agent, and independent PyTorch routing match saved native
weights and IDs exactly; generated CUDA kernels have zero spill. The native
fixture and regression are in `test_staged_softmax_contract.py`; 40 routing and
profile checks plus two explicit profile-routing checks pass.

The rebuilt decode still differs at output 30 for prompt 32 and output 15 for
prompt 128 when starting from native states; 1024/2048 continuations match all
63 outputs. This is diagnostic evidence, not complete-request acceptance.
Full-model numerics and complete-request performance remain open, and the
replacement figures remain unpublished.

## Ordered fused QKV accumulation and independent convolution evaluation

`nn.qkv_parallel_linear` did not carry the ordered reduction contract used by
other native BF16 projections. Its AutoDistribution provider could therefore
accumulate independent K shards, round each partial to BF16, and combine them
with reassociated SIMT additions. From an identical first-decode input at layer
3, this produced 202 differing key and 216 differing value elements for the
32-token prompt.

The native profile now declares `mma_k16` on fused QKV. Packing, Python resume,
AutoDistribution and microkernel selection preserve it. N sharding remains
available; independently accumulated K shards are rejected. The SM90 TMA
consumer carries its FP32 MMA accumulator across every K transfer and performs
one final BF16 store. A tiled MMA implementation covers multiple input rows.
The standalone projection regression failed before the contract existed; the
M=1 TMA and M=32 tiled executions now match independent PyTorch BF16 projections
exactly with zero spill bytes. Thirty contract/profile checks pass, and the
full-model first-decode layer-3 KV cache now matches native exactly for both
32 and 128 tokens. This correction does not yet resolve the full sequence's
token-30/token-15 divergence: the next large state discrepancy begins at layer
4 after full attention.

The neighboring canonicalization checks exposed a separate existing source
placement defect: `fused_weight_n_shares` had been inserted before the local
return of `_fused_weight_type`, making that return unreachable. Local packed
weights consequently produced `None` instead of their fused tensor type. Moving
the local return into its owning function removes the defect; all five formerly
failing cases and ten neighboring canonicalization/owner-order checks pass.
Distributed owner-major materialization retains its existing branch.

The independent CUDA convolution evaluator used accurate Torch SiLU although
the declared device convolution uses exp2-based fast activation. On a small
fixture captured from native `causal_conv1d_fn`, all 64 output channels differed
at BF16 rounding boundaries. CUDA eager now evaluates the same exp2 expression
with ordinary Torch operations. CPU arithmetic-reference behavior is preserved.
The native fixture, generated convolution and neighboring evaluator checks pass
61 tests. This changes the independent oracle, not compiled performance data.

The independent gate evaluator also used accurate exponentials for the inner
softplus and sigmoid, and ignored the declared outer exponential mode. CUDA
eager now uses Torch exp2 for the declared fast inner expression and selects
the outer expression from `alpha_exp_mode`. A small fixture from native
`fused_post_conv_prep` failed before the correction. The gate, delta-rule and
generated-kernel suites pass 189 tests afterwards. Full request output
acceptance remains separate from these arithmetic checks.

## Native FlashAttention probability rounding

The first decode's layer-3 QKV correction made its KV cache exact, but the
following full-attention output still differed. On the actual native query and
33-token cache, FP32 softmax/PV produced 617 differing BF16 outputs; casting
globally normalized probabilities to BF16 produced 987. Casting the
**unnormalized** exponentials to BF16 before PV and normalizing the FP32
accumulator afterwards matches all 4096 native outputs exactly.

The pinned [FA3 tile contract](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/tile_size.h)
uses 80-token blocks for nonlocal BF16 256-dimensional heads. Its mainloop
traverses blocks in reverse, and its softmax retains the probability store
before PV. The numerical profile now explicitly declares `online_bf16`, block
size 80 and reverse traversal. Python resume, partial decomposition, eager
evaluation and kernel selection retain those attributes. A separately rounded
context-shard origin is not equivalent to this ordered traversal; the generated
partial ABI therefore assigns the complete traversal to one context owner and
publishes empty states from the remaining context owners. Independent heads
remain distributed, and the rest of the fused model pipeline remains available.

The native one-head fixture failed before the contract existed. Its eager
regression passes, and the generated two-head/empty-context-owner case matches
the native fixture exactly with zero spill bytes. Forty contract checks and
65 neighboring causal-chunk, statistics, owner-ABI and QKV-fusion checks pass.
At a 129-token cache, reverse 80-token traversal reduces the standalone
difference to eight of 4096 BF16 elements; full-model 64-token acceptance is
still pending and remains the publication gate.

## Further native rounding boundaries

The following component corrections retain the complete workload and the
strict zero-spill requirement. They do not constitute full-request acceptance.

| Location and trigger | Causal correction | Checked-in regression |
| --- | --- | --- |
| `importer/numerics/rotary_cache.py` and the rotary-embedding template: device-generated inverse frequencies and approximate trigonometry differ from the native CPU frequency table and accurate CUDA trigonometry. | Preserve the native FP32 inverse-frequency coefficients and declare accurate cosine/sine evaluation. | `codegen/triton/kernels/rotary_embedding/test_native_rotary_cache.py`: all 3072 positions and 64 rotary dimensions are exact. |
| `kernels/paged_attention_partial/decode.py.jinja`: SIMT QK/PV reductions change BF16 output rounding relative to FA3's ordered MMA. The first full-model MMA implementation also exceeded the register limit of its enclosing program. | Carry the FP32 MMA accumulator across K tiles; stream QK in 64-element tiles and PV in 32-element tiles to end operand lifetimes earlier. Retain the native 80-token reverse traversal and unnormalized BF16 probability store. | `test_native_online_bf16.py`, including native layer-3 and layer-11 fixtures. The complete Initial decode artifact prepares with zero spill; its first-step layer-3/7/11 query and attention outputs are exact. |
| `nn.l2_normalization`: splitting scalar Q/K reductions changes the native contiguous-eight accumulation order. Ordinary Torch sqrt/reciprocal also differs from the declared CUDA approximate sqrt and full division. | Represent contiguous reduction groups explicitly, retain them through vectorization, and evaluate the declared scalar CUDA instructions through PyTorch Jiterator. Accurate arithmetic remains a separate choice. | `test_native_grouped_reduction.py` and `ir/ops/nn/l2_normalization` fixtures. The complete native Q/K preparation for the first two layers is exact; 88 neighboring checks pass. |
| `nn.delta_rule_coefficients::_rounded_mma`: eager widens rounded BF16 Gram and FP16 inverse operands before matrix multiplication, changing their accumulation order. | Keep the rounded operand dtype and request an FP32 accumulator from Torch CUDA batch matmul. | `ir/ops/nn/delta_rule/test_coefficients_mma_rounding.py`: the reduced 12-head fixture fails before and is exact afterwards; 191 neighboring checks pass. Expected coefficients describe the typed MMA schedule, rather than an externally captured native coefficient buffer. |
| `nn.norm_stats` and its local template: a 256-dimensional packed Q/K vector is accumulated as two 128-element tiles instead of native contiguous pairs. | Declare contiguous-pair reduction for Q/K statistics, including packed scalar lanes. Reject reassociation across independent partial owners. RMS fusion retains explicit statistics when its replacement cannot express the ordered group. | `test_native_qk_pair_reduction.py`, NormStats contract tests and the fail-before/pass-after ordered-fusion regression. |
| `nn.delta_rule_log_prefix::evaluate`: `log2_mode='fast'` is ignored and ordinary Torch log2 changes the FP32 decay scan. | Evaluate the declared approximate CUDA log2 independently through PyTorch Jiterator; use accurate libdevice log2 for the accurate generated contract. | `ir/ops/nn/delta_rule/test_log_prefix_evaluation.py`: the eight-element fixture fails before the fix; 204 delta-rule/generated-prefix checks pass afterwards. |

The updated 32-token batched prefill prepares all 221 programs with zero spill,
selects native token 79091, and has exact persistent state through model layers
0–4. Layer 5's recurrent state remains the first differing state. Independent
eager now matches every saved layer-0 activation boundary, including the MoE
output and FP32 residual sum, but still differs in later layers. The 128-token
first decode also has exact layer-3 attention gate, projection, post-norm,
shared/routed output and final FP32 residual sum. These diagnostics are excluded
from performance measurements. All four complete 64-token requests and their
independent eager references remain required before replacing the figures.

## Rounded block-update MMA and native hidden RMS accumulation

`nn.delta_rule_block_update::delta_rule_block_update` widened BF16 operands
before its matrix products. On the saved native layer-1 prefill preparation,
this changed 30 of 4096 BF16 core outputs even though query, key, coefficients
and decay prefix were exact. The evaluator now retains BF16 operands and uses
FP32 CUDA MMA accumulation for all six matrix products. The reduced one-head
fixture in `test_block_update_evaluation.py::test_cuda_block_update_retains_rounded_mma_operand_dtype` fails before this change and is
exact afterwards; 190 related checks pass. Nonzero incoming state and longer
multi-chunk full-model evaluation remain part of complete-request acceptance.

Native hidden RMS kernels accumulate contiguous groups of eight scalars. The
post-attention kernel uses a 2048-column reduction block, while the following
input RMS uses 1024 columns and carries per-column FP32 FMA accumulators across
the two blocks. Their actual selected launch configurations and source were
captured from the pinned native engine, then replayed against its saved inputs
and outputs. A flat tree or independently summed 128-column tiles changes BF16
rounding: the reduced layer-4 post-norm fixture differed at channel 417.

The profile now declares the group and tile-FMA accumulation order explicitly
on `nn.norm_stats`, and accurate inverse-square-root arithmetic on ordinary
`nn.norm_apply`. The evaluator implements that order independently with Torch
operations and a scalar CUDA FMA through Jiterator. Code generation preserves
the same order. NormStats threading retains every numerical attribute across
function boundaries; fusion applies only when its replacement can express the
declared arithmetic. Default contracts keep their existing selections. The
native hidden RMS regression, contract tests and fusion/threading regressions
pass, along with 178 neighboring checks (eight skips, four expected failures).

The rebuilt diagnostic 32-token prefill has 180 programs, prepares with zero
spill, produces native token 79091, and has exact persistent state through
layers 0–5. Independent eager has exact GDN core outputs in layers 0 and 1;
the next discrepancy is in layer-1 output gating. Prepared-state decode now
matches all 63 outputs for 32/1024/2048 tokens, while the 128-token continuation
still differs at output 15. These diagnostics are not performance measurements
and do not satisfy the full-request publication gate.

The next independent-eager discrepancy was isolated to accurate CUDA SiLU:
layer 1's native core, packed QKV/Z projection and RMS inputs are exact, but one
gated BF16 output differs. Feeding the actual native gated output to the
independent projection removes all 25 projection differences. The native
generated expression uses accurate `exp` followed by CUDA `div.full.f32`;
Torch's ordinary activation rounds that division differently at this BF16
midpoint. `math.silu::evaluate` now evaluates its declared accurate CUDA
expression with an independent scalar Jiterator operation. The generated
implementation remains unchanged. The one-head native fixture fails before
the evaluator correction and passes afterwards in both eager and compiled
execution, with zero generated spill; four direct activation checks pass.

`nn.paged_attention::paged_attention_scalar_value` also shortened the last
causal block before eager CUDA QK/PV matmul. CUDA selected a different BF16
accumulation schedule for these short shapes. Native prefill layer 3's query
and complete KV cache were exact, but 46 of 131072 BF16 attention outputs
differed; a reduced causal row failed at head 9, channel 244. Eager now retains
the declared softmax/MMA block size, pads invalid K/V positions with zero and
masks their scores to negative infinity. The real prompt and causal length
remain unchanged. The reduced native fixture passes after the correction;
the generated empty-context-owner regression also covers it. Forty-one nearby
attention, cache, causal-chunk and native-rounding checks pass with zero
generated spill. Native prefill layer 3's attention, projection, post-norm and
FP32 output are now exact in eager. The following input-RMS difference was
subsequently isolated and corrected as described below. Complete 64-token
acceptance remains open.

The native 1024-column input RMS uses a two-level reduction: contiguous groups
of eight, XOR reduction within 32 lanes, then reduction across the remaining
groups. Its fully unrolled block loop also rounds the second block's squares
before applying FMA with the first block. The previous eager implementation
traversed blocks forward and applied a flat XOR tree. On the actual layer-4
input, this changed one BF16 normalized element and 435 QKV projection
elements. Instrumenting the original native kernel preserved all of its BF16
outputs and exposed its FP32 statistics; the reverse block traversal and
32-lane hierarchy matched all 96 captured statistics from three input RMS
calls exactly.

`nn.norm_stats` now represents block traversal and the inner reduction group
explicitly. The native input-RMS profile declares both, and evaluator,
codegen, Python serialization, pattern matching and function threading retain
them. Default contracts are unchanged. The checked-in native input fixture in
`codegen/triton/kernels/norm_stats/test_native_input_hierarchical_tree.py`
fails before the evaluator correction and passes afterwards with exact FP32
statistics and BF16 output in eager and compiled execution, with zero spill.
Forty-nine direct checks and 111 neighboring checks pass (eight skips and four
expected failures). Actual prefill QKV, gates and GDN core/gated outputs are
now exact through model layers 0, 1, 2 and 4.

The next layer-5 discrepancy is another CUDA division boundary, this time in
`nn.gdn_convolution::gated_delta_net_convolution`. Its QKV, A/B and normalized
Q/K were exact, but one convolution value differed at token 16, channel 7445,
which changed 14 subsequent BF16 GDN core outputs. The chronological BF16
products and FP32 sum were correct. Ordinary Torch division rounded SiLU to
`-0.011749268509447575`; native CUDA full division produced the BF16 midpoint
`-0.011749267578125`, which rounds to the other BF16 neighbor. The CUDA eager
path now independently evaluates approximate exp and `div.full.f32` through
Jiterator before the final BF16 cast. The compiled implementation was already
correct and remains unchanged. The native four-token fixture
`data/native_convolution_full_division.json` fails in eager before the fix and
passes afterwards; compiled execution also matches. All 57 convolution
numerical checks pass. Full-model acceptance is being rerun; these component
results remain excluded from performance measurements.

Layer 6 exposed the same full-division distinction in
`nn.delta_rule_gates::evaluate`: its ordinary Torch reciprocal changed 110 of
1024 FP32 beta gates by one ulp. Query, key, value and the accurate alpha gates
were exact. Replaying the typed coefficient and block-update operations with
native beta changed seven BF16 coefficients and removed all 16 differing core
outputs. The eager fast sigmoid now independently retains CUDA approximate
exp and full division; CPU evaluation and generated kernels are unchanged.
The native head-22 fixture in `data/native_delta_rule_fast_sigmoid.json`
fails before the fix at five of 32 beta values and is exact afterwards. Both
old and new native fixtures now require exact FP32 beta agreement. Twenty-three
gate checks pass, including generated storage/distribution variants and the
new native fixture, with strict zero-spill preparation. GDN QKV, A/B, core and
gated output are now exact through model layers 0, 1, 2, 4, 5 and 6. The subsequent attention layer-7 differences were isolated and corrected as
described below. Replacement figures remain unpublished.


The native fused QK RMS kernel reduces each pair of columns, then performs
XOR reductions within 32 lanes before combining the remaining groups. The
previous flat eager reduction changed one BF16 query element at layer 7,
token 11, head 12, channel 224, despite exact projection and normalization
inputs. The native QK profile now declares the 32-lane inner tree. The original
native QK kernel replay with identity RoPE preserves all unaffected query
channels. Its checked-in fixture fails before the correction and passes in
eager and generated execution with zero spill.

FA3's online softmax uses FMA for `score * scale - maximum * scale`, followed
by approximate CUDA exp2; final normalization multiplies by approximate
reciprocal. Unfused eager subtraction and Torch division changed five BF16
layer-7 attention values even with exact native Q/K/V. The CUDA eager path
now independently retains those operations. The reduced native fixture
`data/native_flash_attention_fma_reciprocal.json` fails before the correction
and passes afterwards in both eager and generated execution. Nine direct
QK/attention checks and 97 Qwen3.5 importer checks pass.

Native post RMS additionally seeds each group of eight squares with column
1, then applies FP32 FMA to columns 0, 2, 3, 4, 5, 6 and 7, before its inner
32-lane and outer reductions. Individually rounded squares or a column-0 FMA
seed do not preserve this rounding. Layer 9's post norm changed one BF16
element at token 12, channel 1766, which then changed 2349 layer-10 QKV
values. Read-only instrumentation of seven original native calls retained all
BF16 outputs; the column-1 grouped FMA matched all 224 FP32 statistics and
reciprocal square roots. On native 1024/2048-token inputs the same contract
also reproduces the tested post-RMS boundaries exactly.

`nn.norm_stats` now carries `group_fma` and its group seed explicitly through
inference, evaluation, Python serialization, pattern matching, generated
kernels and function threading. The checked-in native post fixture verifies
exact BF16 output, exact FP32 statistics and zero spill; its legacy-contract
negative control changes one BF16 output. Nearby importer verification also
exposed a stale call-structure assertion and a legacy GLU constructor that
omitted canonical attributes; these now preserve the actual imported call
structure and use the op's canonical attribute normalization.

The latest complete independent eager run after grouped FMA accepts all 64
outputs for prompt 32. Prompt 128 still first differs at output 15, prompt
1024 at output 51, and prompt 2048 at its first output. A separate native
request-order audit retained identical outputs across three rotated rounds.
This earlier diagnostic preceded the GDN prefill normalization correction.
The subsequent native-contract eager run has matching first tokens for all
four prompt lengths, with the first later native differences at output
indices 29, 26 and 50 for prompts 32, 128 and 1024; prompt 2048 matches all 64.
Per the latest requested scope, complete native token equality is comparison
evidence rather than a publication gate. The original nncase BF16 contract
is also being measured against its own independent IR/PyTorch reference.
No diagnostic timings are used as formal request performance results.

True 32-row prefill exposed a distribution defect at
`passes/auto_distributed/norm_providers.py::AddNormStatsCandidateProvider`.
Its projection used broadcast rows and block-cyclic column units of one,
with Sum partials on hierarchy axis 0; the private residual used four-row
block-cyclic owners and contiguous column units of 16. The semantic partial
materialization predicate compares hierarchy-axis numbers, while the fused
kernel reads residuals using the projection's complete owner mapping. That
relation therefore required a reshard the selected kernel did not implement.
The provider now exposes a materialized input in the residual layout so edge
Boxing explicitly reduces and reshards it before the local add/statistics
fusion. Direct owner-preserving collective fusion remains available. The
checked-in 32-by-256 BF16x8 reproducer fails before this change; 27 provider,
normalization and semantic neighbors pass afterward, and the full 32-token
prefill artifact now generates successfully. General semantic materialization
and other combine operations retain their existing contracts.

The decode resource regression also exposed a function pool ABI defect at
`codegen/triton/function_pool_abi.py::function_pool_actuals`: every static
layer call passed a different pointer into the same offset table. LLVM
hoisted all those addresses into the entry, and SASS shows their simultaneous
liveness and spill stores before useful computation. ABI v2 passes the
shared table base and a scalar record displacement; each callee forms its
record address where it performs the existing volatile loads. This preserves
the pooled view offsets, alignment proofs, bank-row displacement and reusable
device functions. The full Agent decode check changes from 136 spill-store
and 232 spill-load bytes to zero, with 139 registers per thread and no stack
frame. Fifteen focused ABI tests include distinct pooled views read through
64 real noinline calls, static and dynamic record indices, and alignment
neighbors. The old general pipeline fixture also has stale concrete-variant
selection points; its failure is recorded separately and is not reported as
an ABI test pass.

Two SM90 GEMV variants additionally requested 16 compute warps for the entire
persistent consumer, reducing its fixed per-thread register budget to 96.
The SM90 model now assigns eight compute warps to those same N/K tile shapes,
preserving the larger matrix tiles and split-K geometry. The QKV MMA template
uses the actual execution warp count rather than its minimum consumer count;
the former mismatch rejected a 16-warp Agent artifact during layout
verification. Generated QKV execution at eight and sixteen warps now passes
numerical and zero-spill checks. These are resource/correctness checks; full
request performance measurements and the replacement figures remain pending.
