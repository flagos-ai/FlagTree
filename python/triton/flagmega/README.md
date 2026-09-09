# FlagMega

FlagMega is a Python-native, staged model compiler for FlagTree, organized around
nncase-style IR, rewriting, target policies, TIR and runtime contracts. Python IR
is an executable serialization format for editing and resuming compilation, not
just a debug dump. The compiler can run unattended; a user or agent can also
inspect a checkpoint, edit it, override a selection and resume through the CLI.

`F.nn.rope(value, cos, sin, rotary_dim=64)` represents partial RoPE directly,
without Slice/Concat nodes. `rotary_dim` is a positive even scalar-coordinate
prefix length, bounded by the input head dimension; omitting it preserves full-head
rotation and the existing serialized contract. Tables cover only that prefix.
The tail is copied unchanged. `F.ntt.vectorized_rope` uses the same scalar units
after packing. RoPE promotes inputs and tables independently to FP32, computes the products
and sum in FP32, and rounds only the result back to the input dtype, as nncase
does. `F.nn.rotary_embedding(..., output_dtype="bfloat16")` expresses a rounded
table directly at its producer's store boundary. Thus a numerical profile can
use BF16 input/output and BF16 tables without surrounding RoPE Cast nodes;
normalization's separate rounding boundaries remain unchanged.

## Workspace implementation summary

| Area | Implemented responsibilities |
| --- | --- |
| `importer` | Checkpoint-backed model import; Qwen3 entry/decode function separation during import, embedding, paged attention, FFN, final projection and sampling. |
| `ir` | Per-operation modules; colocated operation semantics and parameter/type contracts; handwritten functional builders; symbolic dimensions, vector/distributed types, graph functions and TIR. |
| `pattern_match`, `rules` | Typed patterns, parameter descriptors, dataflow rewrites and reusable rule/pass-level tests. |
| `egraph` | Typed equality saturation and OR-Tools CP-SAT extraction, including acyclic extraction constraints; graph/cost/pick diagnostics. |
| `passes` | TargetIndependent, TargetDependent, AutoPacking, AutoDistribution and TIR pass groups; real vectorization, packing, distribution and bufferization transformations. |
| `diagnostics` | Pass/stage Before and After directories, per-function names, dump flags, editable Python IR and readable `.il`/`.script` companions. |
| `evaluator`, `artifacts` | Lazy constant islands compatible with graph/egraph rewriting, reference evaluation, readonly-data materialization/cache, section hashes and artifact validation. |
| `targets` | Target-machine capabilities, selection policies, implementation catalogs and launch/package contracts, separated from model import. |
| `codegen/triton` | Kernel-family/variant/platform templates; function-level producer/consumer regions; reusable decode functions, kernel definitions and tensor-map tables. |
| `runtime` | Prepared launches, argument binding, persistent state, descriptors, compiler scratch and explicit resource/spill validation. |
| `serving` | Standalone artifact-only prompt prefill and decode, chat templates, prefix-cache reuse, sampling, interactive CLI and per-turn performance metrics. |

Bufferization models memory spans, aliases, lifetimes and synchronization; it
uses a CP-SAT allocator by default, or verified first-fit for fast iteration. Distribution
uses a 2D mesh in the validated configuration. Ordinary kernels consume local
shards; communication/boxing implementations own the necessary distributed
coordination.

The supporting TLE changes preserve pipe endpoints and synchronization across
noinline calls, carry shared offsets and compiler scratch across function
boundaries, and maintain legal explicit memory layouts and asynchronous copies.
They include standalone MLIR, Python/GPU and C++ regression tests. TLE-specific
native changes are compile-time guarded; the Python/backend hooks are gated.

Repeated cleanup removed unnecessary native changes and the public
`rematerialize_index` and exact-physical-warp-allocation controls. Physical warp
allocation now uses ordinary four-warp padding. Device calls still retain a
fixed register ABI; logical single-warp producer functions remain supported.
The allocator's remaining native delta is 17 lines against the development
base, including guards/comments.

## CLI and editable IR

Use a FlagTree environment built with NVIDIA/TLE support. The package installs
the `flagmega` CLI by default; `python -m triton.flagmega` is equivalent.
`FLAGTREE_FLAGMEGA=0` disables the optional console entry point and its extra
installation dependencies. Core extraction requires `ortools==9.10.4067`, with
no greedy-extraction fallback; templates use Jinja2. GPU execution additionally
needs the corresponding CUDA/PyTorch environment and model checkpoint.

```sh
python -m triton.flagmega --help
python -m triton.flagmega import --model /path/to/checkpoint --output imported.py
python -m triton.flagmega verify imported.py
python -m triton.flagmega inspect imported.py --json
python -m triton.flagmega compile --input imported.py --checkpoint /path/to/checkpoint \
  --target nvidia-sm90 --output /path/to/artifact --emit-executable \
  --work-dir /path/to/dumps --dump-flags all
python -m triton.flagmega artifact verify /path/to/artifact
```

Additional commands include `stage`, `candidates`, `select`, `diff`, `schedule`,
`resume`, `replay` and `artifact run`; use each command's `--help` for its inputs.
Import defaults to the first decode layer; `--full-model` imports the complete
model. JSON CLI output is an inspection/report format, not a replacement for
the editable Python IR. Only load trusted Python checkpoints: loading executes
their Python builders.

Readable `.il`/`.script` dumps use function-local short SSA names on the left
and original node IDs in trailing `name=...` comments. Constants and weight
references are typed inline operands; splats stay compressed and weights are
never loaded for printing. Constant recipes use the same notation. Computations
with shared users remain single SSA definitions, and each function prints its
own body. These display names do not rename the IR or change executable Python
checkpoints, semantic hashes, selection plans, or generated kernels.

Each function groups proven weight/constant preprocessing into an internal
`weights { ... }` section, using `%w0`, `%w1`, etc.; the compute body has its own
`%0`, `%1` numbering. Frozen recipes also live in the consuming function's
weights section. This is only a display grouping, not a scope, scheduling
region, or claim that an operation has already been folded. No extra dump
files are created. Classification follows immutable dependencies and the op's
constant-evaluation/determinism contracts, not names such as `pack` or `weight`.
DumpManager proves constant parameters across all call sites before splitting
function views, and shares that read-only analysis across their renders. A
standalone callee without caller context keeps unproven parameter expressions
in its compute body. Dynamic inputs, effects and physical TIR calls remain in
the execution body.

After post-boundary vector-layout propagation, `PackResultProducers` moves
embedding feature packing into the weight table (nncase Gather propagation)
and makes rotary-table producers return vector elements directly. It rebuilds
each producer once in dataflow order, preserving shared scalar consumers and
the single position-state read. A subsequent egraph pass folds inverse layout
boundaries. No Pack-to-Bitcast normalization is used to hide activation packing.

An executable artifact contains `ir/final.py`, the readable companion
`ir/final.script`, generated source, readonly data and a checked manifest.
Custom implementation catalogs must match the saved target snapshot when
resuming target verification. Regenerate generated source after API changes;
retired names are not silently accepted by compatibility shims.

Before freezing readonly-data recipes, `LiftConstantParameterExpressions`
uses all call sites to prove constant function arguments, including through
nested wrappers. Pure, deterministic, constant-evaluable expressions over
fixed-size immutable tensors move to each caller; the reusable callee accepts
their results through one refined ABI. Weights may differ between calls and
no `packed_from` annotation is required. Shared users, exported parameters and
runtime call order are preserved. Stop at `lift-constant-parameters` (IR stage
`constant_parameters_lifted`) to edit these ordinary Python expressions before
`FreezeConstantIslands`. Frozen or already lowered TIR must be resumed from an
earlier checkpoint to perform this transformation.

Readonly-data verification streams bounded chunks and checks both the whole
image and every indexed entry. One worker hashes the image while the calling
thread hashes entry slices over the same immutable chunk; each chunk completes
before the next read. This overlaps CPU work without caching trust decisions,
skipping hashes, or making a full-image copy. Worker failures are propagated.

### Bufferization optimization levels

Use `CompileOptions(bufferize_opt_level="fast")` or `--bufferize-opt-level fast`
for deterministic first-fit allocation during iteration. Use `optimized` (the
target default) for final memory planning: CP-SAT minimizes pool high-water,
then optionally reduces proven reuse-only synchronization conflicts without
increasing any pool. First-fit supplies an initial placement, hints and upper
bound. All SAT objectives share the configured per-allocation time budget;
there is no address-sum objective. Pool records retain solver status and bounds,
so a feasible placement is not presented as a proven optimum.

Both levels use the same alias/MemSpan analysis, inclusive lifetimes, alignment,
capacity checks, call ABI verification and synchronization realization. Fast
can require more memory or synchronization; it does not silently fall back to
SAT when a pool does not fit. Exact physical allocation problems are cached
within one bufferization run, including baseline/preferred planning and function
specialization. Readonly data remains linearly allocated.

Changing levels requires resuming the Python IR **before Bufferize**, normally
the `PlanFunctionMemory/After` or `Bufferize/Before` directory. Already allocated
checkpoints cannot simply be relabeled: offsets, call bindings, synchronization
and generated source must all be regenerated. Omitting the flag when resuming
an allocated checkpoint preserves its saved level.

For example, compile an imported decode module for iteration, then resume its
complete pre-Bufferize dump with SAT for the final build:

```sh
python -m triton.flagmega compile --input imported.py --output build/fast \
  --bufferize-opt-level fast --work-dir build/fast-dumps --dump-flags compile,pass-ir
python -m triton.flagmega resume --input /path/to/Bufferize/Before \
  --output build/optimized --bufferize-opt-level optimized \
  --work-dir build/optimized-dumps --dump-flags compile,pass-ir
```

The second input is the function-complete `Before` directory produced by the
first command, not its allocated output. Add `--emit-executable` and
`--checkpoint /path/to/checkpoint` to materialize runnable artifacts; these examples otherwise
compile IR and do not load weights or measure device execution. The same level
option is available on `compile`, `resume`, `replay` and `stage`.

AutoDistribution shares exact type-inference, reshard and realized-cost queries
within a search. Fixed choices are propagated through exact compatibility
relations before CP-SAT variables are constructed; candidate IDs remain
unchanged and distinct weights remain independent decisions. `DistributedSearchGraph.dot`
retains all candidates; `Costs/Pick.dot` contains only the selected subgraph.
The pass analysis manager can reuse an unchanged proposal during uninterrupted
compilation. Pausing/resuming creates a fresh manager, and an intervening custom
pass invalidates this analysis unless it explicitly declares preservation.
Custom passes that mutate target/provider policy must not claim preservation.

### Source-runtime numerical contracts

Import may explicitly select a versioned numerical profile. The default
`nncase` preserves the existing import. Full-model Qwen3 BF16 also supports
`vllm-493bd8323-inductor-level3`, describing that revision/configuration's
normalization, residual, rotary-table and activation rounding boundaries:

```sh
python -m triton.flagmega import --model /path/to/checkpoint --full-model \
  --numerical-profile vllm-493bd8323-inductor-level3 --output imported.py
```

The Python API is `import_model(..., numerical_profile=...)`;
`apply_numerical_profile(module, profile)` supports trusted imported checkpoints.
Unsupported profiles, changing an existing contract, and applying a profile to
lowered IR are errors. `compile --input` respects the saved contract; an explicit
conflicting `--numerical-profile` is rejected. The pinned profile currently
requires the reusable `decode_layer` full-model importer, not the legacy
hidden-output layer importer. This is not a numerical-equivalence claim for
arbitrary vLLM revisions or settings.

Profiles are frontend semantics, not codegen model switches. Ordinary
TargetIndependent passes fuse BF16 projections with FP32 GLU intermediates and
FP32 Q/K normalization/RoPE using explicit operation attributes. They preserve
projection/table/final rounding boundaries and cache effects, including values
returned by functions. CLI compilation and edit/resume need no tutorial imports
or agent intervention to obtain these fusions. Generic correctness fixes and
optimizations live here; workload/hardware selection strategies may remain local.

Local fusions are Pattern-based rules grouped in dataflow fixed points:
`DecomposeComplexOps` includes normalization decomposition, wide GLU, final
NormApply casts and QKV/RoPE/cache formation; `FuseDistributedOps` groups the
gather/reduce normalization and QKV variants. Patterns bind shared operands and
private users; callbacks check type/layout, rounding and effect-order legality.
`is_unary_chain` captures arbitrary-length view chains in both dataflow and
e-graph matching. Multi-output/effectful region edits are dataflow transactions,
not e-graph equalities. Late rules use `rewrite_constants=False` to keep frozen
assets and recipes opaque. Legacy fusion stage names resolve to their grouped
stage for checkpoint resume.

`QKVRoPEWithCache.rotary_dim` and its gather/reduce form use scalar coordinates,
independent of vector lanes (omitting it keeps full-head rotation). Normalization
covers the complete head; internal FP32 RoPE rotates only the prefix and retains
the normalized tail. BF16 normalization boundaries, table storage dtype, both
cache writes and sequence advancement remain part of the operation contract.

`HoistCallInvariantExpressions` is a normal TargetIndependent pass with its own
Before/After checkpoints. It lifts pure expressions of identical immutable SSA
arguments across repeated calls, computes them once per caller, and refines the
shared callee ABI. It does not move state reads or infer equal values from names
or types. Mixed-dtype NormApply vectorization and cast-aware MatMulNormStats
fusion preserve projection/residual rounding separately. Generic distributed
inference exposes target-owned leaf layouts at logical-argument use sites,
allowing shard-local casts without changing the originator's external ABI.

`NormApply.output_dtype` expresses a final conversion separately from input
arithmetic/rounding precision. Normalization still rounds in the input dtype
before converting its output; the ordinary private-output cast rule and both
gather/reduce normalization forms preserve this contract. Optional in-place
aliases require compatible physical types, including in reusable callees.
`MatMulNormStats.addend_cast_dtypes` similarly records an ordered residual
conversion chain executed before addition. The packed-matmul lowering can fold
private cast chains with identical endpoint shard types into this epilogue,
removing intermediate buffers without removing numerical rounding or crossing
communication boundaries. Both attributes are omitted for the original default
contract and round-trip through executable Python IR.

## Standalone text serving

```sh
python -m triton.flagmega.serving.chat_cli --artifact "$ARTIFACT" \
  --checkpoint "$CHECKPOINT" --n-predict 128 --metrics-file .local/chat.jsonl
```

This path does not import vLLM or use a Transformers model forward. It loads
tokenizer/config files from the checkpoint and executes a full-vocabulary,
single-token paged-attention artifact for both prompt ingestion and generation.
Prefill is explicitly a compiled-token causal scan, not batched prefill.
The stored numerical-profile name does not introduce a source-runtime dependency.

The interaction conventions follow the [llama.cpp CLI](https://github.com/ggml-org/llama.cpp/tree/master/tools/cli):
system prompts, tokenizer chat templates, streaming replies, and per-response
timings. Use `/reset`, `/stats`, `/help`, `/exit`, or multiline input ending in
`\`. `--prompt` runs one turn; `--interactive` continues afterward. Raw prompt
or token-file completion, greedy or seeded temperature/top-k/top-p sampling,
EOS handling, and explicit context limits are supported. No implicit history
truncation or context shifting is performed.

Metrics separate loading/JIT/graph preparation, prompt evaluation, cached tokens,
TTFT, decode mean/median/p95 latency, decode throughput, total generation time,
and output throughput. JSONL also records memory usage and artifact/serving-source
identities. Decode counts exclude the first output token produced by prefill.
Interactive callbacks are timed; human input wait and setup are not. CUDA Graph
is the default; `--no-cuda-graph` selects eager execution explicitly.

## Latest validated state

The development workspace validated Qwen3-1.7B on NVIDIA H800, including a
complete first-layer path and the full 28-layer decode model, batch 1. This is
not an acceptance claim for Qwen3.8-27B FP8, all shapes or other backends.

- Single-layer three-step CPU/GPU reference checks passed: tokens
  `2176 / 25 / 2096`, minimum logits cosine `0.9999839067`.
- Full-model three-step nncase reference checks passed: tokens `25 / 220 / 16`,
  minimum cosine `0.9996560216`; logits were byte-identical before and after the
  final warp-allocation cleanup.
- Final full-model resources: 12 physical warps, 143 registers/thread,
  172268 bytes shared, 331776 bytes global scratch; zero reported spill-store,
  spill-load, stack and local-memory bytes.
- The final executable instruction section is 261888 bytes, identical to the
  previously measured 12-warp diagnostic variant. The isolated 9-to-12-warp
  comparison used three rounds of 500 samples per variant/context, contexts
  1/128/1024, shared addresses, alternating order and a 256 MiB cache flush.
  Median changes ranged from -0.373% to +0.143%; no material regression was
  observed in that experiment. This is not a universal zero-overhead claim.

Latest targeted validation before publication:

| Validation | Result |
| --- | --- |
| FlagMega codegen/runtime and target implementation-model tests | 760 passed |
| TLE MLIR suite + native allocator test | 101 + 1 passed |
| Targeted TLE Python/GPU tests | 72 passed, 1 skipped |
| Publication CLI, namespace, template and sampling smoke checks | 30 passed |
| Selected native MLIR/LLVM tests | 205 passed, 13 pre-existing failures; unchanged failure set |
| Single-layer/full-model artifacts | Manifest, IR, readonly data and three-step references passed |

These numbers are targeted results, not a claim that the entire Triton test
matrix is green. Tests are under `python/test/flagmega`, `python/test/tle` and
`third_party/tle/{test,unittest}`. For example:

```sh
python -m pytest python/test/flagmega/codegen python/test/flagmega/runtime \
  python/test/flagmega/targets/nvidia/test_implementation_model.py -q
lit -v build/cmake.linux-x86_64-cpython-3.10/third_party/tle/test \
  build/cmake.linux-x86_64-cpython-3.10/test/Conversion/allocate_warp_groups.mlir
```

## Local development records

The original workspace keeps designs, iteration audits, ablation evidence,
benchmark inputs/results and v364 artifacts under ignored `build/flagmega/`.
Its latest records are `nncase_alignment_audit.md`,
`flagtree_change_necessity_audit.md`, `subwarpgroup_performance_v363.md` and
`subwarpgroup_cleanup_v364.md`. These local records, model weights, generated
binaries and retired-test archives are not included in the Git push. This
README provides a version-controlled summary; fresh checkouts must generate
their own artifacts using the source and an appropriate checkpoint.
