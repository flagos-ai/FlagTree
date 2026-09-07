# FlagMega

FlagMega is a Python-native, staged model compiler for FlagTree, organized around
nncase-style IR, rewriting, target policies, TIR and runtime contracts. Python IR
is an executable serialization format for editing and resuming compilation, not
just a debug dump. The compiler can run unattended; a user or agent can also
inspect a checkpoint, edit it, override a selection and resume through the CLI.

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

Bufferization models memory spans, aliases, lifetimes and synchronization; it
uses a CP-SAT allocator rather than a placeholder memory plan. Distribution
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

An executable artifact contains `ir/final.py`, the readable companion
`ir/final.script`, generated source, readonly data and a checked manifest.
Custom implementation catalogs must match the saved target snapshot when
resuming target verification. Regenerate generated source after API changes;
retired names are not silently accepted by compatibility shims.

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
