---
name: flagmega-developer
description: Develop and optimize FlagMega models through editable Python IR/TIR, custom passes, new kernels or in-place kernel overrides, measured choices, and resumable compilation. Use for agent-driven compiler optimization and serving-runtime integration.
---

# FlagMega developer

## Scope and ownership

Treat the optimization unit as **model × workload × hardware × runtime ABI**.
The agent may edit or define IR and TIR, register passes and fusion strategies,
write kernels, override existing kernel implementations in-place, and choose
distribution or scheduling plans. Existing candidates are a starting point,
not the limit of the agent's authority within the requested optimization task.
The ordinary compiler must also complete compilation without agent intervention.

Keep workload-specific decisions and implementations in a local importable
package. Put reusable numerical contracts, correctness fixes, and generally
valid transformations in FlagMega core with core regression tests. Do not
encode workload identity in generic codegen or turn a measured local choice
into an unconditional compiler default. Preserve native compiler ownership
boundaries; changing a FlagMega kernel does not itself require a Triton/TLE
native compiler change.

## Establish the contract and baseline

Inspect the installed compiler, target, runtime integration, and local APIs.
Fix the input/checkpoint revision, numerical profile, shapes, state layout,
execution mode, device, measurement boundary, and requested acceptance rule
before comparing candidates. A source runtime's compiled and custom-op paths
may implement different rounding; its name alone does not define a contract.

Set `WORK_DIR` to a task-local output directory and `TARGET` to a supported
backend. A typical starting point is:

```sh
python -m triton.flagmega import --model "$CHECKPOINT" --output "$WORK_DIR/imported.py"
python -m triton.flagmega verify "$WORK_DIR/imported.py"
python -m triton.flagmega inspect "$WORK_DIR/imported.py" --json
python -m triton.flagmega compile --input "$WORK_DIR/imported.py" --checkpoint "$CHECKPOINT" \
  --target "$TARGET" --output "$WORK_DIR/baseline" --emit-executable \
  --work-dir "$WORK_DIR/baseline-dumps" --dump-flags compile,pass-ir,egraph-cost
python -m triton.flagmega artifact verify "$WORK_DIR/baseline"
```

Inspect `--help` for importer-specific options, including `--full-model` and
`--numerical-profile` when applicable. Start with a representative subgraph
when useful, then validate the complete requested workload. Preserve the
original import, a correct baseline from the current compiler, and any prior
accepted performance artifact as a separately labeled regression reference.
If a general optimization moves into core, rebuild both sides with that core;
do not disable it or reuse a historical slow baseline to inflate agent gains.

Python checkpoints use real `fm.Module` / `F.math.*` constructors and typed
TIR. They are trusted executable input; `.il` and `.script` are readable
companions, not substitutes for edit/resume. `Before` and `After` are
function-named dump directories. Resume a whole module or complete directory,
including callees and local extension imports.

## Agent control surface

Choose the level that expresses the hypothesis; new TIR and kernels do not
require exhausting every existing candidate first.

- **IR:** use typed constructors; define new operations with named
  `input_parameter(TypePattern)` parameters, colocated inference, evaluation,
  effects, and cost information. Keep handwritten snake_case builders and
  pattern helpers discoverable. Python emission must import local definitions.
- **Passes and fusion:** use `RewriteRule`, `DataflowPass`, `EGraphRulesPass`,
  or `FunctionalPass` through `PassManager`. Keep before/after dumps and add
  a stage when a new resume boundary is needed. Preserve numerical boundaries,
  shared users, effects, and aliasing rather than relying only on algebra.
- **TIR:** construct or rewrite `PrimFunction`, typed buffers, regions, and
  kernel calls in `ir/tir`; the agent may also define new TIR operations or
  kernel families. Supply the necessary traversal, serialization, verification,
  effect/memory analysis, and lowering support. Declare memory spans, aliases,
  workspaces, synchronization, transfer lifetimes, and caller/callee contracts
  so the normal planner can reason about the implementation.
- **New kernels:** author implementation source or Jinja templates, register
  the applicable candidate/lowering/rendering path, and provide explicit
  semantic, layout, capability, and resource contracts. Custom TIR can lower
  to existing primitives or have its own complete codegen path.
- **In-place overrides:** an agent may replace an existing kernel's source,
  template, or renderer, including under its existing implementation identity.
  A compatible replacement need not invent a new op, family, or candidate ID.
  Use the actual replacement/injection path; do not assume duplicate registry
  entries override earlier definitions. Confirm the emitted implementation.
  Keep the override explicit, scoped, and reproducible from the optimization
  package or authoritative implementation source; load it before the affected
  compilation stages in every fresh process. If applicability or ABI changes,
  update the contracts and regenerate dependent plans. Record the override's
  source identity so an unchanged IR hash cannot reuse an old executable.
- **Selection:** inspect proposals and emit `SelectionPlan` against the current
  semantic hash. For coupled distribution changes, solve with only the intended
  agent constraints fixed, then emit a complete consistent plan. Proposal
  defaults may already pin other choices; replacing one record does not unpin
  them. Use target capabilities and actual types to express applicability.

An implementation override is not a post-hoc patch to emitted
`generated_kernels.py`. Regenerate the artifact from the edited implementation
or registered override through normal codegen, verification, and runtime setup.
Do not bypass legality checks or silently fall back when a requested choice
or override cannot satisfy its contract.

## Iteration and acceptance

For performance tuning, regression recovery, changes to numerical/storage
contracts, or final measurement, read
[Performance iteration](references/performance-iteration.md). It covers resume
invalidation, physical ABI pitfalls, isolated correctness checks, and evidence
needed to attribute an improvement.

For redundant Pack/Unpack, Cast, mixed-precision outputs, or broken fusion
chains, also read [Representation optimization](references/representation-optimization.md).
It covers producer-owned layouts, packet geometry, conversion semantics, and
the difference between removing an IR node and removing runtime work.

Use an immutable trial per hypothesis: inspect the costly boundary, make a
scoped change, resume from the earliest affected stage, verify, and measure.
Include local TIR/kernel overrides in the trial's reproducible inputs. Keep
previous results rather than overwriting an apparent loser, and distinguish
compilation time from runtime improvement. Agent measurements can guide choices
without a precise cost model; estimates do not replace legality or acceptance.

Core fixes require a causal location and failing-before/pass-after regression,
including neighboring rules, passes, or implementations that share the same
contract. Cover actual device kernels separately from CPU tests and mocks.
Respect the requested correctness and performance criteria; do not loosen the
reference configuration or claim unmeasured workloads to make a trial pass.

## Reproduction and handoff

Before finalizing, rebuild from the original input plus saved optimization
scripts, TIR/kernel definitions, overrides, and plans in a fresh process. Verify
that the artifact being delivered is the one whose correctness and performance
were measured. Source alone is not a runnable artifact: execution also needs
final IR, readonly data, the manifest, and a compatible compiler/runtime.

For tutorial deliverables, keep public files focused on reproduction scripts,
required local implementations, accepted generated source, and documentation
with measured SVGs. Describe agent gains relative to the current FlagMega
baseline, not core capabilities as local agent optimizations. Keep experiment
notes, raw results, tutorial tests, and intermediate snapshots under gitignored
`.local/`; core regression tests remain in the compiler's normal test tree.
Public reproduction must work without the author's private `.local/` files.

Do not discard rejected trials or raw evidence merely to clean the public
view. Document reproduction steps, the applied optimization, measurement
boundaries, and final results; follow the user's requested language and scope.
