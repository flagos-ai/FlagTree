---
name: flagmega-developer
description: Develop and optimize FlagMega models with a coordinated expert workflow, editable Python IR/TIR, custom passes, kernels and overrides, measured choices, and resumable compilation. Use for model compiler optimization and serving-runtime integration.
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

## Agent workflow

For end-to-end optimization and performance regression recovery, read
[Agent flow](references/agent-flow.md). **Orchestrator** owns dispatch, state and
resources; **Architect** owns the joint technical plan and consolidated review;
**Optimizer** integrates implementation; **Validator** independently checks the
artifact and reports acceptance. Ordinary compilation does not require this flow.

Architect and Optimizer each have **Graph**, **Vector**, **Distribution** and
**Kernel** specialists. Obtain four separate baseline assessments for an
end-to-end task, reusing current pinned assessments on resume. Each hypothesis
has one lead and only the affected supporting experts; a `Joint` report cannot
replace four domain assessments. Use the
[expert responsibilities, directories and handoffs](references/expert-workspaces.md)
and [coverage rules](references/agent-flow.md#expert-dispatch-and-coverage).
Keep final review independent of design/implementation; record any delegation
or independence limitation.

Keep one integrated candidate active. Dispatch independent ready analysis,
implementation and review concurrently; refill capacity when inputs become
ready, without whole-wave barriers. Serialize conflicting writes and exclusive
device use, and honor host isolation during protected measurements. Start source
review on frozen inputs alongside author self-checks; consolidate when the
required evidence arrives.

Prioritize the first decisive result using the
[candidate prerequisites](references/agent-flow.md#candidate-checks-and-acceptance):
build, full numerical checks and ordinary screening begin when their own inputs
and execution checks are ready. Pending diagnostic analysis, review paperwork
or figure export alone does not block them. Known execution defects and
`paused_ncu_error` retain their prescribed blocking scope. Acceptance still
requires current independent review, all agreed checks, profiles and reproduction.
Review only changed contracts and reuse valid evidence; use
[bounded coordination](references/agent-flow.md#iteration-cost-and-bounded-coordination)
and [capture preparation reuse](references/kernel-microprofiling.md#capture-preparation-and-reuse)
to reduce measured iteration cost.

Maintain a [progress dashboard](references/agent-flow.md#progress-dashboard)
with completed results, actual running work, and the next actions with owners
and dependencies. Refresh it when adopting a state transition and link it at
handoff; current work and its verification status remain visible even before
the next performance or diagnostic publication.

Kernel development follows the mandatory
[microprofiling flow](references/kernel-microprofiling.md): successful inspected
`ncu` on NVIDIA for baseline and evaluated revisions, actual WG pipeline traces
covering every physical warp in the declared scope, and swimlane SVGs showing WG
stages **and hardware-module activity** (CUDA Core, TMA, Tensor Core or platform
equivalents). KernelOptimizer owns capture/analysis; a separate KernelArchitect
reviews the evidence. Label sampled hardware resolution and evidence type, keep
asynchronous in-flight envelopes distinct, and quantify producer/consumer waits
and overlap. Any ncu error pauses optimization until a successful inspected
capture; obtain sudo access if needed, reusing existing user authorization and
handling supplied credentials privately. Instrumentation cannot replace ncu.
Use the [IKET-derived tracing method](references/kernel-trace-method.md) when
designing or changing instrumentation, including measured overhead and the
NCU-compatible recorder needed for same-invocation WG/hardware traces.

Before choosing a design, affected performance experts follow
[Expert baselines and SOTA targets](references/expert-baselines.md).
Research and pin the strongest applicable implementation, reuse or traceably
port its code/algorithm/mapping rather than start from scratch, and reproduce a
matched baseline. Record license/source lineage, numerical/ABI adaptations and
the changes beyond that starting point. Each hypothesis needs a causal,
falsifiable prediction of whole-workload benefit. Scope SOTA claims to evaluated
implementations; theoretical bounds support planning. An outperform objective
remains open until a strict matched measured win meets the complete contract.

## Workspace, trace and Git

Before creating or resuming a workspace, read
[Workspace layout and maintenance](references/workspace-layout.md).
Orchestrator chooses the model/workload/hardware delivery directory
`DELIVERY_DIR` and initializes each new run at
`WORK_DIR="$DELIVERY_DIR/.local/runs/$RUN_ID"`. Version baselines and trial
revisions; preserve existing runs and record any legacy path mapping.

Keep full optimization traces, agent assignments/handoffs, raw evidence, drafts
and artifacts in `.local/`. Orchestrator maintains the run's overall trace and
state; agents write their assigned responses and evidence. Git-managed files
include reproduction code, required `local_optimizations/`, the public
`optimization-trace.md`, documented results and figures with compact source
data. Before rendering or exporting, follow the user's output layout recorded in
`workspace.json.artifact_layout`; it overrides the default shareable layout.
If results are local and `figures/` is reserved for final performance charts,
write data/history to `.local/results/` and process/WG/operator figures to
`.local/figures/` or the exact run. Keep final iteration/comparison figures in
`figures/`, update links and render commands, and document the local data needed
to redraw them. Otherwise use the reference's self-contained public export
layout. Code reproduction remains independent of private process records.

Every assignment follows the [Markdown report conventions](references/agent-report-templates.md):
full role reports at major handoffs, compact evidence-linked deltas for bounded follow-ups.
Validator, also called **Verifier**, records every performance measurement in
the [iteration performance history](references/performance-history.md), including
screening, rejected candidates and failed attempts. At handoff publish the
one cumulative performance-versus-iteration line chart covering all qualified
iterations, with comparable workload/contract series and explicit acceptance
status. Orchestrator embeds the latest verified chart in the delivery README
and updates its data/reproduction links and coverage caption at each handoff;
do not defer this until a winning candidate or final delivery.
For each execution-based candidate verification, Verifier also saves one
measured per-operator latency SVG and its raw profile/provenance. Register pending
profiling separately from ordinary checks; individual build/preflight commands
are not new verification rounds. Kernel revisions
also retain ncu reports, raw WG/module traces, swimlane SVGs and independent
overlap analysis; model/operator charts do not replace these artifacts. Reserve
attempt IDs before evaluation; assign
an iteration number only after Verifier passes the complete agreed numerical
contract. Numerical failures and partial diagnostics remain attempts; a
numerically correct performance regression still counts as an iteration.
The public `optimization-trace.md` and every
entry must strictly follow the [trace templates](references/agent-report-templates.md#public-optimization-trace).
Run `scripts/validate_trace.py` on the complete staged public trace before
declaring a publication ready, including its index and entry metadata; checking
only a new entry is insufficient. Verify the installed bytes at handoff and
reconcile its measurements/gates with the Verifier ledger.

Experts follow the [mandatory domain checklists](references/expert-checklists.md)
and record check IDs, inspected values/checkpoints, results and evidence in their
role reports. Vector explicitly audits actual `TensorType.dtype` / `VectorType`
lanes and packet geometry, plus justified scalar cases; vectorized op names or
zero adapter counts alone do not prove correct vectorization.

## Establish the contract and baseline

Inspect the installed compiler, codegen backend, target, runtime integration,
and local APIs. Verify the active interpreter's dependencies against repository
requirements before treating an environment-version failure as a compiler bug;
use an isolated task environment when restoring a pinned dependency.
Fix the input/checkpoint revision, numerical profile, shapes, state layout,
execution mode, device, measurement boundary, and requested acceptance rule
before comparing candidates. A source runtime's compiled and custom-op paths
may implement different rounding; its name alone does not define a contract.

Initialize the run as above, set `BASELINE_DIR` to a fresh baseline version
(initially `"$WORK_DIR/baselines/b0001"`), choose `TARGET` and explicitly set
`CODEGEN_BACKEND` (`triton` or `cpp`). Target availability does not establish
support in both backends: check toolchain, runtime binding and required kernel
families/encodings with a small normal-generated build before a full workload.
A typical starting point is:

```sh
python -m triton.flagmega import --model "$CHECKPOINT" --output "$BASELINE_DIR/imported.py"
python -m triton.flagmega verify "$BASELINE_DIR/imported.py"
python -m triton.flagmega inspect "$BASELINE_DIR/imported.py" --json
python -m triton.flagmega compile --input "$BASELINE_DIR/imported.py" --checkpoint "$CHECKPOINT" \
  --target "$TARGET" --codegen-backend "$CODEGEN_BACKEND" \
  --output "$BASELINE_DIR/artifact" --emit-executable \
  --work-dir "$BASELINE_DIR/dumps" --dump-flags compile,pass-ir,egraph-cost
python -m triton.flagmega artifact verify "$BASELINE_DIR/artifact"
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

## Import at the right granularity

When onboarding a new model, or a familiar model's unfamiliar component, import
each component through the coarsest operation that expresses its semantics.
Prefer an existing fused operation; define a new fused operation when none
covers the semantics. Do not express a component as a wide expansion of small
generic ops (per-head slices, unrolled iteration counts, hand-built reductions)
when a single operation can carry it.

A small-op expansion is not a neutral starting point. It multiplies node count
and compile time, hides the numerical contract across dozens of nodes so there
is nothing to validate against a reference, and constrains distribution: an op
whose legality rules reject split axes forces the whole region to stay
replicated, so every chip recomputes it. A fused operation keeps the contract in
one place, gives passes something to match, and lets iteration constructs live
inside a kernel where the IR has no loop of its own.

Check how the latest upstream serving runtimes express the component before
choosing a decomposition; they have usually already found the coarse form and
the layout it assumes. Confirm the checkout is current upstream rather than a
local fork or an old branch — a stale tree supports confident wrong conclusions
about what exists. Read the reference implementation as well as the kernel: it
defines the numerical contract to validate the new operation against, which a
local expansion cannot provide.

Reuse before defining, but do not stretch an existing operation past its
semantics to avoid a new one. A new fused operation follows the usual ownership
rule: the operation and its contracts belong in FlagMega core with regression
tests, while model-specific parameter choices stay in the local package.

## Fuse indexing into consumers

When Slice, Gather, Reshape or layout adapters materialize a tensor only to
feed another operation, read [Consumer indexing fusion](references/consumer-indexing-fusion.md).
Keep the semantic selection, but implement its proven coordinate map in the
consumer's loads. For related state updates sharing a source and an epoch,
consider one operation that owns the complete update and its synchronization.
Choose between an existing consumer extension, a proven storage view and a new
semantic operation according to the actual contract. Verify removed buffers
and data movement; fewer IR nodes alone do not establish an optimization.

## Add or extend IR

Before adding a graph/TIR operation or changing an existing operation's
attributes, result types, or supported layouts, follow the
[IR extension SOP](references/ir-extension-sop.md). Define the per-result
numerical, representation, distribution, and storage contracts, then audit
their path through construction, edit/resume, transformations, lowering, and
device execution. An op definition and a renderer alone do not complete support.

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

The ordinary search may produce an initial IR with one uniform execution mesh,
mapping and resource configuration. This is a valid starting point, not a restriction
on subsequent agent optimization. The agent may edit the typed IR to change
per-op execution meshes, participant/task mappings, program partitions and
per-program resources. Resume before the earliest affected stage and regenerate
distribution, candidate, boundary ABI, buffer/lifetime, synchronization, launch
and cache decisions as needed; never keep stale physical plans or patch emitted
source to implement the edit. Validate the edited complete module through normal
compilation and device execution. A dynamic (`*`) op may use an ordinary or
persistent implementation, but never fuses with other kernels; adjacent fixed
meshes require full mapping identity and the usual fusion legality proofs.

An implementation override is not a post-hoc patch to emitted
`generated_kernels.py` or `generated_kernels.cpp`. Regenerate the artifact from the edited implementation
or registered override through normal codegen, verification, and runtime setup.
Do not bypass legality checks or silently fall back when a requested choice
or override cannot satisfy its contract.

## IR acceptance

Before accepting an imported component, IR extension, representation
optimization, or partition/scheduling change, apply
[IR acceptance and repair](references/ir-acceptance.md)
to the complete affected module after packing/layout propagation and again
after distribution. **Pack, Slice, Cast, Concat and Unpack follow the same rule:
zero standalone nodes in compute bodies, including the entry, loop/region
bodies and reachable callees.** This includes numeric vectorized Cast variants.
A later zero-copy lowering or required rounding does not waive this structural
gate: implement indexing, conversion and layout in the owning producer/consumer
contract. Pure weight/constant transforms remain
allowed with proven constant provenance and offline materialization, not just
a `weights` label. These restrictions apply to compute bodies, not to proven
offline weight recipes or instructions inside the implementing device kernel.

Check producer output layouts, numerical boundaries, function ABI, constant
handling, distribution, effects, and physical lowering together. Diagnose a
remaining Pack, Slice, Cast, Concat or Unpack as an incomplete producer/consumer
contract, propagation path, or representation choice; repair its owner before
acceptance. Record each operation's compute-body count separately, together with
the inspected checkpoint identity and structural/numerical/device evidence,
including any failed items.

Audit physical writers of every result and state update. Broadcast/replicated
values do not authorize overlapping ordinary stores into one canonical shared
span, even when values match. Prove disjoint writes, elect the appropriate
writer on replicated mesh axes, or use a declared synchronization/atomic
contract. Preserve split ownership, private replicas, partial state-update
regions, per-result tuple ownership and all required publication barriers.

Explicitly assess both kernel/program `partition` decisions and the role
partitions inside `warp_specialize`, using the actual lowered IR and execution
scope. Record each role's work, warp allocation and blocking waits. For a
global producer/consumer megakernel, specialization
should cover the outer model/layer execution so the producer can naturally
advance across layers. A function-local specialization that joins before
returning does not establish this global role lifetime. Justify retained cuts
and assess their effects on weight prefetch, communication overlap, buffer
lifetimes and resource use; unexplained
per-op or per-layer splitting fails IR acceptance. Follow the partition checks
in [IR acceptance and repair](references/ir-acceptance.md#assess-partitioning-and-warp-specialization-scope).
Report the partition assessment in a table, including overlap with both
producer prefetch and independent consumer work, and distinguish demonstrated
overlap from unverified intent. Support runtime overlap claims with the actual
WG and hardware-module traces required by the kernel microprofiling flow.

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

Before tuning a bandwidth-bound kernel, check for target-specific memory
characteristics under `references/hardware/<target>/memory-bandwidth.md` — e.g.
[Iluvatar BI-V150](references/hardware/iluvatar-bi-v150/memory-bandwidth.md)
records that vectorized loads should be avoided on that target (vendor-confirmed):
128-bit loads reach only 46% of HBM peak against 82% for scalar loads, because
wide loads are capped at ~1 load instruction per cycle per MP. That file also
explains why the backend's own 128-bit load cap is correct for grid-barrier
megakernels (pinned to one CTA per SM) yet costs up to 1.8x read bandwidth for
ordinary kernels at higher occupancy. Do not port NVIDIA vectorization habits
to a new target without checking its measured load-width behavior, and do not
attribute a read-bound shortfall to HBM before ruling out the issue-rate limit.

Before attributing a hang, fault, or crash to FlagMega logic (especially
anything touching in-kernel grid barriers), check for a target-specific
known issue under `references/hardware/<target>/known-issues.md` — e.g.
[Iluvatar BI-V150](references/hardware/iluvatar-bi-v150/known-issues.md)
documents a CUDA lazy-module-loading fault that looks exactly like a
probabilistic barrier livelock but is a driver/environment issue with a
required one-line workaround, not a compiler or kernel defect. Add a new
file under `references/hardware/<target>/` when a future investigation roots
out another hardware- or driver-level issue or performance characteristic on
any target.
For repeated layers and indexed checkpoint parameters, read
[Structured repetition and parameter banks](references/structured-repetition.md).
It defines the closed `control.for`/`Yield` contract, bank schema, loop
distribution, and buffer/codegen acceptance rules.

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
required local implementations, accepted generated source, documented results
with measured SVGs and their compact source data, and `optimization-trace.md`
summarizing meaningful trials and decisions, including rejected candidates.
Describe agent gains relative to the current FlagMega baseline, not core
capabilities as local agent optimizations. Keep detailed run traces, agent
interaction documents, raw results, tutorial tests and intermediate snapshots
under gitignored `.local/`; core regression tests remain in the compiler's
normal test tree. Public reproduction must work without the author's private
`.local/` files.

Do not discard rejected trials or raw evidence merely to clean the public
view. Document reproduction steps, the applied optimization, measurement
boundaries, and final results; follow the user's requested language and scope.
