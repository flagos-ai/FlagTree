# Agent flow

Use this flow for end-to-end model optimization and performance regression
recovery. It coordinates the existing compiler controls and acceptance rules;
it does not require changes to the compiler CLI or a permanently running agent
service. A focused implementation task can use only the relevant roles.

## Main roles

| Role | Responsibility | Handoff |
| --- | --- | --- |
| **Orchestrator** | Initialize the task workspace; maintain state, provenance, resource assignments and budget; dispatch the next workflow action based on evidence. | Task contract, state and decisions. |
| **Architect** | Diagnose the whole workload; consult domain architects; reconcile hypotheses, contracts and dependencies into one technical plan; organize independent domain implementation reviews and consolidate their verdict. | Joint hypothesis, expert assignments and consolidated review. |
| **Optimizer** | Coordinate domain optimizers; implement in dependency order; integrate changes; resume normal compilation and perform implementation self-checks. | Reproducible change set, integrated artifact and implementation record. |
| **Validator** | Independently check the artifact as execution prerequisites become ready; verify current review coverage and all required evidence before acceptance. | Acceptance results, raw evidence and failure classification. |

Orchestrator chooses the next workflow action; Architect owns technical
prioritization and whole-workload tradeoffs. Optimizer may report implementation
discoveries that require a plan revision. Validator reports evidence without
changing candidate code or relaxing acceptance thresholds. If a test or
reference is defective, assign a separate causal repair and reestablish the
affected baseline before attributing a candidate's result.

The coordinating agent normally acts as Orchestrator. With delegation available
and allowed, dispatch the other roles and relevant specialists as separate
agents. Reuse their context or send concise handoffs rather than launching all
roles and specialists for every trial.
Respect the environment's concurrency limit. When delegation is unavailable,
perform the same logical handoffs and state the limitation on independence;
self-checks alone are not independent acceptance.

### Parallel dispatch

Dispatch independent ready assignments concurrently, up to the available slots
and resource budget. This applies to domain analysis, disjoint implementation,
independent reviews and CPU evidence processing. A single integrated candidate
does not require serial execution of all work contributing to it.

Orchestrator records ready, running and dependency-waiting work in the current
state, with the actual assignment and awaited input/resource for each wait.
Architect supplies technical priorities and Optimizer supplies implementation
dependencies and source ownership; Orchestrator performs the actual dispatch.
Ready work within the agreed plan needs no additional approval relay through
these coordinating roles. Changed contracts still return to their owners.
When a task finishes, immediately use the free slot for the next ready task;
do not wait for unrelated tasks or a whole wave. Start a review when that
review's complete immutable input is ready, while unrelated work continues.
Consolidation still waits for every required domain result.

When an agent has finished all independent work and only awaits another
assignment's result, save its intermediate evidence and end the active turn.
Mark the assignment as waiting, release its execution capacity, and resume the
same context with a follow-up when the input arrives. Do not keep an active
worker polling for dependencies, or mark its unfinished verdict complete.

Serialize only actual dependency edges, conflicting source writes, exclusive
device use or work that would violate the measurement/isolation contract.
Independent CPU preparation can continue around GPU work when resource
contention will not invalidate its timing or monitoring. Keep heavy competing
host work out of protected timing windows. Use disjoint assignment directories
or worktrees for parallel mutations, and reserve shared-state/ledger/public
writes for their designated single writer. A mere difference in role or phase
is not a reason to serialize otherwise independent tasks.

## Candidate checks and acceptance

Use two connected streams: obtain a decisive result on the ordinary candidate,
and complete the evidence needed to explain and accept it. The execution loop
below defines responsibilities, not a global barrier after every numbered step.
Baseline diagnosis and the four initial domain assessments still apply. Reuse
them when the current inputs and contracts support that reuse.

| Work | Actual prerequisites | Work that can proceed alongside it |
| --- | --- | --- |
| Normal native/artifact build | Frozen implementation and valid compiler inputs; correct invalidation; available build resources | Independent reviews, fixture preparation, recorder analysis |
| Ordinary resource/ABI and numerical checks | Runnable artifact with matched manifest/source/library/readonly data; relevant structural, ownership, synchronization and launch prerequisites; device lease | Domain reviews and CPU diagnostic preparation |
| Ordinary performance screening | Required numerical scope for the comparison, valid resources, frozen timing contract and isolated device | Remaining profile analysis, render/export work and reviews |
| Candidate ncu and WG/module capture | Baseline ncu success, capture-specific source/resource/correctness admission, valid scope/timebase and exclusive device lease | Ordinary CPU work and reviews; GPU jobs remain sequential |
| Interim result publication | Verified measurements, complete NUM qualification for numbered iterations, portable provenance and explicit pending-gate status | Remaining capture, analysis and review; publish provisional results without claiming acceptance |
| Final acceptance and delivery | Current independent domain verdict, complete agreed NUM/resources/performance, required ncu/WG/module/perturbation evidence, reproduction and applicable export checks | No acceptance claim until these gates close |

Before dispatch, Architect names the next falsifying check, expected whole-workload
benefit, rough execution cost and stopping condition. Optimizer supplies a runnable
artifact path and exact build/validation commands, including extension bootstrap
and readonly-data lineage. A source tree or standalone `.so` is not a complete
model artifact. Prepare runtime fixtures and validation inputs during implementation
so missing paths are discovered before a device window is reserved.

Keep reusable measurement entrypoints independent of assignment IDs. Bind the
candidate, parent, source, artifact, cache and complete launch environment through
an explicit input packet. Reuse the frozen measurement method and its review
when only those data bindings change; check the current bindings and affected
deltas without copying or re-auditing unchanged harness code. Before declaring
an execution packet ready, include and validate the environment inherited from
its original launcher, including required libraries, import roots and caches.
An earlier assignment's shell environment must not be an implicit prerequisite.
Before a costly model load or device window, run the available CPU portion of
the actual consumer's validation against the concrete packet. Check required
field shapes and meanings as well as hashes: for example, a native API binding
may require exact ELF symbol rows, whereas a kernel inventory contains metadata.
Reuse this preflight when its packet and consumer are unchanged; a binding repair
reopens only the affected checks and retains the failed execution record.

When normal runtime preparation validates every entry's ABI, bindings and
resource admission before the first compute launch, retain that evidence as
the execution prerequisite in the same numerical/screening invocation. Confirm
that failures prevent launch. A separate resource probe is useful when it is a
cheaper falsifier or addresses a distinct risk; do not require it solely to
repeat the same artifact loading, typed verification and resource checks.

Do not treat a missing report as an execution defect. Source self-checks and the
affected owners must establish the prerequisites for the proposed execution;
known memory, synchronization, numerical or artifact defects block affected work.
Record the prerequisite and its evidence, rather than adding a general approval
wave. Independent reviewers can inspect a frozen revision while it builds or runs
provisional checks. Validator must independently check current evidence before
qualification or acceptance; implementer self-checks do not become independent
passes merely by reusing their result files.

A recorder repair, perturbation study, optional diagnostic fixture, broad trace
gallery or pending figure does not automatically block an uninstrumented candidate
that does not depend on it. Full-model NUM and a leaf witness may run in either
order when their own execution prerequisites hold; both remain required if the
task contract requires both. If a frozen plan imposed a broader dependency,
Architect records a scoped amendment with its technical reason before changing
the order. Preserve that plan and all failed evidence.

Bind each prerequisite to the operation or claim that consumes it. For example,
trace clock calibration gates timestamp interpretation and overlap claims; it
does not gate numerical checks with recording disabled when those checks do not
consume the trace clock. A renderer-only change requires review of the affected
evidence binding and rendered output, without rebuilding an unchanged executable
or repeating its NUM. Deliver independently reviewable subsets as soon as they
are frozen; keep the remaining required checks assigned and pending.

This scheduling rule never overrides `paused_ncu_error`: an actual ncu failure
still stops optimization/build/benchmark dispatch until a successful inspected
capture. Schedule each evaluated Kernel revision's mandatory ncu and complete
declared-scope WG/module capture in the same evaluation cycle as its ordinary
checks; do not turn screening into a sweep of unprofiled candidates. Pending
perturbation or analysis limits trace-based conclusions and keeps microprofile
acceptance open. It does not invalidate separately obtained ordinary timings.

Declare screening scenarios/repeats and rejection criteria before execution.
Run the cheapest check that can reject the hypothesis, then the complete agreed
numerical contract before the first qualified full-workload comparison. Preserve
partial and rejected results as attempts; neither a short test nor a timing win
qualifies an iteration. Record screening as provisional, including any missing
review/profile gates. Repeated formal measurements and final reconstruction
follow the original acceptance contract.

When an existing harness validates a scenario and then times that same validated
trajectory, use that sequence once; do not add a redundant full-model NUM-only
run as a procedural stage. Keep required resets and timing exclusions. Results
from early scenarios remain provisional until every required scenario passes.

Use one linked evidence record per result, with concise handoffs that reference
it. On a repair, review the actual changed contracts and verify lineage for the
rest; reuse unchanged inventories, source research, raw-data audits and render
inputs. Rebuild/retest only invalidated stages, while preserving required full
NUM for a changed executable. Re-render only affected views. A clock, decoder,
aggregation or source change invalidates the corresponding cached evidence.

### Iteration cost and bounded coordination

Use the runner's stage timings to separate build, artifact loading/runtime
preparation, device execution, capture, analysis/review and export. Report time
to the first decisive NUM/timing result separately from final evidence completion.
Wall-clock time inside `load_prepare` alone does not identify hashing, I/O or
driver overhead; add a narrow stage boundary only when that distinction changes
the next action. If preparation dominates, assess reuse before scheduling more
identical loads; follow the [capture preparation rules](kernel-microprofiling.md#capture-preparation-and-reuse).

Orchestrator keeps the current ready/running/waiting work and each wait's actual
dependency and owner. A running command stays one job: retain its tool handle,
poll its output/completion promptly, and use its existing log instead of starting
duplicate commands or asking agents to rediscover its status. If elapsed time
exceeds the estimate, make a bounded status diagnosis; elapsed time alone is not
proof that a job hung or permission to kill unrelated work.

Reviewers audit the changed implementation and evidence, then hand off a decision.
Consolidation checks coverage, lineage and unresolved findings; it does not repeat
each domain audit. Reuse a completed independent raw-data audit when its inputs,
decoder and checks are unchanged. Another handoff or report-format change alone
does not justify rereading large payloads, repeating NUM or reviewing a review.
New findings or invalidated contracts reopen only their affected checks.

Distinguish payload-audit reuse from routine source and small IR metadata reads.
If a review harness fails, preserve the failure, fix the harness and rerun the
affected check using the authoritative typed objects. A necessary retry may
reload small frozen metadata; do not invent a second parser or seek approval
solely to satisfy a self-imposed one-load limit. Cache the successful derived
tables afterwards. This does not authorize model/payload reads, another raw-data
audit, a full compiler rerun or a repeated device measurement.
Likewise, avoiding duplicate descriptor probes does not forbid the descriptor
generation, source rendering or native compilation required inside the ordinary
artifact build. Run that normal path when needed; do not add an artifact-assembly
shortcut merely to avoid executing one of its required stages.

A follow-up names one unresolved question, the evidence needed to resolve it and
a stopping condition. When the bounded work ends, record the conclusion or the
remaining uncertainty and choose the next action; do not extend investigation
merely to fill reports. Pending evidence retains its owner and completion
condition. Required checks remain required, including for rejected candidates.

### Progress dashboard

For an end-to-end run, Orchestrator maintains `DELIVERY_DIR/dashboard.html`
and links it from the delivery README. Show completed work and its result,
actual active assignments with their experts, and the next ready or waiting
work. Keep waiting work separate from actual execution. Each wait names its
dependency, owner and concrete next action. Put the current candidate's change,
verified result and next decisive check near the top, even when publication
still describes an older candidate. Include the latest performance
change and its acceptance status, required-check coverage, and links to the
performance history, operator profile and available WG/hardware swimlanes.
Keep diagnostic progress distinct from measured optimization gains; a rejected
candidate or unfinished matched comparison must remain visibly identified.
When a complete matched comparison is independently qualified, show both
implementations' results, the measured gap, scope and evidence link even when
the candidate loses. Replace any earlier "gap unknown" status at that transition;
keep the latest comparison separate from older published screening curves.

Generate the dashboard from current dispatch state and verified evidence when
work is dispatched, starts, terminates, is reviewed or is published. A reserved
command is not running, and a completed capture is not yet an accepted analysis.
Ingesting a terminal receipt, removing its work from ready/running lists, and
refreshing the status snapshot are one orchestration step. On resume, reconcile
persisted tool handles and prepared commands with their terminal receipts before
displaying current work; an old preparation must not reappear as the next task.
Use a bound terminal receipt over a stale running flag for the same execution;
page generation time alone never establishes that execution is still active.
Distinguish failed executions from successful captures awaiting review. When a
published evidence snapshot trails current dispatch, show its cutoff and keep
the current assignment status separate from that historical review status.
Keep Graph, Vector, Distribution and Kernel visible in an expert coverage view
with their responsibilities, assigned run-relative work directories and review
status, including unchanged domains whose evidence is reused. List actual active
assignments separately; an existing domain report is not a running agent.
Move terminal assignments out of the active list at the same transition. Keep
a short list of recent completions with an owner, concrete result and a pinned
receipt, including work awaiting later acceptance; do not show only the last
published milestone. Ready tasks also name their owner and the next action.
Include the next performance implementation when one has been selected; label
planning, implementation, measurement and acceptance separately. Show the chosen
device execution order while identifying CPU work that can proceed concurrently.
Keep completed cards about the result achieved; remove superseded future-work
sentences from this mutable view and put current dependencies in the next/waiting
list. Preserve the original immutable reports. Collapse older completions so
the active work and next decision remain visible without scrolling through history.
Show available stage durations and their measurement boundaries, especially
time to the first decisive result versus remaining profiling/review/export.
Use actual runner timestamps; mark missing timing as unknown. Command wall time
includes preparation and teardown unless separately measured, and parallel task
durations must not be summed into an iteration's elapsed time. Record each active
task's timezone-aware start timestamp and calculate its elapsed time at the
snapshot timestamp; show unknown when no start was recorded. A static page must
not keep increasing a task timer after its last observation. Do not invent progress
percentages or completion estimates. Slow or waiting work names the next action that can
resolve it, using the bounded-coordination rules above.
When existing receipts contain substeps, show the command total and its main
substeps together with the execution ID. Bind them to that same execution and
make clear that substeps are included in the total. Reuse these records without
launching a new capture just to populate the dashboard.
Show the snapshot timestamp and actual update behavior; do not imply live
monitoring for a static page. Retain a compact public status snapshot and a
renderer that works without private run files. Export only the status fields
and evidence links needed by the reader, excluding credentials and private
environment contents. Write each snapshot and its self-contained HTML using
atomic replacement so a page reload cannot read an incomplete file.
Refreshes are small Orchestrator bookkeeping operations;
they must respect protected measurement windows and not add a review barrier.
At a user handoff, refresh after adopting terminal worker results and give a
direct dashboard link. The main status banner describes actual running work;
put unmet downstream dependencies in the waiting list, unless a failure gate
has explicitly paused the current work.
Keep historical status in its original result record. The dashboard summarizes
the current state: replace superseded task statuses, and use the latest complete
pending-gate snapshot instead of concatenating older todo lists. Each publication
must retain the complete set of unresolved gates, including those outside its
own scope. Ordinary performance and diagnostic releases may advance separately;
bind each card and its counts to the corresponding verified release.
Link the latest published gallery for each retained candidate as well as the
current candidate, with candidate identity and scope coverage kept separate.
Readers should be able to open newly completed swimlanes directly from the dashboard.
When dispatch has advanced beyond the latest publication, maintain its complete
current pending-gate list alongside ready/running/waiting work and prefer it for
the live-work summary. Fall back to publication gates only when that current
list is unavailable. Updating progress does not change historical verdicts or
promote an unpublished candidate to an accepted result.

## Expert groups

Architect consults domain architects; Optimizer coordinates the corresponding
domain optimizers. These are responsibilities activated on demand, not a
requirement to keep eight additional agents running.

| Group | Agent names | Owned contracts and changes | Performance evidence |
| --- | --- | --- | --- |
| **Graph** | `GraphArchitect`, `GraphOptimizer` | Coarse semantic operations, graph rewrites/egraph rules, fusion, shared users, effects, numerical boundaries and constant handling. | Changed FLOPs, materialized bytes, repeated computation and effect on the critical path. |
| **Vector** | `VectorArchitect`, `VectorOptimizer` | Vectorization, packing, result lanes, representation propagation, producer/consumer layouts and function-boundary layout contracts. | Memory transactions, load/store issue rate, relevant memory bandwidth, alignment and resource effects. |
| **Distribution** | `DistributionArchitect`, `DistributionOptimizer` | Execution meshes, task/participant mapping, sharding, partial reductions, Boxing/collectives, ownership and distributed dependencies. | Per-participant load, communication latency/bandwidth, publication/synchronization and demonstrated overlap. |
| **Kernel** | `KernelArchitect`, `KernelOptimizer` | TIR, kernel recipes and microkernels, source/template overrides, tile/warp configuration, pipelines, memory/lifetimes, synchronization, partition and program scheduling. | Mandatory ncu counters, actual per-WG and hardware-module traces/swimlanes, producer/consumer overlap and waits; applicable compute/memory ceilings, instruction mix, resources and residency. |

Expert boundaries follow contracts, not one pass or directory per agent.
Use the [mandatory expert checklists](expert-checklists.md) for concrete checks
and evidence, including actual VectorType coverage and justified scalar tensors.
Assignments and role reports identify check IDs, inspected scope and findings;
unchanged domains require pinned lineage rather than an assumed pass.
Fusion can cross graph, post-distribution and TIR stages. Graph owns semantic
legality; Distribution owns the required owner/reduction relations; Kernel
owns physical realization; Vector owns the representation changes. Read the
current pipeline and target extension points before choosing resume boundaries.

Assign one `lead_expert` and only the necessary `supporting_experts` to each
joint hypothesis. A local recipe change can use Kernel alone; a result-layout
change typically needs Vector plus affected Distribution/Kernel experts. A
new fused operation requires the experts whose construction, numerical,
representation, distribution and lowering contracts change. Follow
[IR extension SOP](ir-extension-sop.md),
[Representation optimization](representation-optimization.md) and
[Consumer indexing fusion](consumer-indexing-fusion.md) when applicable.

For a projection/activation fusion, Graph proves the rounding and activation
semantics, Vector specifies the producer's output layout, Distribution proves
where the complete reduction is available, and Kernel implements the combined
operation within its resource budget. Architect evaluates the combined critical
path; local gains from these experts cannot simply be added together.

### Expert dispatch and coverage

Use [Expert responsibilities, workspaces and handoffs](expert-workspaces.md)
for exact domain boundaries, canonical directories and dispatch/contract/report
templates.

For an end-to-end optimization task, obtain separate GraphArchitect,
VectorArchitect, DistributionArchitect and KernelArchitect assessments of the
same baseline before the main Architect selects the cross-domain optimization
plan. A resumed run can reuse current, pinned domain assessments; identify
missing or stale coverage and obtain only those assessments. On one GPU,
Distribution still assesses execution meshes, CTA/task ownership and reductions;
single-device execution alone is not a reason to omit it.

When delegation is available and allowed, dispatch each required specialist in
its own domain context with a durable assignment and domain report. One request
to "act as Graph/Vector/Distribution/Kernel" and one `domain: Joint` response do
not establish four specialist assessments. A specialist may inspect interfaces
and collaborate across domains, but its report does not silently substitute for
another domain's owner. Joint reports belong to coordination and consolidation.

"On demand" selects affected domains and the current phase; it does not merge
required experts to fit the concurrency limit. Schedule domain assessments,
implementation and fresh independent review contexts in waves. For example,
with four total slots, Orchestrator can dispatch three domain assessments and
schedule the fourth when a slot becomes available. There is no requirement to
run all eight Architect/Optimizer specialists together or to invent work for
unaffected domains. If delegation is genuinely unavailable or prohibited,
record the logical-domain work and independence limitation explicitly.

Orchestrator maintains `expert-coverage.json` for the current baseline or
candidate, binding its exact input/IR/artifact identities. Keep all four domain
entries, with:

- `disposition`: affected, unchanged, not_applicable or unassessed, with the
  contract-based reason; unassessed is unfinished work.
- The actual architect, optimizer and independent-review bindings needed for
  the current phase: agent/context identity, role, domain, assignment ID and
  request/report pins. A planned name without an actual dispatch is pending.
- Current findings, evidence and next action. Unchanged scope links the prior
  domain report and proves its input/contract lineage; a generic review or a
  copied checklist is insufficient.

Use domain-and-role names for newly dispatched expert tasks, such as
`graph_architect`, `vector_optimizer` or `distribution_review`, with a purpose
suffix where useful. For legacy agent aliases, record the actual role/domain
binding in the assignment and coverage record. Names aid visibility; the
assignment, inspected evidence and report establish the work performed.

Before implementation, the main Architect reconciles all four domain entries
and identifies the affected contracts for Orchestrator's optimizer dispatch.
A cross-domain discovery reopens the relevant entry and expert handoff.
Before acceptance, Validator
checks current domain review coverage and independence. Narrow helper work,
such as kernel trace export, can remain Kernel-only while the enclosing
whole-model coverage record retains the other domains and their status.

## Implementation review

As implementation inputs freeze, the relevant domain Architects review them
alongside Optimizer's self-checks, before Validator's acceptance stage. Review is
an Architect responsibility, not a fifth main role. Orchestrator assigns review
work; the main Architect consolidates domain findings and cross-layer decisions.

Use independent `GraphArchitect`, `VectorArchitect`, `DistributionArchitect`
and/or `KernelArchitect` instances or fresh review contexts that did not design
or implement this candidate. Provide the hypothesis, actual diff, complete
affected IR/TIR, generated implementation, target/recipe identities and self-check
evidence. Reviewers inspect these inputs rather than relying only on the
Optimizer's summary. Start a fresh review context from this artifact handoff,
without inheriting the design/implementation conversation. Under limited
concurrency, start reviews when their immutable inputs and agent capacity are
ready; do not wait for unrelated implementation workers. Independent source
checks can run alongside author self-checks on the same frozen revision; waiting
for an author's test script or completed report is unnecessary when the reviewer
already has the source and contract needed for its own check. Only verdicts that
consume missing execution evidence wait for that evidence. Final admission still
requires the applicable self-checks and independent review. If independent
review is unavailable, record that unmet gate; self-checks or a reused author's
context cannot be labeled an
independent review pass. Diagnostics can still proceed without claiming acceptance.

Review the domains affected by the contracts and their dependencies:

- Graph: semantic granularity, fusion legality, numerical/rounding boundaries,
  effects, shared users and constant provenance.
- Vector: actual producer/consumer lanes, packing propagation, conversions,
  alignment and function-boundary representations.
- Distribution: owner/task mapping, partial reductions, required Boxing,
  publication, communication and cross-rank dependencies.
- Kernel: selected TIR/recipe and emitted code, physical ABI, memory/aliases,
  transfer lifetimes, synchronization, partition, resources and cache identity.

The main Architect verifies integration, plan deviations and the conditional
whole-workload benefit. Reviewers also assess resume/invalidation correctness
and whether regressions and self-checks exercise the changed contract without
weakening the reference or thresholds. Distinguish unmeasured performance and
pending device evidence from proven correctness or speed. Review establishes
implementation coverage; measured acceptance remains a separate gate.

Check coverage requires evidence or an explicit limitation for every assigned
check; do not require every row's literal result to be `passed`. A source/artifact
review can pass while device-only benefit measurements such as G6/K6 remain
`not_run`, with the numerical/resource/performance acceptance gates still open.
Missing required implementation or contract proofs remain blocking findings.
For Kernel performance work, pending K6/K10 evidence does not by itself block
ordinary build/NUM/screening with satisfied execution prerequisites, or an admitted
diagnostic capture. It cannot close microprofile acceptance or override the
mandatory ncu failure pause. Apply the
[kernel capture and review gates](kernel-microprofiling.md).

Record reviewer identities/domains and independence, the reviewed hypothesis
revision and source/IR/artifact identities, evidence paths, findings and their
dispositions in `review.json`. Use explicit states such as `changes_requested`,
`passed`, `not_run` or `unavailable`. The consolidated verdict passes only when
all required domains are covered and blocking findings are closed. Validator
checks that this verdict covers the artifact it is about to accept.

Implementation findings return to Optimizer; invalid premises or cross-layer
contracts return to Architect for a revised plan. Preserve findings and fixes.
After a source, IR, recipe or ABI change, record a new candidate revision, rerun
affected self-checks and obtain review covering the new integrated revision.
Affected reviewers re-review the changes; explicitly reconfirm any unchanged
review scope. An earlier pass does not automatically cover the repaired artifact.
Apply the same rule to repairs made after Validator finds a failure.

## Workspace and handoffs

Read [Workspace layout and maintenance](workspace-layout.md) before initialization,
resume or export. Orchestrator creates a fresh run under
`DELIVERY_DIR/.local/runs/<run-id>/`, maintains `task.yaml`, `workspace.json`,
`state.json`, `trace.jsonl` and `trace.md`, and dispatches agents through durable
assignment directories. Baselines live in `baselines/<baseline-id>/`; each
candidate lives in `trials/<trial-id>/revisions/<revision-id>/`.

Keep assignment requests, inputs and responses in
`agents/<agent-id>/<assignment-id>/`. Link these records to the candidate and
overall trace; chat history alone is not a durable handoff. Each revision holds
the records below plus source snapshots, plans, dumps, profiles, raw results,
logs, draft figures and its artifact. Orchestrator owns shared state/trace updates;
workers own their responses and assigned outputs. Send concise summaries with
precise evidence paths and identities rather than copying full dumps into chat.

Every assignment follows its [Markdown response template](agent-report-templates.md),
including specialist Architects/Optimizers and independent Reviewers. Validator
and Verifier name the same acceptance role. Verifier owns the append-only
[performance history](performance-history.md): record every attempt with the
reserved attempt ID and exact artifact, leaving the iteration null until full
agreed numerical validation passes. Verifier sends the qualification receipt
and record IDs to Orchestrator, who then assigns the iteration number.
Orchestrator maintains the public trace using the strict document/entry templates.

Each handoff uses the same trial ID and full input/source, target, IR and
artifact identities where available. An unchanged semantic IR hash does not
identify the executable after a kernel override. Use a new trial ID or recorded
revision when evaluated inputs change; preserve prior raw evidence.

| Record | Minimum content |
| --- | --- |
| `expert-coverage.json` | All four domains; contract-based affected/unchanged/not-applicable/unassessed dispositions; actual per-phase agent/context/role/assignment bindings; current or explicitly reused report pins; independence limits and pending actions. Follow [Expert dispatch and coverage](#expert-dispatch-and-coverage). |
| `hypothesis.md` | Baseline and evidence identities; causal bottleneck; `lead_expert`, `supporting_experts`; proposed changes and dependencies; `contracts_changed`, `resume_from` and invalidated plans; expert baseline/source research, matched contracts and SOTA targets; assumptions, predicted whole-workload benefit, counter-costs, confidence and falsification checks. |
| `implementation.json` | Implemented hypothesis revision; source/change and extension/override identities; actual choices and emitted implementation; resume/reselection history; self-check results; integrated IR/artifact identities and any deviations from the plan; expert target status, implementation proof and remaining gaps. |
| `review.json` | Reviewer identities/domains and independence; reviewed hypothesis revision and source/IR/artifact identities; inspected evidence; findings, fixes and re-review results; domain coverage and consolidated verdict; expert baseline applicability and measured/theoretical claim checks. |
| `validation.json` | Tested input, reference, environment and artifact identities; matching review revision/verdict; per-scenario structural/numerical/resource/performance/reproduction results; raw sample/telemetry paths, actual exit statuses, skips, failure category and unmet criteria; expert comparison IDs, matched measurements/target outcomes and operator-SVG completion or pending owner/action. |

For Kernel work, these records also link the shared
[microprofile handoff](agent-report-templates.md#kernel-microprofile-handoff):
baseline/candidate ncu receipts, raw WG/module trace, swimlane SVG and analysis,
independent review and separate ncu/WG/hardware-interval completion states.
Orchestrator carries `paused_ncu_error` and missing-evidence owner/actions into
state and dispatch; an aggregate performance result cannot override them.

## Analysis and theoretical benefit

Every affected domain expert establishes its best-known implementation baseline
and SOTA target using [Expert baselines](expert-baselines.md). Search primary
sources, freeze comparison contracts and adopt concrete upstream code or a
traceable port before further optimization. Reproduce that starting point and
predeclare the comparison rule; after implementation report changes beyond it,
measured attainment, remaining gap or missing evidence. The main Architect
reconciles local targets with the complete workload. Review checks source reuse,
baseline applicability and implementation; Verifier checks actual comparisons.
A theoretical result keeps measured acceptance open.

Architect first establishes the requested execution boundary and critical path.
Specify and collect matching IR/TIR dumps and whole-workload timelines, then
collect detailed counters for the relevant kernels. Validator or the assigned
device owner can run the diagnostic capture requested by an architect; record
its artifact identity and keep profiled execution separate from formal timing.

For Kernel assignments, perform the
[kernel development loop](kernel-microprofiling.md#kernel-development-loop-and-ownership)
during development: baseline ncu and actual WG/hardware swimlanes → causal
producer/consumer and module-overlap diagnosis → implementation → candidate
recapture → independent microprofile review. KernelOptimizer owns the capture
and analysis even when a designated device owner executes it. Do not defer this
work to the final whole-model profiler. Show actual hardware working intervals,
not just role names or issue-to-wait envelopes.
Before implementing or revising a recorder, KernelOptimizer follows the
[IKET-derived tracing method](kernel-trace-method.md); KernelArchitect reviews
the selected profiler path, event semantics, generated instructions and measured
perturbation. The mandatory NCU and same-execution evidence gates still apply.

Choose the model that explains the evidence: operation/byte accounting for
graph changes; relevant memory-level and issue-rate ceilings for representation;
load balance and communication/dependency models for distribution; compute,
instruction, resource and pipeline limits for kernels. Use target-specific
memory and known-issue references where available.

For Roofline, define units and the applicable arithmetic path:

```text
F = floating-point work; Q = bytes at the modeled memory level
I = F / Q                         [FLOP/byte]
P_roof = min(P_compute, B * I)     [FLOP/second]
T_lower = max(F / P_compute, Q / B)
```

Choose `P_compute` for the actual dtype, instruction path and numerical profile,
and `B` for the relevant memory level and access pattern. Label theoretical
peaks versus calibrated sustainable ceilings. A lower predicted time bound
does not establish that observed runtime will decrease. Include changes in
issue rate, occupancy, spills, synchronization and communication that the
simple model omits. See the
[NVIDIA Roofline guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
for background.

Translate local estimates into whole-workload benefit. For a serial additive
time model with affected fraction `f` and local speedup `s`, the estimated
speedup is `1 / ((1 - f) + f / s)` if the remaining work is unchanged. For a
program DAG or multi-rank workload, recompute the affected critical path and
use the requested rank aggregation; overlapping kernel durations and waits
cannot be summed as end-to-end time.

Every hypothesis states its mechanism, assumptions, expected benefit range or
bound, counter-costs and observations that would falsify it. Derive confidence
from evidence and mark unknowns. Distinguish semantic correctness proofs from
conditional performance predictions. A low-cost exploratory probe may proceed
without a precise gain estimate if it has an explicit mechanism and rejection
rule. Neither a model nor a microbenchmark replaces requested device and
whole-workload acceptance. Rank experiments by expected whole-workload benefit,
evidence, validation cost and risk; do not require exhausting existing candidates
before writing a more appropriate implementation.

## Execution loop and failure routing

1. **Initialize and baseline.** Orchestrator records the task and resources,
   initializes the workspace, ignore rules, trace and assignment records;
   Optimizer builds a versioned ordinary current-compiler baseline; Validator checks
   its correctness and records requested baseline measurements. Preserve the
   original import and historical references separately. A baseline need not
   outperform the reference runtime to be a valid starting point.
2. **Research and adopt industry SOTA.** Each affected performance expert
   inspects primary-source implementations and benchmark entrypoints, freezes
   the strongest applicable source revision and matched contract, and selects
   concrete code/algorithm/mapping/memory components to reuse or port as the
   implementation starting point. Record source/license lineage, required
   numerical/ABI adaptations and baseline reproduction. Improve that adopted
   implementation; an outperform objective remains open until a strict matched
   measured win also satisfies the task's whole-workload acceptance rule. Follow
   [Expert baselines](expert-baselines.md); revisit the starting point when a
   stronger applicable implementation is found.
3. **Analyze and plan.** Architect consults relevant experts on the same frozen
   baseline, reconciles `expert-coverage.json` under the dispatch rules above,
   selects one joint hypothesis and records its dependencies and
   invalidation requirements, external best-known baselines and each affected
   expert's target. For Kernel changes, obtain baseline ncu, per-WG pipeline and
   hardware-module activity evidence with swimlanes before selecting a tuning
   change; missing evidence has an explicit capture task. Orchestrator assigns
   resources for that plan and enforces the ncu failure pause.
4. **Implement and integrate.** Optimizer provides domain implementation tasks,
   dependencies and source ownership. Orchestrator dispatches ready work in
   parallel; dependent work starts at its actual handoff. Optimizer integrates
   one runnable artifact through normal compiler verification.
   Start ordinary checks when their prerequisites hold. KernelOptimizer schedules
   mandatory candidate ncu/trace/swimlane capture in the same evaluation cycle
   and compares the affected pipelines; analysis need not precede ordinary NUM.
   Independent changes can use isolated worktrees in parallel.
   A discovery affecting semantics, ABI or the hypothesis returns to Architect
   for a recorded revision before the affected implementation continues.
5. **Review and repair.** Freeze the implementation revision and send it to
   independent relevant domain Architect reviewers. The main Architect
   consolidates the findings. Optimizer resolves implementation defects;
   Architect revises invalid plans. Rebuild, self-check and re-review affected
   changes until the current candidate has a passing consolidated verdict.
   Review may run alongside build and provisional checks under the prerequisite
   table above. Reuse unchanged domain evidence with checked lineage.
   Preserve unmet review requirements if review cannot be completed. Independent
   Kernel review checks actual raw WG/module events, interval attribution,
   coverage, chart regeneration and overlap calculations; static-only review
   cannot mark the microprofile gate complete.
6. **Validate and measure.** Freeze candidate inputs for the run. Validator
   applies the prerequisite table to IR/numerical/resource checks and ordinary
   screening as their inputs become ready. Use quick screening before expensive
   full measurement where useful. Before acceptance, verify actual domain
   assignments, current review coverage and independence, identity and artifact
   integrity; formal acceptance covers every requested scenario. Record pass,
   fail, skip and not-run separately; self-checks and mocks do not establish
   independent device acceptance.
   Verifier saves samples and a performance-history record before reporting an
   outcome or starting the next attempt, including failed/rejected attempts.
   Only a complete agreed numerical pass qualifies the revision as an iteration;
   Orchestrator records that allocation after receiving Verifier's evidence.
   Partial screening does not qualify, and performance benefit is not required
   to count a numerically correct iteration.
   For each execution-based candidate verification, Verifier schedules profiling
   separately and produces its measured per-operator latency SVG,
   linking raw data, identities and attribution limits from the report.
   Register pending diagnostic work without blocking the next independent
   ordinary check; a single verification may include multiple preflights/scenarios.
   Update one cumulative performance figure containing all qualified iterations
   from the ledger. Verifier hands off the checked SVG, compact public data and
   reproduction command; Orchestrator embeds the latest export in the delivery
   README and updates its coverage/status caption before closing this handoff.
   A private or unlinked diagnostic figure alone does not complete publication.
   Check the Kernel revision's ncu/WG/hardware-interval gates separately and link
   its swimlanes and raw data. Follow [Performance history](performance-history.md)
   for all diagnostic and publication artifacts.
7. **Decide.** Validator classifies results; Architect compares predicted and
   observed behavior and revises the model; Orchestrator dispatches repair,
   a revised hypothesis or the next candidate. Keep evaluated candidates and
   rejected hypotheses identifiable, including why they lost. Reconcile each
   expert's SOTA target and measured/theoretical gap; unmet targets have a next
   owner/action and cannot be marked complete by a source-review pass.
8. **Reproduce and hand off.** For a candidate meeting the requested criteria,
   Optimizer rebuilds from the original input plus saved plans/scripts/overrides
   in a fresh process. Apply the workspace's public-package relocation rules:
   verify reviewed input, IR/generated-code and contract identities; changed
   execution inputs or unresolved differences return to review. Validator
   verifies generation identity and repeats applicable
   acceptance on the deliverable. Orchestrator records completion
   only when the requested criteria and reproduction requirements are met,
   updates the public optimization trace, and exports verified figures/data
   and reproduction files with the recorded identities.
   Include the performance-versus-iteration chart and compact history dataset;
   exporting only the best result is insufficient.

| Failure | Next action |
| --- | --- |
| ncu error, denied counters or unusable capture | Orchestrator pauses optimization dispatch and safely quiesces owned work. Request sudo access if not already authorized/available; use supplied credentials privately, diagnose and retry. Remain paused until ncu succeeds; no static/timing/instrumentation fallback. Follow the mandatory microprofiling failure gate. |
| Missing WG trace, hardware working intervals or swimlane evidence | Keep microprofile acceptance open; assign the missing capture/instrumentation/analysis work. In-flight envelopes, model charts and performance wins do not close this gate. |
| Implementation, compilation or binding defect within a valid plan | Optimizer and the responsible implementation expert repair it, preserving a causal reproducer; re-review the repaired revision before acceptance. |
| Numerical/representation/distribution contract or capability premise is invalid | Architect and affected domain architects revise the joint plan. |
| Exact independent generation differs | Validator retains the first same-history mismatch and actual/reference logits or top-candidate margins, and checks the current same-profile baseline. Architect isolates the changed mechanisms with same-input evidence before assigning a causal repair or justified profile revision. |
| Correct candidate has insufficient performance benefit | Architect reconciles the profile/model and chooses a revised or different hypothesis. |
| Environment, device contention, source identity or artifact pairing is invalid | Orchestrator resolves the condition; do not use the run for performance attribution. |
| Review is missing, stale or has open blocking findings | Orchestrator obtains required reviews or routes repair; Validator does not accept the candidate under the earlier verdict. |

Use the resume/invalidation rules in
[Performance iteration](performance-iteration.md): changed semantics or layouts
invalidate affected downstream analysis; changed distribution reopens coupled
choices; candidate/ABI/workspace changes regenerate proposals and dependent
memory/synchronization plans; compatible source overrides still regenerate
source and device code with a new implementation identity. Rebuild both the
baseline and candidate after shared core fixes so core gains are not attributed
to workload-local choices.

Reserve each device for one active GPU job at a time, including diagnostics,
self-checks and suites. Comparable variants run sequentially on the same devices
with rotated order and retained raw samples. CPU analysis or isolated work on
other devices can proceed within resource limits. Record device ownership in
state and enforce it with the available scheduler/lock; do not mutate a frozen
artifact while it is being validated or measured.

Apply [IR acceptance and repair](ir-acceptance.md),
[Performance iteration](performance-iteration.md) and the skill's reproduction
requirements without weakening their scope. A plateau triggers model/priority
reassessment. Respect the user's continuation requirements and agreed budget;
if work ends with unmet criteria, report them explicitly rather than calling
the candidate accepted or the optimization complete.
