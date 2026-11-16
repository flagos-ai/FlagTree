# Workspace layout

`WORK_DIR` is the model/backend directory, e.g.
`python/tutorials/flagmega/01-qwen3-1.7b-bf16/nvidia-h800`.
`MILESTONE_DIR=WORK_DIR/.local/milestones/mNNN`;
`REVISION_DIR=MILESTONE_DIR/revisions/rNNN`.
The [catalog](../assets/workspace-templates/deliverables.json) defines each delivery's
path, writer, consumers, template and script; use the delivery map below to dispatch.
Inspect one template with workspace.py catalog --kind KIND before preparing it.
prepare --output NAME creates a versioned file within the default delivery directory.

## 1. Workflow

1. Follow the numbered [orchestrator cycle](agents/orchestrator.md#cycle-transitions).
   orchestrator dispatches setup; graph optimizer runs workspace.py init and owns
   physical paths/.gitignore/workspace.json. orchestrator freezes task details and
   maintains dashboard.md as the sole table-based management record.
2. Establish the measured baseline under .local/baseline/ through the same technical
   owners, synchronizing root artifacts/figures as outputs and states change.
   Baseline establishment is not a performance milestone. If reviewer
   confirms the baseline already meets the goal and all acceptance, seal its reviewed snapshot and end;
   otherwise orchestrator opens m000/r000 with fixed metric/baseline/rule and a plan owner.
3. Prepare draft plan.md and tracking.md at revision creation; profile current
   inputs, analyze with experts, then finalize the expert-owned signed plan before
   optimization. reviewer maintains tracking/coverage.
   Each agent receives named inputs, produces its assigned templates/real outputs
   and signs its own phase. Keep no-gain/failed revisions and all raw receipts.
   Follow the [revision boundary](agents/orchestrator.md#revision-boundary): create
   a new revision for a new independent hypothesis, even if final acceptance of a
   previous measured revision remains pending. Keep its result and outstanding
   evidence attached to the owning revision. Preserve frozen execution paths and
   authentic original receipt identities when correcting a late revision assignment;
   record the new ownership and reused evidence explicitly rather than rewriting
   old signed receipts or moving files used by a live job.
4. reviewer verifies gain/acceptance/signatures and issues release.json. graph
   optimizer maintains latest working artifacts; graph profiler maintains latest
   working figures/readme, including kernel profiler's current figures. These updates
   occur immediately at build/export/capture/measurement/review/state changes, with
   visible pending/running/failed/verified status and exact provenance, without
   waiting for release. Reviewer release separately seals accepted snapshots.
   orchestrator consumes verdict/publication
   receipts and marks achieved only under the [gain gate](agents/orchestrator.md#milestone-gate).

## 2. Checklist

<a id="delivery-map"></a>

| Workspace delivery | Producer / sole writer | Agent inputs / consumers | Template and executable |
| --- | --- | --- | --- |
| Root dashboard.md; frozen .local/task.yaml; milestone.json, revision.json; agents/*/request.json | orchestrator | All workers read the dashboard; orchestrator updates it from terminal evidence and manages only by its rows plus flow | Catalog dashboard/task/milestone/revision/request; workspace.py prepare; requests cite dashboard rows/flow steps |
| .gitignore, .local/workspace.json and physical directory setup | graph optimizer | All agents resolve paths, environment and bindings | Catalog gitignore/workspace; workspace.py init |
| .local/cache/weights, kernels, comparison | graph optimizer: weights/full builds; kernel optimizer supplies native-build receipts; graph profiler: comparison data | Optimizers/profilers/reviewer consume pinned inputs/cache, not unversioned assumptions | Catalog cache-manifest + job; real acquisition/build command via workspace.py run, manifest --kind cache |
| Root/milestone/revision artifacts/ (.py/.il and other IR/generated code) | graph optimizer integrates/exports/synchronizes; kernel optimizer supplies owned native code | Experts inspect; profilers execute pinned build/data; reviewer validates/reproduces | Catalog benchmark-build/job/artifact-manifest; real compiler via bound entrypoint, manifest --kind artifacts, sync for current views; publish for reviewed snapshots |
| .local/baseline and revision build/ (native binary/data bindings) | graph optimizer or kernel optimizer in declared disjoint scopes | Profilers/reviewer need full runnable source/data/library/manifest | Catalog job/native-manifest/kernel-native-manifest; full-build manifest belongs to graph optimizer, assignment-native manifest to kernel optimizer; run/manifest --kind native |
| tests/ | reviewer owns independent CI adapter; optimizers supply causal regression cases | CI/reviewer run tests; orchestrator consumes receipts | Catalog ci + bound entrypoint; workspace.py run |
| benchmark/build_artifacts.py | graph optimizer | Profilers/reviewer reproduce the actual full build | Catalog benchmark-build, entrypoint.py + job/run |
| benchmark/{benchmark_model,compare_backends,profile_operators,render_results}.py | graph profiler | reviewer consumes samples, capture/monitoring receipts | Catalog benchmark-model/backends/operators/render; real bindings + job/run/renderers |
| benchmark/profile_kernel.py; profiles/kernel/CAPTURE_ID/{harness,reports,analysis}/ | kernel profiler | Kernel/graph expert diagnoses; reviewer audits NCU/raw/clocks/census/controls | Catalog benchmark-kernel/capture-kernel/controls/kernel-trace + job/run; render_kernel_swimlane.py |
| profiles/{graph,operators}/CAPTURE_ID/ | graph profiler | graph expert diagnoses; reviewer audits scope/clocks/overhead | Catalog capture-graph/graph-profile/figure-data + job/run; render_workspace_figures.py |
| Revision figures/: full model timeline/time-sorted operator bars, milestone/revision histories/backend SVGs; operator/kernel swimlanes | graph profiler: graph/operator/history/backend; kernel profiler: operator/kernel swimlanes | Experts analyze bottlenecks; reviewer audits raw-to-chart; graph profiler consumes kernel SVGs for publication | Catalog figure-graph/figure-operators/figure-milestones/figure-history/figure-backends/kernel-trace; the two renderers |
| Root/milestone figures/, root readme.md; .local/reports/ live packets and .local/views/ synchronization/publication manifests | graph profiler synchronizes latest actual files and status; original profiler authorship retained | Users/agents read current results; orchestrator consumes synchronization receipt; reviewer accepts exact snapshots | Catalog readme/release; readme template, workspace.py sync for current figures/readme, publish for reviewed snapshots |
| Revision plan.md, results/baselines.json; expert contract.md/inventory.json | Designated plan owner: graph expert for graph/joint, kernel expert for kernel-only; each expert owns its assignment contract | Optimizers/profilers/reviewer consume approved hypothesis/contracts/acceptance | Catalog plan/baselines/contract/inventory; plan/agent templates, workspace.py prepare |
| Revision tracking.md, results/tracking-snapshots/, agent-coverage.json | reviewer | All agents read phase identities/events; orchestrator consumes readiness/signature verdict | Catalog tracking/coverage; tracking template, prepare and workspace.py snapshot |
| results/{implementation,kernel-implementation}.json | Respective graph/kernel optimizer | Expert/reviewer inspects self-checks; profilers consume executable pins | Catalog implementation/kernel-implementation; prepare + actual job/manifest receipts |
| results/{review,validation,baseline-comparison,release}.json | reviewer | orchestrator consumes verdict; publishers consume exact approved release | Matching catalog templates; prepare, actual independent jobs and hash-bound release |
| results/{graph-profile,kernel-profile}.json | Respective graph/kernel profiler | Expert/reviewer/orchestrator consume actual capture/analysis gate receipts | Matching catalog templates; prepare + real capture jobs/renderers |
| agents/*/{response.md,handoff.json} | Assigned actual agent | Named recipient/plan owner/reviewer/orchestrator consume signed scoped handoff | Catalog response/handoff; agent template + handoff JSON |
| .local/metrics/performance-history.jsonl; .local/optimization-trace.md; revision results/trace-entry.md | reviewer | graph profiler renders history/comparisons; all agents read immutable provenance | Catalog measurement/trace/trace-entry; append-measurement, trace validator |
| .local/jobs/JOB_ID/{stdout.log,stderr.log,receipt.json}; optional logs/raw telemetry | Actual executing optimizer/profiler/reviewer | reviewer audits; orchestrator consumes terminal state/resource handoff | Catalog job; workspace.py run retains actual exit/output hashes |

1. Root dashboard/artifacts/figures/tests/benchmark/readme/.gitignore are tracked; .local is
   ignored. Root/milestone/revision artifacts contain no weights/packed payloads,
   compilation cache or native binary. These stay in local cache/build, with manifests.
   Every `artifacts/` contains only `generated/` and `ir/`: latest exported
   model/kernel codegen and latest real compiler dumps, respectively. Do not
   accumulate old revisions inside either directory. Manifests, dump indexes,
   publication metadata and historical snapshots stay outside `artifacts/` in
   `.local` view/build/revision locations. Measurement/profile/raw/log payloads,
   tests, benchmark/helper scripts and execution/review receipts also stay there.
   `artifacts/ir/` includes real imported/final IR and key stages such as vectorized,
   packed, distributed and TIR/bufferized snapshots. Preserve chronological pass
   identity and complete function sets in named subdirectories, with source and
   semantic hashes in the external manifest. Export retained compiler dumps directly; keep
   all finer-grained pass dumps locally for diagnosis.
2. Live cumulative milestone/revision SVGs and readme are refreshed on every output or state
   change in .local/reports/REPORT_ID/{results,figures}/: graph profiler writes
   data/SVG/readme and synchronizes root working views immediately; reviewer
   subsequently audits/signs release to seal accepted snapshots. These reuse actual
   measured source records and identify their current review status
   and never modify an achieved milestone or frozen revision. A live history refresh
   merges updated figures into the existing working root view; it preserves other
   graph/operator/backend SVGs. A full working revision synchronization replaces
   its managed view; reviewed publication seals the reviewed snapshot separately.
   Every milestone has milestone.json, artifacts/, figures/ and revisions/.
   Every revision has revision.json, plan.md, tracking.md, artifacts/figures and
   named owners. Add results/profiles/agents/build only as required by real tasks.
3. Latest root working views are actual current file copies, not symlinks into .local;
   review/achievement status is explicit and does not gate visibility. Accepted
   milestone/revision snapshots remain separately reviewed and immutable.
   Artifact publication archives superseded or out-of-contract files under
   `.local/` and enforces the two-directory layout on every refresh. Historical
   evidence remains outside the current artifact view. Other publication kinds
   retire only their earlier managed files, preserving unrelated files. Synchronization
   receipts pin source files and authoritative state. Keep build/NUM/measurement/
   review status separate, including the current attempt's failure and the identity
   of retained last-built source when a new build has not produced source yet.
   Open milestones label the actual revision verdict; achieved milestones freeze
   the selected winning view. Preserve failed/no-gain revision evidence and all
   measurement rows; iteration numbering does not establish achievement.
4. Templates start pending with empty signatures/measurements. IR, kernels,
   weights, reports and raw samples come from real bound compiler/tool commands;
   templates define their manifests/jobs, not fabricated payloads. entrypoint.py
   requires a real MODULE:FUNCTION binding supplied by its owning agent.

Current-view synchronization uses `workspace.py sync --work-dir W --agent OWNER
--kind artifacts|figures|readme --record RECORD.json`. The producer's
`flagmega.working-sync/v1` record names a unique sync_id, source_id, actual
structured status, agent_name/agent_id, current base_view_sha256, files with
workspace-relative source/destination paths and sha256, explicit selected
source_manifest {path, sha256}, and input_pins. A dependency/ancestor pin cannot
identify the current selection; source labels and paths do not replace hashes. Use
retain_previous for a partial or status-only refresh. Source and state pins are
checked before updating the managed root files. Receipts and copied versions stay
immutable under `.local/working/KIND/SYNC_ID/`; `.local/views/root-KIND.json`
identifies the current generation. Artifacts belong to graph optimizer;
figures/readme belong to graph profiler. Refresh within the same flow cycle using
producer/state hooks or an active watcher; do not defer to review or milestone
completion. Reviewed `publish` seals its exact snapshot without rolling back a
newer working view.

For artifact state refresh, the producer can use `workspace.py refresh` as an
event hook or `workspace.py watch --watch-interval 2 --stop-file STOP_FILE` with
the same owner/work-dir/kind and a `flagmega.working-watch/v1` record. Its pinned
source_manifest, files, source identity and dashboard_path identify the selected
source and authoritative management table. The current watch spec is
flagmega.working-watch/v2 and reads dashboard.md; do not create state.json or a
JSON management mirror. A managed watcher needs a real PID/start identity, log,
status/stop control and verified liveness; a stopped test is not continuous
refresh. Unchanged state creates no heartbeat versions. A watcher for an older
source stops when superseded; the next producer installs its new pinned watcher
or hooks, preserving current-source ownership.

<a id="signatures"></a>

1. Plan names applicable agents by phase. orchestrator signs flow scope; affected
   experts/optimizers/profilers approve their scope; reviewer signs validation scope
   and independence. Plan owner alone writes plan/approval references; reviewer
   alone writes tracking, phase snapshots and authentic signoff references.
2. Each actual agent signs its own handoff: name/context/assignment, phase/decision/
   UTC, milestone/revision, plan version/hash, inspected source/manifest scope,
   tracking snapshot sequence/hash and evidence pins. Nobody signs for another
   context. Unaffected scopes need explicit not_applicable reasons.
3. Plan approvals hash the frozen body excluding signature section; tracking
   approvals hash frozen phase snapshots excluding signature section. Save those
   snapshots under results/tracking-snapshots/. New phases use new assignments or
   versioned receipts; never overwrite a file already referenced by hash.
4. Append events/corrections and preserve original failures/verdicts. New events
   do not renew prior approvals; changed plan/source/artifacts invalidate affected
   signatures and need renewed approvals. reviewer audits completeness; orchestrator
   advances flow using that signed verdict. No-gain or unsigned work stays open.

## 3. Index

1. [Agent registry/flow](agents/orchestrator.md), [reviewer/gain and history](agents/reviewer.md).
2. [Delivery templates and owners](../assets/workspace-templates/deliverables.json), [workspace helper](../scripts/workspace.py),
   [entrypoint adapter](../assets/workspace-templates/entrypoint.py), [readme template](../assets/workspace-templates/readme.md).
3. SVG inputs and actual examples are indexed below; all example data is synthetic.
   [Graph/history/backend renderer](../scripts/render_workspace_figures.py),
   [operator/kernel renderer and method](kernel-trace-method.md), [plan](../assets/revision-templates/plan.md),
   [tracking](../assets/revision-templates/tracking.md), [coverage](../assets/agent-templates/agent-coverage.json).
   Latest root views retain these canonical template layouts and coverage before
   review, during failures and while measurements are pending. For missing captures,
   render an explicit pending version of the corresponding template, with no
   fabricated clock window or numeric points. Supplemental status charts do not
   replace required figures. Each figure receipt pins its template and renderer.
   Genuine partial GPU leader captures may use explicit `observed_role_leaders`
   scope and `observed_window_ns`, with pinned source/health/nominal clock evidence
   and one actual CTA's consumer/producer leader envelopes. Unknown resolution
   stays null and calibration/alignment stays pending, visibly labeled. This
   mode excludes unobserved prefix/tail and D2H and cannot establish engine busy,
   all-warp coverage or whole-model time. The existing `whole_model` mode still
   requires its full execution window, known positive resolution and validated
   common clock; do not promote a partial capture to that mode.
   Preserve mutable state/metadata/ledger inputs and template/renderer dependencies
   as immutable snapshots in each render packet. Figure provenance uses those
   snapshot paths; current synchronization additionally pins live state for
   freshness. A later state/source change must not break historical validation.
4. Example below; [NCU layout reference](https://github.com/mit-han-lab/ncu-report-skill/blob/d1887948c7d53690cfe6605f59c1329b8a1c6bb5/reference/00-directory-layout.md) keeps each capture's harness/reports/analysis together.

| Figure | Input template | SVG example | Script / owner |
| --- | --- | --- | --- |
| Graph: full model start-to-end timeline | [graph.json](../assets/workspace-figures/graph.json) | [graph.svg](../assets/workspace-figures/graph.svg) | render_workspace_figures.py / graph profiler |
| Graph: operator-time bars sorted descending | [operators.json](../assets/workspace-figures/operators.json) | [operators.svg](../assets/workspace-figures/operators.svg) | render_workspace_figures.py / graph profiler |
| Operator/kernel microanalysis swimlane | [example.json](../assets/kernel-swimlane/example.json) | [example.svg](../assets/kernel-swimlane/example.svg) | render_kernel_swimlane.py / kernel profiler |
| Milestone curve + concise performance-point table | [milestones.json](../assets/workspace-figures/milestones.json) | [milestones.svg](../assets/workspace-figures/milestones.svg) | render_workspace_figures.py / graph profiler |
| Revision curve + concise performance-point table | [history.json](../assets/workspace-figures/history.json) | [history.svg](../assets/workspace-figures/history.svg) | render_workspace_figures.py / graph profiler |
| FlagMega/vLLM/SGLang/llama.cpp comparison | [backends.json](../assets/workspace-figures/backends.json) | [backends.svg](../assets/workspace-figures/backends.svg) | render_workspace_figures.py / graph profiler |

```sh
python skills/flagmega-developer/scripts/render_workspace_figures.py skills/flagmega-developer/assets/workspace-figures/milestones.json --output /tmp/milestones.svg
```

```text
python/tutorials/flagmega/01-qwen3-1.7b-bf16/nvidia-h800/
├── .local/
│   ├── task.yaml                                 # frozen task detail, orchestrator -> all agents
│   ├── workspace.json                            # graph optimizer -> all agents
│   ├── cache/{weights,kernels,comparison}/        # technical owners -> workers/reviewer
│   ├── baseline/{artifacts,figures,results}/       # baseline evidence, not achieved m
│   ├── milestones/m000/
│   │   ├── milestone.json                        # orchestrator -> all agents
│   │   ├── {artifacts,figures}/                   # publishers -> latest milestone view
│   │   └── revisions/
│   │       ├── r000/                             # no_gain, retained
│   │       └── r001/
│   │           ├── revision.json                 # orchestrator -> assigned owners
│   │           ├── plan.md                       # plan expert -> workers/reviewer
│   │           ├── tracking.md                   # reviewer -> all agents
│   │           ├── artifacts/                    # graph optimizer -> profilers/reviewer
│   │           ├── figures/                      # graph/kernel profiler -> reviewer
│   │           ├── agents/*/                     # request + actual signed response/handoff
│   │           ├── results/                      # named result owners -> downstream
│   │           └── profiles/kernel/c0001/{harness,reports,analysis}/
│   ├── reports/REPORT_ID/{results,figures}/         # live history/readme, independently reviewed
│   ├── jobs/JOB_ID/{stdout.log,stderr.log,receipt.json}
│   ├── metrics/performance-history.jsonl          # reviewer -> graph profiler
│   └── optimization-trace.md                      # reviewer -> all agents
├── artifacts/{generated/,ir/}                    # latest codegen and IR only; metadata in .local
├── dashboard.md                                  # sole management table, orchestrator -> all agents
├── figures/{graph-trace.svg,operator-latency.svg,operators/,performance-vs-milestone.svg,performance-vs-iteration.svg,backend-comparison.svg}
├── tests/check_correctness.py
├── benchmark/{build_artifacts.py,benchmark_model.py,compare_backends.py,profile_operators.py,profile_kernel.py,render_results.py}
├── readme.md
└── .gitignore                                    # /.local
```

```sh
# Run each command in its named owner context; paths here are illustrative.
python skills/flagmega-developer/scripts/workspace.py init --work-dir /tmp/flagmega-example --agent graph_optimizer
python skills/flagmega-developer/scripts/workspace.py prepare --kind milestone --work-dir /tmp/flagmega-example --milestone m000 --agent orchestrator
python skills/flagmega-developer/scripts/workspace.py prepare --kind revision --work-dir /tmp/flagmega-example --milestone m000 --revision r000 --agent orchestrator
python skills/flagmega-developer/scripts/workspace.py prepare --kind plan --work-dir /tmp/flagmega-example --milestone m000 --revision r000 --agent graph_expert
python skills/flagmega-developer/scripts/workspace.py prepare --kind tracking --work-dir /tmp/flagmega-example --milestone m000 --revision r000 --agent reviewer
```

## 4. Lessons

1. A directory is a container; every delivery has an owner and named downstream inputs.
2. Generic runners preserve real execution receipts; the worker owns model/tool bindings and their validity.
3. Pending templates and signature scaffolds cannot prove measurements, review or milestone achievement.
4. Fresh checkout reproduction uses tracked scripts/data acquisition; historical redraw uses retained raw pins. Unavailable backends remain gaps.
