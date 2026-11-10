# Workspace layout and maintenance

Use this reference when initializing, resuming, maintaining or exporting an
optimization run. Orchestrator owns the directory contract and overall trace.
New runs use this layout; honor explicit user paths and record equivalent or
legacy mappings without moving existing artifacts or deleting user files.

## Roots and versions

- `DELIVERY_DIR`: the model/workload/hardware tutorial or delivery directory.
  Reuse the applicable directory under `python/tutorials/flagmega/` when one
  exists. Choose a descriptive new directory when the task needs a new delivery.
- `RUN_ID`: a unique UTC timestamp plus workload slug, for example
  `20261103T142500Z-glm53-h800-tp4-b1`. A run fixes its workload, numerical,
  reference and measurement contracts.
- `WORK_DIR="$DELIVERY_DIR/.local/runs/$RUN_ID"`: private run records and outputs.
- `BASELINE_DIR="$WORK_DIR/baselines/b0001"`: initial baseline. Shared core fixes
  create `b0002`, etc.; retain earlier versions and identify which one every
  comparison uses.
- `TRIAL_DIR="$WORK_DIR/trials/t0001-consumer-layout/revisions/r0001"`: one
  hypothesis and implementation revision. A different hypothesis gets a new
  trial ID; a repair or changed evaluated source/IR/recipe/ABI gets a new revision.

Use IDs that are single path components: letters, digits, `.`, `_` and `-`,
starting with a letter or digit. Allocate IDs exclusively; never reuse a run,
baseline, trial or evaluated revision directory to overwrite its contents.
Changes to workload, numerical/reference configuration or measurement boundary
start a separate run or explicitly recorded comparison cohort; keep their
results separate from the original contract.

## Standard layout

```text
DELIVERY_DIR/                         # Git-managed delivery
  .gitignore
  readme.md                          # reproduction, accepted scope and limits
  optimization-trace.md              # public summary of meaningful trials
  optimize.py, accuracy.py, ...       # actual reproduction/validation entrypoints
  render_results.py                  # regenerates public figures from public data
  local_optimizations/                # required importable implementations/plans
  generated_kernels.py or .cpp        # accepted backend source snapshot, when applicable
  figures/
    <metric>.svg                      # latest public SVG for each figure only
  results/
    <export-id>/                      # compact data, receipts and historical exports
      summary.json or summary.json.gz
      performance-history.csv.gz
      <metric>-<export-id>.svg        # archived version and provenance
  .local/                            # ignored process workspace
    runs/<run-id>/
      task.yaml                      # fixed task/reference/measurement contract
      workspace.json                 # roots, path mappings, caches and versions
      state.json                     # current phase, assignments and gates
      trace.jsonl                    # append-only overall event record
      trace.md                       # human-readable full optimization trace
      metrics/
        performance-history.jsonl    # Verifier's append-only measurement ledger
      agents/<agent-id>/<assignment-id>/
        request.md                   # dispatched task, scope and expected handoff
        inputs.json                  # run/trial/revision and input identities/paths
        response.md                  # agent findings/actions and output references
      baselines/<baseline-id>/
        imported.py
        dumps/, results/, logs/
        validation.json
        artifact/
      trials/<trial-id>/revisions/<revision-id>/
        hypothesis.md
        implementation.json
        review.json
        validation.json
        source/                      # exact scripts/overrides/patches for this revision
        plans/, dumps/, profiles/
        plans/expert-baselines.json  # searched implementations, contracts, targets
        results/, logs/              # raw requests, correctness, resources, telemetry
        results/expert-baseline-comparison.json  # measured/proven outcomes and gaps
        figures/                     # draft or diagnostic figures for this revision
        artifact/
      exports/<export-id>/
        export.json                  # source revision, gates, data/file hashes
        files/                       # staged public files before promotion
        reproduction/
          artifact/, results/, logs/
          reproduction.json          # fresh public-package rebuild and test receipt
    cache/                           # reusable verified readonly/JIT data, if local
    locks/                           # run ownership and device-lease records
    worktrees/                       # isolated implementation checkouts, if used
```

Create only directories and entrypoints the task actually uses. The tree is
the placement contract, not a request to create empty public scripts or fake
results. Large checkpoints and shared caches may stay outside this tree;
record their paths, source identities and reuse policy in `workspace.json`.
Device leases must use a scheduler or lock namespace shared by all GPU jobs;
per-run lock files alone do not coordinate other runs or worktrees.

## Initialization and resume

Before dispatching implementation or diagnostic work, Orchestrator:

1. Selects `DELIVERY_DIR` and inspects its existing files, Git status and ignore
   rules. Ensures `/.local` is ignored by repository-controlled rules; add that
   entry to the delivery's `.gitignore` when needed, preserving other rules.
   Use the slash-free `/.local` pattern so it also covers an external-storage
   symlink: `/.local/` matches directories but leaves that symlink unignored.
   Verify with `git check-ignore -v --no-index <delivery>/.local` and
   `git ls-files -- <delivery>/.local <delivery>/.local/**`. For a real directory,
   also check a child probe; Git rejects child paths beyond a symlink.
   An ignore rule does not untrack files;
   report an existing tracked process path rather than silently removing it
   from Git. Do not force-add ignored process files.
2. Allocates a fresh `RUN_ID` and creates its run root exclusively. On a
   collision, choose another ID; an existing path is a resume target, not an
   initialization target. Create initial `baselines/b0001/` and the run records.
3. Writes `task.yaml`: model/checkpoint and initial compiler/source identity,
   including dirty changes; numerical profile, shapes/scenarios, state and
   runtime ABI; devices/topology/runtime versions; actual reference execution;
   primary metric, guardrails, acceptance thresholds and agreed resources/budget.
4. Writes `workspace.json` with the schema version, roots, run ID, baseline and
   revision paths, external inputs/caches, source/worktree mappings and ignore
   policy. Writes initial `state.json` and an initialization trace event. State
   records pending work, assigned agents/devices, gate status and current best
   accepted artifact; a proposed candidate is not the accepted best.
   Initializes the readable run trace and adds the pending run to the public
   `optimization-trace.md`, preserving existing run summaries and their status.
5. Creates an assignment directory before each dispatch. `request.md` states
   the role, scope, required outputs and permitted write locations; `inputs.json`
   pins contract/source/IR/artifact identities and evidence paths. Create only
   assignments that are actually dispatched.

If the build entrypoint supports a preplanned recipe, Orchestrator may create
`input/recipe.json` before dispatch and pin its original bytes in the request.
The build must admit only a real input directory containing that one ordinary
file, with an explicitly supplied recipe whose parsed JSON matches it. Preserve
the registered bytes; do not replace the file with a reformatted copy. Reject
extra files, empty input directories, symlinks, missing or mismatched recipes,
and existing build outputs before mutating the workspace. Record this admission
contract and its boundary checks when changing a build wrapper; do not delete
preplanned inputs to bypass an existing-output guard.

For resume, verify the task contract, source/target/artifact identities, path
mappings and current device ownership. Reconcile state against durable events
and receipts; interrupted work remains pending until its actual outcome is
known. Preserve earlier records and allocate a new assignment/revision when
inputs change. Never truncate the overall trace or relabel a stale review pass.
Rebuild missing reservation/allocation entries in the state view from their
existing pinned receipts. Verify that qualified assignments and the next
iteration counter agree; a counter alone is not evidence of qualification.
Keep an active unqualified attempt separate from the last qualified iteration.

## Writers and agent interaction records

Use [Agent flow](agent-flow.md#workspace-and-handoffs) for the contents and gate
semantics of hypothesis, implementation, review and validation handoffs.

| File or area | Owner and update rule |
| --- | --- |
| `task.yaml`, `workspace.json` | Orchestrator initializes them; contract/path amendments are versioned and recorded in trace. |
| `state.json`, `trace.jsonl`, `trace.md` | Orchestrator is the single writer; state/view updates use a temporary file and atomic replacement. |
| `metrics/performance-history.jsonl` | Verifier is the single writer; append every attempt and later verdict correction, preserve old records. |
| Assignment `request.md`, `inputs.json` | Orchestrator writes before dispatch; inputs remain pinned for that assignment. |
| `expert-coverage.json` | Orchestrator records all four domain dispositions, exact revision identities and actual per-phase expert/context/assignment/report bindings. Update the current view atomically; preserve prior versions and dispatch/correction events. Joint coordination does not count as separate domain work. |
| Assignment `response.md` and assigned evidence | The assigned agent writes its own response; Orchestrator records the receipt and links it to trace. |
| `hypothesis.md` | Main Architect consolidates expert inputs for the specified revision. |
| `implementation.json`, `source/`, `plans/`, build outputs | Main Optimizer integrates actual domain implementations and their provenance. |
| Domain review responses, `review.json` | Independent reviewers own their responses; main Architect consolidates the review record. |
| `validation.json`, formal `results/` | Validator records actual outcomes and their exact candidate identities. |
| Export receipt and public overall trace | Orchestrator stages/updates them; Validator verifies data, provenance and applicable gates. |

Assignments use distinct IDs even when an agent or role is reused. Record the
actual instance/context and whether a reviewer is independent of design and
implementation. Keep requests, responses, review findings and fix receipts;
tool/chat messages may supplement these records but cannot replace them.
New domain assignments use the canonical `agents/<domain>_<role>/aNNNN/`
directories and files in [Expert workspaces](expert-workspaces.md). Preserve
legacy directory mappings and frozen evidence instead of renaming old runs.
Agents send shared-state updates to Orchestrator instead of concurrently
editing the overall trace or another agent's response.
Generate file pins from the actual bytes and exchange their structured records;
do not hand-transcribe long hashes from chat summaries. Display abbreviations
only as shorthand for a linked full pin. Resolve a mismatch against the frozen
receipt and actual file before using that evidence.

Validate activation schema and bindings against the existing frozen proposal
before creating the activation file. Then pin the written activation and verify
those exact bytes immediately before launching the command. A validator that
returns a file pin must receive an existing file; do not ask it to pin an
activation that has not been written. Preserve failed preparation receipts and
reissue time-sensitive device clearance before retrying a launch.

Use the [role-specific Markdown templates](agent-report-templates.md) for agent
responses and the strict public trace document/entry templates. Orchestrator
reserves attempt IDs, assigns iteration IDs only after complete numerical
qualification, and assigns the active Verifier ledger writer. State records
`next_attempt_index`, `attempt_reservations`, `next_iteration_index` and
`iteration_assignments` separately. Each iteration assignment pins Verifier's
complete numerical receipt and the exact artifact; unqualified revisions keep
a null iteration and explicit reason. Other agents send
diagnostic/self-check performance receipts to Verifier, preserving the actual
measurement author and kind. See [Performance history](performance-history.md).

## Overall optimization trace

`trace.jsonl` is the durable append-only event history. Each meaningful event
has an event ID/sequence, UTC timestamp, run/trial/revision/baseline IDs as
applicable, actor/assignment, phase/action, pinned inputs, output/evidence paths,
status and decision with its reason. Record metric changes with the scenario,
measurement boundary, baseline and actual sample provenance. Append corrections
as new events; preserve earlier failures and verdicts.

Orchestrator updates `trace.md` after a completed assignment, gate, repair,
measurement or next-action decision. This readable view contains:

- The task contract, baseline versions and current scope/status.
- Every meaningful trial/revision in order: bottleneck evidence, conditional
  benefit, experts, implementation, source/IR/artifact identities and handoffs.
- Review findings, repairs, re-review and numerical/resource/performance results,
  including failed, skipped and not-run items; actual measurements versus estimates.
- Links to that revision's raw results, profiles, draft figures and receipts.
- The accepted best, rejected directions with reasons, unresolved work and next action.

`DELIVERY_DIR/optimization-trace.md` is the Git-managed public counterpart.
Summarize all meaningful trial outcomes, including rejected candidates and
regressions, without copying agent transcripts or dumping raw logs. Include
run/trial/revision and baseline identity, change and owner, expected effect,
measured per-scenario outcome, review/acceptance/reproduction status, decision
and links to published results/figures. Update it after meaningful decisions
when those outcomes belong in the delivery; at handoff it must reflect the
actual delivered revision and every unmet criterion.

Public trace and readme link to Git-managed data and figures. Local archive
pointers may be identified as optional evidence, but public commands and
figure rendering cannot require private trace, agent records or `.local/` code.

## Figures and Git-managed files

Honor `workspace.json.artifact_layout` before applying the default shareable
layout below. Record user-selected paths there and in the delivery readme.
For a user-requested local-results/final-figures layout, use
`results: .local/results`, `process_figures: .local/figures`, and
`final_figures: figures`, with an explicit list of final figure names. Keep
diagnostic/WG/operator plots and historical versions in the local archive even
after review; their required analysis and evidence remain in scope. Renderers,
dashboard links and future exports must use that layout, rather than recreate
the old root directories. Preserve frozen evidence bytes and record relocation
mappings; resolve historical pins through the mapping instead of rewriting
their scientific contents. State that redrawing final figures requires the
local data. A later request for a self-contained shareable package is separate.

When simplifying a multi-scenario performance chart, preserve the declared
statistic and data lineage. If the user chooses a geometric mean, first compute
each scenario's throughput from its qualified latency samples, then take the
geometric mean across the same complete scenario set. Select one documented
cohort per iteration by evidence quality and chronology, never measured speed;
exclude diagnostic/control overlays. Record the source measurement IDs and
whether the reference is contemporaneously matched or a fixed reference.

| Content | Placement and Git policy |
| --- | --- |
| Optimization process, complete traces, agent interaction documents | Run records under `.local/runs/`; ignored. |
| Raw requests/state comparisons, telemetry, profiler exports, logs, IR dumps, temporary tensors | The exact revision's `results/`, `profiles/`, `logs/`, `dumps/` under `.local/`; ignored. |
| Draft or per-trial diagnostic figures | The exact revision's private `figures/`; ignored until deliberately exported. |
| Full artifacts, rdata, compiled libraries, checkpoints, caches and worktrees | `.local/` or recorded external storage; ignored/untracked. Generated source alone is not a runnable artifact. |
| Required optimization implementations, overrides, semantic inputs and selection/reproduction scripts | Git-managed `local_optimizations/` and delivery entrypoints. Save failed/temporary implementation snapshots privately. |
| Reusable compiler changes and regressions | Git-managed compiler sources and normal compiler test tree. Tutorial-only tests and temporary probes stay in `.local/`. |
| Public trace, readme, accepted source snapshot, compact shareable result data and formal figures | Git-managed delivery root; latest SVGs in `figures/`, data/receipts/history in `results/`. Preserve actual scope/status; estimates and diagnostic results are labeled accordingly. |

Draft figures are generated from a pinned revision's raw data. Publish compact
shareable data as `results/<export-id>/summary.json` (or an equivalent explicit
public path), with source/artifact/baseline identities, scenario/configuration,
measurement boundaries, sample/warmup counts, relevant samples/statistics and
gate status. Public rendering uses these committed inputs and `render_results.py`.
If the user explicitly places figure data or history in `.local/`, record that
override and its actual path. Keep only latest SVGs public and document that
rerendering requires the private data; do not claim private-free reproduction.
Do not put required public data in `figures/data/`: this repository ignores that
path. Check the destination's ignore rules before export instead of force-adding.

Keep versioned data and SVG exports under `results/<export-id>/`; keep only the
latest SVG for each figure under `figures/`, with stable names such as
`decode-tps-vs-iteration.svg`. Update that SVG, its data link and README caption
together, retaining historical provenance in `results/`. Do not overwrite a historical accepted figure
with a draft and leave its accepted label unchanged. Diagnostic/regression plots
may be exported when useful to the task, with their failed/unaccepted scope
explicit; they do not make a candidate accepted.

At handoff publish the latest `figures/performance-vs-iteration.svg` (or a name
reflecting the chosen metric) with its compact dataset under `results/<export-id>/`.
This is one cumulative figure across qualified
iterations, with workload/contract curves or compact workload panels, rather
than a figure or panel for each iteration. Verifier checks coverage of meaningful
iterations, regressions and unaccepted measurements, with their actual status.
Separate workloads, incompatible cohorts and measurement kinds; an unmeasured
failure has no invented latency. Public rendering reads committed data rather
than the private ledger.

For each execution-based candidate verification, also save
`profiles/operator-latency-<verification-id>.json` and
`figures/operator-latency-<verification-id>.svg` under that exact revision.
Retain previous verification figures. Verifier links raw profiling data,
measurement IDs, source identity, attribution and overhead in its report and
`validation.json`; Orchestrator links them from the trace. Public export copies
the latest relevant SVGs to Git-managed `figures/` and the self-contained
datasets, receipts and versioned exports to `results/<export-id>/`.

Use the [verification boundary](performance-history.md#operator-latency-after-verification)
when grouping preflights, scenarios and repeats. Keep pending profile/export work
in state without making it a prerequisite for unrelated ordinary checks; record
each actual run and failure. Required public handoff checks still apply.

For kernel development, also retain
`profiles/kernel/<capture-id>/` with the raw `.ncu-rep`, counter export, actual
WG/hardware events and identity/clock/coverage manifest, plus
`figures/kernel-swimlane-<capture-id>.svg`. Keep baseline/candidate and repeated
capture identities separate. Include capture commands, failure receipts,
overhead and independent analysis; never include sudo credentials. Public
exports contain self-contained trace data and rendering inputs for the relevant
WG and hardware-module swimlanes. Follow [Kernel microprofiling](kernel-microprofiling.md)
for interval semantics, required coverage and completion gates.

Stage public outputs in `WORK_DIR/exports/<export-id>/files/`. `export.json`
records the source trial/revision, matching review and validation identities,
reproduction status, source dataset and every promoted file's hash. Validator
checks the applicable gates, sample coverage, computed statistics, captions and
artifact/data identity. Promote the coherent file set only after these checks;
public source data embeds the provenance needed without the private receipt.

Before handoff, inspect changed/staged Git paths and confirm `.local/` remains
ignored and untracked. Public reproduction must rebuild from the checkpoint
plus Git-managed code/plans; public figures must render with `.local/` unavailable.
Do not silently move essential code out of an ignored run by changing its path:
copy the reviewed implementation into the importable public package, rebuild
from that package, and verify identity and acceptance before claiming reproduction.

For a byte-identical package relocation, verify source-file content hashes,
importable module names, dependency/target/recipe configuration, typed IR and
generated implementation against the reviewed revision. A filesystem-root
change alone can reuse that review when these execution inputs remain identical;
record the old/new roots and copy hashes rather than rewriting earlier identity.
If source, effective imports/registration, configuration, IR, generated code or
ABI changes, create a new candidate revision and obtain affected re-review.
If equivalence is unknown, do not assume the old review applies.

Build the public-package artifact and save its logs/results under
`WORK_DIR/exports/<export-id>/reproduction/`, leaving the reviewed original
artifact intact. `reproduction.json` links original/rebuilt input and artifact
identities, the matching review, path mappings and actual acceptance results.
Rebuilt artifact metadata or binary hashes may differ; record and test the new
artifact, and investigate code/contract differences before reusing review.
Changes confined to figure rendering or result packaging use a new export ID
and figure checks; they do not revise the model candidate unless they also
change compilation, runtime execution or the measurement contract.

## Maintenance and retention

Freeze implementation inputs, source snapshots, plans, IR and artifacts when
a revision enters review. Preserve completed handoffs, verdicts and raw evidence;
later phases add their own records without overwriting earlier evidence. Only
build outputs of an unfrozen, unreviewed and unmeasured revision are mutable.
A fix creates a new revision; re-review and validation bind to that revision.
Shared core fixes create a new baseline version and rebuild both sides.

Keep state/trace updates serialized, record actual process exits and refresh
paths/identities after resume. Reuse immutable caches only under verified source
and recipe identity. Never clean another agent's active outputs, referenced
readonly data, the active baseline or accepted deliverable.
Protect candidates still pending review, validation, reproduction or export.

Closed trials may shed redundant large build outputs after records, source/final
IR, plans, reproduction inputs and raw evidence are retained. Archive transfers
verify content and update `workspace.json`/trace before any redundant copy is
removed; retain a discoverable archive location. Rejected trials, interaction
documents and overall trace remain available. Mark interrupted or incomplete
runs honestly and preserve their last durable state for resumption.
