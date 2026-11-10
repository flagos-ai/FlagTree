# Agent Markdown templates

Every dispatched agent writes `agents/<agent-id>/<assignment-id>/response.md`.
Use the full role template below for an initial domain assessment, integrated
candidate handoff or acceptance/reproduction decision. For these full reports,
retain the required frontmatter, section order and table columns; replace
placeholders before handoff. Use the user's requested language and explicit
`not_run`, `unavailable` or `not_applicable` with a reason; never invent facts
to fill a template. Bounded follow-ups use the compact form below.

| Role | Template |
| --- | --- |
| Orchestrator / flow agent | [Orchestrator](../assets/report-templates/orchestrator.md) |
| Main and specialist Architects | [Architect](../assets/report-templates/architect.md) |
| Main and specialist Optimizers | [Optimizer](../assets/report-templates/optimizer.md) |
| Independent post-implementation domain reviewer | [Reviewer](../assets/report-templates/reviewer.md) |
| Validator / Verifier | [Verifier](../assets/report-templates/verifier.md) |

All reports identify the actual agent/assignment, run/attempt/iteration/revision and
verdict, then pin inputs, checks/evidence, limits and a specific next handoff.
Link exact source/IR/artifact hashes and durable receipts; do not paste dumps
or chat transcripts. Check tables distinguish CPU/mocks, source/NVCC and GPU
execution, including actual commands/exits and failed/skipped/not-run checks.

For a bounded follow-up, retain the role template's identity/status frontmatter
and use only these three sections:

1. **Decision:** the scoped verdict and the exact prior full report being extended.
2. **Evidence and delta:** changed inputs, checks actually performed, results and
   limitations, linked to their durable evidence. Pin unchanged contracts and
   earlier evidence with verified lineage; do not repeat their inventories,
   baseline research or unaffected checklist tables.
3. **Open work and next handoff:** failures, pending gates, the next executable
   action, its owner and actual prerequisites.

The linked prior report and delta together must cover the assigned scope. If
there is no valid prior report or the scope changes substantially, use the full
template. Keep `handoff.json` and the evidence required for the decision; a
shorter report does not remove checks or establish acceptance. Separate execution
prerequisites from final acceptance gates using the
[candidate flow](agent-flow.md#candidate-checks-and-acceptance). Record pending
profile/review work with its actual dependent action, not a blanket blocked state.

The Markdown handoff must agree with the applicable hypothesis, implementation,
review, validation and performance-history records. A prose pass cannot
override a failed structured gate. Completed reports remain frozen; new inputs
get a new assignment/revision. Do not rewrite old frozen reports for formatting.

Every role separates `attempt_index` from `iteration_index` and states numerical
qualification. Reserve the attempt before evaluation; keep the iteration null
until Verifier passes the complete agreed task numerical scope and Orchestrator
records allocation with the exact artifact and receipt. A partial pass does not
qualify; a complete numerical pass with slower performance does. Frozen legacy
reports retain their original numbering, with an explicit versioned migration
and append-only corrections in the ledger/current trace view.

Verifier's cumulative iteration figure covers the qualified history in one SVG,
with workload curves or compact workload panels. Every execution-based candidate
verification also fills the mandatory **Operator latency figure** section under
the [verification boundary](performance-history.md#operator-latency-after-verification):
actual profiling execution, all-operator attribution/aggregation, raw data and
SVG identities, statistic/unit and call/token counts, and overhead or
unavailable/pending reasons with a next owner/action. A missing actual SVG keeps
that deliverable pending in `validation.json` and the trace. The per-verification
operator figure is distinct from the cumulative history figure.

## Expert-specific evidence

Use the fixed [expert directory and handoff contract](expert-workspaces.md):
the role template below supplies `response.md`, the domain template supplies
the Architect/Optimizer's `contract.md`, and `handoff.json` binds their actual
inputs and evidence. Independent reviewers pin the author's contract instead
of rewriting it.

Graph/Vector/Distribution/Kernel experts use the Architect or Optimizer template
for their role, identifying one assigned primary domain and filling the
domain-contract section. Bind the actual agent/context and assignment to the
current [expert coverage record](agent-flow.md#expert-dispatch-and-coverage).
Joint reports link separately dispatched expert handoffs and explain
integration; one Joint report does not establish four domain specialists.
Reviewers identify their domain and independence from its design and
implementation; reviewers inspect the same affected contracts.
Follow [Expert checklists](expert-checklists.md) and fill the check-coverage table
in the role template. Every check in an assigned domain is accounted for with
an inspected checkpoint/value, actual result/evidence or explicit reason.
Vector reports also link the complete tensor-type inventory defined there;
"vectorization checked" or zero adapter counts are insufficient evidence.

Every role's **Expert baselines and SOTA target** section follows the shared
table in [Expert baselines](expert-baselines.md): sources/search date, matched
contract, metric and baseline, predeclared target, current result/gap, evidence
scope and next action. Architects set targets; Optimizers prove implementation
and report outcomes; Reviewers check applicability and claims; Verifier checks
measured attainment; Orchestrator carries unmet targets into the next dispatch.
Affected performance experts also record the adopted industry implementation
as a starting point: upstream revision/license, concrete reused or ported
symbols and mapping/memory strategy, required numerical/ABI adaptation diff,
matched baseline reproduction, changes beyond the adopted implementation and
the measured gap to outperform it. Keep an outperform objective open until a
strict matched measured win; report theoretical bounds separately.
Performance claims distinguish reported numbers, local measurements and
theoretical proofs. Non-performance assignments give an explicit reason.

| Domain | Evidence when affected |
| --- | --- |
| Graph | Before/after graph and adapter counts, fusion/rounding legality, shared users/effects, reachable callees and whole-workload impact. |
| Vector | Scalar dtype/lanes, logical/physical shapes/strides, indexing/packing/rounding maps, producer/consumer/function ABI and removed buffers/copies. |
| Distribution | Placement/mesh/split units, replicas/partials, physical writer ownership, communication/publication, load balance and critical path. |
| Kernel | Family/candidate/override, emitted code, buffers/aliases/alignment/address width, lifetimes/barriers, launch/warp/resources/cache identity, successful ncu and per-WG/hardware-module trace/swimlanes with measured overlap assessment. |

Optimizer reports self-checks; Reviewer reports independent source/artifact
review; Architect reports conditional benefit. Verifier reports actual measured
acceptance. Orchestrator derives the next action and accepted best from receipts.

## Kernel microprofile handoff

Every role template includes **Kernel microprofile**. For Kernel performance
assignments and joint reports covering them, fill this section using
[Kernel microprofiling](kernel-microprofiling.md). Other assignments give a
specific `not_applicable` reason. Link one pinned capture manifest/analysis
instead of duplicating large raw tables, but explicitly state these outcomes:

| Required evidence | Report content |
| --- | --- |
| ncu gate | Baseline/candidate capture IDs, version/command/settings, exact source/library/workload, actual exit and inspected report/export hashes. On error, state `paused_ncu_error`, authorization availability and next action; never include credentials. |
| Recorder method | Chosen standalone or NCU-compatible path; pinned upstream method/ABI and actual reuse; event/role map, recorder/decoder identities, inspected timestamp/write instructions and raw format. For changes, follow [IKET-derived tracing](kernel-trace-method.md), including measured disabled/minimal/full controls and compatibility limits. |
| WG pipeline | Actual role/group/warp map, captured invocations/SMs/CTAs/tiles, raw trace identity, all-group coverage, dropped/unmatched events and clock validity. |
| Hardware working intervals | Actual modules and scope, event/correlation mapping to WG operations, measured activity versus in-flight/sampled/inferred evidence; unsupported/unused/missing modules and required follow-up. |
| Swimlane and analysis | WG plus hardware SVG/data/renderer identities; overlap denominators, fill/steady-state/drain, waits/bubbles, before/after bottleneck and falsification result. |
| Perturbation and review | Instrumentation identity, overhead/resources, replay/clock limits, independent raw-to-chart/calculation receipt and relation to uninstrumented verification. |
| Completion | Separate ncu, WG coverage, hardware intervals and analysis/review states; missing evidence remains pending with an owner/action. |

KernelOptimizer owns the capture and artifact delivery. KernelArchitect defines
the investigation and interprets the bottleneck; an independent instance audits
the evidence and claims. Orchestrator coordinates device ownership and the ncu
pause/access gate. Verifier checks exact revision coverage and independent
uninstrumented acceptance. A temporary capture helper may assist, but no new
standing agent role is required and accountability stays with KernelOptimizer.

## Public optimization trace

Orchestrator maintains `DELIVERY_DIR/optimization-trace.md` using the strict
[document template](../assets/report-templates/optimization-trace.md).
**Every meaningful entry uses [trace-entry.md](../assets/report-templates/trace-entry.md)
without omitting, renaming or reordering its sections/table columns.** Cover
baseline versions, trials/revisions, repairs, regressions/rejections, environment
or reference changes and exports that affect conclusions. Reserve stable entry
IDs and list entries in actual order; a repair does not erase the failed entry.

Each entry states the problem/evidence, conditional plan, actual changes and
identities, review/validation, every relevant measurement, decision and next
action. Metrics reference Verifier ledger IDs and sample provenance, including
diagnostics and failures. Unmeasured rows have explicit reasons rather than
invented numbers. Public data/figure links use Git-managed paths; optional
private archive pointers are labeled and cannot be reproduction dependencies.

At handoff reconcile entries with overall trace, implementation/review/validation
receipts and performance history. Check required sections/columns and all IDs,
metrics, statuses and figure provenance. For an existing unstructured public
trace, preserve its private snapshot/hash and migrate the meaningful facts to
the template; preserve earlier failed decisions and add correction entries when
facts change. Future entries must follow the template immediately.

Before each trace handoff, run the maintained structural validator:

```bash
python <skill-directory>/scripts/validate_trace.py <delivery-directory>/optimization-trace.md
```

Fix every reported error before handoff. It reads the assets as the schema and
checks document/entry section order, required table columns and metadata fields,
nonempty sections/cells, stable unique IDs/anchors, index-to-entry agreement and
unreplaced placeholders. A format pass does not establish the truth of metrics,
review gates or provenance; reconcile those against the Verifier ledger and
actual evidence separately.
