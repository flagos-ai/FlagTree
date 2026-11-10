# Expert responsibilities, workspaces and handoffs

Use this contract with [Agent flow](agent-flow.md#expert-dispatch-and-coverage).
The optimization scope remains the complete requested workload. Domain
specialists own their contracts; the main Architect reconciles them and the
main Optimizer integrates their implementations. Vector is the vectorization
and representation expert, including packing and layout propagation.

## Responsibility and interface ownership

| Domain | Owns | Required handoff to other domains | Requires another owner's decision |
| --- | --- | --- | --- |
| Graph | Semantic operations; graph/egraph rewrites and fusion; shared users; control/state/effects; rounding and constant-transform legality. | Component/call graph, typed semantic boundaries, legal fusion region and complete-value requirements, runtime work/bytes removed or retained. | Vector chooses concrete lane representation; Distribution chooses participants and reduction placement; Kernel realizes physical execution. |
| Vector | Actual TensorType/VectorType inventory; all lane components and scalar coordinate maps; packing propagation; producer/consumer and function-boundary representation ABI. | Logical-to-physical coordinates, bytes/alignment/tails, every affected producer/consumer ABI, conversion and rounding boundaries. | Graph approves changed semantics; Distribution approves shard/owner units; Kernel proves the emitted accesses implement the representation. |
| Distribution | Mesh/placement, tasks and participants; replicas/partials; complete-reduction points; physical writer ownership; communication/publication and distributed dependencies, including a single GPU's CTAs. | Per-value owners and byte spans, task/CTA mapping, reduction/transfer/publication graph, load balance and scheduling constraints. | Graph owns complete-value and effect semantics; Vector owns lane units; Kernel proves physical synchronization and resident-grid feasibility. |
| Kernel | TIR/recipes/overrides; selected native implementation; tile/warp roles and pipelines; memory/address/alias/lifetime contracts; barriers and launch/resource feasibility; required microprofiling. | Actual selected/emitted implementation, physical ABI/resources, producer/consumer schedule, NCU/WG/module evidence, measured perturbation and performance limits. | Changed arithmetic or rounding returns to Graph/Vector; changed owners, participant mapping or cross-CTA reduction returns to Distribution. |

An expert can inspect neighboring contracts and propose a cross-domain change.
It must name the receiving owner and required agreement; it cannot close that
owner's gate by including a paragraph in its own report. Kernel microprofiling
stays with KernelOptimizer and independent KernelArchitect review. Benchmark
or plotting helpers are bounded assignments within those responsibilities,
not replacements for Graph, Vector or Distribution experts.

## Canonical work directories

All paths below are relative to `WORK_DIR`, the private run root. Each new
assignment uses a fresh `aNNNN` ID within its directory; never overwrite a
completed assignment. Agent/context identity is recorded separately from this
stable filesystem role name.

| Domain | Planning: domain Architect | Implementation: domain Optimizer | Independent review: fresh domain Architect context |
| --- | --- | --- | --- |
| Graph | `agents/graph_architect/aNNNN/` | `agents/graph_optimizer/aNNNN/` | `agents/graph_review/aNNNN/` |
| Vector | `agents/vector_architect/aNNNN/` | `agents/vector_optimizer/aNNNN/` | `agents/vector_review/aNNNN/` |
| Distribution | `agents/distribution_architect/aNNNN/` | `agents/distribution_optimizer/aNNNN/` | `agents/distribution_review/aNNNN/` |
| Kernel | `agents/kernel_architect/aNNNN/` | `agents/kernel_optimizer/aNNNN/` | `agents/kernel_review/aNNNN/` |

Main coordination uses `agents/architect/aNNNN/` and
`agents/optimizer/aNNNN/`; independent whole-workload acceptance uses
`agents/validator/aNNNN/`. Orchestrator owns run state, the coverage record and
overall/public traces. Preserve existing runs' directory names through an
explicit `workspace.json` mapping; do not rename frozen legacy records or count
an old Joint assignment as several domain assignments.

Within each assignment:

| Path | Writer | Purpose |
| --- | --- | --- |
| `request.md`, `inputs.json` | Orchestrator, before dispatch | Role/domain, bounded scope, allowed writes/resources, required outputs and exact input/source/artifact/contract pins. |
| `response.md` | Assigned expert | Role-specific final report, findings, limitations and next recipient; names actual agent/context, domain and assignment. |
| `contract.md` | Domain Architect or Optimizer | Domain interface facts and proposed or implemented changes, using the domain template below. An independent reviewer reads and pins this file rather than modifying it. |
| `evidence/` | Assigned expert | Inventories, extracted facts, comparisons and raw audit results referenced by the report. |
| `source/`, `patches/` | Assigned Optimizer, when needed | Private implementation or a patch against pinned source; integration follows the assignment's allowed paths. |
| `commands/`, `logs/` | Assigned executor, when needed | Actual commands, outputs, exit statuses and preserved failures. |
| `handoff.json` | Assigned expert | Machine-readable result binding, with the fields below; no claimed completion based only on a plan. |

Create optional directories when used. Domain workers write their own
assignment and only the source paths explicitly assigned to them. They do not
edit another expert's report, overall state/trace, public exports or the active
Verifier's ledger. For concurrent implementation, allocate disjoint source
ownership or isolated worktrees; shared interfaces return to the main
Optimizer for ordered integration. GPU execution requires Orchestrator's
device lease and follows the existing NCU failure gate.

## Required templates

| Output | Template |
| --- | --- |
| Dispatch `request.md` | [Expert assignment](../assets/expert-templates/assignment.md) |
| Architect `response.md` | [Architect report](../assets/report-templates/architect.md) |
| Optimizer `response.md` | [Optimizer report](../assets/report-templates/optimizer.md) |
| Independent domain review `response.md` | [Reviewer report](../assets/report-templates/reviewer.md) |
| Graph `contract.md` | [Graph contract](../assets/expert-templates/graph-contract.md), covering G1–G7 |
| Vector `contract.md` | [Vector contract](../assets/expert-templates/vector-contract.md), covering V1–V8 and the complete type inventory |
| Distribution `contract.md` | [Distribution contract](../assets/expert-templates/distribution-contract.md), covering D1–D8 |
| Kernel `contract.md` | [Kernel contract](../assets/expert-templates/kernel-contract.md), covering K1–K10 and the existing microprofile handoff |
| Current `expert-coverage.json` | [Coverage skeleton](../assets/expert-templates/expert-coverage.json) |
| Per-assignment `handoff.json` | [Handoff skeleton](../assets/expert-templates/handoff.json) |

Bounded follow-ups may use the compact response form in
[Agent report conventions](agent-report-templates.md). Pin an unchanged domain
contract in the handoff instead of copying it into another report; changed
contracts still require their own versioned artifact and affected review.

Fill template fields with actual facts or explicit pending/not-applicable
reasons. At planning time, mark future implementation, device measurements and
independent acceptance as pending. Reference large inventories/evidence by
their pinned files rather than duplicating them in both the report and contract.
Reviewer reports bind the exact author contracts, actual implementation and
independence; the reviewer does not produce a replacement design contract.
The review template's `role: Reviewer` denotes this independent domain
Architect review phase, with its actual domain recorded separately.

Keep bounded handoffs concise: record the changed decision, exact inputs,
new evidence/failures and next executable action. Reference unchanged contract
sections and inventories by their existing pins instead of copying them into
every report. A worker can send a frozen result for the next dependent task
before unrelated analysis finishes; mark remaining outputs pending. Follow the
[candidate prerequisites](agent-flow.md#candidate-checks-and-acceptance) instead
of waiting for an entire role or review wave. Reuse never replaces independent
review of an affected contract.

`handoff.json` identifies the actual agent/context, role, domain, assignment,
run and candidate revision; pins its request, inputs, response, applicable
contract and evidence; states findings/open gates; names the next recipient
and required action. `source_review_passed`, `device_verified` and
`performance_accepted` are separate claims. Preserve original pins across
relocation with an explicit mapping; public exports use portable provenance.
Keep hash bindings acyclic: input and contract files do not hash their future
report/handoff. Reports can pin contracts, and the final handoff pins both;
use ordinary relative links for reverse navigation.

## Phase handoffs

1. **Baseline assessment.** Four separate domain Architects inspect the same
   baseline and produce domain reports/contracts. Orchestrator records actual
   bindings in `expert-coverage.json`; the main Architect consolidates their
   evidence and dependencies. Existing applicable domain evidence can be reused
   with verified lineage; missing evidence stays unassessed.
2. **Plan to implementation.** The main Architect freezes the joint hypothesis
   and domain contract versions. Orchestrator dispatches only the affected
   domain Optimizers with explicit source ownership. Each Optimizer records
   its realized contract and implementation/self-check evidence.
3. **Integration to review.** The main Optimizer freezes the integrated source
   and contributing handoffs, then builds the combined runnable artifact.
   Fresh domain Architect review contexts
   independently inspect every affected contract; their assignment directories
   use the `*_review` names above. A designer or implementer cannot independently
   approve its own work by changing its role label.
   Source review can start while native compilation and fixture preparation run;
   ordinary device checks require their actual execution prerequisites.
4. **Validation and acceptance.** Validator starts independent numerical,
   resource and ordinary screening checks when their actual execution
   prerequisites hold, alongside outstanding reviews. Before acceptance, the
   main Architect consolidates current domain verdicts and Validator verifies
   coverage, identity, independence and every agreed check/profile/reproduction
   gate. A domain plan or source-only review does not close those gates.

With four concurrent slots, keep one for Orchestrator and assign the other
three to ready work; refill them individually as tasks finish. Do not impose
a whole-wave barrier, collapse required domains or start idle experts merely
to fill a slot. A focused repair can use one affected domain; the whole-model
coverage record still shows the other domains and their current evidence.
