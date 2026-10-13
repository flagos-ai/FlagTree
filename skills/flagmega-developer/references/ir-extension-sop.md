# IR extension SOP

Use this procedure for a new graph operation, a new TIR operation, or an
extension to an existing operation's attributes, result types, numerical
semantics, or supported layouts. Apply the checks relevant to the proposed
contract; document why a stage is unaffected rather than adding unnecessary
passes or kernel variants. This is a design and implementation checklist, not
authorization to implement a proposal when the user requested design only.

## 1. Define the operation and its contract

Identify the component's reference implementation and configured execution
path. Reuse an operation that expresses the semantics, or define a fused one
when needed; follow the skill's guidance on import granularity and ownership.
Separate reusable contracts from model-specific parameter values.

Write down the following before implementation, separately for each input and
result, including every tuple field:

| Contract | Decisions to record |
| --- | --- |
| Numerical semantics | Algorithm, normalization/reduction axes, epsilon placement, iteration order, accumulation dtype, intermediate rounding, and result dtype. State the required comparison policy and reference path. |
| Schema and API | Named operands, attributes, defaults, validation, handwritten builders and pattern helpers. Reject unsupported combinations explicitly. |
| Tensor representation | Logical shape, scalar dtype, vector lane shape, lane-to-axis mapping, divisibility/alignment, and supported dynamic extents. Distinguish scalar input support from packed output support. |
| Effects and constants | Reads/writes, mutable state, aliases, and whether constant evaluation is valid. Identify which transformations may duplicate, reorder, or lift the operation. |
| Distribution | Legal placement and per-axis policies for each operand/result, reduction/partial semantics, and necessary Boxing or publication edges. |
| Physical ABI | Scalar storage shape/strides, offsets, owner mapping, allocation scope, aliases, lifetime, scratch, synchronization, and caller/callee requirements. |
| Backend coverage | Applicable targets, candidate/kernel variants, capabilities and resource constraints; explicitly describe unsupported cases. |

Packing is a representation decision. An attribute such as `output_lanes`
must name the affected result fields and axes; an output-dtype attribute must
define the actual rounding path. Equal bytes or equal element counts do not
establish either contract.

## 2. Establish a reproducer and acceptance checks

Create a minimal typed graph or kernel that isolates the intended behavior.
For a defect, demonstrate the failing assertion/execution before the fix and
record the causal file, function/pass, and trigger. For new support, demonstrate
the missing type, rewrite, or lowering capability against an existing valid
input; a missing import is not evidence of a numerical or compiler defect.

Choose checks that distinguish the proposed behavior: exact scalar ordering
for a layout-only change, a rounding-sensitive input for a precision change,
or a dependency/address assertion for a propagation or ABI change. Preserve
before/after timing when performance is part of the claim. Follow
[Performance iteration](performance-iteration.md) for acceptance and measurement
boundaries; do not loosen them to make an implementation pass.

## 3. Complete the definition and edit/resume path

Implement or audit the op definition, registration, attribute normalization,
type inference, evaluator, effects, and cost. Update functional constructors,
handwritten helpers and matching patterns together. A permissive tensor type
pattern does not prove that the evaluator or kernel supports vector inputs.
Count scalar work and bytes across all lane components; outer vector element
counts alone can misprice a representation change and alter downstream choices.

Preserve existing default behavior when adding optional attributes. Check
Python emission/reload, attribute canonicalization, semantic hashes, verifier
rules, and old checkpoints without the attribute. Local extensions must be
importable in a fresh process. For a new TIR node, also supply traversal,
serialization, memory/effect analysis, and lowering support.

Keep numerical implementation separate from representation: a layout-only
evaluator can reshape the already rounded result if the declared scalar order
permits it. A wider result must actually retain the wider precision, rather
than storing a widened value that was already rounded narrowly.

## 4. Integrate transformations and function boundaries

Audit target-independent canonicalization, fusion, vectorization, packing,
producer-demand propagation, and constant handling. Add rules at the stage
that owns the transformation, with the required ordering and legality proofs.
For indexing chains feeding a consumer or coordinated state writes, use
[Consumer indexing fusion](consumer-indexing-fusion.md) to select the operation
boundary. Form a semantic region before packing/distribution when this exposes
its true legal layouts; prove indexing, effects and intermediate observations
before merging it. Physical access lowering belongs to the later ABI stage.
Check all applicable execution paths: FlagMega's post-boundary propagation
uses different paths for ordinary and large graphs, so registering a producer
rule on only one path leaves incomplete support.

For packed results, derive widths from actual consumer demands and target
contracts. Follow demand through `GetItem`, tuple, and reusable-call boundaries;
preserve unaffected result fields, scalar consumers, and consumers requesting
other layouts. Rebuild a shared producer once at its valid position instead
of cloning expensive work or moving state reads. Keep necessary adapters and
let proven inverse views fold; do not delete them based on a desired node count.

Resolve a consistent function ABI across definitions and all call sites,
including sequential calls and tuple returns. Check that propagation reaches
a semantic fixed point and that a second application does not add inverse
views. Audit constant recipes only where expressions are pure and invariant.
Read [Representation optimization](representation-optimization.md) for packet
geometry, conversion ordering, and storage-view proofs.

## 5. Connect distribution and candidate selection

Carry the new contract through distributed inference, candidate providers,
legality, costs, selection, and materialization. Packed lanes belong to the
element representation; they do not automatically add a tensor/SBP axis.
Preserve lane units when deriving split divisibility and local scalar extents.
Treat tuple fields independently when their dtypes or distributions differ.

Permit only distributions the algorithm implements. A global reduction needs
the full required data or a correct partial/collective formulation; a layout
extension is not evidence that a split is newly legal. Conversely, do not
force replication merely because it is the easiest renderer path. Keep
required Boxing and reduction edges explicit.

If the contract changes, regenerate affected proposals, selections and later
plans from the earliest valid checkpoint. Do not patch hashes or reuse a
selection solely because its candidate ID survived the change.

## 6. Complete lowering, storage, and device codegen

Audit candidate applicability, lowering, kernel semantic attributes, renderer,
templates/source, bufferization, and runtime argument binding together. Every
selected variant must implement the declared contract or reject it during
selection. Merely relaxing a renderer guard does not implement a layout.

Use the physical ABI's coordinate helpers. For a contiguous final-axis layout
with lane product `L`, scalar index `h` maps to logical coordinate `h // L`
and flattened lane coordinate `h % L`; this formula requires the declared
lane order and storage geometry. Include every lane component rather than
using only `lanes[0]`. Validate dtype, capacity, strides, offsets and owner
mapping, including alias/public-result lifetimes and synchronization.

A packed IR result can be produced by scalar stores into its ABI; it does not
require a separate packing kernel. A Pack/Unpack can become an alias only
after proving scalar byte order, offsets, ownership and lifetime. For a new
TIR operation, expose its memory spans and scratch/transfer requirements to
the ordinary planners instead of encoding hidden storage in the renderer.

## 7. Verify integration and report completion

Apply [IR acceptance and repair](ir-acceptance.md) to the complete affected
producer/consumer chain. After packing/layout propagation and after distribution,
compute bodies must contain no Pack, including reachable callees. Compile-time
weight recipes require proven constant provenance and offline materialization.

Use focused tests at the levels affected by the contract:

- Op/schema/evaluator: valid and invalid shapes, attributes, dtypes and lanes;
  per-field result types; numerical boundaries and old defaults.
- Rule/pass/ABI: shared and exported values, mixed layout consumers, tuple
  projections, repeated calls, effects, and propagation convergence; cover
  ordinary/large-graph paths when registration differs.
- Distribution/lowering: legal and rejected policies, partial handling,
  packed local extents, candidate applicability, and neighboring variants.
- Actual device execution: scalar addresses, tails/dynamic extents when
  supported, selected kernel numerical behavior, and memory/synchronization
  contracts. CPU results and skipped GPU tests are separate evidence.
- Integration: import or construct through the normal pipeline, emit/reload
  Python IR in a fresh process, compile and verify the artifact, then validate
  a representative producer/consumer chain and the requested workload scope.

Inspect final IR/TIR, generated source, buffer plans and resource reports as
needed. A removed Pack may already have been a zero-copy view; report compile
time, materialization, kernel count, memory traffic, and runtime improvement
only where measured. Scan neighboring rules/backends/lowering stages sharing
the changed contract. Retain the minimal reproducer, fail-before/pass-after
evidence for defects, causal explanation, and any unvalidated scope.

## Worked design example: MhcPre packed collapsed output

This example describes the collapsed-output extension. It does not imply
packed-input support or a new split-capable MHC algorithm.

For scalar BF16 streams `[T, C, H]`, an optional `output_lanes=(8,)` would change
only the third field from `bf16[T,H]` to `bf16<8>[T,H/8]`. Require divisibility;
keep `post` and `comb` FP32 and preserve the collapse's accumulation and
rounding. Their distributions and the existing input legality remain intact.

The complete path includes type inference/evaluation and optional-attribute
round trips; `GetItem(2)` consumer-demand propagation in both graph paths;
shared/scalar consumers and function ABI; distributed tuple inference;
candidate/lowering contracts; and output addressing using logical hidden
groups plus lane coordinates. The renderer must validate the third output's
ABI instead of dropping the blanket scalar check without replacing it.
Acceptance compares flattened output values and unaffected tuple fields,
executes the device kernel, and checks that matching consumer Packs fold
without duplicating MhcPre or introducing materialization.
