# Representation optimization

Use this reference when Pack/Unpack, Cast, output precision, or layout adapters
obscure compute or interrupt fusion. Diagnose the representation contract before
choosing a rewrite; a cleaner dump alone is not a performance result.

## Identify the boundary and its owner

Compare the same function before/after target-independent optimization,
vectorization, packing, distribution, and TIR lowering. Trace both inputs and
all users of an unwanted node. Classify it before editing:

| Boundary | Proof or design needed |
| --- | --- |
| Constant/weight transformation | Fold the ordinary pure expression into the readonly-data recipe; preserve source identity and evaluate once. |
| Activation Pack/Unpack | Propagate the demanded layout to the producer and compatible consumers; verify coordinates and function ABI. |
| Storage view | Prove identical byte order, offsets, owner mapping and lifetime; a view must not allocate/copy at bufferization. |
| Numeric Cast | Decide whether rounding is required by the acceptance contract or may be removed. |
| Reshard/publication/reduction | Preserve distributed ownership and synchronization; this is not just a local representation change. |

Moving weight recipes to a separate section inside a function is a display
choice, not constant folding. Do not introduce a parallel family of weight-only
cast/unary/binary/transpose operations: keep ordinary operators compatible with
dataflow/egraph rewriting and evaluate pure constant expressions offline.

## Make producers emit the layout consumers need

Intermediate Pack nodes are legitimate while exploring vectorization choices.
They should not become permanent activation conversions when the producer can
already emit the required packed result. Follow the demand through supported
embedding, slice, reshape, transpose, elementwise, tuple and call boundaries.
At a slice, prove axis mapping and alignment of starts, strides and extents;
equal element counts do not prove that a Pack commutes with it.

Plan related producer, residual and normalization layouts together. Selecting
each operation's vector width independently can insert conversions even when
every local choice appears reasonable. Keep explicit agent choices authoritative;
offer coherent alternatives and ordinary compiler defaults with inspectable
legality. Multiple users may legitimately demand different layouts: do not
duplicate expensive computation, change exported types, or erase effects merely
to reach a zero-Pack count.

Distinguish **logical lane count** from **vector packet geometry**. For example,
with a 16-byte packet contract, widening `f16<8>` to F32 can require
`f32<2,4>` on the same outer tensor shape. Its two lane components map to the
same logical axis. Neither `f32<8>` nor `f32<4>` with a changed outer extent is
the same representation contract. Derive packet width from the target/operand
contract, not model identity or an unqualified global constant.

Track every lane component in type inference, evaluation, split units, scalar
offsets and codegen. Using only `lanes[0]`, requiring one lane component per
logical axis, or assuming equal vector rank across a Cast loses this information.
Do not fix incorrect producer/consumer layout selection by merely renaming Pack
to Bitcast. A genuinely storage-preserving view is valid, but is not a substitute
for propagating the correct production layout.

For reusable functions, propagate one compatible ABI through callees and all
call sites. An output-only layout change can invalidate a later call argument
even when that callee's input signature is unchanged. Resolve substitutions for
every argument, preserve topological order, and test sequential and tuple-return
calls. Check that boundary propagation reaches a semantic fixed point rather
than repeatedly moving inverse views between caller and callee.

## Remove conversions at the semantic stage that owns them

An `A -> B -> A` chain is not generally numerically identical to its input.
When the agreed policy permits dropping that rounding, canonicalize the chain
in target-independent optimization. Do not first encode it as PostOps and then
invent kernel-specific machinery to erase it. Under an exact policy, preserve
the conversion order and fuse only its materialization where legal.

When a consumer needs a wider projection, use the producer's explicit output
dtype if supported. Carry it through scalar/vector/packed op definitions,
handwritten builders and patterns, Python edit/resume, fusion, distribution,
candidate selection and every applicable kernel variant. An FP32 output buffer
does not prove FP32 output semantics: an evaluator or epilogue may still round
the accumulator to a narrow dtype and widen it again. Use a cancellation or
half-ULP test that distinguishes those implementations, not a loose tolerance.

Order rewrites by semantic dependencies. A small bottom-up Cast fold can destroy
the pattern needed to form a larger multi-branch fusion. First recognize the
larger semantic region, then canonicalize remaining output conversions where
appropriate. Prefer pattern-based rules grouped in the dataflow pass; use a
separate phase only for a real ordering dependency, not one pass per fusion.

For conversions that remain required, PreOps/PostOps can keep them in registers
when the op and kernel family implement that boundary. Supply full inference,
evaluation, effect, alias and codegen support; attributes or pretty-printing
alone do not implement a fusion. Private-use and exported-value checks remain
necessary. Never move rounding across a partial reduction or publication edge
without proving the resulting distributed semantics.

## Attribute the performance effect

A materialized Pack or Cast may cost reads/writes, a temporary buffer, a kernel
or device call, synchronization, and a broken producer/consumer pipeline. Its
largest cost can be the larger fusion it prevents. Offline weight packing has
a different cost: compile/materialization time, cache reuse, and later weight
access efficiency. A proven alias view may have no runtime work at all.

Removing a Cast can still increase traffic or registers if it widens a stored
result. Fusion can increase live ranges, spills or instruction footprint.
Different packet geometry can change vector access and lane indexing. These are
measurement hypotheses, not reasons to claim fewer nodes must be faster.

Reinspect selected layouts after a representation change, even when the scalar
work and byte count are unchanged. An approximate objective counting outer
vector elements can change its ranking when one logical group becomes multiple
packets, while publication costs stay fixed. First check whether the desired
local candidate is legal, then compare complete plans. Use an explicit agent
selection and re-solve coupled constraints to measure the alternative; do not
delete a required restore/Boxing node or adjust global cost constants just to
force one benchmark. Keep unrelated selections fixed for a causal comparison.

Validate at three levels: focused op/rule/pass tests with negative cases;
actual selected device kernels and neighboring variants; then independent
stateful acceptance and end-to-end timing under the agreed workload scenarios.
Retain fail-before/pass-after evidence and compare fresh immutable artifacts on
the same device and measurement boundary. Inspect generated source, buffer
plans and resource reports to verify that the intended materialization or
rounding actually disappeared. Report remaining conversions and their reasons,
not an unqualified promise that every Pack/Cast should be eliminated.
