# Structured repetition and parameter banks

Repeated model layers are represented as a closed `control.for` region. The
loop has a scalar induction value, explicit captures, typed `iter_args`, and a
`Yield` result for every carried value. The backedge is a region contract; it
must not be encoded as a cyclic ordinary node graph.

Layer parameters use a `ParameterBankSchema` and rows of checkpoint keys. A
bank is a read-only indexed collection whose index is separate from tensor
axes and distributed SBP. A row may contain heterogeneous fields, but every
row has exactly the same schema. Per-field transforms are named in the schema
and are applied at the bank access/lowering boundary, not copied into every
layer's importer graph.

Importer requirements:

- detect repetition from the model's original layer schedule and preserve
  non-periodic exceptions and order;
- create one body region per compatible schema, with explicit state and
  parameter captures;
- keep cache/state indices distinct from parameter-bank row indices;
- never form a bank by stacking incompatible shapes or dtypes;
- keep checkpoint source identity and row keys in the semantic hash.

The e-graph treats a region as a lexical scope. Body variables are matched by
binding position, not spelling. Rewrites may simplify pure body expressions and
move loop invariants only with an effect and alias proof. A rewrite must
preserve the types and distribution of every `iter_arg` and `Yield` field.
The default extractor counts one body plus the runtime trip count; it never
unrolls a loop merely to improve a local expression cost.

AutoDistributed gives each carried field one fixed loop interface. The initial
value, body argument, yielded value, and loop result must agree, with explicit
reshards when a body layout differs. Operation cost is charged once for the
body and multiplied by the proven trip count; bank storage is charged once as
readonly data.

Bufferize and codegen keep the loop marker and bank root visible. A bank row
access becomes a typed byte-offset view or descriptor lookup; it must not
materialize all rows or introduce one pointer formal per field. Workspace and
state buffers are reused across iterations only after lifetime and
synchronization proofs. Dynamic bounds require a target capability and a
runtime trip-count contract; otherwise the importer must use a constant range.

Acceptance requires zero implicit captures, matching init/iter/yield/result
types, alpha-equivalent serialization after reload, correct zero/one/multiple
trip behavior, order-preserving numerical comparison against an unrolled
reference, and a structural report showing that body node count is independent
of the number of repeated rows. Compute bodies still obey the no-`Pack`
acceptance rule.
