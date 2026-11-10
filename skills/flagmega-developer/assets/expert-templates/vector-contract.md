# Vector and representation contract

| Identity | Value |
| --- | --- |
| Assignment / stage / candidate | <assignment; proposed or implemented; exact revision> |
| Inputs / report | <pinned input manifest; relative response.md link, without a reciprocal hash> |
| Scope / check coverage | <packed/distributed/final checkpoints; V1–V8 evidence> |

## Complete type and lane inventory

| Function / value / tuple field | Role | Actual tensor.dtype | Scalar element / all lanes | Outer shape / scalar coordinate map | Bytes / strides / alignment / shard | Producer-consumer / function ABI | Decision / reason / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <stable identity> | <activation, residual, state, weight, result or interface> | <actual VectorType or scalar type> | <actual values> | <map> | <facts> | <all users and boundaries> | <correct, changed or justified scalar> |

<A pinned JSON/CSV can hold the full inventory. Group equivalent rows only with
explicit membership/counts. Op names or wide loads do not replace actual types.>

## Representation changes and physical ABI

| Value / interface | Before → after lanes and scalar-coordinate map | Byte geometry / tails / masks | Conversion / rounding invariant | Emitted buffer/access evidence |
| --- | --- | --- | --- | --- |
| <identity> | <exact mapping> | <sizes, alignment, coverage> | <contract> | <source/IR/native pin or pending> |

## Cross-domain handoff

| Receiving owner | Required interface / constraint | Input contract version | Agreement / open action |
| --- | --- | --- | --- |
| Graph | <semantic/rounding preservation or requested change> | <pin> | <status> |
| Distribution | <outer/lane shard units and owner byte spans> | <pin> | <status> |
| Kernel | <address, alignment, storage and caller/callee ABI> | <pin> | <status> |

## Benefit and verification boundary

<Actual byte/transaction/issue costs, counter-costs, source/evaluator/device
evidence, applicable upstream strategy and remaining measured gap.>
