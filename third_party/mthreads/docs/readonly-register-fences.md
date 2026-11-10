# Opt-in narrowing of dead tied register identities

`lower_dead_tied_register_fences=True` runs a post-optimization LLVM rewrite.
It defaults to `False`, requires a boolean and participates in the compiler
cache key. `readonly_register_fence_count` reports the number of replacements.

The only accepted call is a side-effecting empty inline assembly with exactly
`=R,0` constraints, a single FP32 operand/result and an unused result. It becomes
a side-effecting empty `R` input-only call at the same location. The input data
dependency, side-effect marker, calling convention, call attributes and metadata
are retained. Tail calls, operand bundles and return attributes are excluded.
Used results, nonempty instructions, other types/constraints and memory clobbers
are not rewritten. The helper is idempotent.

This is not a hardware fence or a memory publication optimization. No hardware
barrier, async wait, memory clobber, or memory operation is removed. An empty
assembly with only register constraints does not itself establish a memory
ordering guarantee. A caller needing ordering through the output keeps a used
result, which the transform explicitly excludes.

Compiler coverage is in `python/test/unit/tle/test_readonly_register_fence.py`:
exact LLVM predicate and negative cases, unchanged nonempty instructions,
used-result preservation, default behavior, option validation and cache identity.
Target-codegen and MP31 runtime checks are required separately before production
use; historical combined-kernel performance is not an isolated speedup claim.
