# Noinline pipe and token calls

Pipe analysis temporarily inlines helpers to bind lifecycle operations to the
actual pipe instance. This PR marks each call scope, lowers synchronization
state, and restores a reusable private function with explicit control-state
operands. Token parameters are replaced in the callee's original operand order,
so different instance creation orders cannot change the call ABI.

Ordinary helper barriers are specialized by their caller's warp-group scope,
while barrier-free helper functions stay shared. The register allocator retains
a fixed ABI across device calls. These native compiler hooks are enclosed in
the __TLE__ feature guard; their non-TLE projection is identical to main.

## Scalar return regression

The extracted restoration pass initially created only void function types and
rejected every result used outside a marked call scope. A noinline pipe helper
that commits a token and returns value + 1 therefore failed compilation, even
though scalar noinline returns are valid Triton calls.

The root locations are TleLowerPipeToNvws.cpp::inlinePipeCall and
TleRestorePipeFunctionCalls.cpp::createOutlinedFunction/runOnOperation.
Return operands previously lost their call boundary during temporary inlining.
A pipe.call_return marker now preserves returned values and result aliases,
including nested calls, multiple results and direct argument returns. The
restoration pass builds the correct function result types, returns mapped
values, rewires external users to tt.call results and validates repeated
instances' returned-value equivalence. Unexpected escaping results remain
diagnosed; no call is silently kept inlined as a fallback.

~~~bash
pytest python/test/tle/integration/test_tle_pipe_scalar_return.py -q
lit -v third_party/tle/test/GPU/test_tle_restore_pipe_scalar_results.mlir
~~~

The GPU reproducer fails before this correction and passes afterward. The IR
fixture covers repeated result-bearing calls and a direct argument return.
The noinline ABI GPU suite verifies one shared final LLVM helper definition
and surviving calls for two distinct fieldless instances with same/different
diagnostic names. Neighboring regressions cover token operand ordering,
warp-count/register ABI, whole-pipe drain, capacity wrap and TMA handoffs.
