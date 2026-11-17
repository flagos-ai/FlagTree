# Opt-in MP31 TME warp issue

`tme_issue_warp=True` selects warp-uniform issue predicates for MP31 TME
transactions. It defaults to `False` and participates in the compiler cache key.
Non-boolean values and enabled use outside MP31 are rejected.

For a warp-aligned issuer ID, transaction-byte addition, copy and no-result
arrival execute in the issuer's warp rather than only its first lane. Barrier
initialization remains single-threaded. An explicitly unaligned issuer ID keeps
its original single-thread predicate. Explicit physical issuer IDs retain the
physical thread domain; implicit issuers retain the existing local thread domain.

This option is an explicit caller contract: predicate and transaction operands
must be warp-uniform, and the selected warp must participate uniformly. It does
not make arbitrary divergent code collective. It changes no expected-byte/count
values, waits, barrier scope, pipe ownership, or TCE commit/wait groups.

## Why use a warp-uniform issue predicate?

The MP31 [MUSA Tensor programming guide][musa-tensor] separates the
transaction-byte counter from the arrival counter. The vendor's [MUTLASS MP31
FMHA producer][mutlass-fmha] calls the TME copy and transaction-barrier
operations in a branch selected by warp ID, not lane ID. Its [pipeline][mutlass-pipe]
counts one producer for that issue group; its [barrier wrapper][mutlass-barrier]
lowers producer transaction accounting and arrival to separate builtins.

- **Add-trans** registers outstanding bytes for the current barrier phase.
  Supply the total bytes of the logical copies attached to that barrier,
  not the number of issuing lanes times that total.
- **Copy** delivers the tile asynchronously; the global-to-shared form uses
  the same barrier ID to report transfer completion. Issuing the copy is not
  sufficient for a consumer to read the destination. Shared-to-global TME
  store still uses its own existing commit and read-wait protocol, not the
  load's transaction-barrier accounting.
- **No-result arrive** contributes an arrival for the issuer group. It does
  not add bytes or perform a copy. A waiter can proceed for that phase only
  after the expected arrivals and tracked transfers have completed. This
  option does not change the phase-returning arrive or the wait operations.

Issuing from the entire selected warp with uniform operands follows the vendor
example's control flow. The public example supports that programming pattern;
it does not establish that every partially active mask is invalid on every
MP31 stepping, or that every lowering has identical behavior for a partial
warp. The default therefore retains the existing single-lane issue path.
Do not increase counts to 32 when enabling the option.

## Validation

Compiler coverage is in `python/test/unit/tle/test_tme_issue_warp.py`: default
and opt-in lowering at 4/8/16 warps, aligned and unaligned explicit IDs,
single-thread initialization, architecture/type rejection and cache identity.
The explicit-ID tests inspect LLVM dialect before target code generation.

The MP31 runtime regression in `python/test/unit/tle/test_tme_issue_warp_device.py`
checks both options with one, two and three stages. It deliberately uses a
runtime loop, puts up to three different tiles in flight before the first
wait, and compares each copied raw FP16 payload and the untouched guard
regions against a host reference. A shorter repeat exercises different
runtime bounds without changing the kernel options.

Aggregate historical kernel results are not independent proof for this
isolated compiler change. This check does not qualify arbitrary divergent
participation or establish a performance gain.

[musa-tensor]: https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/musa_cpp_syntax/tensor_programming/
[mutlass-fmha]: https://github.com/MooreThreads/mutlass/blob/main/experimental/fmha/collective/fmha_collective_tme_warpspecialized.hpp#L265-L327
[mutlass-pipe]: https://github.com/MooreThreads/mutlass/blob/main/include/mutlass/pipeline/mp31_pipeline.hpp#L131-L232
[mutlass-barrier]: https://github.com/MooreThreads/mutlass/blob/main/include/mutlass/arch/barrier.hpp#L133-L181
