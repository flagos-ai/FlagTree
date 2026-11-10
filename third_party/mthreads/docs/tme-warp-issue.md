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

Compiler coverage is in `python/test/unit/tle/test_tme_issue_warp.py`: default and
opt-in lowering at 4/8/16 warps, aligned and unaligned explicit IDs, single-thread
initialization, architecture/type rejection and cache identity. The explicit-ID
tests inspect LLVM dialect before target code generation. Target compilation and
MP31 phase-reuse/runtime testing must pass before treating this option as ready
for production. Aggregate historical kernel results are not independent proof
for an isolated compiler change.
