# Region-owned mbarrier lifetimes

TLE mbarrier allocations own their hardware objects within the defining
structured region. In a function with multiple blocks, ownership starts at
initialization and extends through the blocks it dominates. A caller-provided
descriptor is borrowed. Allocate in an enclosing region or caller when users
need a longer lifetime.

## Root cause and ordering

`third_party/tle/dialect/lib/Transforms/TleLowerBarriers.cpp`,
`TritonTleLowerBarriers::runOnOperation`, previously lowered allocation to
`ttng.init_barrier` and lowered wait/arrival operations without publishing
initialization or retiring the hardware objects. Initialization executes on one
thread. Other participants must observe initialization before waiting or
starting a TMA transaction. NVIDIA mbarrier objects must be invalidated before
their shared bytes can serve an unrelated allocation. Sequential outlined
helpers and allocations inside runtime loops expose these missing boundaries.

The pass records the slots that it initializes, publishes initialization, and
uses shared-memory alias analysis to reject an owned descriptor that escapes
through a return, yield, or CFG edge that leaves the initialized scope. At each
exit from that scope it inserts a rendezvous, invalidates each owned slot, then
inserts another rendezvous. This covers returns reached after initialization,
joins with paths that bypass initialization, and backedges that reexecute the
allocation. Invalidation keeps the shared bytes live until retirement. An outer
allocation remains owned by its outer region when a loop or warp-specialization
partition borrows an alias.

Named barriers are integer hardware resources, rather than mbarrier objects in
shared storage. Their frontend descriptors must not acquire mbarrier
initialization, publication, or retirement. Publication belongs inside the
mbarrier allocation branch; placing it after that branch incorrectly adds a
rendezvous for named-only captures. The regression covers this distinction.

## Reproduce and validate

```sh
lit -v third_party/tle/test/GPU/test_tle_mbarrier_lifetime.mlir \
       third_party/tle/test/GPU/test_tle_mbarrier_lifetime_cfg.mlir \
       third_party/tle/test/GPU/test_tle_mbarrier_lifetime_errors.mlir
pytest -q python/test/tle/unit/test_tle_mbarrier_lifetime.py
```

The CFG test reproduces the failure from #1541 when an early return puts a
MiniMax sparse prefill barrier in a multi-block function body. It checks
retirement only on paths that ran initialization, including the edge into a
shared continuation and the backedge before reinitialization.

With the barrier pass from main `bb8a53cc427f`, both original MLIR files fail:
the positive file lacks publication/retirement, and the diagnostics file accepts
owned descriptors that escape. The Hopper GPU case
`test_owned_mbarrier_storage_reuse[different_helpers-4]` fails with two PTX
`mbarrier.init` instructions and zero `mbarrier.inval` instructions. The initial
extracted patch also fails the named-only warp-specialization regression because
it publishes a nonexistent mbarrier object.

The GPU matrix uses four/eight warps, 64 CTAs, and 32 iterations. It covers
sequential helpers with different shared-memory shapes, outer-owned barriers
borrowed by loop calls, and inner-owned barriers reinitialized on each
iteration. Outputs are compared exactly, and PTX must retire every initialized
mbarrier. The descriptors come from the existing TLE tensor-map test helper.
The test requires NVIDIA Hopper or later and does not run on HIP targets.

The change is confined to TLE-owned lowering, operation/pass contracts, and
regressions. Neighbor validation covers named barrier allocation, borrowed
barriers, TMA copies, pipe lowering, and async WGMMA rings. NVIDIA and AMD are
built together; no native lowering, optimization flag, or hardware limits are
changed.

## C++17 compiler portability

The alias-check lambda must capture a normal local reference. Capturing the
`barriers` variable from a structured binding in the surrounding map loop is a
C++20 extension. The older Clang used by Sunrise rejects that source while
compiling this shared TLE pass as C++17. A strict Clang C++17 syntax check also
reproduces the failure at the lambda's `for (Value alloc : barriers)`.

Use an ordinary map entry and named `function`/`barriers` locals before creating
the lambda. This removes the unsupported capture at its source without changing
ownership analysis, requiring a newer language standard, or modifying a backend.
Validate the actual translation unit with its CMake include/define arguments and
`clang++ -std=gnu++17 -fsyntax-only -Werror=c++20-extensions`; the structured-binding
version fails and the ordinary-reference version passes. The Sunrise CI build
is the corresponding backend regression.

A standalone reproducer for the unsupported capture (`capture.cpp`):

```cpp
#include <utility>
void capture() {
  std::pair<int, int> entry;
  const auto &[function, barriers] = entry;
  auto check = [&] { return barriers; };
  (void)check;
}
```

```sh
clang++ -std=c++17 -Werror=c++20-extensions -fsyntax-only capture.cpp
```
