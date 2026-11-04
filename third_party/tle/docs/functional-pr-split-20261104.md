# Functional FlagTree PR split — 2026-11-04

This index replaces the backend-sized drafts #1521, #1522, and #1523 with small functional reviews. Every implementation PR carries one functional change and has an English title and description. Repository format-bot commits may follow the functional commit. The implementation set contains 6 native PRs, 26 TLE PRs, and 5 FlagCX PRs; this document is a separate documentation PR.

The split starts from `main` at `98403b2a16d0ad857bab013c29d1971960df290f`. Changes originate from `feature/flagmega` at `45e17f38d68cbd344b668e37c041520f8bcbf3e4`. The source branch is unchanged. Model/tutorial code, weights, generated extensions, build products, local reports, and the `triton_shared` gitlink are excluded. No independent MThreads backend change exists in this source range, so no empty PR was opened.

## Review and merge order

Native PRs target main independently. The shared/NVIDIA TLE chain targets main at its first PR; each subsequent PR targets the preceding functional branch. AMD and Iluvatar TLE stacks branch from the shared-offset ABI PR. FlagCX is a separate stack from that same foundation. Merge and retarget in dependency order. The native reproducer drafts document unresolved defects and must not be treated as fixes.

## PR index

| Category | Backend | Function | PR | Base | Files at extraction |
| --- | --- | --- | --- | --- | ---: |
| native | amd | Add trig compatibility and separately rounded multiply APIs | [#1524](https://github.com/flagos-ai/FlagTree/pull/1524) | main | 2 |
| native | iluvatar | Derive num_threads from the target warp size | [#1525](https://github.com/flagos-ai/FlagTree/pull/1525) | main | 2 |
| native | iluvatar | Add an explicit load-vectorization compile option | [#1526](https://github.com/flagos-ai/FlagTree/pull/1526) | main | 11 |
| native | iluvatar | Respect element size when choosing vector access width | [#1527](https://github.com/flagos-ai/FlagTree/pull/1527) | main | 3 |
| native | iluvatar | Isolate scalar spill corruption across device calls | [#1528](https://github.com/flagos-ai/FlagTree/pull/1528) | main | 3 |
| native | iluvatar | Cover nonzero MMA reduction recurrence | [#1529](https://github.com/flagos-ai/FlagTree/pull/1529) | main | 1 |
| tle | common/nvidia | Parse ptxas kernel resource usage | [#1530](https://github.com/flagos-ai/FlagTree/pull/1530) | main | 3 |
| tle | common/nvidia | Preserve shared arena geometry in pointer and buffer views | [#1531](https://github.com/flagos-ai/FlagTree/pull/1531) | [#1530](https://github.com/flagos-ai/FlagTree/pull/1530) | 19 |
| tle | common/nvidia | Retain local-pointer proofs through load/store lowering | [#1532](https://github.com/flagos-ai/FlagTree/pull/1532) | [#1531](https://github.com/flagos-ai/FlagTree/pull/1531) | 7 |
| tle | common/nvidia | Preserve call-site argument hints during inlining | [#1533](https://github.com/flagos-ai/FlagTree/pull/1533) | [#1532](https://github.com/flagos-ai/FlagTree/pull/1532) | 5 |
| tle | common/nvidia | Bound coalescing analysis and propagate gather axis facts | [#1534](https://github.com/flagos-ai/FlagTree/pull/1534) | [#1533](https://github.com/flagos-ai/FlagTree/pull/1533) | 19 |
| tle | common/nvidia | Merge compatible layout proposals without losing consensus | [#1535](https://github.com/flagos-ai/FlagTree/pull/1535) | [#1534](https://github.com/flagos-ai/FlagTree/pull/1534) | 4 |
| tle | common/nvidia | Honor explicit layouts across memory and structured control flow | [#1536](https://github.com/flagos-ai/FlagTree/pull/1536) | [#1535](https://github.com/flagos-ai/FlagTree/pull/1535) | 31 |
| tle | common/nvidia | Track shared byte ranges and borrowed memory across calls | [#1537](https://github.com/flagos-ai/FlagTree/pull/1537) | [#1536](https://github.com/flagos-ai/FlagTree/pull/1536) | 7 |
| tle | common/nvidia | Enforce explicit async-copy ownership and exact fill bits | [#1538](https://github.com/flagos-ai/FlagTree/pull/1538) | [#1537](https://github.com/flagos-ai/FlagTree/pull/1537) | 23 |
| tle | common/nvidia | Expose tensor-map publication and descriptor views | [#1539](https://github.com/flagos-ai/FlagTree/pull/1539) | [#1538](https://github.com/flagos-ai/FlagTree/pull/1538) | 8 |
| tle | common/nvidia | Preserve WGMMA accumulator layouts and async lifetimes | [#1540](https://github.com/flagos-ai/FlagTree/pull/1540) | [#1539](https://github.com/flagos-ai/FlagTree/pull/1539) | 17 |
| tle | common/nvidia | Publish and retire region-owned mbarriers | [#1541](https://github.com/flagos-ai/FlagTree/pull/1541) | [#1540](https://github.com/flagos-ai/FlagTree/pull/1540) | 6 |
| tle | common/nvidia | Preserve physical mesh domains for grid barrier groups | [#1542](https://github.com/flagos-ai/FlagTree/pull/1542) | [#1541](https://github.com/flagos-ai/FlagTree/pull/1541) | 13 |
| tle | common/nvidia | Give reusable pipes structural identity and drain semantics | [#1543](https://github.com/flagos-ai/FlagTree/pull/1543) | [#1542](https://github.com/flagos-ai/FlagTree/pull/1542) | 26 |
| tle | common/nvidia | Restore reusable noinline pipe and token calls | [#1544](https://github.com/flagos-ai/FlagTree/pull/1544) | [#1543](https://github.com/flagos-ai/FlagTree/pull/1543) | 24 |
| tle | common/nvidia | Pass shared device-call arguments as arena offsets | [#1545](https://github.com/flagos-ai/FlagTree/pull/1545) | [#1544](https://github.com/flagos-ai/FlagTree/pull/1544) | 8 |
| flagcx | common/nvidia | Match generic-pointer device-library call signatures | [#1546](https://github.com/flagos-ai/FlagTree/pull/1546) | [#1545](https://github.com/flagos-ai/FlagTree/pull/1545) | 8 |
| flagcx | common/nvidia | Own distributed contexts and bootstrap group cleanup | [#1547](https://github.com/flagos-ai/FlagTree/pull/1547) | [#1546](https://github.com/flagos-ai/FlagTree/pull/1546) | 6 |
| flagcx | common/nvidia | Cache native allocators and expose symmetric tensor allocation | [#1548](https://github.com/flagos-ai/FlagTree/pull/1548) | [#1547](https://github.com/flagos-ai/FlagTree/pull/1547) | 4 |
| flagcx | common/nvidia | Validate indexed barriers against actual window capacity | [#1550](https://github.com/flagos-ai/FlagTree/pull/1550) | [#1548](https://github.com/flagos-ai/FlagTree/pull/1548) | 16 |
| flagcx | common/nvidia | Scope collective helper barriers to warp groups | [#1551](https://github.com/flagos-ai/FlagTree/pull/1551) | [#1550](https://github.com/flagos-ai/FlagTree/pull/1550) | 5 |
| tle | amd | Lower count and phase barriers to LDS state | [#1552](https://github.com/flagos-ai/FlagTree/pull/1552) | [#1545](https://github.com/flagos-ai/FlagTree/pull/1545) | 6 |
| tle | amd | Lower grid barriers and initialize launch scratch | [#1553](https://github.com/flagos-ai/FlagTree/pull/1553) | [#1552](https://github.com/flagos-ai/FlagTree/pull/1552) | 10 |
| tle | amd | Lower static wave partitions and launch their full CTA | [#1554](https://github.com/flagos-ai/FlagTree/pull/1554) | [#1553](https://github.com/flagos-ai/FlagTree/pull/1553) | 9 |
| tle | amd | Lower reusable pipes to the RDNA barrier protocol | [#1555](https://github.com/flagos-ai/FlagTree/pull/1555) | [#1554](https://github.com/flagos-ai/FlagTree/pull/1554) | 11 |
| tle | iluvatar | Convert explicit layouts across control flow | [#1557](https://github.com/flagos-ai/FlagTree/pull/1557) | [#1545](https://github.com/flagos-ai/FlagTree/pull/1545) | 10 |
| tle | iluvatar | Track pipe identity and reusable warp-group barriers | [#1558](https://github.com/flagos-ai/FlagTree/pull/1558) | [#1557](https://github.com/flagos-ai/FlagTree/pull/1557) | 8 |
| tle | iluvatar | Expose SME transport with explicit tile geometry | [#1559](https://github.com/flagos-ai/FlagTree/pull/1559) | [#1558](https://github.com/flagos-ai/FlagTree/pull/1558) | 19 |
| tle | iluvatar | Preserve convergence and scoped FP contraction | [#1560](https://github.com/flagos-ai/FlagTree/pull/1560) | [#1559](https://github.com/flagos-ai/FlagTree/pull/1559) | 8 |
| tle | iluvatar | Publish SME payloads across warp groups | [#1561](https://github.com/flagos-ai/FlagTree/pull/1561) | [#1560](https://github.com/flagos-ai/FlagTree/pull/1560) | 6 |
| tle | iluvatar | Lower physical grid barrier domains | [#1562](https://github.com/flagos-ai/FlagTree/pull/1562) | [#1561](https://github.com/flagos-ai/FlagTree/pull/1561) | 8 |

## Validation and remaining gates

The reconstructed complete NVIDIA/TLE/FlagCX source rebuilt through normal CMake. All 154 TLE MLIR tests passed. The added C++ helper suites passed 6 coalescing-slice and 5 compatible-proposal cases. Host/API checks passed 95 cases, with 7 CUDA skips and 3 subtests. H800 execution passed 57 selected cases: 48 exact masked-fill cases, 7 cyclic/empty pipe cases, and 2 shared-scatter cases. These are complete-source results, not independent builds of each extracted PR.

The complete native AMD API + AMD TLE + shared/NVIDIA TLE + FlagCX combination also rebuilt. All 166 MLIR cases passed, including the 12 AMD additions. The host suite again passed 95 cases, with 7 skips and 3 subtests; the native AMD math compile suite passed 5 cases in a separate pytest process. Separate processes are needed because the two backend conftest files both register `--device`.

The FlagCX-free AMD cumulative branch built successfully and passed all 12 AMD additions. Its full suite passed 158/161; three inherited remote-pointer fixtures have outdated operand-segment syntax. Their corrections are in the FlagCX call-ABI PR #1546. The complete combination includes those corrections and passes the same three tests. No test was disabled or filtered out to obtain the complete result.

Selected Python checkpoints were also checked on their own cumulative branches using the aggregate extension. The corrected FlagCX context checkpoint passed all 13 cleanup/bootstrap/ABI cases. This does not substitute for independent C++ checkpoint builds.

All PRs remain draft. Before readiness, compiler fixes require isolated checkpoint builds, failing-baseline/passing-fix evidence, root-cause review, and neighboring-path checks. AMD and Iluvatar need hardware correctness/reuse/concurrency/performance validation. Iluvatar compiler builds and device tests were not run in this split. FlagCX host tests use the capacity-query SDK overlay; stock SDK wrapper/C ABI compatibility and distributed execution remain unverified. The call-frame and MMA recurrence drafts are failing reproducers with no claimed compiler fix.

## Causal changes and scope

The physical-mesh fix preserves the original node/device/cluster/block topology through views, decodes IDs in that topology, then projects cluster axes. The three existing hierarchical-mesh regressions failed with the original feature implementation and passed with the fix; rank-mask assertions were retained.

Splitting exposed a builder/schema dependency: adding `group_domain_shape` requires updating every `DistributedBarrierOp` builder, including existing communication-space calls. Missing the two empty-domain arguments caused the isolated FlagCX-free build to fail at `triton_tle.cc`; the arguments now belong to the grid-domain PR and the cumulative build passes.

The context checkpoint preserves import dependencies, communicator/memory argument order, ownership checks, CUDA publication, and host rendezvous in one PR. Capacity-specific checks/assertions land in the later indexed-barrier PR. The final capacity assertions are retained; the factory regression also checks the exact `(comm_ptr, mem_ptr)` argument order.

AMD distributed-lowering source selection is gated by `FLAGTREE_TLE`. The final source audit reconstructs all previous backend changes, allowing only this build-gate correction, the additional factory ABI assertion, documentation replacement, and new native regressions. TLE native-source guards and TLE-owned MLIR test placement are retained.

This split adds no full-model performance measurement and makes no new claim against vLLM. Historical model timings are not validation for these extracted compiler PRs. The original feature checkout remains clean; `reports/flagmega` remains absent.

Repository format-bot commits were preserved on PRs #1524, #1526, #1528, and #1530. Python AST and C/C++ token comparisons confirmed formatting-only changes. Stacked bases can temporarily lead their children by a format commit; comparisons use the merge base, and branches should be rebased/retargeted during merging.
