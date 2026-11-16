# Optional fixed-slot publication census

The NVIDIA TLE census is a diagnostic observation mode for a closed warp-specialized call graph. It leaves the existing publication, barrier, pipe and arithmetic operations in place. It is independent of the bounded ring recorder: selecting both modes is rejected. An ordinary program that does not configure the census keeps its original generated and native path.

## API and ownership

An entry calls `tle.gpu.census_configure(occurrences, ctas)` with positive constant counts. A scheduled mesh grid barrier may receive `census=occurrence`, a scalar signed i32. The lowering records three observations by the existing role-local thread zero: after the original role prebarrier and before its arrival atomic, after the successful phase acquire, and after the original role postbarrier. Grid-axis grouping and the original scratch-counter ABI remain unchanged.

`tle.gpu.census_mark(occurrence, point)` records one point numbered 0 through 3. `tle.gpu.census_mapping(role, point)` records an entry/exit SM ID for role 0 or 1. The scheduling layer must assign unique occurrence IDs per invocation and place mapping calls exactly once at each role entry and exit. The TLE native writer does not infer layer IDs or role identities from names.

The role-local thread ID is materialized before normal warp-specialization context rewriting. The writer must not derive leadership by hardcoding a physical producer warp. Normal transitive direct-call analysis forwards a distinct address-space-1 profile argument; it does not reinterpret a helper's previous last argument or the global-counter pointer. Missing configured scratch ownership, indirect/address-taken callees and incompatible ring events are rejected.

## Buffer layout

The normal NVIDIA runtime interprets profile scratch size as a per-CTA reservation. The writer uses the unadvanced root profile pointer. All offsets below are global-buffer offsets:

- Header: 64 bytes, written only by the host. The first four little-endian u32 fields are enable, version=1, occurrence count and CTA count.
- Timestamps: `occurrences * ctas * 4` little-endian u64 values starting at byte 64. The index is `((occurrence * ctas + linear_cta) * 4 + point)`.
- SM mapping: `[2, ctas, 2]` u32 values immediately after the timestamps.
- Per-CTA reservation: `align_up(ceil(logical_bytes / ctas), 64)`. The physical allocation is this reservation times the launch CTA count.

The host fills the complete allocation with 0xff, then writes the header outside the observed invocation. For 310 occurrences and 128 CTAs the logical size is 1,271,872 bytes, the reservation is 9,984 bytes per CTA, and the physical allocation is 1,277,952 bytes. The 6,080 padding bytes remain untouched. Barrier point 3 is reserved and remains UINT64_MAX. Runtime occurrence or launch-grid mismatch produces missing slots; the mandatory host completeness check rejects the observation.

The profile allocation must be disjoint from model tensors, descriptors, global barrier scratch and all other pools. Only its header enable controls recording. Model computation must not consume recorder output. A disabled invocation must leave all profile bytes unchanged.

## Interpretation and validation

The recorder reads `%globaltimer`; mapping reads `%smid`. These are observations at the actual emitted points, not hardware engine activity intervals. Inspect final PTX/SASS and test clock monotonicity and effective observed resolution on the selected device. Do not repair inter-SM timestamps with clock64 offsets. Noncontiguous SM IDs are valid; matching entry and exit IDs do not exclude intermediate migration.

For each barrier occurrence, validate per-CTA point ordering and, within the actual full-grid or axis group, `max(point0) <= min(point1)`. Do not impose a global departure barrier that the program does not have. Producer commit-return and consumer release-return markers have no required cross-role timestamp order. A final release can precede later reduction, activation and output stores; only a marker after the caller's actual drain denotes drain return.

Each invocation owns the same fixed slots. A multi-step decode window may reuse the allocation without clearing it in the hot path; its final contents represent only the last invocation. A separate sentinel-initialized single invocation is required for complete census readback.

Measure ordinary, diagnostic-disabled and the same diagnostic binary enabled separately. Record resource, ABI and runtime binding changes. Do not subtract an average overhead from individual intervals. Missing data, invalid ordering or perturbation comparable to the proposed explanation makes attribution inconclusive. The normal model acceptance contract remains separate from diagnostic full-array/state comparisons.

CPU tests are in `python/test/tle/unit/test_tle_publication_census.py` and `test_tle_census_contract.py`. Native ownership is in `third_party/tle/dialect/include/Tools/PublicationCensus.h` and the grid-barrier lowering. The explicit enable mode currently requires NVIDIA, 32-thread warps, one CTA per cluster, and a configured normal warp-specialized entry profile ABI.
