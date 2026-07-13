# Proton Vendor Adapter Minimal Patch

## Goal

This patch adds the smallest code skeleton needed for someone else to finish vendor runtime profiling support in Proton, starting with a `cann` adapter and leaving a clean path for future adapters.

It intentionally does **not** change current user-visible behavior for:

- `backend="cupti"`
- `backend="roctracer"`
- `backend="instrumentation"`

It only introduces the classes, interfaces, and data structures required to finish the work.

## Background and Current Scope

This work was introduced to make Proton ready for vendor-specific runtime profiling on non-CUDA/non-HIP accelerators, starting with CANN/Ascend. The immediate goal is to land a stable public API, artifact contract, adapter boundary, and degradation behavior before requiring every deployment to have a complete CANN profiling stack available.

The current implementation should be understood as a minimum viable CANN backend path:

- `backend="cann"` is accepted by the Python API.
- Proton can create the expected output bundle:
  - `.hatchet`
  - `.timeline.json`
  - `.meta.json`
  - `.vendor.json`
- CANN/Ascend device and driver extension points exist.
- A CANN runtime profiler path exists and can attempt to use CANN profiling inputs.
- If CANN profiling libraries or exported `msprof/aclprof` summaries are unavailable, the session degrades instead of failing.
- Degradation reasons are written to both metadata and vendor outputs.

This means the patch can pass the minimal acceptance criteria without proving that real CANN profiling data was collected. Real vendor enhancement still depends on the runtime environment providing loadable Ascend/CANN libraries such as `libacl.so` and `libacl_prof.so`, plus usable `msprof/aclprof` export files.

In short:

- Minimal API/artifact/degradation contract: in scope for this patch.
- Full real-device CANN metric collection and correlation: follow-up validation work.

## Expected User Experience

The implementation owner should treat the following Python usage as the target public contract.

### Example: CANN enhanced runtime profiling

```python
import triton.profiler as proton

session_id = proton.start(
    name="profile_run",
    context="shadow",
    data="tree",
    hook="triton",
    backend="cann",
    mode="runtime_base:vendor_metrics=aicore,bandwidth",
)

run_kernels()

proton.finalize()
```

### Meaning of the user-facing arguments

- `backend="cann"`
  Select the CANN vendor adapter explicitly.
- `mode="runtime_base:vendor_metrics=aicore,bandwidth"`
  Request two layers of collection:
  - stable base runtime metrics
  - vendor enhancement metrics from the selected adapter

### Expected outputs

For the example above, the user should receive:

- `profile_run.hatchet`
  Used for hotspot ranking, aggregation by kernel/operator/session, and regression comparison input
- `profile_run.timeline.json`
  Used for chrome trace / perfetto inspection of stream overlap, memcpy overlap, and synchronization gaps
- `profile_run.meta.json`
  Used for `run_id`, backend, device, driver/runtime version, schema version, and effective config
- `profile_run.vendor.json`
  Used for imported vendor metrics and correlation results

### Degradation rule

Even when `vendor_metrics` is requested, the first version must guarantee:

- base runtime metrics remain available and stable
- `.hatchet`, `.timeline.json`, and `.meta.json` are still produced
- vendor enhancement may degrade independently
- degradation reasons must be written into metadata and vendor outputs

## What Was Added

### 1. Artifact and metadata data structures

File:

- `third_party/proton/csrc/include/Data/Artifacts.h`

This file defines the output model Proton will need for the new runtime backend flow:

- `ArtifactKind`
- `VendorMetricState`
- `ArtifactPathSpec`
- `SessionArtifactLayout`
- `RuntimeVersionInfo`
- `DeviceSnapshot`
- `RuntimeTraceEventKey`
- `VendorMetricRequest`
- `VendorMetricAssociation`
- `SessionProfileMetadata`
- `VendorProfileArtifact`

These structures are designed around the target outputs:

- `<base>.hatchet`
- `<base>.timeline.json`
- `<base>.meta.json`
- `<base>.vendor.json`

### 2. Vendor mode parsing

Files:

- `third_party/proton/flagtree_profiler/csrc/include/Profiler/Vendor/Mode.h`
- `third_party/proton/flagtree_profiler/csrc/lib/Profiler/Vendor/Mode.cpp`

This adds a common parser for runtime vendor profiling mode strings such as:

```text
runtime_base:vendor_metrics=aicore,bandwidth
```

Two key structures are defined:

- `VendorProfileOptions`
  Raw parsed request
- `VendorProfilePlan`
  Adapter-normalized execution plan after capability checks

The parser is intentionally generic so later adapters can reuse the same syntax.

### 3. Vendor adapter framework

Files:

- `third_party/proton/flagtree_profiler/csrc/include/Profiler/Vendor/Adapter.h`
- `third_party/proton/flagtree_profiler/csrc/lib/Profiler/Vendor/Adapter.cpp`

This adds the base extension points:

- `VendorMetricsImporter`
- `VendorAdapter`
- `VendorAdapterRegistry`

This is the minimal framework needed to avoid baking `cann` logic directly into `Session.cpp` or `profile.py`.

### 4. Initial CANN adapter and runtime path

Files:

- `third_party/proton/flagtree_profiler/csrc/include/Driver/Ascend/AscendApi.h`
- `third_party/proton/flagtree_profiler/csrc/include/Profiler/Vendor/CannProfiler.h`
- `third_party/proton/flagtree_profiler/csrc/lib/Profiler/Vendor/CannAdapter.cpp`
- `third_party/proton/flagtree_profiler/csrc/lib/Driver/Ascend/AscendApi.cpp`
- `third_party/proton/flagtree_profiler/csrc/lib/Profiler/Vendor/CannProfiler.cpp`

This adapter currently provides:

- adapter name: `cann`
- supported vendor metrics list:
  - `aicore`
  - `bandwidth`
- plan normalization
- CANN runtime profiler wiring
- CANN/Ascend runtime loading helpers
- `aclprof/msprof` summary import hooks
- host-timing fallback when runtime profiling inputs are unavailable

It does **not** guarantee that real CANN metrics are collected in every environment. Real collection requires the deployment to expose the relevant Ascend/CANN runtime libraries and exported profiler summaries.

When those inputs are missing, the implementation must still complete the session and record degradation details.

### 5. Minimal CANN smoke test

File:

- `third_party/proton/flagtree_profiler/test/test_cann_smoke.py`

This test validates the public API and artifact/degradation contract. It intentionally uses a host-timing fallback operation instead of requiring a real Ascend kernel. Passing this test proves the minimal acceptance path, not full real-device profiling.

## How To Finish The Task

The implementation should be completed in the following order.

### Step 1. Validate and harden real CANN profiling

The repository now has a CANN profiler path. The remaining work is to validate and harden it against real CANN deployments, similar in role to:

- `CuptiProfiler`
- `RoctracerProfiler`

Relevant files:

- `third_party/proton/flagtree_profiler/csrc/include/Profiler/Vendor/CannProfiler.h`
- `third_party/proton/flagtree_profiler/csrc/lib/Profiler/Vendor/CannProfiler.cpp`

Requirements:

- inherit from `Profiler` or reuse the `GPUProfiler<T>` pattern if it fits the CANN callback model
- collect stable runtime kernel events
- convert them into existing Proton `KernelMetric`
- guarantee base runtime metrics are still available even when vendor enhancement fails

Current implementation note:

- runtime base prefers CANN-native `task_time/op_summary` import
- host timing fallback can be toggled with:
  - `runtime_host_timing_fallback=true|false`
  - (alias) `runtime_base_host_fallback=true|false`

### Step 2. Extend `DeviceType`

`DeviceType` now has a CANN/Ascend extension point. Keep this path aligned with any future device property queries.

Files to keep aligned:

- `third_party/proton/common/include/Device.h`
- `third_party/proton/csrc/lib/Driver/Device.cpp`

If device property queries are needed, add a matching driver helper implementation.

### Step 3. Make Session output multi-artifact

Current `Session` owns a single `Data` object and finalizes via:

- one `data->dump(...)`

That is not enough for the target output set.

Minimal completion strategy:

1. Keep existing `TreeData` and `TraceData`
2. Add a session-level artifact writer that emits:
   - `<base>.hatchet`
   - `<base>.timeline.json`
   - `<base>.meta.json`
   - `<base>.vendor.json`
3. Keep base runtime collection separate from vendor enhancement import

Recommended implementation approach:

- add a small session-owned output bundle
- avoid overloading a single `Data::dump()` call to mean "write four files"

Files likely to change:

- `third_party/proton/csrc/include/Session/Session.h`
- `third_party/proton/csrc/lib/Session/Session.cpp`
- `third_party/proton/csrc/include/Data/Data.h`
- `third_party/proton/csrc/lib/Data/Data.cpp`

### Step 4. Wire Python API to the new adapter path

After the runtime side exists, update Python entry points.

Files:

- `third_party/proton/proton/profile.py`
- `third_party/proton/proton/proton.py`

Required changes:

- allow `backend="cann"`
- keep `mode="runtime_base:vendor_metrics=..."` as the public contract
- preserve current behavior for existing backends

### Step 5. Implement vendor metric import

Finish `CannMetricsImporter::import(...)`.

The importer should:

- read exported `aclprof/msprof` outputs
- normalize vendor metrics into `VendorProfileArtifact`
- associate metrics back to runtime kernel events

The first version should tolerate partial failure:

- base runtime metrics succeed
- vendor enhancement may degrade
- degradation reason must go to:
  - `meta.json`
  - `vendor.json`

### Step 6. Correlate vendor metrics to base runtime events

Use `RuntimeTraceEventKey` and `VendorMetricAssociation` as the minimum join model.

Recommended matching keys, in descending quality:

1. vendor correlation id / task id if available
2. device id + stream id + kernel name + timestamp window
3. fallback fuzzy timestamp matching with explicit "unmatched" annotation

Do not make `.hatchet` depend on vendor association success.

## Minimal Acceptance Criteria

The implementation is minimally complete when all of the following are true:

1. This API works:

```python
import triton.profiler as proton

session_id = proton.start(
    name="profile_run",
    context="shadow",
    data="tree",
    hook="triton",
    backend="cann",
    mode="runtime_base:vendor_metrics=aicore,bandwidth",
)
```

2. Finalization emits:

- `profile_run.hatchet`
- `profile_run.timeline.json`
- `profile_run.meta.json`
- `profile_run.vendor.json`

3. If vendor enhancement is unavailable, the following still hold:

- `profile_run.hatchet` exists
- `profile_run.timeline.json` exists
- `profile_run.meta.json` records degradation
- `profile_run.vendor.json` is empty or contains degradation details

## AArch64 Server Build and Acceptance Notes

The editable install was validated on an aarch64 Linux server using a dedicated conda environment. A clean shell or a different user account should not assume that the environment is already active.

Recommended setup for the shared server:

```bash
cd /home/secure/zhaoguoxiang/FlagTree

source /home/secure/miniforge3/etc/profile.d/conda.sh
conda activate flagtree-py310

export CC="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export MAX_JOBS=16
```

Build/install:

```bash
python -m pip install -e . --no-build-isolation
```

Basic import checks:

```bash
python -c "import triton; print(triton.__file__)"
python -c "import triton.profiler as proton; print(proton.start)"
```

Minimal CANN acceptance:

```bash
python -m pytest -q third_party/proton/flagtree_profiler/test/test_cann_smoke.py -s
```

Expected result:

```text
1 passed
```

Manual artifact check:

```bash
OUT=/tmp/proton_cann_acceptance_$(date +%s)
export OUT
mkdir -p "$OUT/vendor"

python - <<'PY'
import os
import pathlib
import time
import triton.profiler as proton
from triton._C.libproton import proton as libproton

base = pathlib.Path(os.environ["OUT"]) / "profile_run"
vendor_output = pathlib.Path(os.environ["OUT"]) / "vendor"

sid = proton.start(
    name=str(base),
    context="shadow",
    data="tree",
    hook="triton",
    backend="cann",
    mode=(
        "runtime_base:"
        "vendor_metrics=aicore,bandwidth:"
        f"aclprof_output_path={vendor_output}:"
        "runtime_host_timing_fallback=true"
    ),
)

scope_id = libproton.record_scope()
libproton.enter_op(scope_id, "cann_acceptance_kernel")
time.sleep(0.001)
libproton.exit_op(scope_id, "cann_acceptance_kernel")

proton.finalize(sid)
print(base)
PY

ls -lh "$OUT"/profile_run.*
```

The output must include:

```text
profile_run.hatchet
profile_run.timeline.json
profile_run.meta.json
profile_run.vendor.json
```

Then verify the JSON contract:

```bash
python - <<'PY'
import os
import json
import pathlib

base = pathlib.Path(os.environ["OUT"]) / "profile_run"
paths = {
    "hatchet": base.with_suffix(".hatchet"),
    "timeline": base.with_suffix(".timeline.json"),
    "meta": base.with_suffix(".meta.json"),
    "vendor": base.with_suffix(".vendor.json"),
}

missing = [name for name, path in paths.items() if not path.exists()]
assert not missing, f"missing artifacts: {missing}"

meta = json.loads(paths["meta"].read_text())
vendor = json.loads(paths["vendor"].read_text())
timeline = json.loads(paths["timeline"].read_text().splitlines()[0])

assert meta["backend"] == "cann"
assert meta["runtime_base_enabled"] is True
assert "aicore" in meta["vendor_metrics_enabled"]
assert "bandwidth" in meta["vendor_metrics_enabled"]
assert isinstance(meta["degrade_reasons"], list)
assert isinstance(vendor.get("degrade_reasons", []), list)
assert isinstance(vendor.get("associations", []), list)
assert isinstance(timeline.get("traceEvents", []), list)

print("ACCEPTANCE OK")
print("meta degrade_reasons:", meta["degrade_reasons"])
print("vendor degrade_reasons:", vendor.get("degrade_reasons", []))
print("timeline events:", len(timeline.get("traceEvents", [])))
PY
```

If `ACCEPTANCE OK` is printed, the minimal acceptance criteria are satisfied.

Important caveat:

- Degradation messages such as `Failed to load libacl.so/libacl_prof.so` mean the fallback path was validated, not real CANN profiling.
- Real CANN profiling validation requires the process to load Ascend/CANN profiling libraries and/or consume real `msprof/aclprof` exported summary CSV files.
- `third_party/proton/test/test_api.py` imports `torch` during collection through Triton's internal test helpers, so it requires a PyTorch installation even when running only a CANN-specific `-k` selection.

## Testing Plan

### A. Unit tests for mode parsing

Add tests for:

- empty mode
- `runtime_base`
- `runtime_base:vendor_metrics=aicore`
- `runtime_base:vendor_metrics=aicore,bandwidth`
- malformed token handling
- unsupported metric downgrade in the CANN adapter plan

Recommended test target:

- new C++ unit tests for `parseVendorProfileMode`
- or Python-level tests if a binding/helper is later exposed

### B. Adapter plan tests

Verify `CannAdapter::makePlan(...)`:

- always keeps `runtimeBaseEnabled=true`
- enables supported vendor metrics
- records unsupported metrics in `disabledVendorMetrics`
- records degradation reasons

### C. Session artifact tests

After session wiring is complete, add tests that validate:

- `finalize()` writes all required files
- meta schema version and backend fields are present
- vendor artifact may be empty without failing the session

### D. Runtime collector tests

Use a minimal Triton kernel and verify:

- a kernel event reaches Proton tree output
- the timeline file contains the kernel event
- device/backend info reaches `meta.json`

### E. Degradation tests

Run with vendor enhancement intentionally unavailable.

Expected result:

- session succeeds
- base runtime files are valid
- degradation is visible in metadata

## Recommended Patch Sequence

To minimize risk, land the remaining work as a short series:

1. `CannProfiler` and `DeviceType` support
2. multi-artifact session writing
3. Python `backend="cann"` wiring
4. vendor importer and correlation
5. end-to-end tests

This order keeps each patch reviewable and allows base runtime functionality to become usable before vendor enhancement is complete.
