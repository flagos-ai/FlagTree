# FlagTree Agent Brief

## 1. Scope & Branching
- **Goal**: FlagTree extends Triton into a unified, multi-backend AI compiler. Active work streams cover Triton Language Extensions (TLE), performance tuning, and onboarding of new hardware targets.
- **Branches**: Default branch `main`; current working branch `feature/gpu_tile`. Confirm upstream status before large refactors.
- **Key domains**:
  - `python/triton/experimental/tle` and `third_party/tle`: TLE dialect/lowering work.
  - `third_party/*`: backend-specific integrations (kernels, runtime glue, tests).
  - `lib/`, `include/`, `bin/`: C++ compiler/driver sources.

## 2. Workflow (Suggested)
1. **Sync context**: read `README.md` for backend updates and check `reports/` for architectural constraints.
2. **Plan**: outline affected components (Python API, MLIR pass, runtime) before editing.
3. **Implement**:
   - Python: favor modular helpers in `python/src/`; keep kernels under `python/triton_kernels/`.
   - C++: update both `lib/` (implementation) and `include/` (declarations); register passes/tools in `bin/` as needed.
   - TLE: coordinate changes between `python/triton/experimental/tle` and `third_party/tle` lowering/runtime pieces.
4. **Validate**: run relevant pytest suites and/or `ninja check-*` via `build/cmake.linux-x86_64-cpython-3.10`.
5. **Review readiness**: update docs/README if behavior changes.

## 3. Environment & Toolchain
- **Python env**: use `conda activate flagtree` (or prefix commands with `conda run -n flagtree ...`).
- **Install**: from repo root run `python -m pip install -r python/requirements.txt`, then `./build-nvidia.sh` (editable install wires up C++ extensions + `python/` package). Re-run after touching C++/pybind bindings.
- **Backends**:
  - Set `FLAGTREE_BACKEND` before build (e.g., `export FLAGTREE_BACKEND=nvidia`).
  - LLVM toolchains in `~/.flagtree/<backend>` or via `LLVM_SYSPATH`. Offline packs: `python/scripts/offline_build_unpack.sh`.
- **C++ build**: `./build-nvidia.sh` configures `build/cmake.linux-x86_64-cpython-3.10` and drives `ninja`. For bespoke toolchains use `scripts/build-llvm-project.sh`.

## 4. Repository Map
- **Top-level build files**: `CMakeLists.txt`, `Makefile`, `pyproject.toml`, `setup.py`, `MANIFEST.in`.
- **Python package** (`python/`):
  - `src/triton`, `src/flagtree`: main Python APIs & runtime utilities.
  - `build_helpers.py`, `setup_tools/` (see `setup_tools/utils/`).
  - `tutorials/`, `examples/`: user education.
  - `triton_kernels/`: reference kernels per backend.
- **Compiler sources** (`lib/`, `include/`, `bin/`): MLIR/LLVM passes, tools like `triton-opt`, `triton-llvm-opt`, `triton-reduce`.
- **Docs**: `docs/` (Sphinx). Backends: `docs/backend/`, quickstart: `docs/getting-started/`.
- **Reports**: `reports/` (e.g., `reports/decoupling/FlagTree_Backend_Specialization`).

## 5. TLE-Specific Notes (Recent Learnings)
- **Local memory API**: `tle.local_load`/`tle.local_store` are removed. Use `tle.local_ptr(...)` + `tl.load`/`tl.store` instead.
- **Index semantics**: `tle.local_ptr(buffer, indices_fn, shape)` builds pointer views via a callable.
  - `indices_fn` is invoked with one loop variable per `shape` dimension and must return a tuple of integer tensors/scalars (length == buffer rank).
  - Flattening is **row-major** (last dimension fastest); `shape` defines the pointer tensor shape and may be 0‑dim.
  - Indices can be any integer dtype (signed or unsigned); no explicit `tl.int32` cast is required.
- **Pointer address space**: shared-memory pointers use LLVM address space **3**.
- **Lowering pipeline**:
  - `LocalPointersOp` lowers to shared pointers and applies index deltas during LLVM lowering.
  - Offsets tensor layout must match the result encoding; `TleAssignLocalPointersEncoding` handles this.
  - Insert barriers for local pointer load-after-store hazards via `TleInsertLocalPointerBarriers`.
  - NV backend pass order: call `add_assign_local_pointers_encoding` and `add_insert_local_pointer_barriers` after early memory-space assignment.
- **Lowering legality**: TLE-to-LLVM conversion needs `mlir::gpu::GPUDialect` and `mlir::UnrealizedConversionCastOp` legal in the TLE conversion target, plus `LocalPointersOp` patterns registered.
- **Load/store lowering**: prefer shared vs global PTX based on pointer address space; local_ptr pointers must be treated as shared.

## 6. Testing Matrix
- **Core python**: `conda run -n flagtree pytest python/test -s` (or narrower suites).
- **Backend-specific**: `conda run -n flagtree pytest third_party/<backend>/python/test -s`.
- **TLE unit**: `conda run -n flagtree pytest python/test/tle/unit -s` (includes `test_tle_local_pointer.py`).
- **C++/MLIR**: `conda run -n flagtree ninja -C build/cmake.linux-x86_64-cpython-3.10 check-*`.

## 7. Conventions & Guardrails
- Prefer feature gates or backend capability checks over hard-coded vendor assumptions.
- Keep shared logic in `python/triton` or `lib/`; isolate backend-specific behavior under `third_party/<backend>`.
- **Marker rule**: add `// begin flagtree tle` / `// end flagtree tle` only in **native Triton** files; do **not** add markers inside `third_party/tle`.

## 8. Quick References
- **Docs build**: `cd docs && pip install -r requirements.txt && make html`.
- **LLVM helper scripts**: `scripts/build-llvm-project.sh`, `build-nvidia.sh` for vendor toolchains.
- **Nightly artifacts**: `.flagtree` cache, `reports/v0.1.0/` benchmark baselines.

Keep this brief updated when new backends land or workflow changes (CI, scripts). This file is intended for Codex/GPT agents—embed actionable context whenever you add major tooling or process updates.
