# FlagTree TileIR Backend

## Source

**Imported source**

- Upstream repo: `https://github.com/triton-lang/Triton-to-Tile-IR`
- Upstream commit: `a3befd959b02410cfbdac08d91d817b0ec0b3e33`
- Synced path: `third_party/tileir`

**Pinned dependencies**

- cuda-tile submodule path: `third_party/tileir/third_party/cuda-tile`
- cuda-tile commit: `2e5ccba66fb3afdba34b26cf358418283027c248`
- cuda-tile reference: `v13.3.0`
- tileiras version used for validation: CUDA 13.3, `V13.3.36`

## Build

When `FLAGTREE_BACKEND` is not set, FlagTree builds the default in-tree backend
set, including `nvidia`, `amd`, and `tileir`.
Setting `FLAGTREE_BACKEND=tileir` selects TileIR explicitly while retaining the
NVIDIA and AMD backends required for per-kernel fallback.

TileIR requires CUDA TileIR tooling from CTK 13.3. The validated toolchain is:

- `tileiras` from CUDA 13.3, version `V13.3.36`
- GCC or Clang 13 or newer
- the cuda-tile submodule at the pinned commit above

The TileIR path uses `tileiras`. Native NVIDIA fallback and comparison runs use
FlagTree's normal NVIDIA assemblers, managed by `setup.py` at:

- `third_party/nvidia/backend/bin/ptxas`
- `third_party/nvidia/backend/bin/ptxas-blackwell`

`pip install` and `pip install -e` initialize the pinned cuda-tile submodule
automatically through `setup.py`; no manual submodule command is needed. If you
invoke CMake directly without `setup.py`, initialize the submodule first:

```bash
git submodule update --init --recursive \
  third_party/tileir/third_party/cuda-tile
```

At runtime, TileIR looks for `tileiras` in this order:

1. `$TRITON_TILEIRAS_PATH/tileiras`
2. `triton/backends/nvidia/bin/tileiras` in the installation
3. `tileiras` on `PATH`

Default FlagTree source install:

```bash
MAX_JOBS=32 python3 -m pip install . --no-build-isolation
```

For local editable development, the usual editable install also works when the
build dependencies are already present:

```bash
MAX_JOBS=32 python3 -m pip install -e . --no-build-isolation
```

## Runtime

**Backend discovery**

A normal installation registers the selected backends through the
`triton.backends` entry-point group. No discovery environment variable is
required.

The existing FlagTree option below bypasses entry points and scans the installed
`triton/backends/` directory instead:

```bash
TRITON_BACKENDS_IN_TREE=1
```

This option only changes where Triton discovers already-present backends. It
does not build, install, enable, or route TileIR and is normally unnecessary.

**Runtime routing**

Routing is disabled by default. `FLAGTREE_USE_TILEIR=1` enables per-kernel
selection; it does not force every NVIDIA-targeted Triton kernel onto TileIR:

```bash
FLAGTREE_USE_TILEIR=1
```

On NVIDIA hardware, the initial Triton target is labeled `cuda`. The routing
result is:

```text
non-`cuda` Triton target                      -> unchanged
`cuda` target, Triton kernel uses no TLE      -> TileIR
`cuda` target, supported top-level `tle` subset only -> TileIR
`cuda` target, other or unknown TLE usage     -> native NVIDIA
```

For routing diagnostics:

```bash
FLAGTREE_TILEIR_VERBOSE=1
```

See the [TileIR tutorials](../../python/tutorials/tileir/README.md) for
functional checks and benchmark commands.

The two paths use these names:

| Path | Backend name | Target label | Compiler | Kernel driver |
| --- | --- | --- | --- | --- |
| Native NVIDIA | `nvidia` | `cuda` | `CUDABackend` | `CudaDriver` |
| TileIR | `tileir` | `tileir` | `TileIRBackend` | `TileIRDriver` |

`CudaDriver` provides the initial `cuda` target. Routing either keeps it or
changes it to `tileir`; the final target selects the compiler and kernel driver
shown above.

## Vendor Maintenance

**FlagTree compatibility**

- Target FlagTree/Triton base: `triton_v3.6.x` / Triton 3.6.0
- FlagTree's pinned LLVM is older than the LLVM expected by cuda-tile v13.3.0.
- `scripts/patch_bytecode_utils.sh` is a FlagTree-local compatibility patch for
  that LLVM skew. It rewrites cuda-tile source before build, including:
  - `DenseTypedElementsAttr` to `DenseIntOrFPElementsAttr`
  - `llvm::scope_exit` to `llvm::make_scope_exit`
  - `DenseElementsAttr::isValidRawBuffer` 2-argument calls to the 3-argument API

**FlagTree-local changes**

Compared with the imported upstream commit, FlagTree modifies 8 files and adds
4 entries:

- **Build and provenance:** Modified `README.md`, `CMakeLists.txt`, and
  `scripts/patch_bytecode_utils.sh` to record the vendor base, build the pinned
  cuda-tile source, and support FlagTree's LLVM. Added the `third_party/cuda-tile`
  submodule.
- **Routing and compiler integration:** Modified `backend/compiler.py` for
  backend hooks and lazy `tileiras` lookup. Added `backend/router.py` to choose
  TileIR or native NVIDIA per kernel.
- **TLE frontend:** Added `backend/extend_core.py` and
  `backend/extend_semantic.py`; `CudaTileSemantic` extends `TritonSemantic` with
  the view/token operations exposed as `tle.<name>` through `tl.ext`. Modified
  `backend/code_generator.py` to use it for TileIR kernels and
  `triton_tileir.cc` to expose the corresponding C++ builder methods.
- **Lowering:** Modified `lib/TritonToTileIR/TritonToTileIRPass.cpp` to add
  `math.erf` lowering and align flattened TensorDescriptor arguments with
  FlagTree's ABI.
- **Runtime:** Upstream activates `TileIRDriver` globally with `ENABLE_TILE=1`
  and creates it during import. Modified `backend/driver.py` so it is not a
  global active driver and is created only for a kernel routed to TileIR. The
  file also adapts binary loading and TensorDescriptor arguments to FlagTree's
  runtime ABI.

TileIR-specific routing and frontend behavior must remain in this vendor
directory. Changes under `python/triton` are limited to backend-neutral hooks so
that non-TileIR backends do not import or depend on TileIR implementation code.

**Update checklist**

1. Sync `third_party/tileir` from the chosen Triton-to-Tile-IR commit.
2. Keep `third_party/tileir/third_party/cuda-tile` as an explicit submodule pin.
3. Reapply or verify the FlagTree-local changes above.
4. Recheck the LLVM compatibility patch in `scripts/patch_bytecode_utils.sh`.
5. Run TileIR routing, TLE/TKO, and tutorial validation before updating this file.
