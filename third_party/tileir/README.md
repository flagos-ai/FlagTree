# FlagTree TileIR Backend

## Source

**Imported source**

- Upstream repo: `https://github.com/triton-lang/Triton-to-Tile-IR`
- Upstream release: `v3.6.0-tileir-13.4.0`
- Upstream commit: `a943171a33c3e2f27fa95103f69ba64ae7a99dce`
- Synced path: `third_party/tileir`

**Pinned dependencies**

- cuda-tile checkout path: `third_party/tileir/third_party/cuda-tile`
- cuda-tile commit: `7e8e2e68fa219716103824c01f7303367cf7df8d`
- cuda-tile reference: `v13.4.0`
- TileIR compiler toolchain: `13.4.59`

## Build

When `FLAGTREE_BACKEND` is not set, FlagTree builds its default NVIDIA and AMD
backends. Setting `FLAGTREE_BACKEND=tileir` adds TileIR while retaining both
default backends for per-kernel routing and fallback.

The TileIR build downloads and packages a matching compiler toolchain:

- `tileiras`, `ptxas`, `libnvvm`, and `libdevice` version `13.4.59`
- GCC or Clang 13 or newer
- the cuda-tile checkout at the pinned commit above

The TileIR path uses `tileiras`. Native NVIDIA fallback and comparison runs use
FlagTree's normal NVIDIA assemblers, managed by `setup.py` at:

- `third_party/nvidia/backend/bin/ptxas`
- `third_party/nvidia/backend/bin/ptxas-blackwell`

`pip install` and `pip install -e` fetch the pinned cuda-tile checkout and the
compiler tools automatically when `FLAGTREE_BACKEND=tileir` is set.

At runtime, TileIR looks for `tileiras` in this order:

1. `$TRITON_TILEIRAS_PATH/tileiras`
2. `triton/backends/nvidia/tileir_cuda/bin/tileiras` in the installation
3. `tileiras` on `PATH`

Default FlagTree source install:

```bash
FLAGTREE_BACKEND=tileir MAX_JOBS=32 \
  python3 -m pip install . --no-build-isolation
```

For local editable development, the usual editable install also works when the
build dependencies are already present:

```bash
FLAGTREE_BACKEND=tileir MAX_JOBS=32 \
  python3 -m pip install -e . --no-build-isolation
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
`cuda` target, only TileIR tle_supported TLE  -> TileIR
`cuda` target, other or unknown TLE usage     -> native NVIDIA
```

`is_tle_supported` / `require_tle` take the list of TLE primitives a kernel
uses, pick the backend that list routes to by the same rules, and check it
against that backend's `backend/tle_supported.py` whitelist.

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

- Target FlagTree/Triton base: `main` / Triton 3.6.0
- The Triton-to-Tile-IR release and FlagTree use LLVM commit
  `f6ded0be897e2878612dd903f7e8bb85448269e5`.
- `scripts/patch_bytecode_utils.sh` adapts cuda-tile v13.4.0 to that public LLVM
  API and omits only the unsupported `Float8E5M3FNUType` bytecode entry.

**FlagTree-local changes**

Compared with the imported upstream release, FlagTree substantively modifies 8
files and adds 4 files:

- **Build and provenance:** Modified `README.md` and `CMakeLists.txt` to record
  the vendor base and build the checkout populated by `setup.py`.
- **Routing and compiler integration:** Modified `backend/compiler.py` for
  backend hooks and lazy `tileiras` lookup. Added `backend/router.py` to choose
  TileIR or native NVIDIA per kernel. Added `backend/tle_supported.py` for the
  supported top-level TLE API set.
- **TLE frontend:** Added `backend/extend_core.py` and
  `backend/extend_semantic.py`; `CudaTileSemantic` extends `TritonSemantic` with
  the view/token operations exposed as `tle.<name>` through `tl.ext`. Modified
  `backend/code_generator.py` to use it for TileIR kernels and
  `triton_tileir.cc` to expose the corresponding C++ builder methods.
- **Release builds:** Modified `lib/TritonToTileIR/Utils.cpp` so its unsupported
  reduction-type path remains unreachable when `NDEBUG` removes assertions.
- **Compiler errors:** Modified `backend/errors.py` so `tileiras` failures use
  the backend-neutral compiler error handled by FlagTree's autotuner.
- **Runtime:** Upstream activates `TileIRDriver` globally with `ENABLE_TILE=1`
  and creates it during import. Modified `backend/driver.py` so it is not a
  global active driver and is created only for a kernel routed to TileIR. The
  file also adapts binary loading and TensorDescriptor arguments to FlagTree's
  runtime ABI.

TileIR-specific routing and frontend behavior must remain in this vendor
directory. Changes under `python/triton` are limited to backend-neutral hooks so
that non-TileIR backends do not import or depend on TileIR implementation code.

FlagTree's formatting hooks also normalize 14 imported vendor files. Those
format-only differences do not change TileIR behavior and are kept separate
from the 8 modified files listed above.

**Update checklist**

1. Sync `third_party/tileir` from the chosen Triton-to-Tile-IR release.
2. Update the cuda-tile checkout pin in both `CMakeLists.txt` and
   `python/setup_tools/utils/__init__.py`.
3. Reapply or verify the FlagTree-local changes above.
4. Recheck the LLVM compatibility patch in `scripts/patch_bytecode_utils.sh`.
5. Run TileIR routing, TLE/TKO, and tutorial validation before updating this file.
