# FlagTree Packaging

Debian (.deb) and RPM (.rpm) packaging for FlagTree.

## Status

- nvidia backend: scaffolded (DEB + RPM). Wheel build verified by spike
  (~36 min wall on 4-core, 205 MB wheel, 493 MB installed).
- Other backends (mthreads, metax, amd, iluvatar, cambricon, hcu, xpu):
  not yet wired up. Each needs its own build container with the chip
  vendor toolchain.

## Layout

```
packaging/
├── debian/                       # .deb packaging
│   ├── control                   # python3-flagtree-<backend>
│   ├── rules                     # installs pre-built wheel into staging
│   ├── changelog, compat, source/format
│   └── helpers/
│       ├── Dockerfile.deb        # 2 stages: wheel build → deb assemble
│       └── build-flagtree.sh     # entrypoint: ./build-flagtree.sh nvidia
├── rpm/                          # .rpm packaging
│   ├── specs/flagtree.spec       # uses pre-built wheel via wheel_dir define
│   └── helpers/
│       ├── Dockerfile.rpm        # 2 stages: rpm-base wheel → rpm assemble
│       └── build-flagtree-rpm.sh
└── spike/
    └── Dockerfile-nvidia         # original build-time spike for reference
```

## Why pre-built wheels rather than dh-python source build

FlagTree's Python build invokes CMake/Ninja to compile a substantial C++/MLIR
codebase (LLVM is downloaded at build time unless `LLVM_SYSPATH` is set).
A 36-minute compile in `override_dh_auto_build` is slow, hard to debug, and
collides with `dh-python`'s expectations for a pure-Python `pyproject` build.
The two-stage Docker approach builds the wheel once, then wraps it cheaply
into a `.deb`/`.rpm` — keeping the heavy lifting in a controlled container.

## Build locally

```sh
# DEB
./packaging/debian/build-helpers/build-flagtree.sh nvidia
# Output: ./dist/output/python3-flagtree-nvidia_*.deb

# RPM
./packaging/rpm/helpers/build-flagtree-rpm.sh nvidia
# Output: ./dist-rpm/output/python3-flagtree-nvidia-*.rpm
```

Both commands take 30–40 minutes wall on a typical 4-core machine.

## Namespace conflict

FlagTree installs into the `triton` Python namespace (it is a fork of
`triton-lang/triton`). The package therefore declares:

- `Provides: python3-triton, python3-flagtree`
- `Conflicts: python3-triton` (and other flagtree backend variants)
- `Replaces: python3-triton`

Users with the upstream `python3-triton` installed must remove it first.

## Adding a new backend

1. Verify the backend builds in a Docker container (see
   `spike/Dockerfile-nvidia` as a template).
2. Extend the relevant build helper (`build-flagtree.sh` /
   `build-flagtree-rpm.sh`) — add the backend to the case statement.
3. If the backend needs a different base image (e.g. `mthreads` needs
   the MUSA toolkit), add a backend-specific Dockerfile or stage.
4. Add the backend to the `Conflicts` lists in `debian/control` and
   `rpm/specs/flagtree.spec`.
