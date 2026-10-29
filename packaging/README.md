# FlagTree Packaging

Debian (.deb) and RPM (.rpm) packaging for FlagTree.

## Status

`packaging/backends.conf` lists the backends a package can be built for, one
row each. The row's `wheel` column says where the wheel comes from:

| wheel | meaning | cost |
| --- | --- | --- |
| `source` | compiled from this tree by `Dockerfile.deb`'s `wheel-builder` stage | ~36 min, needs LLVM/CMake/Ninja and the vendor toolchain |
| `index` | downloaded from the FlagOS Python index, where the backend's own `<backend>N.N-delivery.yml` workflow published it | minutes, needs neither a vendor SDK nor vendor hardware |

- nvidia: `source` (DEB + RPM).
- metax, amd, enflame, ppu, tileir: `index`; their wheels are tagged `cp312`,
  which is what `ubuntu:24.04` ships.
- hcu, mthreads, tsingmicro, xpu: `index`; their wheels are tagged `cp310`,
  which is what `ubuntu:22.04` ships.
- ascend: the index publishes it for `cp311` on aarch64 only, so it needs an
  arm64 builder and has no row yet.
- iluvatar: no row. Its published `cp312` wheel carries
  `triton/_C/iluvatarTritonPlugin.so` linked against `libpython3.10.so.1.0`
  (0.6.1 and 0.7.0 alike), which no `cp312` distribution can satisfy.
- A backend whose delivery workflow already publishes a wheel needs a row and
  a build row in `.github/workflows/build-deb.yml`, nothing more. One whose
  wheel has to be compiled here also needs a build container carrying its
  toolchain.

## Layout

```
packaging/
├── backends.conf                 # the backend table: one row per package
├── debian/                       # .deb packaging
│   ├── control.in                # template -> python3-flagtree-<backend>
│   ├── lintian-overrides.in      # template -> <package>.lintian-overrides
│   ├── rules                     # installs pre-built wheel into staging
│   ├── changelog, compat, source/format
│   └── build-helpers/
│       ├── Dockerfile.deb        # wheel build | wheel download → deb assemble
│       ├── build-flagtree.sh     # entrypoint: ./build-flagtree.sh nvidia
│       ├── backends.sh           # reader for backends.conf
│       ├── render-packaging.sh   # control.in -> control for one backend
│       └── fetch-wheel.sh        # download a published wheel (wheel=index)
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
The wheel builders use the distro nlohmann-json headers through
`JSON_SYSPATH=/usr`; backend-specific LLVM and device tools remain pinned to
the versions expected by each backend.

## Build locally

```sh
# DEB
./packaging/debian/build-helpers/build-flagtree.sh nvidia
# Output: ./dist/output/python3-flagtree-nvidia_*.deb

# RPM
./packaging/rpm/helpers/build-flagtree-rpm.sh nvidia
# Output: ./dist-rpm/output/python3-flagtree-nvidia-*.rpm
```

```sh
# DEB for a backend whose wheel is published rather than built here
./packaging/debian/build-helpers/build-flagtree.sh metax
# Output: ./dist/output/python3-flagtree-metax_*.deb
```

A `source` backend takes 30–40 minutes wall on a typical 4-core machine; an
`index` backend takes as long as the download plus the `.deb` assembly.
`debian/control` is rendered inside the container, so neither writes into the
work tree.

## Namespace conflict

FlagTree installs into the `triton` Python namespace (it is a fork of
`triton-lang/triton`). The package therefore declares:

- `Provides: python3-triton, python3-flagtree`
- `Conflicts: python3-triton, python3-flagtree`
- `Replaces: python3-triton`

Conflicting with the virtual `python3-flagtree` each backend provides keeps at
most one backend installed without naming them: a package that conflicts with
a virtual package it provides does not conflict with itself (Debian Policy
7.4). Users with the upstream `python3-triton` installed must remove it first.

## Adding a new backend

1. Add a row to `packaging/backends.conf`. Pick `wheel=index` if the backend's
   delivery workflow publishes a wheel, and set `base`/`python` to the release
   the published wheel's `cp` tag matches — `fetch-wheel.sh` refuses a wheel
   built for another interpreter. Pick `wheel=source` otherwise, and give the
   `wheel-builder` stage the backend's toolchain.
2. If the backend's wheel bundles vendor `.so` files, list the path fragments
   in the row's `excludes` column so `dpkg-shlibdeps` leaves them alone; the
   vendor's own installation supplies those sonames at deploy time.
3. Build it once. If the backend's `compiler.py` reaches for the vendor SDK or
   for `torch` while `triton` is being imported, the build host cannot import
   it at all; set the row's `smoke` column to `install` so the build asserts
   the install and stops there.
4. Add a build row to `.github/workflows/build-deb.yml`.
5. Verify on the backend's hardware: these packages are assembled, and at
   most imported, without it, which does not exercise a single kernel.
