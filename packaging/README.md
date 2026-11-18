# FlagTree Packaging

Debian (.deb) and RPM (.rpm) packaging for FlagTree.

## Status

`packaging/backends.conf` lists the backends a package can be built for, one
row each. The row's `wheel` column (deb) and `rpm_wheel` column (rpm) say
where that line's wheel comes from; an empty column means that line does not
package the backend yet:

| wheel | meaning | cost |
| --- | --- | --- |
| `source` | compiled from this tree by the `wheel-builder` stage | ~36 min, needs LLVM/CMake/Ninja and the vendor toolchain |
| `index` | downloaded from the FlagOS Python index, where the backend's own `<backend>N.N-delivery.yml` workflow published it | minutes, needs neither a vendor SDK nor vendor hardware |

A published wheel can only be packaged where the distribution's `python3` and
architecture match its tags, which is what decides a row's base image — and
why most backends are deb-only today: their wheels are `cp312` / `cp310`,
while the rpm targets ship python 3.11 (openEuler 24.03) and 3.14 (Fedora 43).

- nvidia: `source` (DEB + RPM).
- metax, amd, enflame, ppu, tileir: `index`; their wheels are tagged `cp312`,
  which is what `ubuntu:24.04` ships.
- hcu, mthreads, tsingmicro, xpu: `index`; their wheels are tagged `cp310`,
  which is what `ubuntu:22.04` ships.
- ascend: `index`, RPM only. Its wheel is published for `cp311` on aarch64,
  which is openEuler 24.03 on aarch64; no ubuntu release's default `python3`
  matches it, so the row carries rpm columns alone.
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
│       └── render-packaging.sh   # control.in -> control for one backend
├── build-helpers/                # shared by both lines
│   ├── backends.sh               # reader for backends.conf
│   └── fetch-wheel.sh            # download a published wheel (wheel=index)
├── rpm/                          # .rpm packaging
│   ├── specs/flagtree.spec       # uses pre-built wheel via wheel_dir define
│   └── helpers/
│       ├── Dockerfile.rpm        # wheel build | wheel download → rpm assemble
│       └── build-flagtree-rpm.sh # entrypoint: ./build-flagtree-rpm.sh nvidia
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

# RPM for such a backend; the ascend package is aarch64, so this wants an
# aarch64 host. On another architecture, emulate it (qemu, slower):
RPM_PLATFORM=linux/arm64 ./packaging/rpm/helpers/build-flagtree-rpm.sh ascend
# Output: ./dist-rpm/output/python3-flagtree-ascend-*.aarch64.rpm
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

1. Add a row to `packaging/backends.conf`. Pick `wheel=index` (deb) /
   `rpm_wheel=index` (rpm) if the backend's delivery workflow publishes a
   wheel, and set `base`/`python` or `rpm_base`/`rpm_arch` to the release the
   published wheel's `cp` and platform tags match — `fetch-wheel.sh` refuses a
   wheel built for another interpreter or architecture. Pick `source`
   otherwise, and give the `wheel-builder` stage the backend's toolchain. A
   line whose column stays empty simply does not package the backend.
2. If the backend's wheel bundles vendor `.so` files, list the path fragments
   in the row's `excludes` column so `dpkg-shlibdeps` leaves them alone; the
   vendor's own installation supplies those sonames at deploy time.
3. Build it once. If the backend's `compiler.py` reaches for the vendor SDK or
   for `torch` while `triton` is being imported, the build host cannot import
   it at all; set the row's `smoke` column to `install` so the build asserts
   the install and stops there.
4. Add a build row to `.github/workflows/build-deb.yml` or
   `.github/workflows/build-rpm.yml`. An aarch64 row runs on a hosted arm64
   runner, so it needs no emulation and no self-hosted machine.
5. Verify on the backend's hardware: these packages are assembled, and at
   most imported, without it, which does not exercise a single kernel.
