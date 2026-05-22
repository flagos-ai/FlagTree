# FlagTree RPM spec.
#
# Like the .deb side, this spec assumes a pre-built wheel has been produced
# in a backend-specific build container. The spec then unpacks the wheel
# into the buildroot. Building the wheel from %prep/%build inside the RPM
# build would require LLVM, CMake, Ninja, and the chip vendor toolchain
# in the chroot — heavy, brittle, and duplicates work.
#
# Usage in CI:
#   1. build-flagtree-rpm.sh produces /wheels/flagtree-*.whl
#   2. rpmbuild -ba flagtree.spec --define "wheel_dir /wheels"
#                                 --define "flagtree_backend nvidia"

%{!?flagtree_backend: %global flagtree_backend nvidia}
%{!?wheel_dir: %global wheel_dir %{_sourcedir}}

# TODO(known issue): the FlagTree wheel produced by Stage-1 wheel-builder is
# tagged cp310-cp310-linux_x86_64 (ubuntu:22.04 Python 3.10). Fedora 43
# ships Python 3.14, so `pip install` here rejects the wheel as
# "not a supported wheel on this platform". Two ways forward:
#   - build the wheel against fedora:43's Python 3.14 in a parallel RPM
#     wheel-builder stage (clean but doubles build time), OR
#   - retag the wheel to abi3 stable ABI (upstream change in setup.py)
# Until either lands, the .rpm build will fail; .deb side ships fine.

# Disable byte-compile failures for the bundled vendored Python files.
# Triton ships some files that aren't expected to compile under all interpreters.
%define _python_bytecompile_errors_terminate_build 0

# Skip auto-provides/requires for the bundled native libraries; they reference
# CUDA libs we don't want declared as RPM-level Requires.
%define __requires_exclude libcuda\\.so|libnvidia.*

Name:           python3-flagtree-%{flagtree_backend}
Version:        0.5.0
Release:        1%{?dist}
Summary:        FlagTree compiler with %{flagtree_backend} backend
License:        MIT AND Apache-2.0 WITH LLVM-exception AND BSD-3-Clause AND LicenseRef-NVIDIA-CUDA-EULA
URL:            https://github.com/flagos-ai/FlagTree
# The FlagTree wheel is linux_x86_64 only (LLVM/MLIR backend). Use
# ExclusiveArch instead of BuildArch so rpmbuild refuses to start on a
# non-x86_64 host rather than producing a mislabeled rpm.
ExclusiveArch:  x86_64

# No Source0: the wheel is supplied via --define "wheel_dir ...".

BuildRequires:  python3-devel
BuildRequires:  python3-pip
BuildRequires:  python3-setuptools
BuildRequires:  python3-wheel

Requires:       python3
Requires:       python3-filelock
Requires:       python3-setuptools

Provides:       python3-flagtree
Provides:       python3-triton
Conflicts:      python3-triton
# Backend-variant exclusion: only one flagtree backend per host.
%if "%{flagtree_backend}" != "nvidia"
Conflicts:      python3-flagtree-nvidia
%endif
%if "%{flagtree_backend}" != "mthreads"
Conflicts:      python3-flagtree-mthreads
%endif
%if "%{flagtree_backend}" != "metax"
Conflicts:      python3-flagtree-metax
%endif
%if "%{flagtree_backend}" != "amd"
Conflicts:      python3-flagtree-amd
%endif
%if "%{flagtree_backend}" != "iluvatar"
Conflicts:      python3-flagtree-iluvatar
%endif
%if "%{flagtree_backend}" != "cambricon"
Conflicts:      python3-flagtree-cambricon
%endif
%if "%{flagtree_backend}" != "hcu"
Conflicts:      python3-flagtree-hcu
%endif
%if "%{flagtree_backend}" != "xpu"
Conflicts:      python3-flagtree-xpu
%endif

%description
FlagTree is a unified compiler supporting multiple AI chip backends for
custom Deep Learning operations, forked from triton-lang/triton.

This package contains the FlagTree Python distribution built with the
%{flagtree_backend} backend enabled. It installs into the "triton" Python
namespace and conflicts with the upstream python3-triton package.

%prep
# No source unpack. The wheel was built externally.

%build
# Nothing to compile. The wheel is already a build artifact.

%install
set -e
WHEEL=$(ls %{wheel_dir}/flagtree-*.whl 2>/dev/null | head -n1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: no flagtree-*.whl found under %{wheel_dir}" >&2
    exit 1
fi

PYDIR=%{buildroot}%{python3_sitearch}
mkdir -p "$PYDIR" %{buildroot}%{_bindir}

# Unpack the wheel into the buildroot without running pip's resolver.
python3 -m pip install --no-deps --no-compile --no-index \
    --target="$PYDIR" \
    "$WHEEL"

# Move console scripts to /usr/bin
if [ -d "$PYDIR/bin" ]; then
    mv "$PYDIR/bin/"* %{buildroot}%{_bindir}/ 2>/dev/null || true
    rmdir "$PYDIR/bin" 2>/dev/null || true
fi

# RECORD references absolute paths under --target which become wrong after
# rpmbuild relocates them. Drop it; pip doesn't need RECORD to function.
rm -f "$PYDIR"/flagtree-*.dist-info/RECORD

%files
%license LICENSE
%{python3_sitearch}/triton
%{python3_sitearch}/flagtree-*.dist-info
%{_bindir}/proton*

%changelog
* Mon Apr 27 2026 FlagOS Contributors <contact@flagos.io> - 0.5.0-1
- Initial RPM packaging.
- Package the %{flagtree_backend} backend variant.
