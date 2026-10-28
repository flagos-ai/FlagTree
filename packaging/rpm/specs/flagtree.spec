# FlagTree RPM spec.
#
# Like the .deb side, this spec assumes a pre-built wheel has been produced
# in a backend-specific build container. The spec then unpacks the wheel
# into the buildroot. Building the wheel from %%prep/%%build inside the RPM
# build would require LLVM, CMake, Ninja, and the chip vendor toolchain
# in the chroot — heavy, brittle, and duplicates work.
#
# Usage in CI:
#   1. build-flagtree-rpm.sh produces /wheels/flagtree-*.whl
#   2. rpmbuild -ba flagtree.spec --define "wheel_dir /wheels"
#                                 --define "flagtree_backend nvidia"
#                                 --define "flagtree_arch x86_64"
#
# The wheel must be built with the same Python ABI as the RPM build
# environment. pip validates the wheel tag during %%install, so a cp310 wheel
# (%% escaped in these comments: openEuler's rpm expands macros even in
# comments, and a bare %%install here terminates the preamble.)
# cannot be installed by a cp314 Fedora Python, for example.

%{!?flagtree_backend: %global flagtree_backend nvidia}
%{!?flagtree_arch: %global flagtree_arch x86_64}
%{!?wheel_dir: %global wheel_dir %{_sourcedir}}

# Disable byte-compile failures for the bundled vendored Python files.
# Triton ships some files that aren't expected to compile under all interpreters.
%define _python_bytecompile_errors_terminate_build 0

# Skip auto-provides/requires for the bundled native libraries; they reference
# vendor runtime libraries that ship with the vendor's own installation, not as
# rpms: CUDA for the nvidia backend, CANN (libascendcl.so) for ascend.
%define __requires_exclude libcuda\\.so|libnvidia.*|libascendcl\\.so

Name:           python3-flagtree-%{flagtree_backend}
Version:        0.6.1
Release:        2%{?dist}
Summary:        FlagTree compiler with %{flagtree_backend} backend
License:        MIT AND Apache-2.0 WITH LLVM-exception AND BSD-3-Clause AND LicenseRef-NVIDIA-CUDA-EULA
URL:            https://github.com/flagos-ai/FlagTree
# A FlagTree wheel is architecture-specific (LLVM/MLIR backend), and which
# architecture depends on the backend: the nvidia wheel this spec compiles is
# linux_x86_64, the published ascend wheel is linux_aarch64. The build passes
# the backend's architecture; ExclusiveArch (rather than BuildArch) then makes
# rpmbuild refuse to start on another host instead of producing a mislabeled
# rpm. See the rpm_arch column in packaging/backends.conf.
ExclusiveArch:  %{flagtree_arch}
Source0:        LICENSE

# The wheel itself is supplied via --define "wheel_dir ...".

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
%if "%{flagtree_backend}" != "ascend"
Conflicts:      python3-flagtree-ascend
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

# Guard against Version drifting from the wheel again (0.5.0 rpms used
# to ship a 0.6.0 wheel). A published wheel carries the backend and chip
# generation as a PEP 440 local version (0.6.1+ascend3.5); the part before
# the "+" is what has to match. cut, not shell suffix removal: rpm would eat
# the %% of a ${var%%+*} expansion before the shell ever sees it.
WHEEL_VER=$(basename "$WHEEL" | cut -d- -f2 | cut -d+ -f1)
if [ "$WHEEL_VER" != "%{version}" ]; then
    echo "ERROR: wheel version $WHEEL_VER != spec Version %{version}" >&2
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

# pip writes console-script shebangs from sys.executable, which may resolve to
# /usr/sbin/python3 under rpmbuild. Normalize them to Fedora's Python path so
# RPM auto-requires are satisfiable from the python3 package.
# Which scripts exist depends on the wheel: a wheel built from this tree ships
# proton / proton-viewer, a published one may ship flagtree-profiler instead.
for script in %{buildroot}%{_bindir}/*; do
    [ -e "$script" ] || continue
    sed -i "1s|^#!.*python3$|#!%{__python3}|" "$script"
done

# RECORD references absolute paths under --target which become wrong after
# rpmbuild relocates them. Drop it; pip doesn't need RECORD to function.
rm -f "$PYDIR"/flagtree-*.dist-info/RECORD
# triton hashes _C/libtriton.so for its compile-cache key (triton_key); make
# sure the module carries that name whatever suffix the wheel build used.
if [ ! -e "$PYDIR"/triton/_C/libtriton.so ]; then
    suffixed="$(ls -1 "$PYDIR"/triton/_C/libtriton.cpython-*.so 2>/dev/null | head -1)"
    test -n "$suffixed"
    mv "$suffixed" "$PYDIR"/triton/_C/libtriton.so
fi
test -f "$PYDIR"/triton/_C/libtriton.so

install -D -m 0644 %{SOURCE0} %{buildroot}%{_licensedir}/%{name}/LICENSE

%files
%license %{_licensedir}/%{name}/LICENSE
%{python3_sitearch}/triton
# The flagtree package itself, alongside the triton one it wraps. Absent from
# this list the build fails on unpackaged files, which is how it was found.
%{python3_sitearch}/flagtree
%{python3_sitearch}/flagtree-*.dist-info
%{_bindir}/*

%changelog
* Thu Sep 24 2026 FlagOS Contributors <contact@flagos.io> - 0.6.1-2
- Guarantee triton/_C/libtriton.so exists after unpacking the wheel (compile-cache key hashes it)

* Wed Jul 15 2026 FlagOS Contributors <contact@flagos.io> - 0.6.0-1
- Sync package version with the flagtree wheel (0.6.0).
- Fail the install step when the wheel version does not match the spec.

* Mon Apr 27 2026 FlagOS Contributors <contact@flagos.io> - 0.5.0-1
- Initial RPM packaging.
- Package the %{flagtree_backend} backend variant.
