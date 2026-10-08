# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import shutil
import inspect
import subprocess
import importlib.machinery
import importlib.util
from pathlib import Path

from setuptools import find_packages

XPU_PYTHON_ROOT = "third_party/xpu/python"
BACKEND_ROOT = "third_party/xpu/backend"
# Intrinsic tables: `python/triton/backends/_tables/<tag>.json`.
#
# The tables are the norm for "which intrinsic names this build accepts", and
# they are decoded from the *consumer* toolchains' `IntrinsicImpl.inc`.  The
# LLVM 22 consumer is XTDK's, whose `.inc` is deliberately absent from the
# public `llvm_trust` tarball -- and the public LLVM 22 frontend knows no
# `llvm.xcn.*` / `llvm.xpu.*` names at all, so there is no public counterpart to
# decode either.  Both tables therefore travel pre-decoded in the compiled
# Python package (`xpu-python-impl-so`), and this build only checks they arrived.
_TABLE_TAGS = ("l19", "l22")

FLAGTREE_PYTHON_ROOT = "python"
# Package-level metadata that must **never** be staged into the tree.  It
# travels *inside the tarball* (MIT: "shall be included in all copies") but it
# describes the **package**, not the tree: the repository root already carries
# the LICENSE, and PROVENANCE's subject is the package digest itself, so a copy
# inside the tree is redundant at best and self-referential at worst.  Without
# this filter the stage step drops them into `third_party/xpu/` (and, for the
# flattened installers, into `triton/backends/` and `TritonSDNN/IR/`), where
# nothing tracked them and a `git add -A` would sweep them into a commit.
_PKG_METADATA_FILES = frozenset({"LICENSE", "PROVENANCE", "PROVENANCE.json"})
# The XPU overlay ships its own `triton.experimental.tle`; see
# _merge_xpu_packages for why the main-tree one is not merged in.
TLE_PACKAGE = "triton.experimental.tle"


def _use_xtdk_frontend():
    """True when the legacy XTDK LLVM 22 frontend is requested.

    The sync source (triton@de4bf790) defaults to the *public* LLVM 22 frontend
    and only stages XTDK's LLVM 22 when `TRITON_USE_XTDK_FRONTEND` is set; in
    the default leg the XTDK tarball is not even downloaded.  The frontend
    decides which MLIR the whole build links against, so a build that silently
    picks the other one is not the configuration this branch was validated
    with (see the public-LLVM compatibility layer and gate G15).
    """
    return os.environ.get("TRITON_USE_XTDK_FRONTEND", "").strip().upper() in ("1", "ON", "YES", "TRUE", "Y")


def generate_intrinsic_tables():
    """Check the intrinsic tables this build ships (`_tables/<tag>.json`).

    Both tables travel in the compiled Python package, decoded upstream from the
    consumer toolchains (see `_TABLE_TAGS`), so there is nothing to decode here.
    They still have to be checked: the intrinsic gates load them, and a build
    that ships none used to report every gate green while judging nothing.
    """
    backends_dir = Path(XPU_PYTHON_ROOT) / "triton" / "backends"
    # The decoder ships compiled in a wheel and as plain `.py` in a checkout.
    # Which one is used is reported below: a build that succeeds on the `.py`
    # fallback must not read the same as one that used the compiled form.
    decoder_path = backends_dir / "intrinsic_tables.py"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = backends_dir / ("intrinsic_tables" + suffix)
        if candidate.is_file():
            decoder_path = candidate
            break
    if not decoder_path.is_file():
        raise RuntimeError(f"XPU intrinsic table decoder is missing: {decoder_path}")
    print(f"[XPU] intrinsic table decoder: {decoder_path.name}"
          f"{' (source fallback)' if decoder_path.suffix == '.py' else ''}")
    # The name must match the module's own: a compiled extension exports only
    # `PyInit_<name>`, so a private alias here would fail to load it.
    spec = importlib.util.spec_from_file_location("intrinsic_tables", decoder_path)
    decoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = decoder
    spec.loader.exec_module(decoder)
    for tag in _TABLE_TAGS:
        dest = backends_dir / "_tables" / f"{tag}.json"
        if not dest.is_file():
            raise RuntimeError(f"XPU {tag} intrinsic table is missing: {dest}. It ships in the "
                               f"compiled Python package (xpu-python-impl-so) and this build "
                               f"staged none; without it the intrinsic gates judge nothing.")
        table = decoder.load(dest)
        if table.tag != tag:
            raise RuntimeError(f"XPU intrinsic table {dest} is tagged {table.tag!r}, not {tag!r}")
        if tag == "l22" and not table.family("llvm.xcn."):
            # The l22 table is the XTDK LLVM 22 leg's; the public frontend knows
            # no `llvm.xcn.*` names at all, so a table without them was decoded
            # from the wrong leg -- a wheel that ships one fails every gate.
            raise RuntimeError(f"XPU intrinsic table {dest} carries no llvm.xcn.* names: "
                               f"it was decoded from the public LLVM 22 frontend, not XTDK's")
        print(f"[XPU] verified {dest}: {len(table.names)} {tag} intrinsics")


def get_package_data_tools():
    return ["compile_xpu.h", "compile_xpu.c"]


def skip_package_dir(package):
    return package == "triton" or package.startswith("triton.")


def get_package_dir():
    return {
        "": XPU_PYTHON_ROOT,
    }


def _is_backend_package(package):
    return package == "triton.backends" or package.startswith("triton.backends.")


def _is_language_extra_package(package):
    return package == "triton.language.extra" or package.startswith("triton.language.extra.")


def _merge_xpu_packages(existing_packages):
    packages = []
    seen = set()

    def add(package):
        if package not in seen:
            packages.append(package)
            seen.add(package)

    # The XPU backend ships a complete `triton.*` python tree under
    # third_party/xpu/python/triton. It overlays the main tree so the
    # backend-specific tweaks (and XPU-only modules such as triton.ops and
    # triton.language.extra.xpu) live with the backend instead of polluting
    # the main Triton tree.
    for package in find_packages(where=XPU_PYTHON_ROOT, include=["triton", "triton.*"]):
        add(package)

    # TLE: the XPU overlay carries its own `triton.experimental.tle` (typed raw
    # ops on the XPU dialects + an xpu-clang payload pipeline), which the loop
    # above already picked up. The main tree's TLE is a same-origin fork with an
    # incompatible mechanism (multi-backend regions), so it must NOT be added on
    # top -- two sources for one package makes build_py copy whichever runs last.
    # Non-XPU builds go through default.py and keep the main-tree version.

    for package in existing_packages:
        if (not package.startswith("triton.") or _is_backend_package(package) or _is_language_extra_package(package)
                or package == "triton.profiler" or package.startswith("triton.profiler.")
                or package == "triton.tools.triton_to_gluon_translater"):
            add(package)

    return packages


def _merge_xpu_package_dir(existing_package_dir):
    package_dir = dict(existing_package_dir or {})
    package_dir[""] = XPU_PYTHON_ROOT

    for package in find_packages(where=XPU_PYTHON_ROOT, include=["triton", "triton.*"]):
        rel_package_path = package.replace(".", "/")
        package_dir[package] = f"{XPU_PYTHON_ROOT}/{rel_package_path}"

    # Packages that `_merge_xpu_packages` pulls in from the **main tree** (e.g.
    # `triton.backends.nvidia` / `triton.backends.amd`, which the XPU overlay does
    # not carry) have no directory under XPU_PYTHON_ROOT.  Without an explicit
    # entry setuptools falls back to `package_dir[""]` and aborts with
    #     error: package directory
    #     'third_party/xpu/python/triton/backends/nvidia' does not exist
    # Point them at the main tree explicitly so the merged install resolves.
    for package in find_packages(where=FLAGTREE_PYTHON_ROOT, include=["triton", "triton.*"]):
        if package in package_dir:
            continue
        rel_package_path = package.replace(".", "/")
        if os.path.isdir(os.path.join(FLAGTREE_PYTHON_ROOT, rel_package_path)):
            package_dir[package] = f"{FLAGTREE_PYTHON_ROOT}/{rel_package_path}"

    # No main-tree TLE entry here either: see _merge_xpu_packages.

    return package_dir


def _patch_xpu_cmdclass(existing_cmdclass):
    cmdclass = dict(existing_cmdclass or {})
    original_build_py = cmdclass.get("build_py")
    if original_build_py is None:
        return cmdclass

    class XpuBuildPy(original_build_py):

        def find_data_files(self, package, src_dir):
            # setuptools >= 79 can include symlink directories themselves in the
            # manifest file list (SOURCES.txt), which then causes build_py to
            # attempt to copy a directory as if it were a regular file and fail
            # with "can't copy '...': doesn't exist or not a regular file".
            # Filter out anything that is not a regular file so that only actual
            # .py / binary files are passed to the copy step.
            return [path for path in super().find_data_files(package, src_dir) if Path(path).is_file()]

        def run(self):
            self.force = True
            build_triton_dir = Path(self.build_lib) / "triton"
            if build_triton_dir.exists():
                shutil.rmtree(build_triton_dir)
            return super().run()

    cmdclass["build_py"] = XpuBuildPy
    return cmdclass


def _patch_llvm_exports():
    """Remove the cmake import-file existence check from LLVMExports.cmake.

    Some trust/xtdk-llvm22 packages export targets whose build-only binaries
    (llvm-tblgen/opt/llvm-link/...) are not shipped, which makes the cmake
    "verify imported files exist" loop raise FATAL_ERROR during find_package(MLIR).
    This patch disables that check.
    """
    import glob
    import re
    syspath = os.environ.get('LLVM_SYSPATH', '')
    if not syspath:
        return
    marker = "# [patched] import file check disabled - binaries not shipped in trust package"
    for f in glob.glob(f"{syspath}/lib/cmake/llvm/LLVMExports*.cmake"):
        with open(f) as fh:
            content = fh.read()
        if marker in content:
            continue
        patched = re.sub(r'# Loop over all imported files.*?unset\(_cmake_import_check_targets\)', marker, content,
                         flags=re.DOTALL)
        if patched != content:
            with open(f, 'w') as fh:
                fh.write(patched)
            print(f"[XPU] patched LLVMExports: {f}")


def _prune_stale_sdnn_objects(dst_root, package_root):
    """Remove prebuilt SDNN artifacts left over from an older package.

    Objects removed by a newer internal sync (e.g. Combine.cpp.o in
    v0.3.6.8.0) would otherwise trip xpu_check_object_file_list's FATAL_ERROR
    "NOT included in CMakeLists.txt" when upgrading an existing tree.
    """
    managed_dirs = (
        "lib/Dialect/TritonSDNN",
        "lib/Dialect/LLVMSDNN",
        "lib/Conversion/TritonSDNNToLLVM",
        "lib/Conversion/LinalgToTritonSDNN",
        "lib/Analysis/SDNN",
        "lib/Target/LLVMXPU",
        "device/xpu3",
    )
    for rel_dir in managed_dirs:
        dst_dir = os.path.join(dst_root, rel_dir)
        if not os.path.isdir(dst_dir):
            continue
        for dirpath, _, filenames in os.walk(dst_dir):
            for fn in filenames:
                if not fn.endswith((".o", ".a", ".bc")):
                    continue
                dst_file = os.path.join(dirpath, fn)
                rel = os.path.relpath(dst_file, dst_root)
                if not os.path.exists(os.path.join(str(package_root), rel)):
                    print(f"[XPU] pruning stale SDNN object from older package: {rel}")
                    os.remove(dst_file)


def install_sdnn_objects(cached_path, flagtree_dir):
    """Copy prebuilt SDNN objects from cache to third_party/xpu/."""
    dst_root = os.path.join(flagtree_dir, "third_party", "xpu")
    for item in os.listdir(cached_path):
        if item in _PKG_METADATA_FILES:
            continue
        src = os.path.join(str(cached_path), item)
        dst = os.path.join(dst_root, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)
    _prune_stale_sdnn_objects(dst_root, cached_path)

    # The prebuilt tarball lays libTritonXPUAnalysisSDNN.a under lib/Analysis/SDNN,
    # but lib/Analysis/NewAnalysis/CMakeLists.txt imports it from NewAnalysis/SDNN.
    sdnn_lib_name = "libTritonXPUAnalysisSDNN.a"
    sdnn_src = None
    for cand in (os.path.join(dst_root, "lib", "Analysis", "SDNN",
                              sdnn_lib_name), os.path.join(dst_root, sdnn_lib_name)):
        if os.path.exists(cand):
            sdnn_src = cand
            break
    if sdnn_src is not None:
        sdnn_dst_dir = os.path.join(dst_root, "lib", "Analysis", "NewAnalysis", "SDNN")
        os.makedirs(sdnn_dst_dir, exist_ok=True)
        shutil.copy(sdnn_src, os.path.join(sdnn_dst_dir, sdnn_lib_name))
    else:
        print(f"[XPU] warning: {sdnn_lib_name} not found under {dst_root}")

    required = os.path.join(dst_root, "lib", "Dialect", "TritonSDNN", "Transforms", "DSACopy.cpp.o")
    if not os.path.isfile(required):
        raise RuntimeError(f"[XPU] incomplete SDNN artifact: missing {required}")
    print(f"[XPU] SDNN prebuilt objects installed to {dst_root}")


_RESTORED_SOURCE_MODULES = (
    "triton/backends/xpu/compiler.abi3.so",
    "triton/backends/xpu/driver.abi3.so",
    "triton/tools/tensor_descriptor.abi3.so",
)


def _remove_restored_source_modules(dst_root):
    for rel in _RESTORED_SOURCE_MODULES:
        path = Path(dst_root) / rel
        if path.is_file() or path.is_symlink():
            path.unlink()


def _sweep_stale_python_impl_modules(dst_root, package_modules):
    stale = set()
    for root, _dirs, files in os.walk(dst_root, followlinks=True):
        for name in files:
            if name.endswith(".abi3.so"):
                stale.add(Path(os.path.relpath(os.path.join(root, name), dst_root)))
    unexpected = stale - package_modules
    if unexpected:
        names = ", ".join(str(path) for path in sorted(unexpected))
        raise RuntimeError(f"stale XPU Python modules are not in the new package: {names}")
    for rel in stale:
        (Path(dst_root) / rel).unlink()


def install_python_impl(cached_path, flagtree_dir):
    """Stage the compiled XPU Python implementation into the tree.

    The package mirrors the installed layout under `python/`, so this is a
    straight overlay onto `third_party/xpu/python/`.  Two kinds of file travel
    in it: the compiled modules, and the decoded intrinsic tables
    (`triton/backends/_tables/*.json`), which cannot be decoded at build time
    (see `_TABLE_TAGS`).

    Only compiled modules replace sources: `compiler.py` / `driver.py`
    (byte-identical to upstream) and the package `__init__.py` files stay as
    sources -- a compiled `__init__` loses the package context, so relative
    imports in it resolve against the parent package and fail.
    """
    src_root = os.path.join(str(cached_path), "python")
    if not os.path.isdir(src_root):
        raise RuntimeError(f"[XPU] incomplete python-impl artifact: missing {src_root}")
    dst_root = os.path.join(flagtree_dir, "third_party", "xpu", "python")
    suffixes = tuple(s for s in importlib.machinery.EXTENSION_SUFFIXES if s != ".so")
    tables_rel = os.path.join("triton", "backends", "_tables")
    staged = []
    package_modules = set()
    for root, _dirs, files in os.walk(src_root):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), src_root)
            is_module = name.endswith(suffixes)
            is_table = name.endswith(".json") and rel.startswith(tables_rel + os.sep)
            if not (is_module or is_table):
                continue
            staged.append((os.path.join(root, name), rel, is_module, is_table))
            if is_module:
                package_modules.add(Path(rel))
    if not package_modules:
        raise RuntimeError("[XPU] python-impl artifact carried no compiled module")

    _remove_restored_source_modules(dst_root)
    _sweep_stale_python_impl_modules(dst_root, package_modules)

    modules = tables = 0
    for source, rel, is_module, is_table in staged:
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(source, dst)
        modules += is_module
        tables += is_table
    print(f"[XPU] compiled Python implementation installed: {modules} modules + "
          f"{tables} intrinsic tables -> {dst_root}")


def _sweep_stale_llvm19_staging(flagtree_dir):
    """Drop the staged LLVM 19 trees before re-staging them.

    Every LLVM 19 tool in this staging must come from the same package (see the
    `xtdk-llvm19` entry in `register_cache`): `llc19` produces the kernel
    object, `clang` produces the CRT object, and the two are linked together by
    `ld.lld`.  LLVM 19 and LLVM 22 stamp different ELF section *types* on
    `.XPU.attributes` (0x70000003 vs 0x7000000b); mixing them makes the linker
    refuse the merge with `ld.lld: error: section type mismatch for
    .XPU.attributes`, and the kernel never builds.

    `cache.store(..., files=..., copy_dst_path=...)` skips any file whose
    destination already exists (`FlagTreeCache.check_file` with
    `md5_digest=None` returns True), so a tree staged by an older revision
    keeps its LLVM 22 tools forever.  Removing them first is what makes the
    re-stage actually happen; the copies that follow are local and cheap.

    Also clears the entries an earlier local staging left behind: the old
    `$LLVM_SYSPATH/lib/linux` set (including the `crtbegin`/`crtend` objects
    that are not part of the LLVM 19 package) and the dangling symlinks in
    `llvm19/bin` (their targets were never staged).
    """
    tp = os.path.join(flagtree_dir, "third_party")
    stale = []

    # --- xpu3 elfconv toolchain (the SDNN / dsa path) ---
    base = os.path.join(tp, "xpu", "backend", "xpu3")
    stale += [
        os.path.join(base, "bin", t) for t in ("clang", "xpu-xxd", "xpu3-elfconv", "xpu-kernel.t", "ld.lld",
                                               "llvm-readelf", "llvm-objdump", "llvm-objcopy", "xpu3-crt.xpu")
    ]
    stale += [
        os.path.join(base, "lib", "linux", t)
        for t in ("clang_rt.crtbegin-xpu3.o", "clang_rt.crtend-xpu3.o", "libclang_rt.builtins-xpu3.a",
                  "libclang_rt.builtins-xpu3s.a", "libclang_rt.xpuprintf-xpu3.a", "libclang_rt.xpuprintfs-xpu3.a")
    ]

    # --- the `llc19` toolchain (`llvm19_toolchain` resolves it here) ---
    llvm19 = os.path.join(tp, "xpu", "backend", "llvm19")
    for sub in ("bin", os.path.join("lib", "linux")):
        d = os.path.join(llvm19, sub)
        if os.path.isdir(d):
            stale += [os.path.join(d, n) for n in os.listdir(d)]

    # --- the one shared copy of the LLVM 19 runtime ---
    shared = os.path.join(tp, "_xtdk_llvm19", "lib")
    if os.path.isdir(shared):
        stale += [os.path.join(shared, n) for n in os.listdir(shared)]

    for path in stale:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)


def _write_shared_llvm19_package_marker(flagtree_dir):
    """Make the shared LLVM 19 lib dir an importable package.

    The wheel reaches this directory through the package symlink
    `python/triton/backends/_xtdk_llvm19`; the marker file makes it a package
    so `include_package_data` ships the shared objects with it.
    """
    lib_dir = os.path.join(flagtree_dir, "third_party", "_xtdk_llvm19")
    os.makedirs(lib_dir, exist_ok=True)
    init = os.path.join(lib_dir, "__init__.py")
    if not os.path.exists(init):
        with open(init, "w") as f:
            f.write("# Staged by setup.py: one shared copy of the LLVM 19 runtime\n"
                    "# (libLLVM.so.19.1 / libclang-cpp.so.19.1) for every backend.\n")
    # The symlink the paragraph above describes.  Without it the directory is
    # never discovered as a package, so the wheel ships `opt` / `llc` with an
    # RPATH pointing into a directory the wheel does not contain -- and every
    # staged tool dies with `libLLVM.so.19.1: cannot open shared object file`.
    # A source tree hides this: there the RPATH resolves against the real
    # `third_party/_xtdk_llvm19/lib`.
    link_path = os.path.join(flagtree_dir, "python", "triton", "backends", _XTDK_LLVM19_SHARED)
    if os.path.islink(link_path):
        os.unlink(link_path)
    elif os.path.exists(link_path):
        shutil.rmtree(link_path)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    os.symlink(os.path.abspath(lib_dir), link_path, target_is_directory=True)


_XTDK_LLVM19_SHARED = "_xtdk_llvm19"
_XTDK_LLVM19_SHLIBS = ("libLLVM.so.19.1", "libclang-cpp.so.19.1")


def shared_llvm19_lib_dir(flagtree_dir):
    """The one staged copy of the LLVM 19 shared objects."""
    return os.path.join(flagtree_dir, "third_party", _XTDK_LLVM19_SHARED, "lib")


def _installed_layout_path(flagtree_dir, path):
    """Map a source-tree path to the path the same file gets once installed.

    `build_py` (and therefore every wheel) re-homes
    `third_party/<b>/backend/<...>` to `triton/backends/<b>/<...>`: the
    `backend` component disappears, so the staged tree is *one level
    shallower* than the source tree -- and an `$ORIGIN`-relative path
    computed from the source tree points one level too high in an install
    (at `triton/_xtdk_llvm19`, which does not exist).

    The package symlinks (`python/triton/backends/<b>` ->
    `third_party/<b>/backend`) are the same mapping, which is why a tool
    started through the package path already resolves at the installed
    depth in a source checkout.
    """
    rel = os.path.relpath(path, os.path.join(flagtree_dir, "third_party"))
    parts = rel.split(os.sep)
    if len(parts) > 1 and parts[1] == "backend":
        parts = parts[:1] + parts[2:]
    return os.path.join(flagtree_dir, "python", "triton", "backends", *parts)


def rpath_to_shared_llvm19(flagtree_dir, from_dir):
    """`$ORIGIN`-relative RPATH entries reaching `shared_llvm19_lib_dir()`.

    Both spellings are emitted -- the source-tree depth and the installed
    depth -- because the tools run from *both*: from
    `third_party/xpu/backend/...` in a checkout, and from
    `triton/backends/xpu/...` once packaged, and `$ORIGIN` follows the path
    the tool was started with (`path_to_xtdk_lld()` and the backend call
    sites do start some of them through the package symlink).  A wrong-depth
    entry names a directory that does not exist and is skipped by the
    loader, so carrying both cannot select a different library.
    """
    entries = []
    for from_d, shared_d in ((from_dir, shared_llvm19_lib_dir(flagtree_dir)),
                             (_installed_layout_path(flagtree_dir, from_dir),
                              _installed_layout_path(flagtree_dir, shared_llvm19_lib_dir(flagtree_dir)))):
        rel = os.path.relpath(shared_d, from_d).replace(os.sep, "/")
        entry = "$ORIGIN/" + rel
        if entry not in entries:
            entries.append(entry)
    return ":".join(entries)


def _find_patchelf(cache):
    """Return the patchelf executable to use, or None.

    Resolution order:
      1. patchelf on PATH (developer machines, distro package)
      2. `bin/patchelf` shipped inside the llvm-pub22 prebuilt (the v2
         tarballs bundle a statically linked one: CI images do not carry a
         system patchelf and their network cannot reach PyPI)
    """
    exe = shutil.which("patchelf")
    if exe:
        return exe
    cand = os.path.join(cache.dir_path, "xpu", "llvm-pub22", "bin", "patchelf")
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return None


def _drop_per_directory_llvm19_shlibs(flagtree_dir):
    """Delete per-directory libLLVM/libclang-cpp copies from an older layout.

    Staging no longer writes them (they are byte-identical duplicates across
    directories); trees staged by an earlier setup.py still carry them, and
    build_py copies whatever it finds.  Only runs once the shared copy
    exists, so a failure to stage cannot leave the install without any
    libLLVM.
    """
    shared = shared_llvm19_lib_dir(flagtree_dir)
    if not all(os.path.isfile(os.path.join(shared, n)) for n in _XTDK_LLVM19_SHLIBS):
        return
    dirs = [os.path.join(flagtree_dir, "third_party", "xpu", "backend", "llvm19", "lib")]
    dirs += [
        os.path.join(flagtree_dir, "third_party", "xpu", "backend", arch, "lib") for arch in ("xpu3", "xpu4", "xpu5")
    ]
    for build_lib in Path(flagtree_dir).glob("build/lib*"):
        dirs += [str(build_lib / "triton" / "backends" / "xpu" / "llvm19" / "lib")]
        dirs += [str(build_lib / "triton" / "backends" / "xpu" / arch / "lib") for arch in ("xpu3", "xpu4", "xpu5")]
    for d in dirs:
        for name in _XTDK_LLVM19_SHLIBS:
            p = os.path.join(d, name)
            if os.path.isfile(p) and not os.path.islink(p):
                os.remove(p)


def _repoint_staged_tools_to_shared_llvm19(flagtree_dir, cache):
    """Point every staged LLVM 19 tool's RPATH at the shared runtime.

    Runs after all tool staging, over the directories that hold tools which
    link libLLVM: `xpu/backend/llvm19/bin` (llc19/clang) and
    `xpu/backend/xpu3/bin` (the elfconv tools, ld.lld, llvm-objcopy/...).
    `$ORIGIN/../lib` stays first so a leftover per-directory copy still wins
    -- the layout change must not break a tree staged by an older setup.py.
    """
    patchelf = _find_patchelf(cache)
    _drop_per_directory_llvm19_shlibs(flagtree_dir)
    if patchelf is None:
        print(
            "WARNING: no patchelf found; the staged LLVM 19 tools keep their "
            "DT_RUNPATH, so libLLVM.so.19.1 resolves through LD_LIBRARY_PATH "
            "(prepended by llvm19_toolchain.llvm19_env and the backend call "
            "sites).  An older ambient libLLVM would then win.", flush=True)
        return
    backend_root = os.path.join(flagtree_dir, "third_party")
    tool_dirs = [os.path.join(backend_root, "xpu", "backend", "llvm19", "bin")]
    tool_dirs += [os.path.join(backend_root, "xpu", "backend", arch, "bin") for arch in ("xpu3", "xpu4", "xpu5")]
    for d in tool_dirs:
        if not os.path.isdir(d):
            continue
        rpath = f"$ORIGIN/../lib:$ORIGIN:{rpath_to_shared_llvm19(flagtree_dir, d)}"
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if not os.path.isfile(p) or os.path.islink(p):
                continue
            # --force-rpath on purpose: DT_RPATH is searched *before*
            # LD_LIBRARY_PATH, DT_RUNPATH after it.  Non-ELF entries
            # (xpu-kernel.t, *-crt.xpu) fail silently by design (check=False).
            subprocess.run([patchelf, "--force-rpath", "--set-rpath", rpath, p], check=False, capture_output=True)


def link_elfconv_triton(flagtree_dir):
    # xpu3-elfconv-triton resolves llvm-readelf/objdump/objcopy from xpu3/ (parent of bin/).
    xpu3_dir = os.path.join(flagtree_dir, "third_party", "xpu", "backend", "xpu3")
    bin_dir = os.path.join(xpu3_dir, "bin")
    for tool in ("llvm-readelf", "llvm-objdump", "llvm-objcopy"):
        tool_src = os.path.join(bin_dir, tool)
        tool_dst = os.path.join(xpu3_dir, tool)
        if os.path.exists(tool_src) and not os.path.exists(tool_dst):
            os.symlink(os.path.join("bin", tool), tool_dst)
            print(f"Created symlink: {tool_dst} -> bin/{tool}")


# pybind11 ABI versions provided by each PYBIND11_INTERNALS_VERSION. The prebuilt
# SDNN objects hard-encode a pybind11 ABI (embedded __pybind11_internals_v<N>
# symbol); libtriton must be built against the exact pybind11 the objects were
# built with. The internals version alone is NOT sufficient: pybind11 3.0.1 and
# 3.0.4 both report v11, but 3.0.4 changed the `internals` constructor
# (PR #5870: istate(get_interpreter_state_unchecked()) + tstate.set(nullptr)),
# which crashes the prebuilt SDNN bindings at runtime with a SIGSEGV in the
# pybind11 dispatcher. Pin the exact patch release here.
_PYBIND11_INTERNALS_TO_PIP = {
    4: "pybind11>=2.6,<2.12",
    5: "pybind11>=2.12,<3.0",
    11: "pybind11==3.0.1",
}


def _read_pybind11_internals_from_dir(scan_dir):
    """Return the PYBIND11_INTERNALS_VERSION embedded in the prebuilt SDNN objects."""
    import mmap
    import re
    pat = re.compile(rb"__pybind11_internals_v(\d+)")
    candidates = []
    for root, _, filenames in os.walk(str(scan_dir)):
        for fn in filenames:
            if fn.endswith((".o", ".a", ".so")):
                candidates.append(os.path.join(root, fn))
    candidates.sort(key=lambda p: (os.path.basename(p) != "triton_xpu_sdnn.cc.o", os.path.getsize(p)))
    for path in candidates:
        try:
            with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                m = pat.search(mm)
                if m:
                    return int(m.group(1))
        except (OSError, ValueError):
            continue
    return None


def _installed_pybind11_internals():
    """(internals_version, pybind11_version) for the pybind11 the build will use."""
    import re
    try:
        import pybind11
    except Exception:
        return None, None
    version = getattr(pybind11, "__version__", None)
    hdr = os.path.join(pybind11.get_include(), "pybind11", "detail", "internals.h")
    try:
        m = re.search(r"#\s*define\s+PYBIND11_INTERNALS_VERSION\s+(\d+)", Path(hdr).read_text())
    except OSError:
        return None, version
    return (int(m.group(1)) if m else None), version


def ensure_pybind11_matches_sdnn(scan_dir):
    """Verify the env pybind11 ABI against the prebuilt SDNN objects (check only)."""
    required = _read_pybind11_internals_from_dir(scan_dir)
    if required is None:
        return
    installed, version = _installed_pybind11_internals()

    # The internals version must match, AND the patch release must be the exact
    # one the prebuilt objects were compiled against. For v11 (pybind11 3.0.x)
    # the objects are built with 3.0.1; 3.0.4 passes the internals check but
    # crashes at runtime (PR #5870 constructor change), so pin the exact version.
    required_pip = _PYBIND11_INTERNALS_TO_PIP.get(required)
    if required_pip is not None and required_pip.startswith("pybind11=="):
        required_version = required_pip[len("pybind11=="):]
        if installed == required and version == required_version:
            print(f"[XPU] pybind11 ABI OK: env pybind11 {version} (internals v{installed}) "
                  f"matches prebuilt SDNN objects (internals v{required})")
            return
    elif installed == required:
        print(f"[XPU] pybind11 ABI OK: env pybind11 {version} (internals v{installed}) "
              f"matches prebuilt SDNN objects (internals v{required})")
        return

    pip_spec = _PYBIND11_INTERNALS_TO_PIP.get(required)
    detail = (
        f"[XPU] pybind11 ABI mismatch: prebuilt SDNN objects require "
        f"PYBIND11_INTERNALS_VERSION={required}" +
        (f" with pybind11=={required_version}" if required_pip and required_pip.startswith("pybind11==") else "") +
        f", but the environment's pybind11 {version} provides {installed}." +
        " Building against a mismatched pybind11 makes `import " +
        "triton._C.libtriton` fail or segfault in the pybind11 dispatcher " +
        "('Cannot overload existing non-function object ... with a function " + "of the same name').")
    hint = (f" Install a matching pybind11 first, e.g. `pip install '{pip_spec}'`, then rebuild."
            if pip_spec else " No known pybind11 release maps to that internals version.")
    raise RuntimeError(detail + hint)


def check_pybind11_abi(cache):
    """Verify the env pybind11 ABI matches the prebuilt SDNN objects."""
    scan_dir = None
    try:
        scan_dir = Path(cache.get("xpu-sdnn-objects"))
    except KeyError:
        scan_dir = None
    if scan_dir is None or not scan_dir.is_dir():
        scan_dir = Path(cache.flagtree_dir) / "third_party" / "xpu"
    ensure_pybind11_matches_sdnn(scan_dir)


def collect_xpu_backend_package_data(backend):
    files = []
    driver_c = Path(backend.backend_dir) / "driver.c"
    if driver_c.exists():
        files.append("driver.c")
    xpu3_dir = Path(backend.backend_dir) / "xpu3"
    if xpu3_dir.is_dir():
        files.extend(str(path.relative_to(backend.backend_dir)) for path in xpu3_dir.rglob("*") if path.is_file())
    return files


def ensure_xpu_launch_static_lib(backend):
    src = Path(backend.src_dir) / "device" / "xpu3" / "liblaunch.a"
    if not src.exists():
        print(f"[XPU] liblaunch.a not found at {src}; packaged launcher may fail to link", file=sys.stderr)
        return
    dst = Path(backend.backend_dir) / "xpu3" / "lib" / "liblaunch.a"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[XPU] copied liblaunch.a: {src} -> {dst}", file=sys.stderr)


def _compiled_python_packages():
    """Packages under the XPU surface that carry compiled extension modules.

    `build_py` copies `.py` modules on its own, but a non-`.py` file sitting
    next to them is only picked up by `package_data` -- so every package that
    holds a compiled module has to be named. Computed from the tree rather
    than hand-listed, so the set follows the actual scope.

    Only interpreter-specific suffixes count: a bare `.so` is also what plain
    shared libraries use (`libtriton.so`, the staged LLVM runtime), and
    matching those would name packages that hold no Python module at all.
    """
    suffixes = tuple(s for s in importlib.machinery.EXTENSION_SUFFIXES if s != ".so")
    # Both roots are walked down to the package root itself, so `relative_to`
    # yields a package-relative path that the prefix can be joined onto.
    roots = ((Path(XPU_PYTHON_ROOT) / "triton", "triton"), (Path(BACKEND_ROOT), "triton.backends.xpu"))
    for root, prefix in roots:
        if not root.is_dir():
            continue
        for so in sorted(root.rglob("*")):
            if not so.is_file() or not so.name.endswith(suffixes):
                continue
            # `package_data` silently does nothing for a name setuptools does
            # not know as a package, so only yield directories that carry an
            # `__init__.py` (the root itself is always a package).
            if so.parent != root and not (so.parent / "__init__.py").exists():
                continue
            rel = so.parent.relative_to(root)
            yield prefix if str(rel) == "." else f"{prefix}.{str(rel).replace(os.sep, '.')}"


def get_package_data(backends):
    package_data = {
        "triton": ["FLAGTREE_BACKEND"], "triton.backends": ["_tables/*.json"],
        # The one shared copy of the LLVM 19 runtime.  The package is
        # reached through the symlink `python/triton/backends/_xtdk_llvm19`
        # (see `_write_shared_llvm19_package_marker`), and the staged
        # `opt` / `llc` carry an $ORIGIN-relative RPATH into it -- so a
        # wheel that omits these objects ships tools that die with
        # `libLLVM.so.19.1: cannot open shared object file`.
        # `package_data` is what actually copies them: a symlinked
        # package directory does not block these globs (tested), but
        # nothing else picks up a non-`.py` file.  The glob also covers
        # any library added to that directory later.
        "triton.backends._xtdk_llvm19": ["lib/*.so*"]
    }
    for backend in backends:
        if backend.name == "xpu":
            files = collect_xpu_backend_package_data(backend)
            if files:
                package_data["triton.backends.xpu"] = files
            break
    # Compiled XPU Python implementation: `.so` files sit where the `.py`
    # modules used to, and only `package_data` copies them into the wheel.
    for package in _compiled_python_packages():
        package_data.setdefault(package, [])
        if "*.so" not in package_data[package]:
            package_data[package].append("*.so")
    return package_data


def overlay_runtime_so(cache, build_py_command=None, backends=None):
    """Overwrite third_party/xpu/backend/xpu3/so with the fixed runtime .so set."""
    try:
        src = Path(cache.get("xpu-runtime-so"))
    except KeyError:
        src = None
    if src is None or not src.is_dir():
        print(f"[XPU] runtime-so overlay skipped: {src} not found")
    else:
        dst = Path(cache.flagtree_dir) / "third_party" / "xpu" / "backend" / "xpu3" / "so"
        if dst.exists():
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = src / item
            d = dst / item
            if s.is_dir():
                shutil.copytree(s, d, symlinks=True)
            else:
                shutil.copy2(s, d)
        print(f"[XPU] runtime-so overlay applied: {src} -> {dst}")

    if build_py_command is None or backends is None:
        return
    build_py_command.distribution.package_data = build_py_command.distribution.package_data or {}
    for backend in backends:
        if backend.name == "xpu":
            ensure_xpu_launch_static_lib(backend)
            files = collect_xpu_backend_package_data(backend)
            if files:
                # Re-scan the collected files, but keep the patterns
                # `get_package_data` added for this package -- replacing the
                # list outright drops them, and the `*.so` one is what carries
                # the compiled Python implementation (`compiler` / `driver`
                # among them), so the wheel would ship a backend that cannot
                # import.
                data = build_py_command.distribution.package_data
                existing = data.get("triton.backends.xpu") or []
                data["triton.backends.xpu"] = files + [p for p in existing if p not in files]
            build_py_command.package_data = build_py_command.distribution.package_data
            build_py_command.__dict__.pop("data_files", None)
            break


_XPU_LIBSTDCXX_PTH_NAME = "zzz_flagtree_xpu_libstdcxx.pth"
_XPU_LIBSTDCXX_PTH_BODY = ("import sys; exec(\"try:\\n"
                           " import os, ctypes\\n"
                           " _p = os.path.join(sys.prefix, 'lib', 'libstdc++.so.6')\\n"
                           " if os.path.isfile(_p) and b'GLIBCXX_3.4.30' in open(_p, 'rb').read(4194304):\\n"
                           "  ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)\\n"
                           "except Exception:\\n"
                           " pass\")\n")


def write_site_pth(dest_dir):
    """Write the xpu libstdc++ preload .pth into dest_dir (build_lib root => site-packages)."""
    if not dest_dir:
        return
    try:
        os.makedirs(dest_dir, exist_ok=True)
        out = os.path.join(dest_dir, _XPU_LIBSTDCXX_PTH_NAME)
        with open(out, "w") as f:
            f.write(_XPU_LIBSTDCXX_PTH_BODY)
        print(f"[XPU] wrote libstdc++ preload pth: {out}")
    except OSError as exc:
        print(f"[XPU] could not write libstdc++ preload pth: {exc}")


def _warn_external_llvm_syspath():
    """Loudly report an exported `LLVM_SYSPATH`, which both legs honour.

    Both LLVM entries guard on `LLVM_SYSPATH` being unset -- the upstream
    convention for "use my own LLVM build".  With two legs that escape hatch
    also decides *which* frontend the build gets, and the two are not
    interchangeable: public LLVM and XTDK LLVM disagree on MLIR
    (`RegionSuccessor` / `RegionBranchPoint`) and on the XTDK-only XPU/XCN
    targets, so the wrong one fails to configure or builds a different
    compiler than this branch was validated with.  A stale export -- e.g. the
    `LLVM_SYSPATH=<llvm_trust>` that older in-tree docs recommend -- silently
    pins the fallback frontend.
    """
    path = os.environ.get("LLVM_SYSPATH", "").strip()
    if not path:
        return
    want = "xtdk" if _use_xtdk_frontend() else "public"
    if "llvm-pub" in path:
        got = "public"
    elif "llvm_trust" in path or "xtdk" in path:
        got = "xtdk"
    else:
        got = "unknown"
    if got != want:
        print(
            f"[XPU][WARN] LLVM_SYSPATH is already set to {path!r} ({got}) but the "
            f"{'fallback' if want == 'xtdk' else 'default'} leg expects the {want} frontend; "
            f"both LLVM downloads are skipped and the build will use the exported tree. "
            f"Unset LLVM_SYSPATH to let the build stage the {want} package"
            f"{'' if want == 'xtdk' else ', or set TRITON_USE_XTDK_FRONTEND=1 to select XTDK on purpose'}.",
            file=sys.stderr)


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    """Register all XPU cache artifacts and post-install hooks."""
    is_xpu = "xpu" == flagtree_backend
    if is_xpu:
        _warn_external_llvm_syspath()
    use_xtdk_frontend = _use_xtdk_frontend()
    cache.store(
        file="llvm_trust",
        condition=is_xpu,
        # XTDK LLVM 22.  Two jobs, and only the first is optional:
        #   1. the *fallback* frontend (`TRITON_USE_XTDK_FRONTEND=1`) -- it is
        #      the only leg that links the XTDK XPU/XCN target backends and
        #      provides XTDKDL, hence the env hook is gated on that flag;
        #   2. the source of `xpu3-elfconv-triton`, which ships in no other
        #      package (see the copy below), so the download always happens.
        # Ordered BEFORE `llvm-pub22` on purpose: both entries guard on
        # `LLVM_SYSPATH` being unset, and whichever stages first would otherwise
        # short-circuit the other -- staging public first silently dropped this
        # package and with it the elfconv wrapper.
        url="https://klx-sdk-release-public.su.bcebos.com/XTriton/llvm22/20260615/xtdk-llvm22-ubuntu2004_x86_64.tar.gz",
        pre_hook=lambda: check_env('LLVM_SYSPATH'),
        post_hook=(lambda path: (set_llvm_env(path), _patch_llvm_exports())) if use_xtdk_frontend else
        (lambda path: None),
        version="20260615",
    )
    cache.store(
        file="llvm-pub22",
        condition=is_xpu and not use_xtdk_frontend,
        # The default frontend: the same public LLVM 22 tarball the sync source
        # pins (`v1/triton/llvm-pub/22.1.8/`, content-pinned there).  It is what
        # the public-LLVM compatibility layer and the whole XPU build were validated
        # against; picking XTDK instead makes the two trees disagree on MLIR
        # (`RegionSuccessor` / `RegionBranchPoint` and the XTDK-only XPU/XCN
        # targets) -- see gate G15 and evidence 2026-09-24-public-llvm-not-durable.
        # Served from the XTriton/ mirror: the original path-style URL
        # (`su.bcebos.com/klx-sdk-release-public/v1/...`) is unreachable through
        # the CI runner's proxy.  The mirror is byte-identical (md5
        # 43e662cfae936ba8524d082d70b0308a, 1,492,666,872 B) and was verified
        # by a full re-download after upload.
        url="https://klx-sdk-release-public.su.bcebos.com/XTriton/llvm-pub-22.1.8-ubuntu2004_x86_64-v2.tar.gz",
        pre_hook=lambda: check_env('LLVM_SYSPATH'),
        post_hook=set_llvm_env,
        version="22.1.8-v2",
    )
    cache.store(file="xre-Linux-x86_64", condition=is_xpu,
                url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz",
                copy_dst_path='python/_deps/xre3', version="v0.3.0")
    cache.store(file="xpu-device-libs", condition=is_xpu,
                url="https://klx-sdk-release-public.su.bcebos.com/XTriton/xpu-device-libs-ubuntu-x64_v0.3.6.1.1.tar.gz",
                version="v0.3.6.1.1")
    cache.store(files=("liblaunch_shared.so", "libLLVM-15.so", "libclang-cpp.so.15", "libxpujitc.so"), condition=is_xpu,
                copy_src_path=f"{cache.dir_path}/{flagtree_backend}/xpu-device-libs",
                copy_dst_path=f"third_party/{flagtree_backend}/device")
    cache.store(
        file="xpu-sdnn-objects", condition=is_xpu,
        # v0.3.6.9.0: **public-LLVM leg** (the toolchain the
        # public-LLVM compatibility layer and the rest of this branch are built
        # against).  Numbered from the newest *published* package (v0.3.6.8.1,
        # verified present on BOS) -- locally built candidates do not consume a
        # version number, they are invisible to consumers.
        # Earlier local drafts on the XTDK leg were cut under the same name; they
        # were never uploaded.  Their SDNN objects and `libTritonSharedForXPU.a`
        # reference the XTDK MLIR signature
        # `...visitRegionSuccessors(...RegionBranchPoint...)` and the out-of-line
        # `APFloatBase::PPCDoubleDouble()`, neither of which the public LLVM 22
        # provides -- `import triton` died on the first missing symbol.
        # See evidence 2026-09-24-import-broken-sdnn-xdk-leg.
        url="https://klx-sdk-release-public.su.bcebos.com/XTriton/xpu-sdnn-objects_v0.3.6.9.0.tar.gz",
        # Only the first 8 hex chars of the digest are compared (see
        # setup_helper.check_file). The value was verified by re-downloading the
        # artifact: for multipart uploads BOS's ETag (prefixed with '-') is not
        # the file's md5, so the ETag cannot serve as a digest.
        # Re-issued under the same version: the closed-source packaging added
        # gluon_ir_sdnn.cc.o and abi_table.cc.o to this package.  A rebuilt
        # candidate does not consume a version number, so only the digest moves.
        # Re-issued under the same version twice: the first re-issue added
        # gluon_ir_sdnn.cc.o and abi_table.cc.o, the second restored
        # libTritonSharedForXPU.a, which the first one dropped.  That `.a` is a
        # CMake IMPORTED target with no build rule, so a cold-cache build died
        # on it while every warm cache (which still had the old copy) passed.
        # A rebuilt candidate does not consume a version number, so only the
        # digest moves.
        # Third re-issue (2026-10-07): split gluon_ir_sdnn.cc.o restored
        # <pybind11/stl.h> (vector args were un-callable); triton_xpu_sdnn.cc.o
        # v12 -> v11.
        md5_digest="e92ded7a", version="v0.3.6.9.0",
        post_hook=lambda path: install_sdnn_objects(path, cache.flagtree_dir))
    cache.store(
        file="xpu-python-impl-so", condition=is_xpu,
        # The XPU Python implementation, compiled to extension modules.  abi3
        # (`Py_LIMITED_API`) is what makes this shippable as a prebuilt artifact
        # at all: one `.so` serves every CPython >= 3.8, so the package does not
        # have to track the builder's interpreter version.
        # The `.py` it replaces are absent from the tree on purpose -- this
        # download is the only source for them, so a fetch failure must surface
        # rather than fall back to readable sources that no longer exist.
        url="https://klx-sdk-release-public.su.bcebos.com/XTriton/xpu-python-impl-so_v0.3.6.9.0.tar.gz",
        # Only the first 8 hex chars are compared (see setup_helper.check_file).
        # Verified by re-downloading the artifact: BOS's ETag is prefixed with
        # '-' for multipart uploads and is not the file's md5.
        md5_digest="5cd84a56", version="v0.3.6.9.0",
        post_hook=lambda path: install_python_impl(path, cache.flagtree_dir))
    cache.store(
        file="xtdk-llvm19",
        condition=is_xpu,
        # The per-arch elfconv toolchain, and it must be LLVM 19 -- not the
        # LLVM 22 frontend `llvm_trust` stages.  `make_elf` lowers the kernel
        # through `llvm19_toolchain` (`downgrade_ir` -> `llc19`), so the kernel
        # object carries LLVM 19's `.XPU.attributes` section type
        # (0x70000003); the CRT object is compiled by `$CLANG_PATH/clang` and
        # must agree, or the link dies with `section type mismatch`.  This pin
        # is the *desensitized* build (version.txt `Date: 20260916` / commit
        # cdc64fae); the internal tree stays on 20260827 until the change is
        # carried back.  `trust/latest/` may be re-uploaded in place -- the
        # digest pin turns that into a loud failure instead of a silent stage
        # (a dated fixed path is requested from the provider).
        url="https://klx-sdk-release-public.su.bcebos.com/xtdk_llvm19/trust/latest/xtdk-llvm19-ubuntu2004_x86_64.tar.gz",
        md5_digest="d1eb1e77",
        version="20260916",
    )
    cache.store(
        file="xtdk-llvm19-printf",
        condition=is_xpu,
        # The device printf runtime, pinned to its own release rather than taken
        # from the toolchain tree: the protocol it speaks has to match the
        # deployed host-side reader, and the toolchain snapshot carries the OLD
        # protocol (stride 2048B, 0xF0000 window, no bounds guard) -- staging it
        # makes `test_print` fail wholesale with "saw 0 time(s)".
        url="https://klx-sdk-release-public.su.bcebos.com/v1/triton/XTDK/20260403/xtdk-llvm19-ubuntu2004_x86_64.tar.gz",
        version="20260403",
    )
    _llvm19_pkg = f"{cache.dir_path}/{flagtree_backend}/xtdk-llvm19"
    _printf_pkg = f"{cache.dir_path}/{flagtree_backend}/xtdk-llvm19-printf"
    if is_xpu:
        _sweep_stale_llvm19_staging(cache.flagtree_dir)
        _write_shared_llvm19_package_marker(cache.flagtree_dir)
    cache.store(
        # The `llc19` toolchain.  `llvm19_toolchain.get_llvm19_bin_dir()` resolves
        # it here and raises "LLVM 19 toolchain not found" if `llc` is absent --
        # and it is absent on a fresh checkout, so this staging is what makes the
        # `downgrade_ir` -> `llc19` half of `make_elf` work at all.
        files=("llc", "llvm-mc", "clang", "clang++", "clang-offload-bundler", "ld.lld", "llvm-readelf", "llvm-objcopy",
               "llvm-objdump", "xpu-elfconv", "xpu2-elfconv", "xpu3-elfconv", "xpu4-elfconv", "xpu5-elfconv", "xpu-xxd",
               "xpu-crt.xpu", "xpu3-crt.xpu", "xpu4-crt.xpu", "xpu-kernel.t"), condition=is_xpu,
        copy_src_path=f"{_llvm19_pkg}/bin", copy_dst_path="third_party/xpu/backend/llvm19/bin")
    # `opt` (read-in) and `llvm-link` (tle.raw merge) are not in the desensitized
    # package: the read-in degrades when `opt` is absent (see compiler.py) and
    # tle.raw payload kernels fail loudly without `llvm-link`.
    cache.store(
        # clang's resource dir for the LLVM 19 side (the tools above read it
        # through `$ORIGIN/../lib/clang/19`).
        files=("19", ), condition=is_xpu, copy_src_path=f"{_llvm19_pkg}/lib/clang",
        copy_dst_path="third_party/xpu/backend/llvm19/lib/clang")
    cache.store(files=("libclang_rt.builtins-xpu3.a", "libclang_rt.builtins-xpu3s.a"), condition=is_xpu,
                copy_src_path=f"{_llvm19_pkg}/lib/linux", copy_dst_path="third_party/xpu/backend/llvm19/lib/linux")
    cache.store(files=("libclang_rt.xpuprintf-xpu3.a", "libclang_rt.xpuprintfs-xpu3.a"), condition=is_xpu,
                copy_src_path=f"{_printf_pkg}/lib/linux", copy_dst_path="third_party/xpu/backend/llvm19/lib/linux")
    cache.store(
        # ONE shared copy of the LLVM 19 runtime for the whole install.  The
        # staged tools are dynamically linked against `libLLVM.so.19.1`; without
        # this they only load when the caller happens to prepend the right dir
        # to `LD_LIBRARY_PATH` (which `llvm19_toolchain.llvm19_env` does), and
        # running one directly dies with "cannot open shared object file".
        files=("libLLVM.so.19.1", "libclang-cpp.so.19.1"), condition=is_xpu, copy_src_path=f"{_llvm19_pkg}/lib",
        copy_dst_path="third_party/_xtdk_llvm19/lib")
    cache.store(
        # `xpu3-elfconv-triton` is the one tool NOT in the LLVM 19 package: the
        # Triton-customized wrapper only ships in the XTDK LLVM 22 tarball.
        # Sourced from that package's own cache dir rather than `$LLVM_SYSPATH`:
        # in the default (public-frontend) leg the env points at the public
        # LLVM, which does not carry this wrapper.
        files=("xpu3-elfconv-triton", ), condition=is_xpu,
        copy_src_path=f"{cache.dir_path}/{flagtree_backend}/llvm_trust/bin",
        copy_dst_path="third_party/xpu/backend/xpu3/bin")
    cache.store(
        # `xpu3-crt.xpu` is the ELF-conversion CRT: `xpu{arch}-elfconv-triton` looks for it at
        # `$CLANG_PATH/xpu{arch}-crt.xpu` and, when absent, falls back to decoding a base64 blob
        # embedded in its own script -- which ships with the __CRT_SRC_B64__ placeholder unreplaced
        # outside the SDK packaging step, so the build dies with `base64: invalid input` on every
        # kernel (not just tle.raw/dsa ones). Ship the file so the fallback is never taken.
        files=("clang", "xpu-xxd", "xpu3-elfconv", "xpu-kernel.t", "ld.lld", "llvm-readelf", "llvm-objdump",
               "llvm-objcopy", "xpu3-crt.xpu"), condition=is_xpu,
        copy_src_path=f"{cache.dir_path}/{flagtree_backend}/xtdk-llvm19/bin",
        copy_dst_path="third_party/xpu/backend/xpu3/bin")
    if is_xpu:
        link_elfconv_triton(cache.flagtree_dir)
        # Every staged LLVM 19 tool now resolves the shared copy: DT_RPATH where
        # patchelf exists, LD_LIBRARY_PATH (llvm19_env and the backend call
        # sites) otherwise.
        _repoint_staged_tools_to_shared_llvm19(cache.flagtree_dir, cache)
    cache.store(files=("libclang_rt.builtins-xpu3.a", "libclang_rt.builtins-xpu3s.a"), condition=is_xpu,
                copy_src_path=f"{_llvm19_pkg}/lib/linux", copy_dst_path="third_party/xpu/backend/xpu3/lib/linux")
    cache.store(
        # From the printf release, not the toolchain tree -- see the
        # `xtdk-llvm19-printf` entry above.
        files=("libclang_rt.xpuprintf-xpu3.a", "libclang_rt.xpuprintfs-xpu3.a"), condition=is_xpu,
        copy_src_path=f"{_printf_pkg}/lib/linux", copy_dst_path="third_party/xpu/backend/xpu3/lib/linux")
    cache.store(
        # clang's resource-dir header tree.  The kernel pipeline itself does not
        # need it (it compiles bitcode), but a `tle.raw` payload does: it
        # `#include`s "xpu/kernel/xtdk.h", which only lives here.  Without it an
        # installed (non-checkout) triton fails with
        # `fatal error: 'xpu/kernel/xtdk.h' file not found`.
        files=("include", ), condition=is_xpu,
        copy_src_path=f"{cache.dir_path}/{flagtree_backend}/xtdk-llvm19/lib/clang/19",
        copy_dst_path="third_party/xpu/backend/xpu3/lib/clang/19")
    cache.store(files=("include", "so"), condition=is_xpu, copy_src_path=f"{cache.dir_path}/xpu/xre-Linux-x86_64",
                copy_dst_path="third_party/xpu/backend/xpu3")
    cache.store(
        file="xpu-runtime-so",
        condition=is_xpu,
        url="https://klx-sdk-release-public.su.bcebos.com/XTriton/xpu-runtime-so_v0.3.6.2.0.tar.gz",
        version="v0.3.6.2.0",
    )
    if is_xpu:
        overlay_runtime_so(cache)


def _wrap_setup(original_setup):
    if getattr(original_setup, "_xpu_python_root_patched", False):
        return original_setup

    def setup_with_xpu_python_root(*args, **kwargs):
        kwargs["packages"] = _merge_xpu_packages(kwargs.get("packages", []))
        kwargs["package_dir"] = _merge_xpu_package_dir(kwargs.get("package_dir", {}))
        kwargs["cmdclass"] = _patch_xpu_cmdclass(kwargs.get("cmdclass", {}))
        return original_setup(*args, **kwargs)

    setup_with_xpu_python_root._xpu_python_root_patched = True
    setup_with_xpu_python_root._xpu_original_setup = original_setup
    return setup_with_xpu_python_root


def _patch_setup_for_xpu_python_root():
    patched = False

    frame = inspect.currentframe()
    while frame is not None:
        setup_func = frame.f_globals.get("setup")
        if callable(setup_func):
            frame.f_globals["setup"] = _wrap_setup(setup_func)
            patched = True
        frame = frame.f_back

    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "setup"):
        main_module.setup = _wrap_setup(main_module.setup)
        patched = True

    main_file = getattr(main_module, "__file__", "") if main_module is not None else ""
    if not patched and os.path.basename(main_file) == "setup.py":
        raise RuntimeError("xpu setup hook could not find setup() to patch")


_patch_setup_for_xpu_python_root()
