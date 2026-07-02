import os
import sys
import shutil
import inspect
from pathlib import Path

from setuptools import find_packages

XPU_PYTHON_ROOT = "third_party/xpu/python"
FLAGTREE_PYTHON_ROOT = "python"
TLE_PACKAGE = "triton.experimental.tle"


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

    # TLE is a FlagTree-only feature kept in the main tree; the XPU overlay
    # does not carry it, so source it from the main tree.
    for package in find_packages(where=FLAGTREE_PYTHON_ROOT, include=[TLE_PACKAGE, f"{TLE_PACKAGE}.*"]):
        add(package)

    for package in existing_packages:
        if (not package.startswith("triton.") or _is_backend_package(package) or _is_language_extra_package(package)
                or package == "triton.profiler" or package.startswith("triton.profiler.")):
            add(package)

    return packages


def _merge_xpu_package_dir(existing_package_dir):
    package_dir = dict(existing_package_dir or {})
    package_dir[""] = XPU_PYTHON_ROOT

    for package in find_packages(where=XPU_PYTHON_ROOT, include=["triton", "triton.*"]):
        rel_package_path = package.replace(".", "/")
        package_dir[package] = f"{XPU_PYTHON_ROOT}/{rel_package_path}"

    for package in find_packages(where=FLAGTREE_PYTHON_ROOT, include=[TLE_PACKAGE, f"{TLE_PACKAGE}.*"]):
        rel_package_path = package.replace(".", "/")
        package_dir[package] = f"{FLAGTREE_PYTHON_ROOT}/{rel_package_path}"

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


def install_sdnn_objects(cached_path, flagtree_dir):
    """Copy prebuilt SDNN objects from cache to third_party/xpu/."""
    dst_root = os.path.join(flagtree_dir, "third_party", "xpu")
    for item in os.listdir(cached_path):
        src = os.path.join(str(cached_path), item)
        dst = os.path.join(dst_root, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)

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
    print(f"[XPU] SDNN prebuilt objects installed to {dst_root}")


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
# symbol); libtriton must be built against a pybind11 whose internals version
# matches.
_PYBIND11_INTERNALS_TO_PIP = {
    4: "pybind11>=2.6,<2.12",
    5: "pybind11>=2.12,<3.0",
    11: "pybind11>=3.0,<3.1",
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
    if installed == required:
        print(f"[XPU] pybind11 ABI OK: env pybind11 {version} (internals v{installed}) "
              f"matches prebuilt SDNN objects (internals v{required})")
        return

    pip_spec = _PYBIND11_INTERNALS_TO_PIP.get(required)
    detail = (f"[XPU] pybind11 ABI mismatch: prebuilt SDNN objects require "
              f"PYBIND11_INTERNALS_VERSION={required}, but the environment's pybind11 "
              f"{version} provides {installed}. Building against a mismatched pybind11 makes "
              f"`import triton._C.libtriton` fail with "
              f"'Cannot overload existing non-function object ... with a function of the same name'.")
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


def overlay_runtime_so(cache):
    """Overwrite third_party/xpu/backend/xpu3/so with the fixed runtime .so set."""
    try:
        src = Path(cache.get("xpu-runtime-so"))
    except KeyError:
        return
    if not src.is_dir():
        print(f"[XPU] runtime-so overlay skipped: {src} not found")
        return
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


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    """Register all XPU cache artifacts and post-install hooks."""
    is_xpu = "xpu" == flagtree_backend
    cache.store(
        file="llvm_trust",
        condition=is_xpu,
        url="https://klx-sdk-release-public.su.bcebos.com/XTriton/llvm22/20260615/xtdk-llvm22-ubuntu2004_x86_64.tar.gz",
        pre_hook=lambda: check_env('LLVM_SYSPATH'),
        post_hook=lambda path: (set_llvm_env(path), _patch_llvm_exports()),
        version="20260615",
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
    cache.store(file="xpu-sdnn-objects", condition=is_xpu,
                url="https://klx-sdk-release-public.su.bcebos.com/XTriton/xpu-sdnn-objects_v0.3.6.2.0.tar.gz",
                post_hook=lambda path: install_sdnn_objects(path, cache.flagtree_dir))
    cache.store(
        files=("clang", "xpu-xxd", "xpu3-elfconv", "xpu3-elfconv-triton", "xpu-kernel.t", "ld.lld", "llvm-readelf",
               "llvm-objdump", "llvm-objcopy"), condition=is_xpu,
        copy_src_path=f"{os.environ.get('LLVM_SYSPATH','')}/bin", copy_dst_path="third_party/xpu/backend/xpu3/bin")
    if is_xpu:
        link_elfconv_triton(cache.flagtree_dir)
    cache.store(
        files=("libclang_rt.builtins-xpu3.a", "libclang_rt.builtins-xpu3s.a", "clang_rt.crtbegin-xpu3.o",
               "clang_rt.crtend-xpu3.o", "libclang_rt.xpuprintf-xpu3.a", "libclang_rt.xpuprintfs-xpu3.a"),
        condition=is_xpu, copy_src_path=f"{os.environ.get('LLVM_SYSPATH','')}/lib/linux",
        copy_dst_path="third_party/xpu/backend/xpu3/lib/linux")
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
