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

import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages

from python.setup_tools.utils.default import FlagCXRegistrar, printinfo

OPS_PYTHON_ROOT = "third_party/iluvatar/python"
OPS_DISCOVERY_ROOT = f"{OPS_PYTHON_ROOT}/triton"
OPS_PACKAGE = "triton.ops"

__all__ = ["handle_flagcx", "relocate_flagcx"]

_FLAGCX_RELATIVE_PATH = os.path.join("iluvatar", "tle", "third_party", "flagcx")


def relocate_flagcx(*args, **kwargs):
    from python.setup_tools.utils import flagtree_submodules
    from python.setup_tools.utils.tools import flagtree_configs

    flagcx = flagtree_submodules.get("flagcx")
    if flagcx is None:
        return
    flagcx.dst_path = os.path.join(flagtree_configs.flagtree_submodule_dir, _FLAGCX_RELATIVE_PATH)


def _first_existing_file(candidates):
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return str(path)
    return None


def _has_corex_device_compile_support(path):
    """Check for the files required by CoreX device compilation."""
    path = Path(path).expanduser()
    return ((path / "include" / "cuda_runtime.h").is_file()
            and (path / "nvvm" / "libdevice" / "libdevice.compute_bi.10.bc").is_file())


def _resolve_corex_clang():
    llvm_path = os.environ.get("LLVM_SYSPATH")
    return _first_existing_file((
        os.environ.get("FLAGCX_ILUVATAR_CLANG"),
        str(Path(llvm_path) / "bin" / "clang") if llvm_path else None,
        shutil.which("clang"),
        "/usr/local/corex/bin/clang",
        "/home/corex/sw_home/local/corex/bin/clang",
    ))


def _resolve_corex_sdk(clang_path):
    clang_sdk = None
    if clang_path:
        clang_sdk = Path(clang_path).expanduser().resolve().parent.parent
    return next(
        (str(Path(candidate).expanduser()) for candidate in (
            os.environ.get("FLAGCX_ILUVATAR_SDK"),
            os.environ.get("COREX_HOME"),
            os.environ.get("COREX_SDK"),
            str(clang_sdk) if clang_sdk else None,
            "/usr/local/corex",
            "/home/corex/sw_home/local/corex",
        ) if candidate and _has_corex_device_compile_support(candidate)),
        None,
    )


def _is_ccl_home(path):
    path = Path(path).expanduser()
    include_dir = path / "include"
    lib_dirs = (path / "lib", path / "lib64")
    has_header = (include_dir / "nccl.h").is_file()
    has_library = any(
        list(lib_dir.glob("libnccl.so*")) + list(lib_dir.glob("libixccl.so*"))
        for lib_dir in lib_dirs
        if lib_dir.is_dir())
    return has_header and has_library


def _resolve_ccl_home(sdk_path):
    return next(
        (str(Path(candidate).expanduser()) for candidate in (
            os.environ.get("FLAGCX_ILUVATAR_CCL_HOME"),
            os.environ.get("CCL_HOME"),
            os.environ.get("NCCL_HOME"),
            os.environ.get("IXCCL_HOME"),
            sdk_path,
        ) if candidate and _is_ccl_home(candidate)),
        None,
    )


def _resolve_flagcx_toolchain():
    clang_path = _resolve_corex_clang()
    sdk_path = _resolve_corex_sdk(clang_path)
    ccl_path = _resolve_ccl_home(sdk_path)
    return sdk_path, clang_path, ccl_path


def _ops_packages():
    return [f"triton.{package}" for package in find_packages(where=OPS_DISCOVERY_ROOT, include=["ops", "ops.*"])]


def get_extra_install_packages():
    return _ops_packages()


def get_package_dir():
    return {package: f"{OPS_PYTHON_ROOT}/{package.replace('.', '/')}" for package in _ops_packages()}


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    cache.store(
        file="iluvatar-llvm22-x86_64",
        condition=("iluvatar" == flagtree_backend),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm22-x86_64_v0.6.1.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )


def _build_setup_hook():
    patched_attr = "_iluvatar_ops_packages_patched"

    def wrap_setup(original_setup):
        if getattr(original_setup, patched_attr, False):
            return original_setup

        def setup_with_iluvatar_ops(*args, **kwargs):
            packages = list(kwargs.get("packages", []))
            for package in _ops_packages():
                if package not in packages:
                    packages.append(package)
            kwargs["packages"] = packages

            package_dir = dict(kwargs.get("package_dir", {}))
            package_dir.update(get_package_dir())
            kwargs["package_dir"] = package_dir
            return original_setup(*args, **kwargs)

        setattr(setup_with_iluvatar_ops, patched_attr, True)
        setup_with_iluvatar_ops._iluvatar_ops_original_setup = original_setup
        return setup_with_iluvatar_ops

    return wrap_setup


def _patch_setup(wrap_setup):
    patched = False
    frame = inspect.currentframe()
    while frame is not None:
        setup_func = frame.f_globals.get("setup")
        if callable(setup_func):
            frame.f_globals["setup"] = wrap_setup(setup_func)
            patched = True
        frame = frame.f_back

    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "setup"):
        main_module.setup = wrap_setup(main_module.setup)
        patched = True

    if not patched:
        raise RuntimeError("iluvatar setup hook could not find setup() to patch")


class IluvatarFlagCXRegistrar(FlagCXRegistrar):
    """Iluvatar-specific FlagCX build and runtime file handling."""

    SUPPORTED_ARCHES = ("ivcore11", )

    def _get_iluvatar_arch(self):
        arch = os.environ.get("FLAGCX_ILUVATAR_ARCH", "ivcore11").strip().lower()
        if arch not in self.SUPPORTED_ARCHES:
            raise RuntimeError(f"Unsupported Iluvatar FlagCX device target {arch!r}; "
                               f"supported targets: {', '.join(self.SUPPORTED_ARCHES)}")
        return arch

    def _set_path(self, external):
        relocate_flagcx()
        super()._set_path(external)
        self.iluvatar_arch = self._get_iluvatar_arch()
        self.cache_lib_dir = Path(external["cache"].dir_path) / "flagcx" / self.iluvatar_arch
        self.cache_lib_dir.mkdir(parents=True, exist_ok=True)
        for lib_name in (self.bitcode_name, self.shared_lib_name):
            setattr(self, f"{lib_name.split('.')[0]}_cache_path", self.cache_lib_dir / lib_name)

    def get_compile_cmds(self):
        device_dir = Path(self.flagcx_src_dir) / "bindings" / "ir" / "iluvatar"
        if not device_dir.is_dir():
            raise FileNotFoundError(f"FlagCX Iluvatar device IR build entry not found: {device_dir}. "
                                    "The pinned FlagCX revision must provide bindings/ir/iluvatar.")

        sdk_path, clang_path, ccl_path = _resolve_flagcx_toolchain()
        if not sdk_path or not clang_path:
            raise RuntimeError("Unable to locate the Iluvatar CoreX SDK and clang. "
                               "Set FLAGCX_ILUVATAR_SDK/FLAGCX_ILUVATAR_CLANG or expose "
                               "the CoreX toolchain through LLVM_SYSPATH/PATH.")

        device_cmd = [
            "make",
            "-C",
            str(device_dir),
            f"ILUVATAR_ARCH={self.iluvatar_arch}",
            f"DEVICE_HOME={sdk_path}",
            f"COREX_CLANG={clang_path}",
        ]

        host_cmd = ["make", "USE_ILUVATAR=1", "-j", str(os.cpu_count())]
        host_cmd.append(f"DEVICE_HOME={sdk_path}")
        if ccl_path:
            host_cmd.append(f"CCL_HOME={ccl_path}")

        return {
            self.bitcode_name: device_cmd,
            self.shared_lib_name: host_cmd,
        }

    def _init_submodules(self):
        if getattr(self, "_submodules_ready", False):
            return
        printinfo(f"Initializing FlagCX submodules in {self.flagcx_src_dir}...")
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=self.flagcx_src_dir,
            check=True,
        )
        self._submodules_ready = True

    def _compile_and_cache(self):
        self._init_submodules()
        return super()._compile_and_cache()

    def _copy_required_files(self):
        src = Path(self.flagcx_src_dir) / "plugin" / "interservice" / "flagcx_wrapper.py"
        dst = Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "flagcx_wrapper.py"
        shutil.copy(src, dst)
        printinfo(f"flagcx_wrapper.py copied from {src} to {dst}")

        dst = Path(self.flagtree_dir) / "third_party" / self.backend_name / "backend" / "flagcx_wrapper.py"
        shutil.copy(src, dst)
        printinfo(f"flagcx_wrapper.py copied from {src} to {dst}")

        dst = Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "include"
        src = Path(self.flagcx_src_dir) / "flagcx" / "include"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        printinfo(f"FlagCX headers copied from {src} to {dst}")


def handle_flagcx(*args, **kwargs):
    global registrar
    registrar = IluvatarFlagCXRegistrar(kwargs)
    registrar.run()


_patch_setup(_build_setup_hook())
