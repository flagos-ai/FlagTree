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
import platform
import shutil
import sys
import sysconfig
import functools
from pathlib import Path
import hashlib
from distutils.sysconfig import get_python_lib
from . import utils
import importlib.util
import importlib.metadata
from typing import List, Tuple
from setuptools import find_packages
from .utils.tools import flagtree_configs as configs

downloader = utils.tools.DownloadManager()
configs = configs
flagtree_backend = configs.flagtree_backend


def get_console_colors() -> Tuple[str, str]:
    if platform.system() == "Windows":
        return "", ""
    return "\033[1;33m", "\033[0m"


def get_flagtree_version(git_commit_hash_fn):
    pypi_key_md5 = "ed98ae2a2ba0429b189537c0d3dbef43"
    key = os.environ.get("FLAGTREE_PYPI_KEY", "")
    flagtree_ver = os.environ.get("FLAGTREE_WHEEL_VERSION", "")
    if flagtree_ver:
        if hashlib.md5(key.encode()).hexdigest() == pypi_key_md5:
            return flagtree_ver
        return flagtree_ver + git_commit_hash_fn().replace("+", ".")
    if flagtree_backend:
        return "0.6.0+" + flagtree_backend + git_commit_hash_fn().replace("+", ".")
    return "0.6.0" + git_commit_hash_fn()


def get_long_description():
    readme_path = Path(__file__).resolve().parents[2] / "README.md"
    return readme_path.read_text(encoding="utf-8")


def init_backends(backend_installer):
    if flagtree_backend:
        if flagtree_backend in ("aipu", "tsingmicro", "enflame", "rpu", "thrive", "sunrise", "tileir", "ppu"):
            backends = [
                *backend_installer.copy(configs.default_backends + tuple(configs.extend_backends)),
                *backend_installer.copy_externals(),
            ]
        else:
            backends = [
                *backend_installer.copy(configs.extend_backends),
                *backend_installer.copy_externals(),
            ]
    else:
        backends = [
            *backend_installer.copy(configs.default_backends),
            *backend_installer.copy_externals(),
        ]
    return backends


# flagtree: extend yield "triton.backends.{backend.name}"
def get_backend_packages(backend):
    package_prefix = f"triton.backends.{backend.name}"
    excluded_dirs = set()
    if backend.name == "nvidia" and flagtree_backend not in ("", "nvidia", "tileir"):
        excluded_dirs = {"bin", "include", "lib.cupti"}

    for root, dirs, _files in os.walk(backend.backend_dir):
        relative_dir = os.path.relpath(root, backend.backend_dir)
        dirs[:] = sorted(directory for directory in dirs
                         if directory != "__pycache__" and directory.isidentifier() and (
                             directory if relative_dir == "." else f"{relative_dir.replace(os.sep, '.')}.{directory}"
                         ) not in excluded_dirs)
        package = package_prefix
        if relative_dir != ".":
            package += "." + relative_dir.replace(os.sep, ".")
        yield package, root


def get_generated_backend_packages(backend):
    """Declare data directories created after setuptools scans packages."""
    if backend.name == "nvidia" and flagtree_backend not in ("", "nvidia", "tileir"):
        return

    generated_package_suffixes = {
        "nvidia": (
            "bin",
            "include",
            "include.Openacc",
            "include.Openmp",
            "include.cooperative_groups",
            "include.cooperative_groups.details",
            "include.crt",
            "lib.cupti",
        ),
        "xpu": (
            "xpu3",
            "xpu3.bin",
            "xpu3.include",
            "xpu3.include.crt",
            "xpu3.include.cuda_etbl",
            "xpu3.include.xpu",
            "xpu3.include.xpurt_priv",
            "xpu3.lib",
            "xpu3.lib.linux",
            "xpu3.so",
        ),
    }
    package_prefix = f"triton.backends.{backend.name}"
    for suffix in generated_package_suffixes.get(backend.name, ()):
        source_dir = Path(backend.backend_dir).joinpath(*suffix.split("."))
        if source_dir.is_dir():
            yield f"{package_prefix}.{suffix}"


def refresh_generated_backend_packages(build_py_command, backends):
    packages = list(build_py_command.distribution.packages or [])
    known_packages = set(packages)
    for backend in backends:
        if backend.is_external:
            continue
        for package in get_generated_backend_packages(backend):
            if package not in known_packages:
                packages.append(package)
                known_packages.add(package)
    build_py_command.distribution.packages = packages
    build_py_command.packages = packages


set_llvm_env = lambda path: set_env(
    {
        'LLVM_INCLUDE_DIRS': Path(path) / "include",
        'LLVM_LIBRARY_DIR': Path(path) / "lib",
        'LLVM_SYSPATH': path,
        'PYTHONPATH': os.pathsep.join([str(Path(path) / "python_packages" / "mlir_core"),
                                       os.getenv("PYTHONPATH", "")]),
    })


def install_extension(*args, **kargs):
    backend_spec_install_extension_fn = get_hook_instance("install_extension")
    if backend_spec_install_extension_fn:
        backend_spec_install_extension_fn(*args, **kargs)


def get_backend_cmake_args(*args, **kargs):
    if "editable_wheel" in sys.argv:
        editable = True
    else:
        editable = False
    handle_plugin_backend(editable)
    try:
        cmake_args = configs.activated_module.get_backend_cmake_args(*args, **kargs)
    except Exception:
        cmake_args = []
    if editable:
        cmake_args += ["-DEDITABLE_MODE=ON"]
    return cmake_args


def customize_gluon_cmake_args():
    if flagtree_backend != "iluvatar":
        return []
    enabled = os.getenv("TRITON_ILU_BUILD_GLUON", "").upper() in ["ON", "1", "YES", "TRUE", "Y"]
    return [f"-DTRITON_BUILD_GLUON={'ON' if enabled else 'OFF'}"]


def get_device_name():
    return configs.device_alias_map[flagtree_backend]


def get_extra_packages():
    packages = []
    try:
        packages = configs.activated_module.get_extra_install_packages()
    except Exception:
        packages = []
    return packages


def get_package_data_tools():
    package_data = ["compile.h", "compile.c"]
    try:
        package_data += configs.activated_module.get_package_data_tools()
    except Exception:
        package_data
    return package_data


def dir_rollback(deep, base_path):
    while (deep):
        base_path = os.path.dirname(base_path)
        deep -= 1
    return Path(base_path)


def get_hook_instance(hook_name):
    if not configs.activated_module or not hook_name:
        return None
    hook_instance = getattr(configs.activated_module, hook_name, None)
    return hook_instance if callable(hook_instance) else None


def enable_flagtree_third_party(name):
    if name in ["triton_shared", "flagcx"]:
        return os.environ.get(f"USE_{name.upper()}", 'OFF') == 'ON'
    else:
        return os.environ.get(f"USE_{name.upper()}", 'ON') == 'ON'


def download_flagtree_third_party(name, condition, required=False, hook=None):
    if condition:
        if enable_flagtree_third_party(name):
            submodule = utils.flagtree_submodules[name]
            downloader.download(module=submodule, required=required)
            hook_call = get_hook_instance(hook)
            if hook_call:
                hook_call(configs=configs, backend=submodule, cache=cache)

        else:
            print(f"\033[1;33m[Note] Skip downloading {name} since USE_{name.upper()} is set to OFF\033[0m")


def post_install():
    backend_spec_post_install_fn = get_hook_instance("post_install")
    if backend_spec_post_install_fn:
        backend_spec_post_install_fn()


def write_flagtree_backend_file(triton_pkg_dir=None):
    if triton_pkg_dir is None:
        triton_pkg_dir = Path(__file__).resolve().parents[1] / "triton"
    os.makedirs(triton_pkg_dir, exist_ok=True)
    dest_file = Path(triton_pkg_dir) / "FLAGTREE_BACKEND"
    dest_file.write_text(flagtree_backend)


def write_backend_file_to_build_lib(build_lib):
    # xpu-only: ensure triton/FLAGTREE_BACKEND lands in the wheel: build_py only
    # copies .py by default, so this extension-less marker (read by
    # triton._flagtree_backend to make XPUDriver.is_active() return True
    # without any env var) was missing from the install, causing
    # "0 active drivers". Write it into build_lib/triton so it is packaged.
    if flagtree_backend == "xpu":
        try:
            write_flagtree_backend_file(os.path.join(build_lib, "triton"))
        except Exception as exc:  # noqa: BLE001
            print(f"[flagtree] could not write build_lib FLAGTREE_BACKEND: {exc}")


class FlagTreeCache:

    def __init__(self):
        self.flagtree_dir = str(Path(__file__).resolve().parents[2])
        self.dir_name = ".flagtree"
        self.sub_dirs = {}
        self.cache_files = {}
        self.dir_path = self._get_cache_dir_path()
        self._create_cache_dir()
        if flagtree_backend:
            self._create_subdir(subdir_name=flagtree_backend)

    @functools.lru_cache(maxsize=None)
    def _get_cache_dir_path(self) -> Path:
        _cache_dir = os.environ.get("FLAGTREE_CACHE_DIR")
        if _cache_dir is None:
            _cache_dir = Path.home() / self.dir_name
        else:
            _cache_dir = Path(_cache_dir)
        return _cache_dir

    def _create_cache_dir(self) -> Path:
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path, exist_ok=True)

    def _create_subdir(self, subdir_name, path=None):
        if path is None:
            subdir_path = Path(self.dir_path) / subdir_name
        else:
            subdir_path = Path(path) / subdir_name

        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        self.sub_dirs[subdir_name] = subdir_path

    def _md5(self, file_path):
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as file:
            while chunk := file.read(4096):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def check_file(self, file_name=None, url=None, path=None, md5_digest=None):
        origin_file_path = None
        if url is not None:
            origin_file_name = url.split("/")[-1].split('.')[0]
            origin_file_path = self.cache_files.get(origin_file_name, "")
        if path is not None:
            _path = path
        else:
            _path = self.cache_files.get(file_name, "")
        empty = (not os.path.exists(_path)) or (origin_file_path and not os.path.exists(origin_file_path))
        if empty:
            return False
        if md5_digest is None:
            return True
        else:
            cur_md5 = self._md5(_path)
            return cur_md5[:8] == md5_digest

    def clear(self):
        shutil.rmtree(self.dir_path)

    def reverse_copy(self, src_path, cache_file_path, md5_digest):
        if src_path is None or not os.path.exists(src_path):
            return False
        if os.path.exists(cache_file_path):
            return False
        copy_needed = True
        if md5_digest is None or self._md5(src_path) == md5_digest:
            copy_needed = False
        if copy_needed:
            print(f"copying {src_path} to {cache_file_path}")
            if os.path.isdir(src_path):
                shutil.copytree(src_path, cache_file_path, dirs_exist_ok=True)
            else:
                shutil.copy(src_path, cache_file_path)
            return True
        return False

    def store(self, file=None, condition=None, url=None, copy_src_path=None, copy_dst_path=None, files=None,
              md5_digest=None, pre_hook=None, post_hook=None, version=None):

        if not condition or (pre_hook and pre_hook()):
            return
        is_url = False if url is None else True
        path = self.sub_dirs[flagtree_backend] if flagtree_backend else self.dir_path

        if files is not None:
            for single_files in files:
                self.cache_files[single_files] = Path(path) / single_files
        else:
            self.cache_files[file] = Path(path) / file
            if url is not None:
                origin_file_name = url.split("/")[-1].split('.')[0]
                self.cache_files[origin_file_name] = Path(path) / file
            if copy_dst_path is not None:
                dst_path_root = Path(self.flagtree_dir) / copy_dst_path
                dst_path = Path(dst_path_root) / file
                if self.reverse_copy(dst_path, self.cache_files[file], md5_digest):
                    return

        if is_url:
            cache_path = self.cache_files[file]
            need_download = not self.check_file(file_name=file, url=url, md5_digest=md5_digest)
            # Version check: re-download if cached version doesn't match expected
            if not need_download and version is not None:
                version_file = Path(cache_path) / "version.txt"
                if version_file.exists():
                    cached_ver = version_file.read_text().strip()
                    if cached_ver != version:
                        print(
                            f"[cache] version mismatch for '{file}': cached='{cached_ver}', expected='{version}', re-downloading..."
                        )
                        shutil.rmtree(cache_path)
                        need_download = True
                # If no version.txt (legacy cache), keep using it
            if need_download:
                downloader.download(url=url, path=path, file_name=file)
                if version is not None:
                    cache_path = self.cache_files[file]
                    version_file = Path(cache_path) / "version.txt"
                    if os.path.isdir(cache_path):
                        version_file.write_text(version)

        if copy_dst_path is not None:
            file_lists = [file] if files is None else list(files)
            for single_file in file_lists:
                dst_path_root = Path(self.flagtree_dir) / copy_dst_path
                os.makedirs(dst_path_root, exist_ok=True)
                dst_path = Path(dst_path_root) / single_file
                if not self.check_file(path=dst_path, md5_digest=md5_digest):
                    if copy_src_path:
                        src_path = Path(copy_src_path) / single_file
                    else:
                        src_path = self.cache_files[single_file]
                    print(f"copying {src_path} to {dst_path}")
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy(src_path, dst_path)
        post_hook(self.cache_files[file]) if post_hook else False

    def get(self, file_name) -> Path:
        return self.cache_files[file_name]


cache = FlagTreeCache()

# -----flagtree-tle-raw-----flagtree-mlir---


class LLVMDetector:
    ENV_VARS = [
        "LLVM_INCLUDE_DIRS",
        "LLVM_LIBRARY_DIR",
        "LLVM_SYSPATH",
    ]

    @classmethod
    def env_vars(cls) -> List[str]:
        return [k for k in cls.ENV_VARS if k in os.environ]

    @staticmethod
    def is_whl_installed(pkg_name: str) -> bool:
        try:
            importlib.metadata.version(pkg_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    @staticmethod
    def get_paths_from_whl(pkg_name: str) -> Tuple[str, str, str]:
        module_spec = importlib.util.find_spec(pkg_name)
        if module_spec is None:
            raise RuntimeError(f"LLVM wheel '{pkg_name}' found via metadata but import failed.")

        if module_spec.origin:
            pkg_root = os.path.dirname(module_spec.origin)
        elif module_spec.submodule_search_locations:
            pkg_root = module_spec.submodule_search_locations[0]
        else:
            raise RuntimeError(f"LLVM wheel '{pkg_name}' is found but has no filesystem location")

        # New wheel structure: LLVM artifacts are under mlir/llvm_artifact/
        llvm_artifact_dir = os.path.join(pkg_root, "llvm_artifact")
        if os.path.isdir(llvm_artifact_dir):
            llvm_root = llvm_artifact_dir
        else:
            # Fallback: legacy structure where artifacts are directly under the package root
            llvm_root = pkg_root

        include_dir = os.path.join(llvm_root, "include")
        lib_dir = os.path.join(llvm_root, "lib")
        return include_dir, lib_dir, llvm_root


def check_llvm_via_mlir(pkg_name: str = "mlir") -> bool:
    mlir_installed = LLVMDetector.is_whl_installed(pkg_name)
    llvm_envs = LLVMDetector.env_vars()

    # flagtree llvm rule1 : mlir whl installed & set llvm env → fail
    if mlir_installed and llvm_envs and not os.environ.get("USE_FLAGTREE_MLIR_BUILD"):
        raise RuntimeError("[FATAL] LLVM wheel is installed, but LLVM-related environment variables are set:\n"
                           f"  {llvm_envs}\n"
                           "Please unset these env vars to avoid conflicts.")

    # flagtree llvm rule2：mlir whl installed & no llvm env → use mlir whl
    if mlir_installed:
        include_dir, lib_dir, llvm_root = LLVMDetector.get_paths_from_whl(pkg_name)
        # env variables will not appear out of python process
        os.environ["USE_FLAGTREE_MLIR_BUILD"] = "1"
        os.environ["LLVM_SYSPATH"] = llvm_root
        os.environ["LLVM_INCLUDE_DIRS"] = include_dir
        os.environ["LLVM_LIBRARY_DIR"] = lib_dir
        return True

    # flagtree llvm rule3: no mlir whl → use llvm env (fallback to legacy logic)
    return False


# --------------------------


# flagtree backend specialization
class SpecPackageHelper:

    @staticmethod
    def get_spec_packages():
        spec_install_dir = os.path.join("python", "triton", "spec")
        yield "triton.spec", spec_install_dir

        spec_dirs = sorted(
            (entry for entry in os.scandir(spec_install_dir) if entry.is_dir() and entry.name != "__pycache__"),
            key=lambda entry: entry.name)
        for spec_dir in spec_dirs:
            name = spec_dir.name
            source_dir = spec_dir.path
            for root, dirs, _files in os.walk(source_dir):
                dirs[:] = sorted(directory for directory in dirs if directory != "__pycache__")
                relative_dir = os.path.relpath(root, source_dir)
                package = f"triton.spec.{name}"
                if relative_dir != ".":
                    package += "." + relative_dir.replace(os.sep, ".")
                yield package, root

    @staticmethod
    def get_excluded_packages():
        return ["triton.spec", "triton.spec.*"]


def get_spec_packages():
    yield from find_packages(
        where="python",
        include=["triton", "triton.*"],
        exclude=SpecPackageHelper.get_excluded_packages(),
    )

    for package, _source_dir in SpecPackageHelper.get_spec_packages():
        yield package

    # These directories have no __init__.py; include them to avoid warnings.
    yield "triton._C"
    yield "triton._C.libtriton"
    yield "triton.tools.triton_to_gluon_translater"

    tle_include_dir = Path("python/triton/experimental/tle/language/include")
    if tle_include_dir.is_dir():
        # FlagCX headers are copied here while setup.py is running. The
        # directory intentionally has no __init__.py, so find_packages cannot
        # discover it even though setuptools includes it as package data.
        yield "triton.experimental.tle.language.include"

    if flagtree_backend == "xpu":
        yield "triton.language.extra.xpu"


def get_package_data(backends):
    hook_call = get_hook_instance("get_package_data")
    if not hook_call:
        return {}
    write_flagtree_backend_file()
    return hook_call(backends)


def get_excluded_package_data():
    cache_patterns = [
        "__pycache__/*",
        "**/__pycache__/*",
        "*.py[cod]",
        "**/*.py[cod]",
    ]
    excluded_package_data = {
        "": cache_patterns,
        "triton": ["spec/*"],
    }
    if flagtree_backend not in ("", "nvidia", "tileir"):
        excluded_package_data["triton.backends.nvidia"] = [
            "bin/*",
            "bin/**/*",
            "include/*",
            "include/**/*",
            "lib/cupti/*",
            "lib/cupti/**/*",
        ]
    return excluded_package_data


class CommonUtils:

    @staticmethod
    def unlink():
        cur_path = dir_rollback(2, __file__)
        if "editable_wheel" in sys.argv:
            installation_dir = cur_path
        else:
            installation_dir = get_python_lib()
        backends_dir_path = Path(installation_dir) / "triton" / "backends"
        # raise RuntimeError(backends_dir_path)
        if not os.path.exists(backends_dir_path):
            return
        for name in os.listdir(backends_dir_path):
            exist_backend_path = os.path.join(backends_dir_path, name)
            if not os.path.isdir(exist_backend_path):
                continue
            if name.startswith('__'):
                continue
            if os.path.islink(exist_backend_path):
                os.unlink(exist_backend_path)
            if os.path.exists(exist_backend_path):
                shutil.rmtree(exist_backend_path)

    @staticmethod
    def skip_package_dir(package):
        if 'backends' in package or 'profiler' in package:
            return True
        try:
            return configs.activated_module.skip_package_dir(package)
        except Exception:
            return False

    @staticmethod
    def get_package_dir(packages):
        package_dict = {}
        if flagtree_backend and flagtree_backend not in configs.plugin_backends:
            connection = []
            backend_triton_path = f"./third_party/{flagtree_backend}/python/"
            for package in packages:
                if CommonUtils.skip_package_dir(package):
                    continue
                pair = (package, f"{backend_triton_path}{package}")
                connection.append(pair)
            package_dict.update(connection)
        try:
            package_dict.update(configs.activated_module.get_package_dir())
        except Exception:
            pass
        return package_dict


def handle_flagtree_backend():
    global ext_sourcedir
    if flagtree_backend:
        print(f"\033[1;32m[INFO] FlagtreeBackend is {flagtree_backend}\033[0m")
        configs.extend_backends.append(flagtree_backend)
        if "editable_wheel" in sys.argv and flagtree_backend not in configs.plugin_backends:
            ext_sourcedir = os.path.abspath(f"./third_party/{flagtree_backend}/python/{configs.ext_sourcedir}") + "/"


def handle_plugin_backend(editable):
    plugin_mode = os.getenv("FLAGTREE_PLUGIN")
    if (plugin_mode and plugin_mode.upper() not in ["0", "OFF"]) or not flagtree_backend:
        return
    flagtree_backend_dir = cache.sub_dirs[flagtree_backend]
    flagtree_plugin_so = flagtree_backend + "TritonPlugin.so"
    src_build_plugin_path = flagtree_backend_dir / flagtree_plugin_so
    if not src_build_plugin_path.exists():
        return
    if editable is False:
        dst_build_plugin_dir = Path(sysconfig.get_path("purelib")) / "triton" / "_C"
        if not os.path.exists(dst_build_plugin_dir):
            os.makedirs(dst_build_plugin_dir)
        dst_build_plugin_path = dst_build_plugin_dir / flagtree_plugin_so
        shutil.copy(src_build_plugin_path, dst_build_plugin_path)
    dst_install_plugin_dir = Path(__file__).resolve().parent.parent / "triton" / "_C"
    if not os.path.exists(dst_install_plugin_dir):
        os.makedirs(dst_install_plugin_dir)
    shutil.copy(src_build_plugin_path, dst_install_plugin_dir)


def set_env(env_dict: dict):
    for env_k, env_v in env_dict.items():
        os.environ[env_k] = str(env_v)


def check_env(env_val):
    return os.environ.get(env_val, '') != ''


def register_backend_cache():
    hook_call = get_hook_instance("register_cache")
    if hook_call:
        hook_call(cache=cache, flagtree_backend=flagtree_backend, check_env=check_env, set_llvm_env=set_llvm_env)


def check_pybind11_abi():
    hook_call = get_hook_instance("check_pybind11_abi")
    if hook_call:
        hook_call(cache=cache)


def overlay_backend_runtime_so(build_py_command=None, backends=None):
    # Re-apply the fixed xpu runtime .so overlay after cmake: device/CMakeLists.txt
    # copies the stale liblaunch_shared.so/libxpujitc.so into backend/xpu3/so during
    # build_ext, so we overwrite them again before build_py packages the wheel.
    hook_call = get_hook_instance("overlay_runtime_so")
    if hook_call:
        hook_call(cache=cache, build_py_command=build_py_command, backends=backends)


def write_backend_site_pth(dest_dir):
    # xpu-only: drop a site .pth that preloads a GLIBCXX_3.4.30-capable libstdc++
    # before torch, so kernel launch needs no manual LD_LIBRARY_PATH/LD_PRELOAD.
    # Written into build_lib root so it lands at the site-packages root of the wheel.
    hook_call = get_hook_instance("write_site_pth")
    if hook_call:
        hook_call(dest_dir)


def uninstall_triton():
    is_bdist_wheel = any(cmd in sys.argv for cmd in ['bdist_wheel', 'egg_info', 'sdist'])
    if is_bdist_wheel:
        return
    try:
        import pkg_resources
        import subprocess
        try:
            pkg_resources.get_distribution('triton')
            print("Detected existing 'triton' package. Uninstalling to avoid conflicts...")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "triton"])
            print("Successfully uninstalled 'triton'.")
        except pkg_resources.DistributionNotFound:
            print("'triton' package not found, no need to uninstall.")
    except Exception as e:
        print(f"Warning: Failed to check/uninstall triton: {e}")


offline_handler = utils.OfflineBuildManager()
if offline_handler.is_offline:
    if utils.is_skip_cuda_toolkits():
        print(f"[INFO] Skipping CUDA toolkits for {flagtree_backend} backend in offline build.")
    else:
        print("[INFO] FlagTree Offline Build: Use offline build for triton origin toolkits")
        offline_handler.handle_triton_origin_toolkits()
    offline_build = True
else:
    print('[INFO] FlagTree Offline Build: No offline build for triton origin toolkits')
    offline_build = False

cache = FlagTreeCache()

download_flagtree_third_party("flir", condition=(flagtree_backend in configs.use_flir), required=True)
'''
   FlagCX is a third-party library adopted by the tle distributed system,
   refer to https://github.com/flagos-ai/FlagCX
'''

download_flagtree_third_party("flagcx", condition=(not flagtree_backend), hook="handle_flagcx", required=True)

download_flagtree_third_party("tileir", condition=(flagtree_backend == "tileir"), required=True)

handle_flagtree_backend()

register_backend_cache()
