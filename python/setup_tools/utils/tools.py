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
import shutil
from pathlib import Path
import tarfile
import zipfile
from io import BytesIO
import urllib.request
from dataclasses import dataclass
import json
import subprocess
import time
import functools
import hashlib

from python.build_helpers import get_base_dir
import platform
from typing import Mapping
from types import MappingProxyType
import importlib.util
from dataclasses import field
from . import flagtree_submodules


def _get_flagtree_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def get_hook_instance(hook_name):
    if not configs.activated_module or not hook_name:
        return None
    hook_instance = getattr(configs.activated_module, hook_name, None)
    return hook_instance if callable(hook_instance) else None


def enable_flagtree_third_party(name):
    return os.environ.get(f"USE_{name.upper()}", 'ON') == 'ON'


def download_flagtree_third_party(name, condition, required=False, hook=None):
    if condition:
        if enable_flagtree_third_party(name):
            submodule = flagtree_submodules[name]
            downloader.download(module=submodule, required=required)
            hook_call = get_hook_instance(hook)
            if hook_call:
                hook_call(configs=configs, backend=submodule, cache=cache)

        else:
            print(f"\033[1;33m[Note] Skip downloading {name} since USE_{name.upper()} is set to OFF\033[0m")


@dataclass
class FlagtreeConfigs:
    default_backends: tuple = ("nvidia", "amd")
    plugin_backends: tuple = ("cambricon", "ascend", "aipu", "tsingmicro", "enflame", "hcu", "thrive")
    use_cuda_toolkit_backends: tuple = ('aipu', 'tileir')
    language_extra_backends: tuple = ('xpu', 'mthreads', "cambricon")
    ext_sourcedir: str = "triton/_C/"
    flagtree_root_dir: str = field(default_factory=_get_flagtree_root)
    flagtree_backend: str = field(default_factory=lambda: os.environ.get("FLAGTREE_BACKEND", ""))
    flagtree_plugin: str = field(default_factory=lambda: os.environ.get("FLAGTREE_PLUGIN"))
    extend_backends: list = field(default_factory=list)
    activated_module: any = None
    flagtree_submodule_dir: str = ''
    device_alias_map: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({
        "xpu": "xpu",
        "mthreads": "musa",
        "ascend": "ascend",
        "cambricon": "mlu",
        "thrive": "thrive",
        "metax": "metax",
        "sunrise": "sunrise",
    }))

    def set_extend_backends(self, backend):
        if backend not in self.default_backends:
            self.extend_backends.append(backend)

    def __post_init__(self):
        backends = list(self.default_backends)
        _backends = [backend for backend in backends if os.environ.get(f"USE_{backend.upper()}", "ON").upper() != "OFF"]
        self.default_backends = tuple(_backends)
        self.flagtree_submodule_dir = os.path.join(self.flagtree_root_dir, "third_party")
        self.activated_module = self._activate_device_module()

    def _activate_device_module(self, suffix=".py"):
        backend = self.flagtree_backend or "default"
        module_path = Path(os.path.dirname(__file__)) / backend
        module_path = str(module_path) + suffix
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except (AttributeError, FileNotFoundError, ImportError, ModuleNotFoundError):
            pass
        return module


flagtree_configs = FlagtreeConfigs()
configs = flagtree_configs
flagtree_backend = flagtree_configs.flagtree_backend


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


@dataclass
class NetConfig:
    max_retry: int = 4
    timeout: int = 300
    user_agent: str = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
    headers: dict = None


@dataclass
class Module:
    name: str
    url: str
    commit_id: str = None
    dst_path: str = None


def dir_rollback(deep, base_path):
    while (deep):
        base_path = os.path.dirname(base_path)
        deep -= 1
    return Path(base_path)


def is_skip_cuda_toolkits():
    return flagtree_configs.flagtree_backend and (flagtree_configs.flagtree_backend
                                                  not in flagtree_configs.use_cuda_toolkit_backends)


def remove_triton_in_modules(model):
    model_path = model.dst_path
    triton_path = os.path.join(model_path, "triton")
    if os.path.exists(triton_path):
        shutil.rmtree(triton_path)


def decompress(url, content, dst_path, file_name=None):
    file_bytes = BytesIO(content)
    file_names = []
    if url.endswith(".zip"):
        with zipfile.ZipFile(file_bytes, "r") as file:
            file.extractall(path=dst_path)
            file_names = file.namelist()
    else:
        with tarfile.open(fileobj=file_bytes, mode="r|*") as file:
            file.extractall(path=dst_path)
            file_names = file.getnames()
    os.rename(Path(dst_path) / file_names[0], Path(dst_path) / file_name)


def get_triton_cache_path():
    user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


class DownloadManager:

    def __init__(self):
        self.src_list = {}
        self.current_url = None
        self.current_dst_path = None
        self.current_file_name = None
        self.module_offline_handler = OfflineBuildManager()
        NetConfig.headers = {'User-Agent': NetConfig.user_agent}

    def download(self, url=None, path=None, file_name=None, mode=None, module=None, required=False):
        if self.module_offline_handler.is_offline_build():
            self.offline_copy(module, required)
            return

        if url:
            self.init_single_src_settings(url, path, file_name, mode)
        if mode == "git" or module:
            return self.git_clone(module, required)
        else:
            return self.general_download(is_decompress=True)

    def offline_copy(self, module, required):
        src_path = os.path.join(self.module_offline_handler.offline_build_dir, module.name)
        succ = os.path.exists(src_path)
        try:
            if succ:
                print(f"[INFO] Offline Build: Found {module.name} at {src_path}")
                self.module_offline_handler.src = os.path.join(self.module_offline_handler.offline_build_dir,
                                                               module.name)
                self.module_offline_handler.copy_to_flagtree_project({"dst_path": module.dst_path})
            else:
                print(f"[INFO] Offline Build: {module.name} is not found in offline build directory.")
        except Exception:
            if (required):
                raise RuntimeError(f"[ERROR] Failed to copy {module.name} from offline build directory.")
            print(f"[WARNING] Failed to copy {module.name} from offline build directory.")
            pass

    def init_single_src_settings(self, url, path, file_name, mode):
        self.current_url = url
        self.current_dst_path = path
        self.current_file_name = file_name
        self.src_list[self.current_url] = {"mode": mode, "path": path, "status": None, "content": None}

    def set_status(self, status, content):
        self.src_list[self.current_url]['status'] = status
        self.src_list[self.current_url]['content'] = content

    def backoff(self, module, retry_count):
        delay = 2**(NetConfig.max_retry - retry_count)
        print(f"\n[{NetConfig.max_retry - retry_count}] retry to clone "
              f"{module.name} after {delay}s...")
        time.sleep(delay)

    def git_clone(self, module, required=False):
        if module is None:
            return
        if not os.path.exists(module.dst_path):
            succ = self.clone_module(module)
        else:
            print(f'Found third_party {module.name} at {module.dst_path}\n')
            return True
        if not succ and required:
            raise RuntimeError(
                f"[ERROR]: Failed to download {module.name} from {module.url}, It's most likely the network!")
        remove_triton_in_modules(module)

    def py_clone(self, module):
        try:
            import git
        except ImportError:
            return False
        retry_count = NetConfig.max_retry
        while (retry_count):
            try:
                repo = git.Repo.clone_from(module.url, module.dst_path)
                if module.commit_id:
                    repo.git.checkout(module.commit_id)
                return True
            except git.GitCommandError:
                retry_count -= 1
                self.backoff(module, retry_count)
        return False

    def sys_clone(self, module):
        retry_count = NetConfig.max_retry
        has_specialization_commit = module.commit_id is not None
        while (retry_count):
            try:
                subprocess.run(["git", "clone", module.url, module.dst_path], check=True)
                if has_specialization_commit:
                    subprocess.run(["git", "checkout", module.commit_id], cwd=module.dst_path, check=True)
                return True
            except subprocess.CalledProcessError:
                retry_count -= 1
                self.backoff(module, retry_count)
        return False

    def clone_module(self, module):
        succ = True if self.py_clone(module) else self.sys_clone(module)
        if not succ:
            print(f"[ERROR]: Failed to clone {module.name} from {module.url}")
            return False
        print(f"[INFO]: Successfully cloned {module.name} to {module.dst_path}")
        return True

    def general_download_impl(self, request):
        with urllib.request.urlopen(request, timeout=NetConfig.timeout) as response:
            content = response.read()
            return content

    def general_download(self, is_decompress=True):
        request = urllib.request.Request(self.current_url, None, NetConfig.headers)
        current_retry_count = NetConfig.max_retry
        content = None
        print(f'downloading {self.current_url} ...')
        while (current_retry_count):
            try:
                content = self.general_download_impl(request)
                break
            except Exception:
                current_retry_count -= 1
                residue = NetConfig.max_retry - current_retry_count
                if current_retry_count == 0:
                    break
                # Transient network errors (connection reset, BOS hiccups) are
                # common for the multi-GB toolchain artifacts; retrying them
                # back-to-back just burns all attempts within seconds.
                delay = min(120, 10 * 2**(residue - 1))
                print(f"\n [Note]: [{residue}] retry to download {self.current_url} in {delay}s...")
                time.sleep(delay)
        if current_retry_count == 0:
            self.set_status(status='fail', content=None)
            raise RuntimeError("The download failed, probably due to network problems!")

        self.set_status(status='succ', content=content)

        if is_decompress:
            decompress(self.current_url, content=content, dst_path=self.current_dst_path,
                       file_name=self.current_file_name)


downloader = DownloadManager()


class OfflineBuildManager:

    def __init__(self):
        self.is_offline = self.is_offline_build()
        self.offline_build_dir = os.environ.get("FLAGTREE_OFFLINE_BUILD_DIR") if self.is_offline else None
        self.triton_cache_path = get_triton_cache_path()

    def is_offline_build(self) -> bool:
        return os.getenv("TRITON_OFFLINE_BUILD", "OFF") == "ON" or os.getenv("FLAGTREE_OFFLINE_BUILD_DIR")

    def copy_to_flagtree_project(self, kargs):
        dst_path = os.path.join(_get_flagtree_root(),
                                kargs['dst_path']) if 'dst_path' in kargs and kargs['dst_path'] else None
        src_path = self.src
        if not dst_path:
            return False
        src_path = self.src
        print(f"[INFO] Copying from {src_path} to {dst_path}")
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy(src_path, dst_path)

    def handle_flagtree_hook(self, kargs):
        if 'post_hook' in kargs and kargs['post_hook']:
            kargs['post_hook'](self.src)

    def handle_triton_origin_toolkits(self):

        # detect system/arch/version, the same with setup.py
        system = platform.system()
        arch = platform.machine()
        arch = {"arm64": "sbsa", "aarch64": "sbsa"}.get(arch, arch)
        supported = {"Linux": "linux", "Darwin": "linux"}
        system = supported[system]
        nvidia_version_path = os.path.join(get_base_dir(), "cmake", "nvidia-toolchain-version.json")
        with open(nvidia_version_path, "r") as nvidia_version_file:
            # parse this json file to get the version of the nvidia toolchain
            NVIDIA_TOOLCHAIN_VERSION = json.load(nvidia_version_file)

        ptxas_cache_path = os.path.join("nvidia/nvcc",
                                        f"cuda_nvcc-{system}-{arch}-{NVIDIA_TOOLCHAIN_VERSION['ptxas']}-archive")
        ptxas_blackwell_cache_path = os.path.join(
            "nvidia/nvcc", f"cuda_nvcc-{system}-{arch}-{NVIDIA_TOOLCHAIN_VERSION['ptxas-blackwell']}-archive")
        cudacrt_cache_path = os.path.join("nvidia/nvcc",
                                          f"cuda_nvcc-{system}-{arch}-{NVIDIA_TOOLCHAIN_VERSION['cudacrt']}-archive")
        triton_origin_toolkits = [
            ptxas_cache_path, ptxas_blackwell_cache_path, cudacrt_cache_path, "nvidia/nvdisasm", "nvidia/cuobjdump",
            "nvidia/cudart", "nvidia/cupti", "json"
        ]
        for toolkit in triton_origin_toolkits:
            toolkit_cache_path = os.path.join(self.triton_cache_path, toolkit)
            if os.path.exists(toolkit_cache_path):
                continue
            src_path = os.path.join(self.offline_build_dir, toolkit)
            if os.path.exists(src_path):
                print(f"[INFO] Copying {toolkit} from {src_path} to {toolkit_cache_path}")
                shutil.copytree(src_path, toolkit_cache_path, dirs_exist_ok=True)
            else:
                raise RuntimeError(
                    f"\n\n \033[31m[ERROR]:\033[0m The {flagtree_configs.flagtree_backend} offline build dependency \033[93m{src_path}\033[0m does not exist.\n"
                )

    def validate_offline_build_dir(self, path, required=False):
        if (not path or not os.path.exists(path)) and required:
            raise RuntimeError(
                "\n\n\033[31m[ERROR]:\033[0m If you want to use the offline build method\n"
                "please set FLAGTREE_OFFLINE_BUILD_DIR as the path of the offline dependency package\n"
                "or please \033[31mdo not use\033[0m the environment variable \033[93mTRITON_OFFLINE_BUILD !\033[0m \n\n"
            )

    def validate_offline_build_deps(self, path, kargs, required=False):
        url = kargs.get('url', None)
        if (not path or not os.path.exists(path)) and required:
            raise RuntimeError(
                f"\n\n \033[31m[ERROR]:\033[0m The {flagtree_configs.flagtree_backend} offline build dependency \033[93m{path}\033[0m does not exist.\n"
                f" And you can download the dependency package from the  \n \033[93m{url}\033[0m \n"
                f" then extract it to the \033[93m{self.offline_build_dir}\033[0m directory you specified !\033[0m\n\n")

    def validate_offline_build(self, path, required=False, is_base_dir=False, kargs=None):
        if is_base_dir:
            self.validate_offline_build_dir(path, required)
        else:
            self.validate_offline_build_deps(path, kargs, required)

    def single_build(self, *args, **kargs):
        if not self.is_offline:
            return False
        required = kargs['required'] if 'required' in kargs else False
        self.validate_offline_build(self.offline_build_dir, required, is_base_dir=True)
        self.src = os.path.join(self.offline_build_dir, kargs['src']) if 'src' in kargs else None
        self.validate_offline_build(self.src, required, kargs=kargs)
        print(f"[INFO] Building in offline mode using directory: {self.src}")
        self.copy_to_flagtree_project(kargs)
        self.handle_flagtree_hook(kargs)
        if is_skip_cuda_toolkits():
            print(f"[INFO] Skipping CUDA toolkits for {flagtree_configs.flagtree_backend} backend in offline build.")
        else:
            self.handle_triton_origin_toolkits()
        return True
