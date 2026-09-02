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

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from python.build_helpers import get_base_dir


def _get_flagtree_root() -> str:
    return str(Path(__file__).resolve().parents[3])


@dataclass
class FlagtreeConfigs:
    default_backends: tuple = ("nvidia", "amd")
    plugin_backends: tuple = ("cambricon", "ascend", "aipu", "tsingmicro", "enflame", "hcu", "thrive")
    use_cuda_toolkit_backends: tuple = ("aipu", "tileir")
    language_extra_backends: tuple = ("xpu", "mthreads", "cambricon")
    ext_sourcedir: str = "triton/_C/"
    flagtree_root_dir: str = field(default_factory=_get_flagtree_root)
    flagtree_backend: str = field(default_factory=lambda: os.environ.get("FLAGTREE_BACKEND", ""))
    flagtree_plugin: str = field(default_factory=lambda: os.environ.get("FLAGTREE_PLUGIN"))
    extend_backends: list = field(default_factory=list)
    activated_module: object = None
    flagtree_submodule_dir: str = ""
    device_alias_map: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({
        "xpu": "xpu",
        "mthreads": "musa",
        "ascend": "ascend",
        "cambricon": "mlu",
        "thrive": "thrive",
        "metax": "metax",
        "sunrise": "sunrise",
    }))

    def __post_init__(self):
        self.default_backends = tuple(backend for backend in self.default_backends
                                      if os.environ.get(f"USE_{backend.upper()}", "ON").upper() != "OFF")
        self.flagtree_submodule_dir = os.path.join(self.flagtree_root_dir, "third_party")
        self.activated_module = self._activate_device_module()

    def _activate_device_module(self, suffix=".py"):
        module_path = Path(__file__).parent / f"{self.flagtree_backend or 'default'}{suffix}"
        spec = importlib.util.spec_from_file_location("flagtree_backend_setup", module_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except (AttributeError, FileNotFoundError, ImportError, ModuleNotFoundError):
            pass
        return module


flagtree_configs = FlagtreeConfigs()


@dataclass
class NetConfig:
    max_retry: int = 4
    timeout: int = 300
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
    headers: dict = None


@dataclass
class Module:
    name: str
    url: str
    commit_id: str = None
    dst_path: str = None


def is_skip_cuda_toolkits():
    return flagtree_configs.flagtree_backend and (flagtree_configs.flagtree_backend
                                                  not in flagtree_configs.use_cuda_toolkit_backends)


def remove_triton_in_modules(module):
    triton_path = os.path.join(module.dst_path, "triton")
    if os.path.exists(triton_path):
        shutil.rmtree(triton_path)


def decompress(url, content, dst_path, file_name=None):
    file_bytes = BytesIO(content)
    if url.endswith(".zip"):
        with zipfile.ZipFile(file_bytes, "r") as archive:
            archive.extractall(path=dst_path)
            names = archive.namelist()
    else:
        with tarfile.open(fileobj=file_bytes, mode="r|*") as archive:
            archive.extractall(path=dst_path)
            names = archive.getnames()
    os.rename(Path(dst_path) / names[0], Path(dst_path) / file_name)


def get_triton_cache_path():
    user_home = os.getenv("TRITON_HOME")
    if not user_home:
        user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH")
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
        NetConfig.headers = {"User-Agent": NetConfig.user_agent}

    def download(self, url=None, path=None, file_name=None, mode=None, module=None, required=False):
        if self.module_offline_handler.is_offline_build():
            self.offline_copy(module, required)
            return
        if url:
            self.current_url = url
            self.current_dst_path = path
            self.current_file_name = file_name
        if mode == "git" or module:
            return self.git_clone(module, required)
        return self.general_download()

    def offline_copy(self, module, required):
        if module is None:
            return
        src_path = os.path.join(self.module_offline_handler.offline_build_dir, module.name)
        try:
            if not os.path.exists(src_path):
                raise FileNotFoundError(src_path)
            self.module_offline_handler.src = src_path
            self.module_offline_handler.copy_to_flagtree_project({"dst_path": module.dst_path})
        except Exception:
            if required:
                raise RuntimeError(f"Failed to copy required offline dependency {module.name}") from None

    def _backoff(self, module, retry_count):
        delay = 2**(NetConfig.max_retry - retry_count)
        print(f"\nretrying clone of {module.name} after {delay}s...")
        time.sleep(delay)

    def git_clone(self, module, required=False):
        if module is None:
            return None
        if os.path.exists(module.dst_path):
            return True
        success = self._clone_module(module)
        if not success and required:
            raise RuntimeError(f"Failed to download required dependency {module.name} from {module.url}")
        if success:
            remove_triton_in_modules(module)
        return success

    def _clone_module(self, module):
        retry_count = NetConfig.max_retry
        while retry_count:
            try:
                subprocess.run(["git", "clone", module.url, module.dst_path], check=True)
                if module.commit_id:
                    subprocess.run(["git", "checkout", module.commit_id], cwd=module.dst_path, check=True)
                return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                retry_count -= 1
                if retry_count:
                    self._backoff(module, retry_count)
        return False

    def general_download(self):
        request = urllib.request.Request(self.current_url, None, NetConfig.headers)
        retry_count = NetConfig.max_retry
        while retry_count:
            try:
                with urllib.request.urlopen(request, timeout=NetConfig.timeout) as response:
                    content = response.read()
                decompress(self.current_url, content, self.current_dst_path, self.current_file_name)
                return
            except Exception:
                retry_count -= 1
        raise RuntimeError("The download failed, probably due to network problems!")


class OfflineBuildManager:

    def __init__(self):
        self.is_offline = self.is_offline_build()
        self.offline_build_dir = os.environ.get("FLAGTREE_OFFLINE_BUILD_DIR") if self.is_offline else None
        self.triton_cache_path = get_triton_cache_path()

    def is_offline_build(self):
        return os.getenv("TRITON_OFFLINE_BUILD", "OFF") == "ON" or bool(os.getenv("FLAGTREE_OFFLINE_BUILD_DIR"))

    def copy_to_flagtree_project(self, kwargs):
        dst_path = kwargs.get("dst_path")
        if dst_path and not os.path.isabs(dst_path):
            dst_path = os.path.join(_get_flagtree_root(), dst_path)
        if not dst_path:
            return False
        if os.path.isdir(self.src):
            shutil.copytree(self.src, dst_path, dirs_exist_ok=True)
        else:
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, dst_path)
        return True

    def handle_triton_origin_toolkits(self):
        system = {"Linux": "linux", "Darwin": "linux"}[platform.system()]
        arch = {"arm64": "sbsa", "aarch64": "sbsa"}.get(platform.machine(), platform.machine())
        version_path = os.path.join(get_base_dir(), "cmake", "nvidia-toolchain-version.json")
        with open(version_path) as version_file:
            versions = json.load(version_file)
        toolkits = [
            os.path.join("nvidia", "nvcc", f"cuda_nvcc-{system}-{arch}-{versions['ptxas']}-archive"),
            os.path.join("nvidia", "nvcc-blackwell",
                         f"cuda_nvcc-{system}-{arch}-{versions['ptxas-blackwell']}-archive"),
            "nvidia/cuobjdump",
            "nvidia/nvdisasm",
            "nvidia/cudart",
            "nvidia/cupti",
            "json",
        ]
        for toolkit in toolkits:
            destination = os.path.join(self.triton_cache_path, toolkit)
            if os.path.exists(destination):
                continue
            source = os.path.join(self.offline_build_dir, toolkit)
            if not os.path.exists(source):
                raise RuntimeError(f"Offline build dependency {source} does not exist")
            shutil.copytree(source, destination, dirs_exist_ok=True)

    def single_build(self, *args, **kwargs):
        if not self.is_offline:
            return False
        if not self.offline_build_dir or not os.path.exists(self.offline_build_dir):
            if kwargs.get("required", False):
                raise RuntimeError("FLAGTREE_OFFLINE_BUILD_DIR does not exist")
            return False
        self.src = os.path.join(self.offline_build_dir, kwargs["src"]) if kwargs.get("src") else None
        if self.src and not os.path.exists(self.src):
            if kwargs.get("required", False):
                raise RuntimeError(f"Offline build dependency {self.src} does not exist")
            return False
        self.copy_to_flagtree_project(kwargs)
        if kwargs.get("post_hook"):
            kwargs["post_hook"](self.src)
        if not is_skip_cuda_toolkits():
            self.handle_triton_origin_toolkits()
        return True
