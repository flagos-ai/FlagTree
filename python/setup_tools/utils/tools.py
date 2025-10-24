import os
import shutil
from pathlib import Path
import tarfile
import zipfile
from io import BytesIO
import urllib.request
from dataclasses import dataclass

flagtree_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
flagtree_submodule_dir = os.path.join(flagtree_root_dir, "third_party")
flagtree_backend = os.environ.get("FLAGTREE_BACKEND")
use_cuda_toolkit = ["aipu"]


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
    spec_submodule: str = None


def dir_rollback(deep, base_path):
    while (deep):
        base_path = os.path.dirname(base_path)
        deep -= 1
    return Path(base_path)


def is_skip_cuda_toolkits():
    return flagtree_backend and (flagtree_backend not in use_cuda_toolkit)


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
    """
        DownloadManager Class Overview:

        This class manages the downloading of files and modules, supporting both general HTTP downloads and Git repository cloning.

        Key functionalities:
        1. Initialize download source information, including URL, destination path, file name, and download mode.
        2. Support multiple download modes:
            - General file download (HTTP/HTTPS) with optional automatic decompression.
            - Git repository cloning, using either the Python git library or system commands.
        3. Implement a retry mechanism to improve reliability, with the maximum retry count configured via NetConfig.max_retry.
        4. Track the status ('succ'/'fail') and content of each download source.
"""

    def __init__(self):
        self.src_list = {}
        self.current_url = None
        self.current_dst_path = None
        self.current_file_name = None
        NetConfig.headers = {'User-Agent': NetConfig.user_agent}

    def download(self, url=None, path=None, file_name=None, mode=None, module=None, required=False):
        if url:
            self.init_single_src_settings(url, path, file_name, mode)
        if mode == "git" or module:
            return self.git_clone(module, required)
        else:
            return self.general_download(is_decompress=True)

    def normalize_url(self, origin_url, default_scheme="https"):
        try:
            from urllib.parse import urlparse
        except ImportError:
            return origin_url  # Fallback if urlparse is not available, return as is

        parsed_url = urlparse(origin_url)
        if parsed_url.scheme:
            return origin_url
        elif origin_url.startswith("/"):
            return "file://" + origin_url
        else:
            return f"{default_scheme}://{origin_url}"

    def init_single_src_settings(self, url, path, file_name, mode):
        self.current_url = self.normalize_url(url)
        self.current_dst_path = path
        self.current_file_name = file_name
        self.src_list[self.current_url] = {"mode": mode, "path": path, "status": None, "content": None}

    def set_status(self, status, content):
        self.src_list[self.current_url]['status'] = status
        self.src_list[self.current_url]['content'] = content

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
        has_specialization_commit = module.commit_id is not None
        has_specialization_submodule = module.spec_submodule is not None
        while (retry_count):
            try:
                repo = git.Repo.clone_from(module.url, module.dst_path)
                if has_specialization_commit:
                    repo.git.checkout(module.commit_id)
                if has_specialization_submodule:
                    submodule = repo.submodules[has_specialization_submodule]
                    submodule.update(init=True, recursive=False)
                return True
            except Exception:
                retry_count -= 1
                print(f"\n[{NetConfig.max_retry - retry_count}] retry to clone {module.name} to  {module.dst_path}")
        return False

    def sys_clone(self, module):
        retry_count = NetConfig.max_retry
        has_specialization_commit = module.commit_id is not None
        has_specialization_submodule = module.spec_submodule is not None
        while (retry_count):
            try:
                os.system(f"git clone {module.url} {module.dst_path}")
                if has_specialization_commit:
                    os.system(f"cd {module.dst_path}")
                    os.system(f"git checkout {module.commit_id}")
                    os.system("cd -")
                if has_specialization_submodule:
                    os.system(f"cd {module.dst_path}")
                    os.system(f"git submodule update --init --recursive --force {module.spec_submodule}")
                    os.system("cd -")
                return True
            except Exception:
                retry_count -= 1
                print(f"\n[{NetConfig.max_retry - retry_count}] retry to clone {module.name} to  {module.dst_path}")
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
                print(f"\n [Note]: [{residue}] retry to downloading and extracting {self.current_url}")
        if current_retry_count == 0:
            self.set_status(status='fail', content=None)
            raise RuntimeError("The download failed, probably due to network problems!")

        self.set_status(status='succ', content=content)

        if is_decompress:
            decompress(self.current_url, content=content, dst_path=self.current_dst_path,
                       file_name=self.current_file_name)


class OfflineBuildManager:
    """
        OfflineBuildManager Class Overview:

        This class manages building and dependency handling in offline mode, allowing projects to be built
        without requiring active network access. It primarily deals with copying prebuilt dependencies
        from a specified offline build directory into the project and validating their existence.

        Key functionalities:
        1. Detect offline build mode based on environment variables (`TRITON_OFFLINE_BUILD` or `FLAGTREE_OFFLINE_BUILD_DIR`).
        2. Store relevant paths, such as the offline build directory and Triton cache path.
        3. Handle optional post-processing hooks (`post_hock`) after copying files.
        4. Manage Triton origin toolkits (e.g., `ptxas`, `nvdisasm`, `cuobjdump`) by copying them from offline sources to the local cache.
        5. Provide a single-build interface that integrates validation, copying, post-hooks, and toolkit handling for offline builds.
"""

    def __init__(self):
        self.is_offline = self.is_offline_build()
        self.offline_build_dir = os.environ.get("FLAGTREE_OFFLINE_BUILD_DIR") if self.is_offline else None
        self.triton_cache_path = get_triton_cache_path()

    def is_offline_build(self) -> bool:
        return os.getenv("TRITON_OFFLINE_BUILD", "OFF") == "ON" or os.getenv("FLAGTREE_OFFLINE_BUILD_DIR")

    def copy_to_flagtree_project(self, kargs):
        dst_path = os.path.join(flagtree_root_dir,
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

    def handle_flagtree_hock(self, kargs):
        if 'post_hock' in kargs and kargs['post_hock']:
            kargs['post_hock'](self.src)

    def handle_triton_origin_toolkits(self):
        triton_origin_toolkits = ["ptxas", "nvdisasm", "cuobjdump", "cudacrt", " cudart", "pybind11", "json"]
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
                    f"\n\n \033[31m[ERROR]:\033[0m The {flagtree_backend} offline build dependency \033[93m{src_path}\033[0m does not exist.\n"
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
                f"\n\n \033[31m[ERROR]:\033[0m The {flagtree_backend} offline build dependency \033[93m{path}\033[0m does not exist.\n"
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
        self.handle_flagtree_hock(kargs)
        if is_skip_cuda_toolkits():
            print(f"[INFO] Skipping CUDA toolkits for {flagtree_backend} backend in offline build.")
        else:
            self.handle_triton_origin_toolkits()
        return True
