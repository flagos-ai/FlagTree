import os
import shutil
import sys
import functools
from pathlib import Path
import hashlib
from distutils.sysconfig import get_python_lib
from . import utils

extend_backends = []
default_backends = ["nvidia", "amd"]
plugin_backends = ["cambricon", "ascend", "aipu", "tsingmicro"]
ext_sourcedir = "triton/_C/"
flagtree_backend = os.getenv("FLAGTREE_BACKEND", "").lower()
flagtree_plugin = os.getenv("FLAGTREE_PLUGIN", "").lower()
offline_build = os.getenv("FLAGTREE_PLUGIN", "OFF")
device_mapping = {"xpu": "xpu", "mthreads": "musa", "ascend": "ascend", "sunrise":"sunrise"}
activated_module = utils.activate(flagtree_backend)
downloader = utils.tools.DownloadManager()

set_llvm_env = lambda path: set_env({
    'LLVM_INCLUDE_DIRS': Path(path) / "include",
    'LLVM_LIBRARY_DIR': Path(path) / "lib",
    'LLVM_SYSPATH': path,
})


def install_extension(*args, **kargs):
    try:
        activated_module.install_extension(*args, **kargs)
    except Exception:
        pass


def get_backend_cmake_args(*args, **kargs):
    try:
        return activated_module.get_backend_cmake_args(*args, **kargs)
    except Exception:
        return []


def get_device_name():
    return device_mapping[flagtree_backend]


def get_extra_packages():
    packages = []
    try:
        packages = activated_module.get_extra_install_packages()
    except Exception:
        packages = []
    return packages


def get_package_data_tools():
    package_data = ["compile.h", "compile.c"]
    try:
        package_data += activated_module.get_package_data_tools()
    except Exception:
        package_data
    return package_data


def dir_rollback(deep, base_path):
    while (deep):
        base_path = os.path.dirname(base_path)
        deep -= 1
    return Path(base_path)


def enable_flagtree_third_party(name):
    if name in ["triton_shared"]:
        return os.environ.get(f"USE_{name.upper()}", 'OFF') == 'OFF'
    else:
        return os.environ.get(f"USE_{name.upper()}", 'ON') == 'ON'


def download_flagtree_third_party(name, condition, required=False, hock=None):
    if condition:
        if enable_flagtree_third_party(name):
            submodule = utils.flagtree_submodules[name]
            downloader.download(module=submodule, required=required)
            if callable(hock):
                hock(third_party_base_dir=utils.flagtree_submodule_dir, backend=submodule,
                     default_backends=default_backends)

        else:
            print(f"\033[1;33m[Note] Skip downloading {name} since USE_{name.upper()} is set to OFF\033[0m")


def post_install():
    try:
        activated_module.post_install()
    except Exception:
        pass


class FlagTreeCache:

    def __init__(self):
        self.flagtree_dir = os.path.dirname(os.getcwd())
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
              md5_digest=None, pre_hock=None, post_hock=None):

        if not condition or (pre_hock and pre_hock()):
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

        if is_url and not self.check_file(file_name=file, url=url, md5_digest=md5_digest):
            downloader.download(url=url, path=path, file_name=file)

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
        post_hock(self.cache_files[file]) if post_hock else False

    def get(self, file_name) -> Path:
        return self.cache_files[file_name]


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
            return activated_module.skip_package_dir(package)
        except Exception:
            return False

    @staticmethod
    def get_package_dir(packages):
        package_dict = {}
        if flagtree_backend and flagtree_backend not in plugin_backends:
            connection = []
            backend_triton_path = f"../third_party/{flagtree_backend}/python/"
            for package in packages:
                if CommonUtils.skip_package_dir(package):
                    continue
                pair = (package, f"{backend_triton_path}{package}")
                connection.append(pair)
            package_dict.update(connection)
        try:
            package_dict.update(activated_module.get_package_dir())
        except Exception:
            pass
        return package_dict


def handle_flagtree_backend():
    global ext_sourcedir
    if flagtree_backend:
        print(f"\033[1;32m[INFO] FlagtreeBackend is {flagtree_backend}\033[0m")
        extend_backends.append(flagtree_backend)
        if "editable_wheel" in sys.argv and flagtree_backend not in plugin_backends:
            ext_sourcedir = os.path.abspath(f"../third_party/{flagtree_backend}/python/{ext_sourcedir}") + "/"


def set_env(env_dict: dict):
    for env_k, env_v in env_dict.items():
        os.environ[env_k] = str(env_v)


def check_env(env_val):
    return os.environ.get(env_val, '') != ''


offline_handler = utils.OfflineBuildManager()
if offline_handler.is_offline:
    print("[INFO] FlagTree Offline Build: Use offline build for triton origin toolkits")
    offline_handler.handle_triton_origin_toolkits()
    offline_build = True
else:
    print('[INFO] FlagTree Offline Build: No offline build for triton origin toolkits')
    offline_build = False

download_flagtree_third_party("triton_shared", hock=utils.default.precompile_hock, condition=(not flagtree_backend))

download_flagtree_third_party("flir", condition=(flagtree_backend == "aipu"), hock=utils.aipu.precompile_hock,
                              required=True)

handle_flagtree_backend()

cache = FlagTreeCache()

# iluvatar
cache.store(
    file="iluvatar-llvm18-x86_64",
    condition=("iluvatar" == flagtree_backend),
    url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.3.0.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

cache.store(
    file="iluvatarTritonPlugin.so", condition=("iluvatar" == flagtree_backend) and (flagtree_plugin == ''), url=
    "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64_v0.3.0.tar.gz",
    copy_dst_path=f"third_party/{flagtree_backend}", md5_digest="015b9af8")

# klx xpu
cache.store(
    file="XTDK-llvm18-ubuntu2004_x86_64",
    condition=("xpu" == flagtree_backend),
    url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

cache.store(file="xre-Linux-x86_64", condition=("xpu" == flagtree_backend),
            url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz",
            copy_dst_path='python/_deps/xre3')

cache.store(
    files=("clang", "xpu-xxd", "xpu3-crt.xpu", "xpu-kernel.t", "ld.lld", "llvm-readelf", "llvm-objdump",
           "llvm-objcopy"), condition=("xpu" == flagtree_backend),
    copy_src_path=f"{os.environ.get('LLVM_SYSPATH','')}/bin", copy_dst_path="third_party/xpu/backend/xpu3/bin")

cache.store(files=("libclang_rt.builtins-xpu3.a", "libclang_rt.builtins-xpu3s.a"),
            condition=("xpu" == flagtree_backend), copy_src_path=f"{os.environ.get('LLVM_SYSPATH','')}/lib/linux",
            copy_dst_path="third_party/xpu/backend/xpu3/lib/linux")

cache.store(files=("include", "so"), condition=("xpu" == flagtree_backend),
            copy_src_path=f"{cache.dir_path}/xpu/xre-Linux-x86_64", copy_dst_path="third_party/xpu/backend/xpu3")

# mthreads
cache.store(
    file="mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64",
    condition=("mthreads" == flagtree_backend),
    url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64_v0.1.0.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

cache.store(
    file="mthreadsTritonPlugin.so", condition=("mthreads" == flagtree_backend) and (flagtree_plugin == ''), url=
    "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.3.0.tar.gz",
    copy_dst_path=f"third_party/{flagtree_backend}", md5_digest="2a9ca0b8")

# ascend
cache.store(
    file="llvm-b5cc222d-ubuntu-arm64",
    condition=("ascend" == flagtree_backend),
    url="https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-b5cc222d-ubuntu-arm64.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

# aipu
cache.store(
    file="llvm-a66376b0-ubuntu-x64-clang16-lld16",
    condition=("aipu" == flagtree_backend),
    url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

# tsingmicro
cache.store(
    file="tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64",
    condition=("tsingmicro" == flagtree_backend),
    url=
    "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

cache.store(
    file="tx8_deps",
    condition=("tsingmicro" == flagtree_backend),
    url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_release_20250814_195126_v0.2.0.tar.gz",
    pre_hock=lambda: check_env('TX8_DEPS_ROOT'),
    post_hock=lambda path: set_env({
        'LLVM_SYSPATH': path,
    }),
)

# hcu
cache.store(
    file="hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64",
    condition=("hcu" == flagtree_backend),
    url=
    "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)
