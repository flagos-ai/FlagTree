import os
from .tools import flagtree_configs, DownloadManager, Module

downloader = DownloadManager()

submodules = (Module(name="ascendnpu-ir", url="https://gitcode.com/Ascend/AscendNPU-IR.git",
                     commit_id="04045a06ec7c9592b17de659307d5debe7be590a",
                     dst_path=os.path.join(flagtree_configs.flagtree_submodule_dir, "ascendnpu-ir")), )


def precompile_hook_flir(*args, **kargs):
    default_backends = kargs["default_backends"]
    default_backends_list = list(default_backends)
    if 'amd' in default_backends:
        default_backends_list.remove('amd')
    default_backends_list.append('flir')
    default_backends = tuple(default_backends_list)
    kargs["default_backends"] = default_backends
    get_submodule()


def get_submodule():
    [downloader.download(module=submodule, required=False) for submodule in submodules]


def is_compile_ascend_npu_ir():
    return os.getenv("ASCEND_NPU_IR_COMPILE", "1") == "1"
