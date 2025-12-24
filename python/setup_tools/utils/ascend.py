import os
from setup_tools.utils.tools import flagtree_submodule_dir, DownloadManager, Module

downloader = DownloadManager()

submodules = (Module(name="ascendnpu-ir", url="https://gitcode.com/qq_42979146/AscendNPU-IR.git",
                     commit_id="ascend_with_flir", dst_path=os.path.join(flagtree_submodule_dir, "ascendnpu-ir")), )


def precompile_hook_flir(*args, **kargs):
    default_backends = kargs["default_backends"]
    if 'amd' in default_backends:
        default_backends.remove('amd')
    default_backends.append('flir')
    get_submodule()


def get_submodule():
    [downloader.download(module=submodule, required=False) for submodule in submodules]


def is_compile_ascend_npu_ir():
    return os.getenv("ASCEND_NPU_IR_COMPILE", "1") == "1"
