import os
import sys

sys.path.append(os.path.dirname(__file__))
from tools import flagtree_root_dir, DownloadManager, Module  #noqa: E402

downloader = DownloadManager()
flagtree_submodule_dir = os.path.join(flagtree_root_dir, "third_party")


def get_extra_install_packages():
    return [
        "triton/language/extra/cann",
        "triton/language/extra/kernels",
        "triton/extension",
        "triton/extension/buffer",
        "triton/extension/buffer/language",
    ]


def get_package_dir():
    package_dict = {}
    ascend_ext_base = "../third_party/ascend/python/triton/extension"
    package_dict["triton/extension"] = ascend_ext_base
    package_dict["triton/extension/buffer"] = f"{ascend_ext_base}/buffer"
    package_dict["triton/extension/buffer/language"] = f"{ascend_ext_base}/buffer/language"
    return package_dict


def handle_editable_install_mode(is_editable=True):
    prefix_dir = os.path.join(flagtree_root_dir, "third_party")
    project_dir = flagtree_root_dir
    required_path_mapping = {f"{project_dir}/python/triton/extension": f"{prefix_dir}/ascend/python/triton/extension"}
    for dst, src in required_path_mapping.items():
        if not os.path.exists(src):
            continue
        if not is_editable and os.path.islink(dst):
            os.unlink(dst)

        if is_editable and not os.path.exists(dst):
            print(f"[INFO] For editable install: creating symlink from {src} to {dst}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)


submodules = (Module(name="AscendNPU-IR", url="https://gitcode.com/Ascend/AscendNPU-IR.git", commit_id="5a3921f8",
                     dst_path=os.path.join(flagtree_submodule_dir, "ascend/AscendNPU-IR")), )


def precompile_hook_flir(*args, **kargs):
    default_backends = kargs["default_backends"]
    default_backends_list = list(default_backends)
    if 'nvidia' in default_backends:
        default_backends_list.remove('nvidia')
    if 'amd' in default_backends:
        default_backends_list.remove('amd')
    default_backends_list.append('flir')
    default_backends = tuple(default_backends_list)
    kargs["default_backends"] = default_backends
    get_submodule()
    return default_backends


def get_submodule():
    [downloader.download(module=submodule, required=False) for submodule in submodules]


def is_compile_ascend_npu_ir():
    return os.getenv("ASCEND_NPU_IR_COMPILE", "1") == "1"
