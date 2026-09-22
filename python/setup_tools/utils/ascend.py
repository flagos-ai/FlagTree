import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from tools import flagtree_configs, DownloadManager, Module  #noqa: E402

downloader = DownloadManager()
flagtree_submodule_dir = flagtree_configs.flagtree_submodule_dir


def get_extra_install_packages():
    return [
        "triton/language/extra/cann",
        "triton/language/extra/kernels",
        "triton/extension",
        "triton/extension/buffer",
        "triton/extension/buffer/language",
        "triton/experimental/tle/language/dsa/ascend",
    ]


def get_package_dir():
    package_dict = {}
    ascend_ext_base = "../third_party/ascend/python/triton/extension"
    package_dict["triton/extension"] = ascend_ext_base
    package_dict["triton/extension/buffer"] = f"{ascend_ext_base}/buffer"
    package_dict["triton/extension/buffer/language"] = f"{ascend_ext_base}/buffer/language"

    # flagtree tle ascend
    flagtree_tle_ascend_base = "../python/triton/experimental/tle/language/dsa"
    package_dict["triton/experimental/tle/language/dsa/ascend"] = f"{flagtree_tle_ascend_base}/ascend"

    return package_dict


def handle_editable_install_mode(is_editable=True):
    prefix_dir = flagtree_configs.flagtree_submodule_dir
    project_dir = flagtree_configs.flagtree_root_dir
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


# AscendNPU-IR must stay in sync with the bishengir tools shipped inside the
# CANN package, so the pin is selected by the CANN version on the build machine.
ASCEND_NPU_IR_PINS = {
    "9.1.0": ("tle-3.5.x-cann9.1.0-dev", "3545d1cba9b1bdd3cb300724d4626282ad1679ee"),
}
ASCEND_NPU_IR_DEFAULT_PIN = ("tle-3.5.x-cann9.0.0-dev", "a205c957")


def get_cann_version():
    # Same convention as backend/utils.py:get_cann_version_file_hash: prefer the
    # sourced CANN environment, then the standard /usr/local/Ascend install.
    arch = platform.machine()
    ascend_roots = []
    if os.getenv("ASCEND_HOME_PATH"):
        ascend_roots.append(Path(os.environ["ASCEND_HOME_PATH"]))
    ascend_roots.append(Path("/usr/local/Ascend/ascend-toolkit/latest"))
    for root in ascend_roots:
        info_path = root / f"{arch}-linux" / "ascend_toolkit_install.info"
        try:
            for line in info_path.read_text().splitlines():
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return ""


def get_ascend_npu_ir_pin():
    cann_version = get_cann_version()
    pin = ASCEND_NPU_IR_PINS.get(cann_version, ASCEND_NPU_IR_DEFAULT_PIN)
    if cann_version:
        print(f"[INFO] CANN {cann_version} detected, pinning AscendNPU-IR to {pin[0]}@{pin[1][:8]}")
    else:
        print("[WARNING] CANN version not detected (source <ascend-toolkit>/set_env.sh or install under "
              f"/usr/local/Ascend), pinning AscendNPU-IR to {pin[0]}@{pin[1][:8]}")
    return pin


# submodules = (Module(name="AscendNPU-IR", url="https://github.com/Ascend/AscendNPU-IR.git", commit_id="4c304921",
#                      dst_path=os.path.join(flagtree_submodule_dir, "ascend/AscendNPU-IR")), )
_ascend_npu_ir_branch, _ascend_npu_ir_commit = get_ascend_npu_ir_pin()
submodules = (Module(name="AscendNPU-IR", url="https://github.com/flagos-ai/FlagTree-AscendNPU-IR.git",
                     branch=_ascend_npu_ir_branch, commit_id=_ascend_npu_ir_commit,
                     dst_path=os.path.join(flagtree_submodule_dir, "ascend/AscendNPU-IR")), )


def precompile_hook_flir(*args, **kargs):
    default_backends = kargs["default_backends"]
    kargs["default_backends"] = default_backends
    get_submodule()
    return default_backends


def get_submodule():
    [downloader.download(module=submodule, required=False) for submodule in submodules]
    warn_stale_ascend_npu_ir_checkout()


def warn_stale_ascend_npu_ir_checkout():
    # DownloadManager reuses dst_path as-is when it already exists, so a
    # checkout left on a different pin (e.g. after switching the CANN package)
    # would silently shadow the pin selected for this build.
    module = submodules[0]
    if not os.path.isdir(os.path.join(module.dst_path, ".git")):
        return
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=module.dst_path, capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip().startswith(module.commit_id):
        return
    print(f"[WARNING] {module.dst_path} is at {head.stdout.strip()[:8]} but this build pins "
          f"AscendNPU-IR to {module.commit_id[:8]}; remove the directory (or run "
          f"'git checkout {module.commit_id}' inside it) before rebuilding")


def is_compile_ascend_npu_ir():
    return os.getenv("ASCEND_NPU_IR_COMPILE", "1") == "1"
