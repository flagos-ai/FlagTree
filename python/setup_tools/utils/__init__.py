from pathlib import Path
import importlib.util
import os
from . import tools, ascend, cambricon, xpu
from .tools import OfflineBuildManager, flagtree_submodule_dir

flagtree_submodules = {
    "triton_shared":
    tools.Module(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                 commit_id="380b87122c88af131530903a702d5318ec59bb33",
                 dst_path=os.path.join(flagtree_submodule_dir, "triton_shared")),
    "ascend":
    tools.Module(name="ascend", url="https://gitcode.com/FlagTree/triton-ascend.git",
                 dst_path=os.path.join(flagtree_submodule_dir,
                                       "triton_ascend"), spec_submodule="third_party/ascendnpu-ir"),
}


def activate(backend, suffix=".py"):
    if not backend:
        backend = "default"
    module_path = Path(os.path.dirname(__file__)) / backend
    module_path = str(module_path) + suffix
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = ["OfflineBuildManager", "tools", "ascend", "cambricon", "xpu"]
