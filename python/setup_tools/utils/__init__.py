from pathlib import Path
import importlib.util
import os
from . import tools, default
from .tools import flagtree_configs, OfflineBuildManager

flagtree_submodules = {
    "triton_shared":
    tools.Module(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                 commit_id="5842469a16b261e45a2c67fbfc308057622b03ee",
                 dst_path=os.path.join(flagtree_configs.flagtree_submodule_dir, "triton_shared")),
    "flir":
    tools.Module(name="flir", url="https://github.com/flagos-ai/flir.git",
                 commit_id="eb9f151a8fd6f7ee0950a53fb3edca0c5a8b7c67",
                 dst_path=os.path.join(flagtree_configs.flagtree_submodule_dir, "flir")),
    "flagprism":
    tools.Module(name="FlagPrism", url="https://github.com/flagos-ai/FlagPrism.git", branch="main",
                 dst_path=os.path.join(flagtree_configs.flagtree_submodule_dir, "FlagPrism")),
}


def activate(backend, suffix=".py"):
    if not backend:
        backend = "default"
    module_path = Path(os.path.dirname(__file__)) / backend
    module_path = str(module_path) + suffix
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        pass
    return module


__all__ = ["default", "activate", "flagtree_submodules", "OfflineBuildManager", "tools"]
