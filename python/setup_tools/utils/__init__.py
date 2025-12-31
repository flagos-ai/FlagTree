from pathlib import Path
import importlib.util
import os
from . import tools, default, aipu
from .tools import flagtree_submodule_dir, OfflineBuildManager

flagtree_submodules = {
    "triton_shared":
    tools.Module(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                 commit_id="5842469a16b261e45a2c67fbfc308057622b03ee",
                 dst_path=os.path.join(flagtree_submodule_dir, "triton_shared")),
    "flir":
    tools.Module(name="flir", url="https://github.com/FlagTree/flir.git",
                 dst_path=os.path.join(flagtree_submodule_dir, "flir")),
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


__all__ = ["aipu", "default", "activate", "flagtree_submodules", "OfflineBuildManager", "tools"]
