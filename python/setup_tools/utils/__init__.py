import os
from . import tools, default, xpu
from .tools import flagtree_configs, OfflineBuildManager, is_skip_cuda_toolkits

flagtree_submodules = {
    "triton_shared":
    tools.Module(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                 commit_id="380b87122c88af131530903a702d5318ec59bb33",
                 dst_path=os.path.join(flagtree_configs.flagtree_submodule_dir, "triton_shared")),
}

__all__ = [
    "default", "tsingmicro", "xpu", "tools", "flagtree_submoduels", "activate", "OfflineBuildManager",
    "is_skip_cuda_toolkits"
]
