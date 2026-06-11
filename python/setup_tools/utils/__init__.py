from pathlib import Path
import importlib.util
import os
from . import tools, default, aipu
from .tools import flagtree_configs, OfflineBuildManager


class SubmoduleRegistrar:

    def __init__(self, name=None, url=None, commit_id=None, relative_path=None, update=False,
                 submodules: dict | tuple | list = None):
        self._registered = dict()
        if submodules is not None:
            for submodule in submodules:
                self._register_submodule(submodule)
        else:
            self._register_submodule(
                {"name": name, "url": url, "commit_id": commit_id, "relative_path": relative_path, "update": update})

    def append(self, name, url, commit_id=None, relative_path=None, update=False):
        self._register_submodule(
            {"name": name, "url": url, "commit_id": commit_id, "relative_path": relative_path, "update": update})

    def _register_submodule(self, submodule):
        sub_dir = flagtree_configs.flagtree_submodule_dir
        name = submodule["name"]
        url = submodule["url"]
        commit_id = submodule.get("commit_id", None)
        relative_path = submodule.get("relative_path", None)
        dst_path = os.path.join(sub_dir, relative_path) if relative_path else os.path.join(sub_dir, name)
        module = tools.Module(name=name, url=url, commit_id=commit_id, dst_path=dst_path)
        self._registered[name] = module


global submodule_registrar
submodule_registrar = SubmoduleRegistrar(submodules=(
    {
        "name": "triton_shared", "url": "https://github.com/microsoft/triton-shared.git", "commit_id":
        "5842469a16b261e45a2c67fbfc308057622b03ee"
    },
    {"name": "flir", "url": "https://github.com/FlagTree/flir.git"},
    {"name": "flagcx", "url": "https://github.com/flagos-ai/FlagCX.git", "relative_path": "tle/third_party/flagcx"},
))


def get_submodules(name):
    return submodule_registrar._registered.get(name)


flagtree_submodules = submodule_registrar._registered


def activate(backend, suffix=".py"):
    backend = backend or "default"
    module_path = Path(os.path.dirname(__file__)) / backend
    module_path = str(module_path) + suffix
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = ["aipu", "default", "activate", "flagtree_submodules", "OfflineBuildManager", "tools", "submodule_registrar"]
