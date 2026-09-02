# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from pathlib import Path

from . import default, tools
from .tools import OfflineBuildManager, flagtree_configs


class SubmoduleRegistrar:

    def __init__(self, submodules=()):
        self._registered = {}
        for submodule in submodules:
            self.append(**submodule)

    def append(self, name, url, commit_id=None, relative_path=None, update=False):
        del update
        root = flagtree_configs.flagtree_submodule_dir
        destination = os.path.join(root, relative_path or name)
        self._registered[name] = tools.Module(name, url, commit_id, destination)


submodule_registrar = SubmoduleRegistrar((
    {
        "name": "triton_shared",
        "url": "https://github.com/microsoft/triton-shared.git",
        "commit_id": "5842469a16b261e45a2c67fbfc308057622b03ee",
    },
    {
        "name": "flir",
        "url": "https://github.com/FlagTree/flir.git",
    },
    {
        "name": "flagcx",
        "url": "https://github.com/flagos-ai/FlagCX.git",
        "relative_path": "tle/third_party/flagcx",
    },
    {
        "name": "tileir",
        "url": "https://github.com/NVIDIA/cuda-tile",
        "relative_path": "tileir/third_party/cuda-tile",
        "commit_id": "2e5ccba66fb3afdba34b26cf358418283027c248",
    },
))
flagtree_submodules = submodule_registrar._registered


def get_submodules(name):
    return flagtree_submodules.get(name)


def activate(backend, suffix=".py"):
    module_path = Path(__file__).parent / f"{backend or 'default'}{suffix}"
    spec = tools.importlib.util.spec_from_file_location("flagtree_backend_setup", module_path)
    module = tools.importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = [
    "default",
    "activate",
    "flagtree_submodules",
    "OfflineBuildManager",
    "tools",
    "submodule_registrar",
]
