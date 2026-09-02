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

import importlib
import os

_spec_module = None


def _get_spec_module():
    global _spec_module
    from ._flagtree_backend import FLAGTREE_BACKEND
    if _spec_module is not None:
        return _spec_module
    if not FLAGTREE_BACKEND:
        return None
    try:
        _spec_module = importlib.import_module(f"triton.spec.{FLAGTREE_BACKEND}.triton")
    except ImportError:
        return None
    return _spec_module


def spec_path(path_list: list):
    from ._flagtree_backend import FLAGTREE_BACKEND
    if not path_list or not FLAGTREE_BACKEND:
        return
    current_path = path_list[0].replace(os.sep, "/")
    marker = "/triton"
    index = current_path.find(marker)
    if index == -1:
        return
    triton_root = current_path[:index + len(marker)]
    relative_path = current_path[index + 1 + len(marker):]
    backend_path = os.path.join(triton_root, "spec", FLAGTREE_BACKEND, "triton", relative_path)
    if os.path.isdir(backend_path) and backend_path not in path_list:
        path_list.insert(0, backend_path)


def spec_call(function_name: str, *args, **kwargs):
    module = _get_spec_module()
    if module is not None and hasattr(module, function_name):
        return getattr(module, function_name)(*args, **kwargs)
    return None


def spec_func(function_name: str):
    module = _get_spec_module()
    if module is not None and hasattr(module, function_name):
        return getattr(module, function_name)
    return None


def bind_language_extension_symbols_to_tl(extension):
    import triton.language as tl
    for name in getattr(extension, "__all__", ()):
        if hasattr(extension, name) and not hasattr(tl, name):
            setattr(tl, name, getattr(extension, name))
