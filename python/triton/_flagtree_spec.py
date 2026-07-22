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


def _triton_root() -> str | None:
    current_path = os.path.abspath(__file__).replace(os.sep, "/")
    marker = "/triton"
    idx = current_path.find(marker)
    if idx == -1:
        return None
    return current_path[:idx + len(marker)]


def _get_spec_module():
    global _spec_module
    from ._flagtree_backend import FLAGTREE_BACKEND
    if _spec_module is not None:
        return _spec_module
    if not FLAGTREE_BACKEND:
        return None
    try:
        _spec_module = importlib.import_module(f"triton.spec.{FLAGTREE_BACKEND}")
    except ImportError:
        return None
    return _spec_module


# flagtree backend path specialization
def spec_path(path_list: list):
    from ._flagtree_backend import FLAGTREE_BACKEND
    if not path_list or not FLAGTREE_BACKEND:
        return
    current_path = path_list[0].replace(os.sep, "/")
    marker = "/triton"
    idx = current_path.find(marker)
    if idx == -1:
        return
    triton_root = current_path[:idx + len(marker)]
    rel_path = current_path[idx + 1 + len(marker):]
    backend_path = os.path.join(triton_root, "spec", FLAGTREE_BACKEND, rel_path)
    if os.path.isdir(backend_path) and backend_path not in path_list:
        path_list.insert(0, backend_path)


# flagtree backend call specialization
def spec_call(function_name: str, *args, **kwargs):
    mod = _get_spec_module()
    if mod is not None and hasattr(mod, function_name):
        return getattr(mod, function_name)(*args, **kwargs)
    return None


# flagtree backend func specialization
def spec_func(function_name: str):
    mod = _get_spec_module()
    if mod is not None and hasattr(mod, function_name):
        return getattr(mod, function_name)
    return None


# flagtree language extension
def bind_language_extension_symbols_to_tl(extension):
    import triton.language as tl

    names = getattr(extension, "__all__", None)
    if not names:
        return

    for name in names:
        if not hasattr(extension, name):
            continue
        if hasattr(tl, name):
            continue
        setattr(tl, name, getattr(extension, name))
