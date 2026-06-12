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
    if not FLAGTREE_BACKEND:
        return None
    triton_root = _triton_root()
    if triton_root is None:
        return None
    spec_dir = os.path.join(triton_root, "backends", FLAGTREE_BACKEND, "spec")
    if not os.path.isdir(spec_dir):
        return None
    if _spec_module is not None:
        return _spec_module
    try:
        _spec_module = importlib.import_module(f"triton.backends.{FLAGTREE_BACKEND}.spec")
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
    backend_path = os.path.join(triton_root, "backends", FLAGTREE_BACKEND, "spec", "triton", rel_path)
    if os.path.isdir(backend_path) and backend_path not in path_list:
        path_list.insert(0, backend_path)


# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
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
