import os
import sys
import importlib


class BaseHintHandler:
    # dynamicly find method
    def trigger(self, hook_name, *args, **kwargs):
        if hasattr(self, hook_name):
            method = getattr(self, hook_name)
            if callable(method):
                try:
                    return method(*args, **kwargs)

                except TypeError as e:
                    import inspect

                    try:
                        sig = inspect.signature(method)
                        expected = str(sig)
                    except Exception:
                        expected = "(unknown)"

                    actual_args = f"{len(args)} positional"
                    actual_kwargs = f"keys={list(kwargs.keys())}" if kwargs else "no keywords"

                    print(f"\n[Hint Trigger Mismatch] {self.__class__.__name__}.{hook_name}")
                    print(f"  > Expect : {expected}")
                    print(f"  > Actual : {actual_args}, {actual_kwargs}")
                    print(f"  > Reason : {e}\n")

                    raise e
        print("no capable method in backend handler")
        return None


class HintManager:

    def __init__(self, backend_name):
        self.backend_name = backend_name
        # load Handler with backend name
        self.handler = self._load_handler(backend_name)

    def _load_handler(self, backend):
        if backend == 'npu':
            try:
                module = importlib.import_module("third_party.ascend.backend.ascend_hint_handler")
                return module.AscendHintHandler()
            except ImportError as e:
                print(f"[FlagTree] Warning: Failed to load Ascend Hint Handler: {e}", file=sys.stderr)
                return BaseHintHandler()
        elif backend == 'aipu':
            from .backends.aipu import AipuHintHandler
            return AipuHintHandler()
        else:
            return BaseHintHandler()


# supported backend with matched version
SUPPORTED_CONFIG = {
    "cuda": {"3.5"},
    "npu": {"3.2"},
    "aipu": {"3.3"},
}

# mapping name
BACKEND_ALIASES = {
    "ascend": "npu",
    "huawei": "npu",
    "nv": "cuda",
}


def normalize_backend_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    return BACKEND_ALIASES.get(name, name)


def hint_get_flagtree_backend() -> str:
    detected_backend = ""

    import torch
    import triton

    # Priority 1: Triton Driver
    try:
        from triton.runtime import driver
        if hasattr(driver, 'active') and hasattr(driver.active, 'get_active_torch_device'):
            device = driver.active.get_active_torch_device()
            if isinstance(device, torch.device):
                detected_backend = device.type
            # unimplemented support
            elif isinstance(device, str):
                detected_backend = device
    except ImportError:
        pass

    # Priority 2: Torch Global State
    if not detected_backend:
        candidates = list(SUPPORTED_CONFIG.keys())
        # cuda priority least
        candidates.sort(key=lambda x: 1 if x == "cuda" else 0)

        # 3. parse according to benefit
        for candidate in candidates:
            module_name = candidate
            module = getattr(torch, module_name, None)
            if module and hasattr(module, "is_available") and module.is_available():
                detected_backend = candidate
                break

    # Priority 3: Environment Variable (need to remove!!!)
    if not detected_backend:
        detected_backend = os.environ.get("FLAGTREE_BACKEND", "")

    # (Normalization and Validation)
    canonical_backend = normalize_backend_name(detected_backend)

    if not canonical_backend or canonical_backend not in SUPPORTED_CONFIG:
        return ""

    # verify name and version match
    try:
        current_triton_version = ".".join(triton.__version__.split(".")[:2])
        supported_versions = SUPPORTED_CONFIG[canonical_backend]
        if current_triton_version not in supported_versions:
            msg = (f"[Flagtree] Hint ignored: Detected backend '{canonical_backend}' but current Triton version "
                   f"'{current_triton_version}' matches no supported versions {supported_versions}.")
            print(msg, file=sys.stderr)
            return ""
    except Exception:
        pass

    return canonical_backend


# lazy load after first call hint trigger
_global_hint_manager = None


def hint_trigger(hook_name, *args, **kwargs):
    global _global_hint_manager

    if _global_hint_manager is None:
        _global_hint_manager = HintManager(hint_get_flagtree_backend())
    return _global_hint_manager.handler.trigger(hook_name, *args, **kwargs)
