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
        return None


class HintManager:

    def __init__(self, backend_name):
        self.backend_name = backend_name
        # load Handler with backend name
        self.handler = self._load_handler(backend_name)

    def _load_handler(self, backend):
        if backend == 'npu':
            try:
                module = importlib.import_module("triton.backends.ascend.ascend_hint_handler")
                return module.AscendHintHandler()
            except ImportError as e:
                print(f"[FlagTree] Warning: Failed to load Ascend Hint Handler: {e}", file=sys.stderr)
                return BaseHintHandler()
        elif backend == 'aipu':
            try:
                module = importlib.import_module("triton.backends.aipu.aipu_hint_handler")
                return module.AipuHintHandler()
            except ImportError as e:
                print(f"[FlagTree] Warning: Failed to load aipu Hint Handler: {e}", file=sys.stderr)
                return BaseHintHandler()
        elif backend == 'cuda':
            try:
                module = importlib.import_module("triton.backends.nvidia.nvidia_hint_handler")
                return module.NvidiaHintHandler()
            except ImportError as e:
                print(f"[FlagTree] Warning: Failed to load Nvidia Hint Handler: {e}", file=sys.stderr)
                return BaseHintHandler()
        else:
            return BaseHintHandler()


# supported backend with matched version
SUPPORTED_BACKENDS = ["aipu", "npu", "cuda"]

# TODO : npu will have conflicts if more backend involved
# mapping name
BACKEND_ALIASES = {
    "ascend": "npu",
    "huawei": "npu",
    "nvidia": "cuda",
}


def normalize_backend_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    return BACKEND_ALIASES.get(name, name)


def hint_get_flagtree_backend() -> str:
    detected_backend = ""

    import torch

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

    # TODO : some backend may not support priority 1, so keep priority 2 is necessary
    # Priority 2: Torch Global State
    if not detected_backend:
        check_priority = ["aipu", "npu", "cuda"]

        # 3. parse according to benefit
        for candidate in check_priority:
            module = getattr(torch, candidate, None)
            if module and hasattr(module, "is_available") and module.is_available():
                detected_backend = candidate
                break

    # (Normalization and Validation)
    canonical_backend = normalize_backend_name(detected_backend)

    if not canonical_backend or canonical_backend not in SUPPORTED_BACKENDS:
        return ""

    return canonical_backend


# lazy load after first call hint trigger
_global_hint_manager = None


def hint_trigger(hook_name, *args, **kwargs):
    global _global_hint_manager

    if _global_hint_manager is None:
        _global_hint_manager = HintManager(hint_get_flagtree_backend())
    return _global_hint_manager.handler.trigger(hook_name, *args, **kwargs)
