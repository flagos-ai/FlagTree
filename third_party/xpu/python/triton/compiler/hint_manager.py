import sys
import importlib

from triton._flagtree_backend import FLAGTREE_BACKEND


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
        if backend == 'ascend':
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
        elif backend == 'nvidia':
            try:
                module = importlib.import_module("triton.backends.nvidia.nvidia_hint_handler")
                return module.NvidiaHintHandler()
            except ImportError:
                # print(f"[FlagTree] Warning: Failed to load Nvidia Hint Handler: {e}", file=sys.stderr)
                return BaseHintHandler()
        else:
            return BaseHintHandler()


# lazy load after first call hint trigger
_global_hint_manager = None


def hint_trigger(hook_name, *args, **kwargs):
    global _global_hint_manager

    if _global_hint_manager is None:
        # NVIDIA builds have no FlagTree backend marker.
        backend_name = FLAGTREE_BACKEND or "nvidia"
        _global_hint_manager = HintManager(backend_name)
    return _global_hint_manager.handler.trigger(hook_name, *args, **kwargs)
