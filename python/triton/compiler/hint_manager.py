# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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
        elif backend == 'sunrise':
            try:
                module = importlib.import_module("triton.backends.sunrise.sunrise_hint_handler")
                return module.SunriseHintHandler()
            except ImportError as e:
                print(f"[FlagTree] Warning: Failed to load Sunrise Hint Handler: {e}", file=sys.stderr)
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
