#
# Copyright 2024 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys

from triton.backends.compiler import GPUTarget
from triton.runtime.driver import DriverBase
from .backend import GCUBackend, GCUDriver, ty_to_cpp


def _ensure_transfer_to_gcu():
    """Import torch_gcu.transfer_to_gcu once torch is fully initialised.

    Called from multiple entry points (is_active, __init__, get_active_torch_device).
    Skips silently when torch is still mid-initialisation to avoid circular imports,
    and retries on the next call.
    """
    if getattr(_ensure_transfer_to_gcu, "_done", False):
        return
    torch_mod = sys.modules.get("torch")
    if torch_mod is None or not hasattr(torch_mod, "__version__"):
        return
    _ensure_transfer_to_gcu._done = True
    try:
        from torch_gcu import transfer_to_gcu
    except Exception:
        pass


def _patch_cuda_is_available():
    """Replace ``torch.cuda.is_available`` with a one-shot wrapper that
    calls ``_ensure_transfer_to_gcu()`` before the first real invocation.

    Why this is needed
    ------------------
    ``transfer_to_gcu`` monkey-patches ``torch.cuda.*`` so that calls
    like ``torch.cuda.is_available()`` return True on GCU hardware.
    Normally ``torch_gcu`` performs this patching during
    ``import torch`` (via the device-backend auto-load mechanism), but
    a circular-import chain prevents it::

        import torch
          -> auto-load torch_gcu
            -> torch._inductor
              -> import triton
                -> triton.backends.enflame  (we are here)
                  -> import torch_gcu       (circular! torch not ready)

    So after ``import torch`` finishes, ``torch.cuda.*`` is still the
    original un-patched version and ``is_available()`` returns False on
    GCU machines.

    How it works
    ------------
    This function runs at **module level** of ``driver.py``, which is
    loaded during the ``import triton`` step in the chain above.  At
    that point ``torch.cuda`` already exists in ``sys.modules``, so we
    can replace ``is_available`` with a thin wrapper.

    The wrapper is **self-removing**: on the very first call it

    1. Restores the original ``is_available`` (so the wrapper never
       runs again - zero overhead for all subsequent calls).
    2. Calls ``_ensure_transfer_to_gcu()``, which imports
       ``torch_gcu.transfer_to_gcu``.  This re-patches
       ``torch.cuda.*`` with GCU-aware versions.
    3. Calls the now-current ``cuda_mod.is_available()`` — which is the
       GCU-aware version if step 2 succeeded, or the original otherwise.

    This guarantees that even very early call-sites such as
    ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
    (which evaluate at module-import time, before any triton driver is
    instantiated) will trigger GCU registration first.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    cuda_mod = getattr(torch_mod, "cuda", None)
    if cuda_mod is None:
        return
    if getattr(cuda_mod, "_orig_is_available", None) is not None:
        return

    orig_is_available = cuda_mod.is_available

    def _wrapped_is_available():
        cuda_mod.is_available = orig_is_available
        _ensure_transfer_to_gcu()
        return cuda_mod.is_available()

    cuda_mod._orig_is_available = orig_is_available
    cuda_mod.is_available = _wrapped_is_available


_patch_cuda_is_available()


class _GCUDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(_GCUDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._driver = GCUDriver()
        self.utils = self._driver.utils
        self.backend = "gcu"
        self.get_current_stream = self._driver.get_current_stream
        self.get_current_device = self._driver.get_current_device
        self.launcher_cls = self._driver.launcher_cls
        _ensure_transfer_to_gcu()

    def get_active_torch_device(self):
        import torch
        _ensure_transfer_to_gcu()
        return torch.device("gcu", self.get_current_device())

    def get_device_properties(self, device):
        return self._driver.get_device_properties(device)

    def get_stream(self, idx=None):
        return self._driver.get_stream(id)

    def get_arch(self):
        return self._driver.get_arch()

    def get_current_target(self):
        arch = self._driver.get_arch()
        warp_size = self._driver.get_warp_size()
        return GPUTarget(self.backend, arch.split(':')[0], warp_size)

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    @staticmethod
    def is_active():
        _ensure_transfer_to_gcu()
        return True

    def get_benchmarker(self):
        return self._driver.get_benchmarker()

    def get_device_interface(self):
        import torch
        return torch.gcu

    def get_empty_cache_for_benchmark(self):
        import torch
        # It's the same as the Nvidia backend.
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='gcu')

    def clear_cache(self, cache):
        cache.zero_()
