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
from triton.backends.driver import DriverBase
from triton.backends.enflame.backend import GCUBackend, GCUDriver, ty_to_cpp


def _patch_cuda_device_interface():
    """Patch CudaInterface.synchronize to use the GCU-aware torch.cuda.synchronize.

    CudaInterface binds ``synchronize = staticmethod(torch.cuda.synchronize)``
    at class-definition time. If CudaInterface was imported before
    ``transfer_to_gcu`` patched ``torch.cuda.synchronize``, the class still
    holds a reference to the original (broken on GCU) function.  This function
    updates CudaInterface to use the now-patched version.
    """
    di_mod = sys.modules.get("torch._dynamo.device_interface")
    if di_mod is None:
        return
    cuda_iface = getattr(di_mod, "CudaInterface", None)
    if cuda_iface is None:
        return
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    cuda_mod = getattr(torch_mod, "cuda", None)
    if cuda_mod is None:
        return
    cuda_iface.synchronize = staticmethod(cuda_mod.synchronize)


def _ensure_transfer_to_gcu():
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
    _patch_cuda_device_interface()


def _monkey_cuda_patch():
    """Patch ``torch.cuda`` so GCU-transparent tooling doesn't break.

    Three patches are applied:

    1. ``is_available`` -> one-shot wrapper that calls
       ``_ensure_transfer_to_gcu()`` before the first real invocation.
       This handles ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
       evaluated at module-import time.

    2. ``init`` -> one-shot wrapper that triggers GCU transfer, then no-op.
       Tests such as ``test_tle_gpu_local_ptr.py`` call ``torch.cuda.init()``
       inside module-scoped autouse fixtures and ``pytest.skip`` on failure.

    3. ``_lazy_init`` -> one-shot wrapper that triggers GCU transfer, then no-op.
       ``torch.empty(device="cuda")`` and similar factories internally call
       ``_lazy_init()``, which crashes with "Torch not compiled with CUDA
       enabled" on GCU.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    cuda_mod = getattr(torch_mod, "cuda", None)
    if cuda_mod is None:
        return

    # --- is_available (one-shot wrapper) ---
    if getattr(cuda_mod, "_orig_is_available", None) is None:
        orig_is_available = cuda_mod.is_available

        def _wrapped_is_available():
            cuda_mod.is_available = orig_is_available
            _ensure_transfer_to_gcu()
            return cuda_mod.is_available()

        cuda_mod._orig_is_available = orig_is_available
        cuda_mod.is_available = _wrapped_is_available

    # --- init -> one-shot wrapper that triggers GCU transfer, then no-op ---
    # Also applies any deferred CUDA patches that require torch to be loaded.
    if getattr(cuda_mod, "_orig_init", None) is None:
        orig_init = cuda_mod.init

        def _wrapped_init():
            cuda_mod.init = orig_init
            _ensure_transfer_to_gcu()
            cuda_mod.init()

        cuda_mod._orig_init = orig_init
        cuda_mod.init = _wrapped_init


_monkey_cuda_patch()


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
