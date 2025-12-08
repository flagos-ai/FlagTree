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
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.backends.enflame.backend import GCUBackend, GCUDriver


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
