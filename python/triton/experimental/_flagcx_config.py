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
"""Shared FlagCX discovery and backend interfaces.

Configuration discovery only inspects files. It does not import PyTorch, load
the host library, or initialize a device. Device-library linking stays in each
compiler backend.
"""

from contextlib import contextmanager
from importlib.machinery import PathFinder
import os
from pathlib import Path
import sys


class FlagcxRuntimeConfig:
    bt_name = "libflagcx_device.bc"
    shared_name = "libflagcx.so"
    include_name = "include"
    flagcx_cache_dir = Path.home() / ".flagtree" / "flagcx"
    require_bitcode = True
    use_wheel = True
    use_cache = True

    def __init__(self, backend_dir, path_order=0):
        self.backend_dir = Path(backend_dir)
        # Keep the installed package path even when backend_dir is a symlink.
        self.triton_path = self.backend_dir.parent.parent
        self.bitcode_path = None
        self.shared_lib_path = None
        self.include_path = None
        self.flagcx_whl_path = None
        self.is_available = False
        keys = ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST", "USE_TLE_DISTRIBUTED")
        self.enabled = not any(os.environ.get(key, "ON").strip().lower() in ("off", "0", "false") for key in keys)
        if not self.enabled:
            return

        module = os.environ.get("FLAGCX_MODULE_PATH")
        self.module_path = Path(module).expanduser() if module else None
        if self.use_wheel and self.module_path is None:
            self.flagcx_whl_path = self._get_flagcx_wheel_path()

        bitcode = self._get_bitcode_paths()
        shared = self._get_shared_lib_paths()
        includes = self._get_include_paths()
        self.bitcode_path = bitcode[path_order] if bitcode else None
        self.shared_lib_path = shared[path_order] if shared else None
        self.include_path = includes[path_order] if includes else None
        complete = bool(self.shared_lib_path and self.include_path and (self.bitcode_path or not self.require_bitcode))
        if self.module_path is not None and not complete:
            required = f"{self.shared_name} and {self.include_name}/"
            if self.require_bitcode:
                required = f"{self.bt_name}, " + required
            raise RuntimeError(f"Incomplete FlagCX bundle: {self.module_path}. Expected {required}.")
        self.is_available = complete and (self.backend_dir / "flagcx_wrapper.py").is_file()

    def _get_flagcx_wheel_path(self):
        spec = PathFinder.find_spec("flagcx", sys.path)
        return Path(spec.origin).parent if spec is not None and spec.origin else None

    def get_needed_package(self):
        return (self.bt_name, self.shared_name, self.include_name)

    def _get_paths(self, key, name, installed, *, directory=False):
        override = os.environ.get(key)
        if override:
            path = Path(override).expanduser()
            exists = path.is_dir() if directory else path.is_file()
            if not exists:
                kind = "directory" if directory else "file"
                raise RuntimeError(f"{key} does not point to an existing {kind}: {path}")
            return [path.resolve()]

        if self.module_path is not None:
            paths = [self.module_path / name]
        else:
            paths = []
            if self.flagcx_whl_path is not None:
                paths.append(self.flagcx_whl_path / "lib" / name)
            paths.append(installed)
            if self.use_cache:
                paths.append(self.flagcx_cache_dir / name)
        return [path.resolve() for path in paths if (path.is_dir() if directory else path.is_file())]

    def _get_bitcode_paths(self):
        return self._get_paths("FLAGCX_BITCODE_PATH", self.bt_name, self.backend_dir / "lib" / self.bt_name)

    def _get_shared_lib_paths(self):
        return self._get_paths("FLAGCX_LIB_PATH", self.shared_name, self.triton_path / "_C" / self.shared_name)

    def _get_include_paths(self):
        return self._get_paths("FLAGCX_INCLUDE_PATH", self.include_name,
                               self.triton_path / "experimental" / "tle" / "language" / self.include_name,
                               directory=True)


class FlagCXBackendAdapter:
    """Runtime operations supplied by a backend, imported only when needed."""

    @property
    def device_type(self):
        raise NotImplementedError

    @property
    def distributed_backend_name(self):
        raise NotImplementedError

    @property
    def allocator_class(self):
        raise NotImplementedError

    @property
    def device_module(self):
        import torch
        return torch.get_device_module(self.device_type)

    @contextmanager
    def compile_allocator_guard(self):
        """Override for temporary backend build setup, restoring it on exit."""
        yield


class CudaFlagCXBackendAdapter(FlagCXBackendAdapter):
    """CUDA-compatible PyTorch APIs used by NVIDIA, HIP and CoreX."""

    device_type = "cuda"
    distributed_backend_name = "nccl"

    @property
    def allocator_class(self):
        from torch.cuda.memory import CUDAPluggableAllocator
        return CUDAPluggableAllocator


class Distributed:
    """Map discovered device bitcode to a backend's external-library key."""

    def __init__(self, config, library_key="libflagcx", *, require_runtime=True):
        self.config = config
        self.library_key = library_key
        self.require_runtime = require_runtime

    @property
    def extern_libs(self):
        config = self.config
        if config.enabled and config.bitcode_path and (config.is_available or not self.require_runtime):
            return {self.library_key: str(config.bitcode_path)}
        return {}

    def get_extern_libs(self):
        return self.extern_libs

    @property
    def is_available(self):
        return self.config.is_available
