"""
FlagCX distributed runtime configuration module.

This module provides the base classes for flagcx integration:
  - FlagcxRuntimeConfig: Resolves and validates flagcx runtime paths
  - FlagCXBackendAdapter: Backend-specific runtime integration
  - Distributed: Extern library mappings for compilation

Expected directory layout (~/.flagtree/flagcx/):
    ├── libflagcx_device.bc   (bitcode for device-side code)
    ├── libflagcx.so          (shared library)
    └── include/              (header files)

Typical usage (backend implementation):

    from triton.experimental._flagcx_config import FlagcxRuntimeConfig, FlagCXBackendAdapter

    class MyFlagcxRuntimeConfig(FlagcxRuntimeConfig):
        def _is_available_impl(self):
            from .flagcx_wrapper import FLAGCXLibrary  # noqa: F401
            return True

        def _get_bitcode_paths(self):
            return [Path(__file__).parent / 'lib' / 'libflagcx_device.bc']

    class MyFlagCXBackendAdapter(FlagCXBackendAdapter):
        @property
        def distributed_backend_name(self) -> str: return 'mycl'
        @property
        def device_type(self) -> str: return 'mydevice'
        @property
        def allocator_class(self): return MyAllocator

    flagcx_rt_conf = MyFlagcxRuntimeConfig()
    backend_adapter = MyFlagCXBackendAdapter()

"""

import os
from contextlib import contextmanager
from pathlib import Path


class FlagcxRuntimeConfig:
    """Base class for flagcx runtime configuration.

    Subclasses must implement:
        _is_available_impl() -> bool: Check if flagcx is available
        _get_bitcode_paths() -> list[Path]: Return available bitcode paths
    """

    shared_name = 'libflagcx.so'
    include_name = 'include'
    flagcx_cache_dir = Path.home() / '.flagtree' / 'flagcx'
    triton_path = Path(__file__).parent.parent
    bitcode_path: str | None = None

    # --- pure virtual: subclass provides ---
    def _is_available_impl(self):
        raise NotImplementedError

    # --- virtual with default impl ---
    def _is_available(self):
        for key in ('USE_FLAGCX', 'USE_DIST', 'USE_DISTRIBUTED', 'USE_TLE_DIST', 'USE_TLE_DISTRIBUTED'):
            if os.environ.get(key) in ('OFF', '0', 'false'):
                return False
        try:
            return self._is_available_impl()
        except Exception:
            return False

    def _check_path_available(self, paths, name):
        """Validate paths and raise helpful error if not found."""
        available = [Path(p) for p in paths if p and Path(p).exists()]
        if not available:
            raise RuntimeError(f"\nCannot find available '{name}' file or lib\n"
                               f"If you already have the required '{name}'\n"
                               "please set the corresponding environment variables\n"
                               "(e.g. FLAGCX_BITCODE_PATH, FLAGCX_LIB_PATH, FLAGCX_INCLUDE_PATH)\n"
                               f"or set FLAGCX_MODULE_PATH to the directory containing them\n"
                               f"Searched paths: {paths}\n ")
        return available

    def _resolve_flagcx_module_path(self):
        """Auto-configure paths from FLAGCX_MODULE_PATH if set."""
        module_path = os.environ.get("FLAGCX_MODULE_PATH", None)
        if module_path:
            module_path = Path(module_path)
            os.environ.update({
                "FLAGCX_BITCODE_PATH": str(module_path / 'libflagcx_device.bc'),
                "FLAGCX_LIB_PATH": str(module_path / self.shared_name),
                "FLAGCX_INCLUDE_PATH": str(module_path / self.include_name),
            })

    def _get_shared_lib_paths(self):
        paths = (
            os.environ.get('FLAGCX_LIB_PATH'),
            self.triton_path / '_C' / self.shared_name,
            self.flagcx_cache_dir / self.shared_name,
        )
        return self._check_path_available(paths, self.shared_name)

    def _get_include_paths(self):
        paths = (
            os.environ.get('FLAGCX_INCLUDE_PATH'),
            self.triton_path / 'experimental' / 'tle' / 'language' / 'include',
            self.flagcx_cache_dir / self.include_name,
        )
        return self._check_path_available(paths, self.include_name)

    def _get_bitcode_paths(self):
        return self._check_path_available(
            (self.triton_path / 'backends' / 'nvidia' / 'backend' / 'lib' / 'libflagcx_device.bc',
             self.flagcx_cache_dir / 'libflagcx_device.bc'),
            'libflagcx_device.bc',
        )

    def __init__(self):
        self.is_available = self._is_available()
        if self.is_available:
            self._resolve_flagcx_module_path()
            self.shared_lib_path = self._get_shared_lib_paths()[0]
            self.include_path = self._get_include_paths()[0]
            bp = self._get_bitcode_paths()
            self.bitcode_path = str(bp[0]) if bp else None


class FlagCXBackendAdapter:
    """Backend-specific runtime integration for flagcx.

    Subclasses must implement:
        device_type -> str: device type name (e.g. "cuda", "musa")
        distributed_backend_name -> str: torch.distributed backend name
        allocator_class -> type: PyTorch allocator class

    Optional overrides:
        compile_allocator_guard(): Context manager for build-time setup
    """

    # Virtual methods
    @property
    def device_type(self) -> str:
        raise NotImplementedError()

    @property
    def distributed_backend_name(self) -> str:
        raise NotImplementedError()

    @property
    def allocator_class(self):
        raise NotImplementedError()

    @contextmanager
    def compile_allocator_guard(self):
        yield


class Distributed:
    """Extern library mappings for flagcx compilation."""

    def __init__(self, config):
        self.config = config

    def get_extern_libs(self):
        if self.config.is_available and self.config.bitcode_path:
            return {'libflagcx': str(self.config.bitcode_path)}
        return {}

    @property
    def is_available(self):
        return self.config.is_available
