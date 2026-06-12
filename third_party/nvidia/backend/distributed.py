import os
from pathlib import Path
from dataclasses import dataclass
'''
FlagCX distributed runtime configuration module.

This module is responsible for:
1. Locating the local FlagCX runtime installation directory;
2. Configuring the FlagCX bitcode and shared library paths required
   by Triton/TLE distributed execution;
3. Providing extern library mappings for compilation/runtime stages.

Expected directory layout:
  ~/.flagtree/flagcx/
      ├── libflagcx_device.bc
      └── libflagcx.so
      └── include


Main components:

- FlagCXConfig:
    Resolves and validates the FlagCX runtime paths, including:
      - libflagcx_device.bc
      - libflagcx.so

Typical usage:

  dist = Distributed()
  extern_libs = dist.get_extern_libs()

  triton.compile(..., extern_libs=extern_libs)
'''


@dataclass
class FlagcxRuntimeConfig:
    bt_name: str = 'libflagcx_device.bc'
    shared_name: str = 'libflagcx.so'
    include_name: str = 'include'
    flagcx_cache_dir = Path.home() / ".flagtree" / "flagcx"
    triton_path = Path(__file__).parent.parent.parent

    def _is_available(self):
        env_keys = ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST", "USE_TLE_DISTRIBUTED")
        user_action = True
        for key in env_keys:
            user_action = os.environ.get(key) not in ('OFF', '0', 'false')
            if not user_action:
                break
        if not user_action:
            return False
        try:
            from .flagcx_wrapper import (FLAGCXLibrary,  # noqa: F401
                                         flagcxDevCommRequirements,  # noqa: F401
                                         flagcxUniqueId,  # noqa: F401
                                         FLAGCX_WIN_COLL_SYMMETRIC,  # noqa: F401
                                         )
            return True
        except ImportError:
            return False

    def __init__(self, path_order=0):
        self.is_available = self._is_available()
        if self.is_available:
            self._find_flagcx_module_path()
            self.bitcode_path = self._get_bitcode_paths()[path_order]
            self.shared_lib_path = self._get_shared_lib_paths()[path_order]
            self.include_path = self._get_include_paths()[path_order]

    def _check_path_available(self, paths):
        available_paths = [Path(p) for p in paths if p and p.exists()]
        if len(paths) == 0:
            raise RuntimeError(f"There are no available {self.bt_name} path in this {available_paths}")
        return available_paths

    def _find_flagcx_module_path(self):
        module_path = os.environ.get("FLAGCX_MODULE_PATH", None)
        if module_path:
            module_path = Path(module_path)
            os.environ.update({
                "FLAGCX_BITCODE_PATH": str(module_path / self.bt_name), "FLAGCX_LIB_PATH":
                str(module_path / self.shared_name), "FLAGCX_INCLUDE_PATH": str(module_path / self.include_name)
            })

    def _get_bitcode_paths(self):
        paths = (
            os.environ.get("FLAGCX_BITCODE_PATH"),
            Path(__file__).parent / "lib" / self.bt_name,
            self.flagcx_cache_dir / self.bt_name,
        )
        return self._check_path_available(paths)

    def _get_shared_lib_paths(self):
        paths = (
            os.environ.get("FLAGCX_LIB_PATH"),
            self.triton_path / "_C" / self.shared_name,
            self.flagcx_cache_dir / self.shared_name,
        )
        return self._check_path_available(paths)

    def _get_include_paths(self):

        paths = (
            os.environ.get("FLAGCX_INCLUDE_PATH"),
            self.triton_path / "experimental" / "tle" / "language" / "include",
            self.flagcx_cache_dir / self.include_name,
        )
        return self._check_path_available(paths)


flagcx_rt_conf = FlagcxRuntimeConfig()


class Distributed:

    def __init__(self):
        self.extern_libs = {}
        if flagcx_rt_conf.is_available:
            self.extern_libs["libflagcx"] = str(flagcx_rt_conf.bitcode_path)

    def get_extern_libs(self):
        return self.extern_libs
