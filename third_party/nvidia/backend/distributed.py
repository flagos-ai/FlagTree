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


def flagcx_packages_detector() -> bool:
    from .flagcx_wrapper import (FLAGCXLibrary,  # noqa: F401
                                 flagcxDevCommRequirements,  # noqa: F401
                                 flagcxUniqueId,  # noqa: F401
                                 FLAGCX_WIN_COLL_SYMMETRIC,  # noqa: F401
                                 )
    return True


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
                return False
        try:
            return flagcx_packages_detector()
        except ImportError:
            return False

    def get_needed_package(self):
        return (self.bt_name, self.shared_name, self.include_name)

    def __init__(self, path_order=0):
        self.is_available = self._is_available()
        if self.is_available:
            self._find_flagcx_module_path()
            self.bitcode_path = self._get_bitcode_paths()[path_order]
            self.shared_lib_path = self._get_shared_lib_paths()[path_order]
            self.include_path = self._get_include_paths()[path_order]

    def _check_path_available(self, paths, name):
        available_paths = [Path(p) for p in paths if p and Path(p).exists()]
        if len(available_paths) == 0:
            raise RuntimeError(f"\nCannot find available '{name}' file or lib\n"
                               f"If you already have the required '{name}'\n"
                               "please set the corresponding environment variables\n"
                               "(e.g. FLAGCX_BITCODE_PATH, FLAGCX_LIB_PATH, FLAGCX_INCLUDE_PATH)\n"
                               f"or set FLAGCX_MODULE_PATH to the directory containing them\n"
                               f"Searched paths: {paths}\n ")
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
        return self._check_path_available(paths, self.bt_name)

    def _get_shared_lib_paths(self):
        paths = (
            os.environ.get("FLAGCX_LIB_PATH"),
            self.triton_path / "_C" / self.shared_name,
            self.flagcx_cache_dir / self.shared_name,
        )
        return self._check_path_available(paths, self.shared_name)

    def _get_include_paths(self):

        paths = (
            os.environ.get("FLAGCX_INCLUDE_PATH"),
            self.triton_path / "experimental" / "tle" / "language" / "include",
            self.flagcx_cache_dir / self.include_name,
        )
        return self._check_path_available(paths, self.include_name)


flagcx_rt_conf = FlagcxRuntimeConfig()


class Distributed:

    def __init__(self):
        self.extern_libs = ({"libflagcx": str(flagcx_rt_conf.bitcode_path)} if flagcx_rt_conf.is_available else {})

    def get_extern_libs(self):
        return self.extern_libs

    @property
    def is_available(self):
        return flagcx_rt_conf.is_available
