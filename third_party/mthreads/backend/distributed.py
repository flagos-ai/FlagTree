"""MThreads FlagCX runtime discovery and device-library linking.

Self-contained for main (does not require the shared `_flagcx_config` from
HCU FlagCX PR #1441).
"""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.machinery import PathFinder
from pathlib import Path
import functools
import hashlib
import json
import os
import sys


def flagcx_packages_detector() -> bool:
    from .flagcx_wrapper import (  # noqa: F401
        FLAGCXLibrary,
        flagcxDevCommRequirements,
        flagcxUniqueId,
        FLAGCX_WIN_COLL_SYMMETRIC,
    )
    return True


@dataclass
class FlagcxRuntimeConfig:
    bt_name: str = "libflagcx_device.bc"
    shared_name: str = "libflagcx.so"
    include_name: str = "include"
    flagcx_cache_dir = Path.home() / ".flagtree" / "flagcx"
    # Arch-specific bitcode must not be shared across MUSA builds via cache.
    use_cache: bool = False
    triton_path = Path(__file__).parent.parent.parent

    def _get_flagcx_wheel_path(self):
        spec = PathFinder.find_spec("flagcx", sys.path)
        return Path(spec.origin).parent if spec is not None else None

    def _is_available(self):
        for key in ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST",
                    "USE_TLE_DISTRIBUTED"):
            if os.environ.get(key) in ("OFF", "0", "false"):
                return False
        # Do not import flagcx_wrapper/torch at module import time: the mthreads
        # compiler imports this module during triton backend discovery, and
        # torch_musa autoload would re-enter a partially initialized triton.
        # Path probes are enough for compile-time linking; the wrapper is used
        # later from communication.py once torch is fully loaded.
        if self._probe_bitcode():
            return True
        try:
            return flagcx_packages_detector()
        except Exception:
            return False

    def _probe_bitcode(self):
        for path in (
            os.environ.get("FLAGCX_BITCODE_PATH"),
            Path(__file__).parent / "lib" / self.bt_name,
        ):
            if path and Path(path).is_file():
                return True
        return False

    def get_needed_package(self):
        return (self.bt_name, self.shared_name, self.include_name)

    def __init__(self, path_order=0):
        self.flagcx_whl_path = self._get_flagcx_wheel_path()
        self.is_available = self._is_available()
        if not self.is_available and not self._probe_bitcode():
            self.bitcode_path = None
            self.shared_lib_path = None
            self.include_path = None
            return
        self._find_flagcx_module_path()
        try:
            self.bitcode_path = self._get_bitcode_paths()[path_order]
        except RuntimeError:
            self.bitcode_path = None
        try:
            self.shared_lib_path = self._get_shared_lib_paths()[path_order]
        except RuntimeError:
            self.shared_lib_path = None
        try:
            self.include_path = self._get_include_paths()[path_order]
        except RuntimeError:
            self.include_path = None
        # Available for host runtime if so+include exist; bitcode alone is OK
        # for compile-only linking.
        self.is_available = bool(self.bitcode_path) or bool(
            self.shared_lib_path and self.include_path)

    def _check_path_available(self, paths, name):
        available_paths = [Path(p) for p in paths if p and Path(p).exists()]
        if not available_paths:
            raise RuntimeError(
                f"\nCannot find available '{name}' file or lib\n"
                f"Searched paths: {paths}\n")
        return available_paths

    def _find_flagcx_module_path(self):
        module_path = os.environ.get("FLAGCX_MODULE_PATH") or os.environ.get("FLAGCX_PATH")
        if module_path:
            module_path = Path(module_path)
            # Prefer build tree layout: lib/libflagcx.so + include/
            lib = module_path / "lib" / self.shared_name
            inc = module_path / "include"
            bt = module_path / "lib" / self.bt_name
            if lib.is_file():
                os.environ.setdefault("FLAGCX_LIB_PATH", str(lib))
            if inc.is_dir():
                os.environ.setdefault("FLAGCX_INCLUDE_PATH", str(inc))
            if bt.is_file():
                os.environ.setdefault("FLAGCX_BITCODE_PATH", str(bt))

    def _get_flagcx_package_paths(self, pkg_name):
        if self.flagcx_whl_path is None:
            return None
        pkg_path = self.flagcx_whl_path / "lib" / pkg_name
        return pkg_path if pkg_path.exists() else None

    def _get_bitcode_paths(self):
        paths = (
            os.environ.get("FLAGCX_BITCODE_PATH"),
            self._get_flagcx_package_paths(self.bt_name),
            Path(__file__).parent / "lib" / self.bt_name,
            self.flagcx_cache_dir / self.bt_name if self.use_cache else None,
        )
        return self._check_path_available(paths, self.bt_name)

    def _get_shared_lib_paths(self):
        paths = (
            os.environ.get("FLAGCX_LIB_PATH"),
            self._get_flagcx_package_paths(self.shared_name),
            self.triton_path / "_C" / self.shared_name,
            self.flagcx_cache_dir / self.shared_name if self.use_cache else None,
        )
        return self._check_path_available(paths, self.shared_name)

    def _get_include_paths(self):
        paths = (
            os.environ.get("FLAGCX_INCLUDE_PATH"),
            self._get_flagcx_package_paths(self.include_name),
            self.triton_path / "experimental" / "tle" / "language" / self.include_name,
            self.flagcx_cache_dir / self.include_name if self.use_cache else None,
        )
        return self._check_path_available(paths, self.include_name)


flagcx_rt_conf = FlagcxRuntimeConfig()


class MusaFlagCXBackendAdapter:
    """MUSA/torch_musa APIs used by FlagCX TLE communication."""

    device_type = "musa"
    distributed_backend_name = "mccl"

    @property
    def allocator_class(self):
        try:
            from torch_musa.core.memory import MUSAPluggableAllocator
            return MUSAPluggableAllocator
        except ImportError:
            from torch_musa.core.memory import CUDAPluggableAllocator
            return CUDAPluggableAllocator

    @property
    def device_module(self):
        import torch
        import torch_musa  # noqa: F401 — registers torch.musa
        if hasattr(torch, "musa"):
            return torch.musa
        return torch.get_device_module(self.device_type)

    @contextmanager
    def compile_allocator_guard(self):
        yield


backend_adapter = MusaFlagCXBackendAdapter()


class Distributed:

    def __init__(self):
        # Key "flagcx" matches flagcxDev* declarations in the device bitcode.
        path = flagcx_rt_conf.bitcode_path
        self.extern_libs = {"flagcx": str(path)} if path else {}

    def get_extern_libs(self):
        return self.extern_libs

    @property
    def is_available(self):
        return bool(flagcx_rt_conf.shared_lib_path and flagcx_rt_conf.include_path)


def _normalize_musa_arch(arch):
    text = str(arch).split(":")[0].strip().lower()
    if text.startswith("mp_"):
        text = text[3:]
    return text


def link_flagcx(llvm_mod, extern_libs, llvm, arch):
    """Link FlagCX device bitcode when the kernel references FlagCX symbols."""
    path = dict(extern_libs or ()).get("flagcx")
    if not path:
        return
    manifest = Path(path).resolve().parent / "metadata.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        if metadata.get("backend") != "mthreads":
            raise RuntimeError(f"FlagCX bundle is not built for mthreads: {manifest}")
        expected = metadata.get("arch")
        if expected and _normalize_musa_arch(arch) != _normalize_musa_arch(expected):
            raise RuntimeError(
                f"FlagCX bitcode targets {expected}, but this kernel targets {arch}. "
                "Rebuild FlagTree with DEVICE_BITCODE_ARCH set to the kernel GPU architecture.")
    llvm.link_extern_libs(llvm_mod, [path], True)
    for fn in llvm_mod.get_functions():
        name = fn.name
        if name.startswith("flagcx") or "flagcx" in name:
            try:
                fn.remove_fn_attr("noinline")
            except Exception:
                pass
            try:
                fn.remove_fn_attr("optnone")
            except Exception:
                pass
            fn.add_fn_attr("alwaysinline", "")


@functools.lru_cache(maxsize=128)
def _file_hash(filename, identity):
    digest = hashlib.sha256()
    with open(filename, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extern_libs_hash(extern_libs):
    result = []
    for name, filename in sorted(extern_libs):
        path = Path(filename).resolve()
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        result.append((name, filename, _file_hash(str(path), identity)))
    return tuple(result)
