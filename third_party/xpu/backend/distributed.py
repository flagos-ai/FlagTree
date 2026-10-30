"""Locate host FlagCX files and optional XPU device bitcode."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.machinery import PathFinder
from pathlib import Path


def flagcx_packages_detector() -> bool:
    try:
        from . import flagcx_wrapper
    except (ImportError, OSError):
        return False
    return all(
        hasattr(flagcx_wrapper, name)
        for name in ("FLAGCXLibrary", "flagcxDevCommRequirements", "flagcxUniqueId", "FLAGCX_WIN_COLL_SYMMETRIC"))


class FlagcxRuntimeConfig:
    bt_name = "libflagcx_device.bc"
    shared_name = "libflagcx.so"
    include_name = "include"

    def __init__(self, path_order=0):
        self.bitcode_path = None
        self.shared_lib_path = None
        self.include_path = None
        self.host_files_available = False
        self.is_available = False

        env_keys = ("USE_FLAGCX", "USE_DIST", "USE_DISTRIBUTED", "USE_TLE_DIST", "USE_TLE_DISTRIBUTED")
        disabled = any(os.environ.get(key, "").strip().lower() in ("off", "0", "false") for key in env_keys)
        if disabled:
            return

        module_dir = os.environ.get("FLAGCX_MODULE_PATH")
        module_dir = Path(module_dir).expanduser() if module_dir else None
        wheel_dir = self._get_flagcx_wheel_path()
        cache_dir = Path(os.environ.get("FLAGTREE_CACHE_DIR", Path.home() / ".flagtree")) / "flagcx" / "xpu"
        backend_dir = Path(__file__).parent
        source_triton_dir = backend_dir.parent / "python" / "triton"
        triton_dir = source_triton_dir if source_triton_dir.is_dir() else backend_dir.parent.parent

        def in_dir(directory, name):
            return directory / name if directory else None

        self.bitcode_path = self._first_existing(
            (os.environ.get("FLAGCX_BITCODE_PATH"), in_dir(module_dir, self.bt_name), backend_dir / "lib" /
             self.bt_name, wheel_dir / "lib" / self.bt_name if wheel_dir else None, cache_dir / self.bt_name),
            path_order, file=True)
        self.shared_lib_path = self._first_existing(
            (os.environ.get("FLAGCX_LIB_PATH"), in_dir(module_dir,
                                                       self.shared_name), backend_dir / "lib" / self.shared_name,
             wheel_dir / "lib" / self.shared_name if wheel_dir else None, cache_dir / self.shared_name), path_order,
            file=True, name=self.shared_name)
        self.include_path = self._first_existing(
            (os.environ.get("FLAGCX_INCLUDE_PATH"), in_dir(
                module_dir, self.include_name), triton_dir / "experimental" / "tle" / "language" / "include",
             wheel_dir / "lib" / self.include_name if wheel_dir else None, cache_dir / self.include_name), path_order,
            file=False)
        self.host_files_available = bool(self.shared_lib_path and self.include_path and flagcx_packages_detector())
        self.is_available = bool(self.bitcode_path and self.host_files_available)

    @staticmethod
    def _get_flagcx_wheel_path():
        spec = PathFinder.find_spec("flagcx", sys.path)
        return Path(spec.origin).parent if spec and spec.origin else None

    @staticmethod
    def _first_existing(candidates, path_order, *, file, name=None):
        paths = []
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if name and path.is_dir():
                path /= name
            exists = path.is_file() if file else path.is_dir()
            if exists:
                paths.append(path)
        return paths[path_order] if path_order < len(paths) else None

    def get_needed_package(self):
        return (self.bt_name, self.shared_name, self.include_name)


flagcx_rt_conf = FlagcxRuntimeConfig()

_implicit_context = None


def register_implicit_context(ctx):
    global _implicit_context
    _implicit_context = ctx


def clear_implicit_context():
    global _implicit_context
    if _implicit_context is not None:
        cls = type(_implicit_context)
        cls._init_count = max(0, cls._init_count - 1)
    _implicit_context = None


def get_implicit_context():
    if _implicit_context is None or not _implicit_context.is_lite_mode:
        raise RuntimeError("FLAGTREE_LITE_DIST requires one registered FlagCX distributed context")
    return _implicit_context


class Distributed:
    """Compatibility interface for backends that can link FlagCX device IR."""

    def get_extern_libs(self):
        return ({"libflagcx": str(flagcx_rt_conf.bitcode_path)}
                if flagcx_rt_conf.is_available and flagcx_rt_conf.bitcode_path else {})

    @property
    def is_available(self):
        return flagcx_rt_conf.is_available


def has_flagcx_extern_deps(kernel_ir: str) -> bool:
    return re.search(r"(?m)^\s*declare\b[^\n]*@flagcx[A-Za-z0-9_.$]+\s*\(", kernel_ir) is not None


def link_flagcx_device_ir(kernel_ir: str, bitcode_path: str) -> str:
    """Link XPU FlagCX device bitcode after the LLVM 22 -> 19 downgrade."""
    from triton.backends import llvm19_toolchain

    staged_tool = Path(llvm19_toolchain.get_llvm19_bin_dir("xpu")) / "llvm-link"
    configured_tool = os.environ.get("FLAGCX_XPU_LLVM_LINK")
    xtdk_home = os.environ.get("XTDK_HOME")
    clang_path = os.environ.get("FLAGCX_XPU_CLANG")
    candidates = (staged_tool, Path(xtdk_home) / "bin" / "llvm-link" if xtdk_home else None,
                  Path(clang_path).expanduser().parent / "llvm-link" if clang_path else None,
                  shutil.which("llvm-link-19"))
    tool = configured_tool or next((str(path) for path in candidates if path and Path(path).is_file()), None)
    if not tool or not Path(tool).is_file():
        raise RuntimeError("XPU FlagCX device calls require LLVM 19 llvm-link; "
                           "set FLAGCX_XPU_LLVM_LINK to an LLVM 19 llvm-link executable")
    version = subprocess.run([str(tool), "--version"], capture_output=True, text=True,
                             env=llvm19_toolchain.llvm19_env("xpu"), check=True).stdout
    if not re.search(r"\bversion 19(?:\.|\b)", version):
        raise RuntimeError(f"XPU FlagCX requires llvm-link version 19, got: {version.splitlines()[0]}")
    with tempfile.TemporaryDirectory(prefix="xpu_flagcx_") as directory:
        source = Path(directory) / "kernel.ll"
        linked = Path(directory) / "linked.ll"
        source.write_text(kernel_ir)
        result = subprocess.run([str(tool), "-S", str(source), bitcode_path, "-o",
                                 str(linked)], capture_output=True, text=True, env=llvm19_toolchain.llvm19_env("xpu"))
        if result.returncode:
            raise RuntimeError(f"Failed to link XPU FlagCX device bitcode {bitcode_path}: {result.stderr.strip()}")
        linked_ir = linked.read_text()
        signatures = {
            "flagcxDevCommGetIntraRank": ("ptr addrspace(1)", ),
            "flagcxDevCommGetIntraSize": ("ptr addrspace(1)", ),
            "flagcxGetIntraPointerC": ("ptr addrspace(1)", "i64", "i32"),
            "flagcxIntraBarrierArriveS": ("ptr addrspace(1)", "i32", "i32", "i1", "i32"),
            "flagcxIntraBarrierWaitS": ("ptr addrspace(1)", "i32", "i32", "i1", "i32"),
            "flagcxIntraBarrierSyncS": ("ptr addrspace(1)", "i32", "i32", "i1", "i32"),
        }
        symbols = set(
            re.findall(
                r"@(flagcx(?:DevCommGetIntra(?:Rank|Size)|IntraBarrier(?:Sync|Arrive|Wait)S|GetIntraPointerC))\(",
                kernel_ir))
        for symbol in symbols:
            definition = re.search(rf"(?m)^define\b[^\n]*@{re.escape(symbol)}\(", linked_ir)
            if not definition:
                raise RuntimeError(f"XPU FlagCX device bitcode {bitcode_path} does not define {symbol}")
            params = []
            depth = 0
            for char in linked_ir[definition.end():].split("\n", 1)[0]:
                if char == ")" and depth == 0:
                    break
                params.append(char)
                depth += (char == "(") - (char == ")")
            actual = "".join(params).split(",")
            expected = signatures[symbol]
            if len(actual) != len(expected) or any(not re.match(r"^\s*" + re.escape(ty) + r"(?:\s|$)", param)
                                                   for param, ty in zip(actual, expected)):
                raise RuntimeError(f"XPU FlagCX device bitcode {bitcode_path} has an incompatible {symbol} ABI")
            if symbol == "flagcxGetIntraPointerC" and "ptr addrspace(1)" not in definition.group(0).split("@")[0]:
                raise RuntimeError(f"XPU FlagCX device bitcode {bitcode_path} has an incompatible {symbol} return type")
        return linked_ir
