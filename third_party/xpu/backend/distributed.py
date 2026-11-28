"""Locate host FlagCX files and optional XPU device bitcode."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.machinery import PathFinder
from pathlib import Path


def _xpu_cache_dir():
    cache_dir = os.environ.get("FLAGTREE_CACHE_DIR")
    return (Path(cache_dir).expanduser() if cache_dir else Path.home() / ".flagtree") / "xpu"


def find_xshmem_device_lib():
    archive_name = "libxshmem_device.a"
    explicit = os.environ.get("FLAGCX_XSHMEM_DEVICE_LIB")
    if explicit:
        archive = Path(explicit).expanduser()
        if not archive.is_file():
            raise FileNotFoundError(f"FLAGCX_XSHMEM_DEVICE_LIB does not point to an archive: {archive}")
        return archive

    ccl_home = next((Path(os.environ[name]).expanduser()
                     for name in ("FLAGCX_XPU_CCL_HOME", "XCCL_HOME", "BKCL_HOME", "CCL_HOME")
                     if os.environ.get(name)), None)
    cache_dir = _xpu_cache_dir()
    default_sdk = cache_dir / "sdk" / "xccl"
    if ccl_home is None:
        ccl_home = next((sdk for sdk in (default_sdk, Path("/usr/local/xccl")) if sdk.is_dir()), None)
    archive = ccl_home / "lib" / archive_name if ccl_home else cache_dir / "lib" / archive_name
    if not archive.is_file():
        raise FileNotFoundError(f"XPU FlagCX distributed kernels need {archive}. "
                                f"Install the XCCL SDK under {default_sdk} or set FLAGCX_XSHMEM_DEVICE_LIB.")
    return archive


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

        if os.environ.get("USE_FLAGCX", "ON").strip().upper() not in ("ON", "1", "TRUE"):
            return

        module_dir = os.environ.get("FLAGCX_MODULE_PATH")
        module_dir = Path(module_dir).expanduser() if module_dir else None
        wheel_dir = self._get_flagcx_wheel_path()
        cache_dir = _xpu_cache_dir()
        lib_dir = cache_dir / "lib"
        legacy_cache_dir = cache_dir.parent / "flagcx" / "xpu"
        backend_dir = Path(__file__).parent
        source_triton_dir = backend_dir.parent / "python" / "triton"
        triton_dir = source_triton_dir if source_triton_dir.is_dir() else backend_dir.parent.parent
        language_dir = triton_dir / "experimental" / "tle" / "language"

        def find_artifact(key, name, package_path, *, directory=False):
            value = os.environ.get(key)
            override = Path(value).expanduser() if value else None
            if key == "FLAGCX_LIB_PATH" and override and override.is_dir():
                override /= name
            exists = Path.is_dir if directory else Path.is_file
            if override is not None and not exists(override):
                raise FileNotFoundError(f"{key} points to a missing XPU FlagCX artifact: {override}")
            candidates = (
                override,
                module_dir / name if module_dir else None,
                package_path,
                lib_dir / name,
                wheel_dir / "lib" / name if wheel_dir else None,
                legacy_cache_dir / name,
            )
            paths = [path for path in candidates if path is not None and exists(path)]
            return paths[path_order] if path_order < len(paths) else None

        self.bitcode_path = find_artifact("FLAGCX_BITCODE_PATH", self.bt_name, backend_dir / "lib" / self.bt_name)
        self.shared_lib_path = find_artifact("FLAGCX_LIB_PATH", self.shared_name, backend_dir / "lib" / self.shared_name)
        self.include_path = find_artifact("FLAGCX_INCLUDE_PATH", self.include_name, language_dir / self.include_name,
                                          directory=True)
        # Importing the TLE package here would recurse into this runtime config.
        self.host_files_available = bool(self.shared_lib_path and self.include_path
                                         and (language_dir / "flagcx_wrapper.py").is_file())
        self.is_available = bool(self.bitcode_path and self.host_files_available)

    @staticmethod
    def _get_flagcx_wheel_path():
        spec = PathFinder.find_spec("flagcx", sys.path)
        return Path(spec.origin).parent if spec and spec.origin else None

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
            "flagcxDevCommGetIntraRank": ("i32", ("ptr addrspace(1)", )),
            "flagcxDevCommGetIntraSize": ("i32", ("ptr addrspace(1)", )),
            "flagcxGetIntraPointerC": ("ptr addrspace(1)", ("ptr addrspace(1)", "i64", "i32")),
            "flagcxGetIntraPointerWithAccessC": ("ptr addrspace(1)", ("ptr addrspace(1)", "i64", "i32", "i32")),
            "flagcxIntraBarrierArriveS": ("void", ("ptr addrspace(1)", "i32", "i32", "i1", "i32")),
            "flagcxIntraBarrierWaitS": ("void", ("ptr addrspace(1)", "i32", "i32", "i1", "i32")),
            "flagcxIntraBarrierSyncS": ("void", ("ptr addrspace(1)", "i32", "i32", "i1", "i32")),
            "flagcxDevInit": ("void", ("ptr addrspace(1)", "ptr")),
            "flagcxDevSignalInc": ("void", ("ptr addrspace(1)", "i32", "i32", "i32", "i32", "i32", "i32")),
            "flagcxDevSignalAdd": ("void", ("ptr addrspace(1)", "i32", "i32", "i32", "i64", "i32", "i32", "i32")),
            "flagcxDevWaitSignal": ("void", ("ptr addrspace(1)", "i32", "i64", "i32", "i32", "i32", "i32")),
            "flagcxDevWaitSignalMeetShadow": ("void", ("ptr addrspace(1)", "i32", "i32", "i32", "i32", "i32")),
            "flagcxDevWaitCounter": ("void", ("ptr addrspace(1)", "i32", "i64", "i32", "i32", "i32", "i32")),
        }
        symbols = set(re.findall(r"@(flagcx[A-Za-z0-9_.$]+)\s*\(", kernel_ir))
        for symbol in sorted(signatures.keys() & symbols):
            expected_return, expected_params = signatures[symbol]
            definition = re.search(rf"(?m)^define\b([^\n]*)@{re.escape(symbol)}\(", linked_ir)
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
            if len(actual) != len(expected_params) or any(
                    not re.match(r"^\s*" + re.escape(ty) + r"(?!\s+addrspace\()(?:\s|$)", param)
                    for param, ty in zip(actual, expected_params)):
                raise RuntimeError(f"XPU FlagCX device bitcode {bitcode_path} has an incompatible {symbol} ABI")
            if not re.search(r"\s" + re.escape(expected_return) + r"\s*$", definition.group(1)):
                raise RuntimeError(f"XPU FlagCX device bitcode {bitcode_path} has an incompatible {symbol} return type")
        return linked_ir
