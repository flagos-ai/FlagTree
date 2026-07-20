# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

from __future__ import annotations

import ctypes
import functools
import os
import re
import shlex
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Final

import torch
from triton import knobs
from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]
from triton.experimental.tle.raw.runtime import RawJITFunction
from triton.experimental.tle.raw.source_store import register_source

# TODO: Temporarily shell out to clang; replace with LLVM Python bindings later.
_MIN_CLANG_MAJOR = 20

# ---------------------------------------------------------------------------
# Clang toolchain: knobs override -> discover, always version-check (>= 20)
# ---------------------------------------------------------------------------


def _parse_clang_major(clang: str) -> int | None:
    try:
        out = subprocess.check_output([clang, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"clang version (\d+)\.", out)
    if match is None:
        match = re.search(r"version (\d+)\.", out)
    return int(match.group(1)) if match else None


def _clang_meets_min_version(clang: str) -> bool:
    major = _parse_clang_major(clang)
    return major is not None and major >= _MIN_CLANG_MAJOR


def _discover_clang_binaries() -> list[str]:
    """Prefer newer versioned binaries, then plain ``clang``."""
    found: list[str] = []
    seen: set[str] = set()
    for name in ("clang-22", "clang-21", "clang-20", "clang"):
        path = shutil.which(name)
        if path is None:
            continue
        resolved = str(Path(path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(path)
    return found


@functools.lru_cache()
def _resolve_clang() -> str:
    """Use user CLANG if usable; otherwise discover. Every candidate is version-checked."""
    tried: list[str] = []
    user_clang = knobs.nvidia.tle_raw_clang
    if user_clang:
        tried.append(user_clang)
        if _clang_meets_min_version(user_clang):
            return user_clang

    for candidate in _discover_clang_binaries():
        if candidate in tried:
            continue
        tried.append(candidate)
        if _clang_meets_min_version(candidate):
            return candidate

    detail = ", ".join(tried) if tried else "<none>"
    raise RuntimeError(f"TLE raw CUDA requires clang >= {_MIN_CLANG_MAJOR}. "
                       f"Tried: {detail}. Install clang-20+ or set CLANG to a suitable binary.")


# ---------------------------------------------------------------------------
# Clang compile flags (--cuda-path, includes, optional CLANG_FLAGS)
# ---------------------------------------------------------------------------


def _cuda_home() -> Path:
    return Path(os.getenv("CUDA_HOME", "/usr/local/cuda"))


def _nvidia_backend_include() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "nvidia" / "backend" / "include"
        if candidate.is_dir():
            return candidate
    return None


def _default_clang_flags() -> list[str]:
    cuda_home = _cuda_home()
    flags = [f"--cuda-path={cuda_home}", f"-I{cuda_home / 'include'}"]
    backend_include = _nvidia_backend_include()
    if backend_include is not None:
        flags.append(f"-I{backend_include}")
    try:
        from triton.experimental.tle.raw.nvshmem.utils import try_get_nvshmem_home
        nvshmem_home = try_get_nvshmem_home()
        if nvshmem_home is not None:
            flags.append(f"-I{nvshmem_home / 'include'}")
    except Exception:
        pass
    return flags


def _clang_flags() -> list[str]:
    extra = knobs.nvidia.tle_raw_clang_flags or ""
    return [*_default_clang_flags(), *shlex.split(extra)]


def _get_cuda_gpu_arch() -> str:
    arch = os.getenv("TLE_CUDA_ARCH")
    if arch:
        return f"--cuda-gpu-arch={arch}"
    major, minor = torch.cuda.get_device_capability()
    return f"--cuda-gpu-arch=sm_{major}{minor}"


# ---------------------------------------------------------------------------
# Sanitize clang LLVM IR for this Triton's parser
# ---------------------------------------------------------------------------


def _sanitize_clang_ir(ir: str) -> str:
    # Newer clang emits attributes that this Triton branch's LLVM parser does
    # not understand yet. They are not needed by TLE raw device function import.
    ir = ir.replace(" nocreateundeforpoison", "")
    ir = ir.replace(" contract", "")

    def _replace_hex_float(match: re.Match[str]) -> str:
        hex_digits = match.group(1)
        bits = int(hex_digits, 16)
        if len(hex_digits) == 16:
            value = struct.unpack("!d", bits.to_bytes(8, byteorder="big"))[0]
        elif len(hex_digits) == 8:
            value = struct.unpack("!f", bits.to_bytes(4, byteorder="big"))[0]
        else:
            return match.group(0)
        return repr(value)

    return re.sub(r"f0x([0-9A-Fa-f]+)", _replace_hex_float, ir)


# ---------------------------------------------------------------------------
# NVSHMEM: post-compile cumodule init hook
# ---------------------------------------------------------------------------

_cumodule_hook_installed = False
_nvshmemx_cumodule_init = None


def _get_nvshmemx_cumodule_init():
    global _nvshmemx_cumodule_init
    if _nvshmemx_cumodule_init is not None:
        return _nvshmemx_cumodule_init

    from triton.experimental.tle.raw.nvshmem.utils import (
        get_nvshmem_home,
        resolve_nvshmem_host_library,
    )
    library = ctypes.CDLL(str(resolve_nvshmem_host_library(get_nvshmem_home())))
    fn = library.nvshmemx_cumodule_init
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _nvshmemx_cumodule_init = fn
    return fn


def _install_cumodule_hook():
    global _cumodule_hook_installed
    if _cumodule_hook_installed:
        return

    def hook(*args, **kwargs):
        key = kwargs["key"]
        function = kwargs["fn"].jit_function
        device = kwargs["compile"]["device"]
        kernel = function.device_caches[device][0].get(key)
        assert kernel is not None
        kernel._init_handles()
        result = _get_nvshmemx_cumodule_init()(ctypes.c_void_p(kernel.module))
        assert result == 0, f"nvshmemx_cumodule_init failed: {result}"

    knobs.runtime.jit_post_compile_hook = hook
    _cumodule_hook_installed = True


# ---------------------------------------------------------------------------
# Dialect runtime
# ---------------------------------------------------------------------------


class CUDAJITFunction(RawJITFunction):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(fn, **kwargs)
        self.code: Final[str] = file.read_text()
        self.region_dialect: Final[str] = "cuda"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"
        self.source_file: Final[str] = str(file)

        if self.library == "nvshmem":
            from triton.experimental.tle.raw.nvshmem.utils import enable_nvshmem_device_bc
            enable_nvshmem_device_bc(True)
        if self.library == "nvshmem" or "nvshmem" in self.code:
            _install_cumodule_hook()

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("deferred tle_raw CUDA source requires extern_func_name= "
                               "(the device function symbol in the .cu file)")
        return register_source(
            region_dialect=self.region_dialect,
            extern_func_name=self.extern_func_name,
            source=self.code,
            hint=hint,
            extra={"source_file": self.source_file},
        )

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = "",
                              extern_func_name: str = ""):
        return super().create_region_by_llvm(builder, llvm, handles, alias_indices, hint, extern_func_name)

    def create_region_deferred(self, builder, source_id: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_deferred(
            source_id,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

    def make_llvm(self, mlir_context) -> str:
        build = subprocess.run(
            [
                _resolve_clang(),
                "-x",
                "cuda",
                "--cuda-device-only",
                _get_cuda_gpu_arch(),
                "-emit-llvm",
                "-O2",
                "-S",
                "-",
                "-o",
                "-",
                *_clang_flags(),
            ],
            input=self.code.encode(),
            capture_output=True,
        )
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")
        llvm_context = llvm.context()
        module = parse_llvm_ir(_sanitize_clang_ir(build.stdout.decode()), llvm_context, mlir_context)
        return f"{module}"


def compile_deferred_pending_source(entry: dict, *, context) -> str:
    source_text = entry["source"]

    class _CudaSourceFile:

        def read_text(self):
            return source_text

    cuda_fn = CUDAJITFunction(
        fn=None,
        file=_CudaSourceFile(),
        extern_func_name=entry.get("extern_func_name"),
        deferred=True,
    )
    return cuda_fn.make_llvm(context)
