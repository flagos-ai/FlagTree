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
from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.iluvatar.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]
from triton.experimental.tle.raw.runtime import RawJITFunction
from triton.experimental.tle.raw.source_store import register_source

# TODO: Temporarily shell out to clang; replace with LLVM Python bindings later.
_MIN_CLANG_MAJOR = 18
# Registered target name of the Iluvatar GPGPU backend in corex clang.
_ILUVATAR_LLVM_TARGET = "bi"

# ---------------------------------------------------------------------------
# Clang toolchain: CLANG override -> discover, always check version and target
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


def _has_iluvatar_target(clang: str) -> bool:
    # A stock clang of the right version still cannot build for corex, so the
    # registered-target list is the check that actually matters here.
    try:
        out = subprocess.check_output([clang, "-print-targets"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return False
    return re.search(rf"^\s*{_ILUVATAR_LLVM_TARGET}\s+-", out, re.MULTILINE) is not None


def _clang_is_usable(clang: str) -> bool:
    major = _parse_clang_major(clang)
    return major is not None and major >= _MIN_CLANG_MAJOR and _has_iluvatar_target(clang)


def _discover_clang_binaries() -> list[str]:
    """The corex SDK ships a plain ``clang``; keep versioned names as fallbacks."""
    found: list[str] = []
    seen: set[str] = set()
    for name in ("clang", "clang-22"):
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
    """Use user CLANG if usable; otherwise discover. Every candidate is checked."""
    tried: list[str] = []
    user_clang = os.getenv("CLANG")
    if user_clang:
        tried.append(user_clang)
        if _clang_is_usable(user_clang):
            return user_clang

    for candidate in _discover_clang_binaries():
        if candidate in tried:
            continue
        tried.append(candidate)
        if _clang_is_usable(candidate):
            return candidate

    detail = ", ".join(tried) if tried else "<none>"
    raise RuntimeError(f"TLE raw CUDA on iluvatar requires a corex clang >= {_MIN_CLANG_MAJOR} "
                       f"with the '{_ILUVATAR_LLVM_TARGET}' target registered. "
                       f"Tried: {detail}. Source the corex SDK or set CLANG to its clang.")


# ---------------------------------------------------------------------------
# Clang compile flags
# ---------------------------------------------------------------------------

_CAPABILITY_TO_ARCH = {71: "ivcore11", 80: "ivcore20", 81: "ivcore30"}


def _get_iluvatar_gpu_arch() -> str:
    arch = os.getenv("TLE_ILUVATAR_ARCH")
    if arch:
        return f"--cuda-gpu-arch={arch}"
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    if capability not in _CAPABILITY_TO_ARCH:
        raise RuntimeError(f"TLE raw CUDA on iluvatar does not know the gpu-arch for "
                           f"capability {capability}; set TLE_ILUVATAR_ARCH explicitly.")
    return f"--cuda-gpu-arch={_CAPABILITY_TO_ARCH[capability]}"


def _clang_flags() -> list[str]:
    extra = os.getenv("CLANG_FLAGS") or ""
    return shlex.split(extra)


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
# Dialect runtime
# ---------------------------------------------------------------------------


class CorexJITFunction(RawJITFunction):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(fn, **kwargs)
        if self.library:
            raise RuntimeError(f"tle_raw library={self.library!r} is not supported on iluvatar; "
                               "only plain corex device sources are.")
        self.code: Final[str] = file.read_text()
        self.region_dialect: Final[str] = "cuda"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"
        self.source_file: Final[str] = str(file)

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("deferred tle_raw corex source requires extern_func_name= "
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
                "ivcore",
                "--cuda-device-only",
                _get_iluvatar_gpu_arch(),
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

    class _CorexSourceFile:

        def read_text(self):
            return source_text

    corex_fn = CorexJITFunction(
        fn=None,
        file=_CorexSourceFile(),
        extern_func_name=entry.get("extern_func_name"),
        deferred=True,
    )
    return corex_fn.make_llvm(context)
