from __future__ import annotations
import os
import re
import shlex
import struct
from pathlib import Path
import subprocess
from typing import Any, Final

import torch

from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]
from triton.experimental.tle.raw.source_store import register_source

# TODO: We use cli tools to compile CUDA code temporarily, and plan to replace it with LLVM components Python bindings in the future.
CLANG = os.getenv("CLANG", "clang")
CLANG_FLAGS = shlex.split(os.getenv("CLANG_FLAGS", ""))

_USE_DEFERRED_LOWERING = False


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


def _get_cuda_gpu_arch() -> str:
    arch = os.getenv("TLE_CUDA_ARCH")
    if arch:
        return f"--cuda-gpu-arch={arch}"
    major, minor = torch.cuda.get_device_capability()
    return f"--cuda-gpu-arch=sm_{major}{minor}"


class CUDAJITFunction(object):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k != "extern_func_name"})
        self.fn: Final[Any] = fn
        self.code: Final[str] = file.read_text()
        self.region_dialect: Final[str] = "cuda"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"
        self.source_file: Final[str] = str(file)
        self.extern_func_name = kwargs.get("extern_func_name", None)
        self.__triton_builtin__: Final[bool] = True

    @property
    def deferred(self) -> bool:
        return _USE_DEFERRED_LOWERING

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError(
                "deferred tle_raw CUDA source requires extern_func_name= "
                "(the device function symbol in the .cu file)"
            )
        return register_source(
            region_dialect=self.region_dialect,
            extern_func_name=self.extern_func_name,
            source=self.code,
            hint=hint,
            extra={"source_file": self.source_file},
        )

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_by_llvm_func(
            llvm,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

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
                CLANG,
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
                *CLANG_FLAGS,
            ],
            input=self.code.encode(),
            capture_output=True,
        )
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")
        llvm_context = llvm.context()
        module = parse_llvm_ir(_sanitize_clang_ir(build.stdout.decode()), llvm_context, mlir_context)
        return f"{module}"
