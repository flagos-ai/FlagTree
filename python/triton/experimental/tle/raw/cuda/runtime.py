from __future__ import annotations
import os
from pathlib import Path
import subprocess
from typing import Any, Final

from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]

# TODO: We use cli tools to compile CUDA code temporarily, and plan to replace it with LLVM components Python bindings in the future.
CLANG = os.getenv("CLANG", "clang")


class CUDAJITFunction(object):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Any] = fn
        self.code: Final[str] = file.read_text()
        self.__triton_builtin__: Final[bool] = True

    def make_llvm(self, mlir_context) -> str:
        build = subprocess.run(
            [
                CLANG,
                "-x",
                "cuda",
                "--cuda-device-only",
                "-emit-llvm",
                "-S",
                "-",
                "-o",
                "-",
            ],
            input=self.code.encode(),
            capture_output=True,
        )
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")
        llvm_context = llvm.context()
        module = parse_llvm_ir(build.stdout.decode(), llvm_context, mlir_context)
        return f"{module}"
