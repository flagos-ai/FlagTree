from __future__ import annotations
import os
from pathlib import Path
import subprocess
from typing import Any, Final
import torch

from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]

# TODO: We use cli tools to compile CUDA code temporarily, and plan to replace it with LLVM components Python bindings in the future.
CLANG = os.getenv("CLANG", "clang")
NVCC = os.getenv("NVCC", "nvcc")


class CUDAJITFunction(object):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__()
        self.fn: Final[Any] = fn
        self.code: Final[str] = file.read_text()
        self.file: Final[Path] = file
        self.libs = kwargs.get("library", {})
        self.__triton_builtin__: Final[bool] = True

    def make_llvm(self, mlir_context) -> str:
        build = subprocess.run(
            [
                CLANG,
                "-x",
                "cuda",
                "--cuda-device-only",
                "-emit-llvm",
                "-O2",
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

    def make_cubin(self):
        src = self.file
        dst = Path(src).with_suffix('.o')
        include_dirs = []
        for lib_name, lib_path in self.libs.items():
            # TODO: Remove the method of passing information by setting environment variables.
            os.environ[(lib_name + "_home").upper()] = lib_path
            include_dirs.append(os.path.join(lib_path, "include"))
        include_flags = [f"-I{inc_dir}" for inc_dir in include_dirs]
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = f"-arch=sm_{prop.major}{prop.minor}"
        build = subprocess.run([NVCC, "-rdc=true", arch, *include_flags, "--extended-lambda", "-c", "-o", dst, src],
                               capture_output=True)
        assert build.returncode == 0, (f"nvcc failed\nstderr:\n{build.stderr.decode()}")
        # TODO: Remove the method of passing information by setting environment variables.
        os.environ["USE_NVCC"] = 'True'
        os.environ["CUDA_CUBIN"] = str(dst)
        return
