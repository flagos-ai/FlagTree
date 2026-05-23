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
PTXAS = os.getenv("PTXAS", "ptxas")


class CUDAJITFunction(object):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__()
        self.fn: Final[Any] = fn
        self.code: Final[str] = file.read_text()
        self.file: Final[Path] = file
        self.libs = kwargs.get("library", None)
        self.macros = kwargs.get("macro", None)
        self.__triton_builtin__: Final[bool] = True

    def make_llvm(self, mlir_context) -> str:
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = f"--cuda-gpu-arch=sm_{prop.major}{prop.minor}"
        build = subprocess.run(
            [
                CLANG,
                "-x",
                "cuda",
                "--cuda-device-only",
                arch,
                "-emit-llvm",
                "-I/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/torch/include",
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
        
        macro_flags = []
        for macro_name, macro_value in self.macros.items():
            macro_flags.append(f"-D{macro_name}={macro_value}")
        
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = f"-arch=sm_{prop.major}{prop.minor}"
        
        # clang cubin
        # build1 = subprocess.run([
        #     CLANG, 
        #     "-fgpu-rdc", 
        #     "-c",
        #     "--cuda-device-only", 
        #     "--cuda-gpu-arch=sm_90", 
        #     "-O3", 
        #     *include_flags, 
        #     "-o", 
        #     dst, 
        #     src
        # ], capture_output=True)
        # print("clang cuda")
        # assert build1.returncode == 0, (f"clang failed\nstderr:\n{build1.stderr.decode()}")
        
        # nvcc -> cubin
        build = subprocess.run([NVCC, "-rdc=true", arch, "-O3", *include_flags, *macro_flags, "--extended-lambda", "-c", "-o", dst, src],
                               capture_output=True)
        assert build.returncode == 0, (f"nvcc failed\nstderr:\n{build.stderr.decode()}")
        
        # nvcc -> ptx -> ptxas -> cubin
        # max_reg_per_block = 65536
        # maxnreg = max_reg_per_block // (4 * 32)
        # NVCC_GENCODE = f"-gencode=arch=compute_{capability}{suffix},code={arch}"
        # nvcc_cmd = [
        #         NVCC, "-rdc=true", f"-maxrregcount={maxnreg}", "-ccbin", "g++", NVCC_GENCODE, 
        #         "-I/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/torch/include",
        #         fsrc.name, "-ptx", "-c", "-o", fptx.name
        #     ]
        
        # TODO: Remove the method of passing information by setting environment variables.
        os.environ["USE_NVCC"] = 'True'
        os.environ["CUDA_CUBIN"] = str(dst)
        return
